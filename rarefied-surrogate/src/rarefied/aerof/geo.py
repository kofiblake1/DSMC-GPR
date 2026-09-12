"""Write a gmsh .geo for an extruded 2D shape, and drive the gmsh CLI.

Stage [4a] of PIPELINE.md. Cluster-only: needs the gmsh CLI (see gmsh_env.sh).

Geometry conventions:

  * by default the 2D shape lies in the x-y plane, extruded one layer in +z,
    matching the FRG short-course mesh and AERO-F's freestream convention
    (see MeshParams.extrude_axis)
  * the outer boundary is a single circle, all of it one physical group

ORIENTATION: input curves are **clockwise** (the project convention, M13,
because SPARTA requires it). gmsh's Plane Surface wants its outer loop
counter-clockwise, so this module reverses the curve internally before
writing points. That reversal is a private implementation detail and is not
load-bearing: AERO-F repairs boundary-face and tet orientation itself at load
time (GeoSource.C:340 -> Elem::checkVolume), and did so for 11004 of 21152
faces on the reference circle run while producing correct physics. Station
indices are defined on the *input* (clockwise) curve, never on this reversal.

Physical group names encode a trailing surface id, because sower reads the
surface id from the text after the last underscore in the element-set name
(FluidDomain.C:202-217). Without it the id is 0 and the surface cannot be
targeted by an AERO-F SurfaceData block -- which matters because StickMoving
is an *adiabatic* wall (BcDef.h:23), and only a SurfaceData override promotes
it to isothermal. An adiabatic wall has zero heat flux by definition.

The side planes are SlipFixed rather than Symmetry deliberately: sower maps
"Symmetry" to code 6 (FluidDomain.C:63) but AERO-F defines 6 as
BC_POROUS_WALL_FIXED and 11 as BC_SYMMETRY (BcDef.h), with no remap found on
the read path. Code 2 (SlipFixed) is unambiguous and is physically identical
to a symmetry plane for a 2D-equivalent run.

No Recombine is used, so extruding a triangulated surface yields tetrahedra.
That is required: gmsh2top accepts only element types 2 (tri) and 4 (tet).
"""

import os
import subprocess
from dataclasses import dataclass, asdict, field

import numpy as np

from rarefied.geometry import curves as _curves

_HERE = os.path.dirname(os.path.abspath(__file__))
GMSH_WRAPPER = os.path.join(_HERE, "gmsh_env.sh")

# sower BC names -> (physical group name, physical tag). The tag is the gmsh
# physical id; the id sower actually uses is the integer after the underscore.
GROUP_WALL = ("StickMoving_1", 1)
GROUP_FARFIELD = ("InletFixed_2", 2)
GROUP_SIDE_LO = ("SlipFixed_3", 3)
GROUP_SIDE_HI = ("SlipFixed_4", 4)
GROUP_VOLUME = ("FluidMesh", 100)

SURFACE_GROUPS = [GROUP_WALL, GROUP_FARFIELD, GROUP_SIDE_LO, GROUP_SIDE_HI]


@dataclass
class MeshParams:
    """All lengths are expressed relative to the shape's characteristic
    diameter D (the diameter of the circle of equal area), so the sizing rule
    is scale invariant across a dataset."""

    farfield_factor: float = 20.0    # outer radius   = 20 D
    # 0.05 D, matching the FRG short-course mesh. Governs near-wall element
    # quality: the one-cell extrusion makes tets anisotropic when the span
    # differs greatly from the wall spacing. Measured on the reference circle,
    # aspect ratio median/max was 28.7/288 at 1.0 D versus 7.8/28.5 at 0.05 D.
    thickness_factor: float = 0.05
    size_max_factor: float = 1.0     # far-field target size = 1 D

    # PRIMARY near-wall spacing knob, as a fraction of D. Decision M4 requires
    # this to come from a wall-gradient convergence study (refine until wall
    # heat flux and shear stop changing), NOT from y+ and NOT from the station
    # count. That study has not been run, so case configs carry FILL_IN and the
    # fallback below applies.
    #
    # Fallback when 0: perimeter / n_surf, i.e. the wall cell matches the wall
    # point spacing. Convenient and self-consistent, but it couples mesh
    # resolution to station count, which M4 and M6 need to be independent --
    # so it is a fallback, not the documented path.
    size_min_factor: float = 0.0

    # Number of shared surface stations (M6's registration layer, ~30). Only
    # affects `stations()`; it does not influence the mesh. 0 => every wall node
    # is a station.
    n_station: int = 0

    # Target cell-to-cell growth ratio away from the wall. For a linear size
    # law h(d) = h0 + m*d, the ratio h(d+h)/h(d) at the wall is exactly 1 + m,
    # so the slope *is* the growth rate and DistMax follows from it:
    #     dist_max = (size_max - size_min) / growth_rate
    # Setting DistMax independently (NACA0012.geo uses 2.0 on a unit chord)
    # implies a growth rate of ~0.5, i.e. 50% per cell, which is far too
    # coarse a transition. 0.10-0.20 is the usual range.
    growth_rate: float = 0.15
    dist_max_factor: float = 0.0     # >0 overrides the derived dist_max

    # Which axis the one-cell extrusion runs along; the shape occupies the
    # other two. "z" puts the shape in x-y, which is what AERO-F's freestream
    # convention wants for a 2D-equivalent run: DistTimeState.C:321-323 gives
    #     u = V cos(alpha) cos(beta),  v = V cos(alpha) sin(beta),  w = V sin(alpha)
    # so with Alpha = 0 the flow lies in the x-y plane and w = 0, i.e. nothing
    # crosses the side planes. This also matches the FRG short-course mesh
    # (FolderCases1To4/sources/2DNaca.top, x-y with z in [0, 0.05]).
    # "y" reproduces pavery's NACA0012.geo instead (shape in x-z).
    extrude_axis: str = "z"
    algorithm: int = 6               # 6 = Frontal-Delaunay (better quality)
    optimize: bool = True

    # Override D instead of deriving it from the polygon's enclosed area. The
    # discrete polygon under-measures the area it approximates (a 300-gon on a
    # 1 m circle gives D = 0.99996), so for reference cases with an exact known
    # size set this explicitly and keep the derived lengths clean.
    reference_diameter: float = 0.0   # 0 => derive from the enclosed area

    def resolve(self, curve):
        """Concrete lengths for a specific curve."""
        d = self.reference_diameter if self.reference_diameter > 0.0 \
            else _curves.characteristic_length(curve)
        per = _curves.perimeter(curve)
        n = len(curve)
        if self.size_min_factor > 0.0:
            size_min = self.size_min_factor * d
            size_min_source = "size_min_factor"
        else:
            size_min = per / n
            size_min_source = "fallback: perimeter/n_surf (see M4 -- should be "
            size_min_source += "set from a wall-gradient convergence study)"
        size_max = self.size_max_factor * d
        if self.dist_max_factor > 0.0:
            dist_max = self.dist_max_factor * d
        else:
            dist_max = (size_max - size_min) / self.growth_rate
        return {
            "characteristic_diameter": d,
            "diameter_from_area": _curves.characteristic_length(curve),
            "perimeter": per,
            "n_surf": n,
            "n_station": self.n_station or n,
            "size_min": size_min,
            "size_min_source": size_min_source,
            "size_max": size_max,
            "dist_max": dist_max,
            # The growth ratio actually implied by the resolved sizes, so an
            # explicit dist_max override cannot silently degrade the grading.
            "implied_growth_rate": (size_max - size_min) / dist_max,
            "farfield_radius": self.farfield_factor * d,
            "thickness": self.thickness_factor * d,
            "extrude_axis": self.extrude_axis,
            # Indices of the two in-plane axes, so quality.py and quicklook.py
            # do not have to re-derive the orientation.
            "plane_axes": (0, 1) if self.extrude_axis == "z" else (0, 2),
            "extrude_axis_index": 2 if self.extrude_axis == "z" else 1,
        }


def stations(curve, n_station=0):
    """Indices into `curve` of the shared surface stations (M6).

    Stations are a coarse, evenly-spaced subsample of the wall points, so every
    station id is an *exact mesh node* -- stage [4c] can read the base state at
    a station directly, with no interpolation. Index 0 is the +x-ray anchor
    established by `curves.anchor_origin`, so ids are reproducible across
    shapes and comparable to the SPARTA side.

    Deliberately separate from mesh resolution: `n_surf` sets how well the wall
    is resolved (M4's convergence study), `n_station` sets how many points the
    two solvers register against (M6's ~30). Returns all wall indices when
    n_station is 0 or >= n_surf.
    """
    n = len(curve)
    if n_station <= 0 or n_station >= n:
        return np.arange(n)
    step = n / float(n_station)
    return np.unique(np.floor(np.arange(n_station) * step).astype(int))


def write_geo(path, curve, params=None):
    """Write the .geo for `curve` (clockwise). Returns the resolved geometry dict.

    `curve` must be in the project's clockwise convention (M13). gmsh wants a
    counter-clockwise outer loop, so the point order is reversed here before
    writing; see the module docstring for why that is safe and why station
    indices stay defined on the clockwise input.
    """
    params = params or MeshParams()
    g = params.resolve(curve)
    n = g["n_surf"]
    R = g["farfield_radius"]

    # Reverse CW -> CCW for gmsh only. Kept local to this function so nothing
    # downstream can mistake it for the canonical ordering.
    if _curves.signed_area(curve) < 0.0:
        curve = curve[::-1]
        g["gmsh_point_order"] = "reversed from clockwise input for gmsh"
    else:
        g["gmsh_point_order"] = "input was already counter-clockwise"

    # Point/curve tag layout. Everything is explicit so the Extrude return
    # list can be indexed deterministically (see below).
    p_shape = list(range(1, n + 1))            # 1 .. n
    l_shape = list(range(1, n + 1))            # 1 .. n
    p_ctr = n + 1
    p_far = [n + 2, n + 3, n + 4, n + 5]
    l_far = [n + 1, n + 2, n + 3, n + 4]

    # Map plane coords (u, v) into 3D, and the extrusion vector.
    if g["extrude_axis"] == "z":
        def xyz(u, v):
            return (u, v, 0.0)
        ext = "0.0, 0.0, %.17g" % g["thickness"]
        plane_label = "x-y plane, z = 0"
    else:
        def xyz(u, v):
            return (u, 0.0, v)
        ext = "0.0, %.17g, 0.0" % g["thickness"]
        plane_label = "x-z plane, y = 0"

    L = []
    a = L.append
    a("// Extruded 2D shape mesh -- generated by rarefied.aerof.geo")
    a("// Do not edit by hand; regenerate from the curve instead.")
    a("SetFactory(\"Built-in\");")
    a("")
    a("// ---- shape boundary (%s) ----" % plane_label)
    for i, (u, v) in enumerate(curve):
        a("Point(%d) = {%.17g, %.17g, %.17g};" % ((p_shape[i],) + xyz(u, v)))
    a("")
    for i in range(n):
        a("Line(%d) = {%d, %d};" % (l_shape[i], p_shape[i], p_shape[(i + 1) % n]))
    a("")
    # Two nodes per segment pins the wall nodes to exactly the curve points,
    # so the wall discretisation is exactly n_surf points at known locations.
    a("Transfinite Curve {%d:%d} = 2;" % (l_shape[0], l_shape[-1]))
    a("Curve Loop(1) = {%d:%d};" % (l_shape[0], l_shape[-1]))
    a("")
    a("// ---- farfield circle ----")
    a("Point(%d) = {%.17g, %.17g, %.17g};" % ((p_ctr,) + xyz(0.0, 0.0)))
    a("Point(%d) = {%.17g, %.17g, %.17g};" % ((p_far[0],) + xyz(R, 0.0)))
    a("Point(%d) = {%.17g, %.17g, %.17g};" % ((p_far[1],) + xyz(0.0, R)))
    a("Point(%d) = {%.17g, %.17g, %.17g};" % ((p_far[2],) + xyz(-R, 0.0)))
    a("Point(%d) = {%.17g, %.17g, %.17g};" % ((p_far[3],) + xyz(0.0, -R)))
    for i in range(4):
        a("Circle(%d) = {%d, %d, %d};" % (
            l_far[i], p_far[i], p_ctr, p_far[(i + 1) % 4]))
    a("Curve Loop(2) = {%d, %d, %d, %d};" % tuple(l_far))
    a("")
    a("// outer loop first, shape as a hole")
    a("Plane Surface(1) = {2, 1};")
    a("")
    a("// ---- size field: fine at the wall, ramping outward ----")
    a("Field[1] = Distance;")
    a("Field[1].CurvesList = {%d:%d};" % (l_shape[0], l_shape[-1]))
    a("Field[1].Sampling = 3;")
    a("Field[2] = Threshold;")
    a("Field[2].InField = 1;")
    a("Field[2].SizeMin = %.17g;" % g["size_min"])
    a("Field[2].SizeMax = %.17g;" % g["size_max"])
    a("Field[2].DistMin = 0.0;")
    a("Field[2].DistMax = %.17g;" % g["dist_max"])
    a("Background Field = 2;")
    a("")
    a("// The field is the only size source; otherwise gmsh propagates the")
    a("// fine wall spacing outward and the element count explodes.")
    a("Mesh.MeshSizeExtendFromBoundary = 0;")
    a("Mesh.MeshSizeFromPoints = 0;")
    a("Mesh.MeshSizeFromCurvature = 0;")
    a("Mesh.Algorithm = %d;" % params.algorithm)
    a("Mesh.ElementOrder = 1;")
    a("Mesh.Optimize = %d;" % (1 if params.optimize else 0))
    a("Mesh.MshFileVersion = 2.2;  // gmsh2top cannot read 4.1")
    a("")
    a("// ---- extrude one layer in +%s (no Recombine => tetrahedra) ----"
      % g["extrude_axis"])
    a("out[] = Extrude {%s} { Surface{1}; Layers{1}; };" % ext)
    a("")
    # Extrude returns: out[0] = top surface, out[1] = volume, then one lateral
    # surface per boundary curve in the order the curve loops were declared on
    # Plane Surface(1) -- here loop 2 (4 farfield arcs) then loop 1 (n lines).
    # This ordering is ASSUMED; quality.py verifies it by comparing each
    # group's area against its analytic value and fails loudly if it is wrong.
    i_far0, i_far1 = 2, 5
    i_wall0, i_wall1 = 6, 6 + n - 1
    a("// out[0] = top surface, out[1] = volume, out[2:] = lateral surfaces")
    a("// lateral order follows Plane Surface(1) = {2, 1}: farfield then shape")
    a("Physical Surface(\"%s\", %d) = {out[{%d:%d}]};" % (
        GROUP_WALL[0], GROUP_WALL[1], i_wall0, i_wall1))
    a("Physical Surface(\"%s\", %d) = {out[{%d:%d}]};" % (
        GROUP_FARFIELD[0], GROUP_FARFIELD[1], i_far0, i_far1))
    a("Physical Surface(\"%s\", %d) = {1};" % (GROUP_SIDE_LO[0], GROUP_SIDE_LO[1]))
    a("Physical Surface(\"%s\", %d) = {out[0]};" % (GROUP_SIDE_HI[0], GROUP_SIDE_HI[1]))
    a("Physical Volume(\"%s\", %d) = {out[1]};" % (GROUP_VOLUME[0], GROUP_VOLUME[1]))
    a("")

    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")

    g["geo_file"] = path
    g["params"] = asdict(params)
    return g


def run_gmsh(geo_path, msh_path, log_path=None, timeout=3600):
    """Mesh a .geo into MSH 2.2 by invoking the gmsh CLI in its own module env.

    gmsh is never imported into this process: gmsh/4.10.1 forces gcc/10.1.0,
    which breaks the py312 numpy stack this pipeline runs on.
    """
    cmd = [GMSH_WRAPPER, "-3", geo_path, "-format", "msh2", "-o", msh_path]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, timeout=timeout)
    if log_path:
        with open(log_path, "w") as f:
            f.write(" ".join(cmd) + "\n\n" + proc.stdout)
    if proc.returncode != 0:
        tail = "\n".join(proc.stdout.strip().splitlines()[-25:])
        raise RuntimeError("gmsh failed (exit %d):\n%s" % (proc.returncode, tail))
    if not os.path.exists(msh_path):
        raise RuntimeError("gmsh reported success but wrote no mesh: %s" % msh_path)

    warnings = [ln for ln in proc.stdout.splitlines()
                if "Warning" in ln or "Error" in ln]
    return {"log": proc.stdout, "warnings": warnings}
