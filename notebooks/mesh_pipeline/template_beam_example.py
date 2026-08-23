"""Worked, runnable example of the 1D-beam <-> 2D-SPARTA-surface coupling path.

This is NOT a fill-in-the-blanks template like template_beam.py - it is a
permanent, documented, reproducible example. It builds a small rectangular
outline directly (no gmsh dependency) representing the thickness-t swept
cross-section of a body, projects it onto a straight 6-node beam centerline,
and writes the AERO-S beam mesh + per-node mapping file needed by
Hetero.d/SpartaBeamExchanger (AERO-S side) and fix_AERO_S_beam (SPARTA side).

The rest of the files needed for an actual coupled run (StructureFile,
coupled_sparta_input, config.txt, air.species/air.vss, run.sh) already exist
alongside this script's output in BEAM_COUPLING_EXAMPLE/ - see the README.md
there for how to run it and what to expect. This script only needs to be
re-run if you want to regenerate/modify the geometry.

Run:
    python template_beam_example.py
"""

from beam_export import export_beam_coupled_mesh
from materials import Material

TITLE = "beam_coupling_example"
OUT_DIR = "BEAM_COUPLING_EXAMPLE"

# -----------------------------------------------------------------------------
# SPARTA OUTLINE: a thin rectangle, half-width W (thickness direction, x) by
# half-span L (beam-axis direction, y), built directly rather than via gmsh -
# a beam-coupling case only needs the outline's ordered boundary points, and
# writing them by hand keeps this example gmsh-free and fast to regenerate.
# -----------------------------------------------------------------------------
W = 0.1   # half-width
L = 0.5   # half-span
NX = 2    # segments per short (width) side
NY = 8    # segments per long (span) side

points = []
for i in range(NX):                                  # bottom (y=-L)
    points.append((-W + 2 * W * i / NX, -L))
for i in range(NY):                                   # right (x=+W)
    points.append((W, -L + 2 * L * i / NY))
for i in range(NX):                                   # top (y=+L)
    points.append((W - 2 * W * i / NX, L))
for i in range(NY):                                   # left (x=-W)
    points.append((-W, L - 2 * L * i / NY))

n = len(points)
sparta_filename = f"{OUT_DIR}/{TITLE}_sparta.txt"
import os
os.makedirs(OUT_DIR, exist_ok=True)
with open(sparta_filename, "w") as f:
    f.write("Refined Surface (Ordered)\n\n")
    f.write(f"{n} points\n{n} lines\n\n")
    f.write("Points\n\n")
    for i, (x, y) in enumerate(points, 1):
        f.write(f"{i} {x} {y}\n")
    f.write("\nLines\n\n")
    for i in range(n):
        nxt = (i + 1) % n
        f.write(f"{i + 1} {i + 1} {nxt + 1}\n")

# -----------------------------------------------------------------------------
# BEAM CENTERLINE: straight, along y, spanning the same [-L, L] range as the
# outline, with 6 nodes (5 EulerBeam elements).
# -----------------------------------------------------------------------------
N_BEAM_NODES = 6
beam_points = [(0.0, -L + 2 * L * i / (N_BEAM_NODES - 1)) for i in range(N_BEAM_NODES)]

# -----------------------------------------------------------------------------
# MATERIAL: a stiff, arbitrary-units beam section. area/izz correspond to the
# rectangle's actual cross-section (2W wide, unit depth) - these are not
# tuned to any physical validation case, just picked to keep this example's
# deflection small, bounded, and quick to see change over a handful of
# coupling steps.
# -----------------------------------------------------------------------------
material = Material(
    id=1,
    area=2 * W * 1.0,
    youngs_modulus=2.0e5,
    poisson_ratio=0.0,
    density=1.0e2,
    ixx=1e-9,
    iyy=1e-9,
    # AERO-S's EulerBeam resists in-plane (local-y) bending via Izz, not Iyy
    # (see Element.d/Beam.d/modmstif6.f: ke(2,2) - the local-y translation
    # stiffness - uses Iz; local-y ends up as the in-plane/gap direction
    # given how EFRAMES + buildFrame() construct the local frame here).
    izz=(2 * W) ** 3 * 1.0 / 12.0,  # bending stiffness in-plane, unit depth
)

# Cantilever: pin all 6 dofs at the root (node 0); pin out-of-plane dofs
# (Zdisp=3, Xrot=4, Yrot=5) everywhere else to enforce strict 2D bending.
fixed_dofs = {0: [1, 2, 3, 4, 5, 6]}
for i in range(1, N_BEAM_NODES):
    fixed_dofs[i] = [3, 4, 5]

if __name__ == "__main__":
    generated = export_beam_coupled_mesh(
        title=TITLE,
        out_dir=OUT_DIR,
        sparta_surface_file=sparta_filename,
        beam_points=beam_points,
        materials=[material],
        default_material_id=1,
        fixed_dofs=fixed_dofs,
    )
    generated["sparta"] = sparta_filename
    for label, path in generated.items():
        print(f"{label}: {path}")
