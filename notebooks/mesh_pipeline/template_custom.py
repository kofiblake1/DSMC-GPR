"""Template: build ANY geometry with gmsh, then export AERO-S + SPARTA + mapping files.

This is the general-purpose template for shapes that aren't a plain rectangle
or circle (airfoils, aircraft cross-sections, arbitrary polygons, ...). Build
the geometry directly with the gmsh API in the GEOMETRY section below, end up
with a single meshed plane surface tag in `surface`, then fill in MATERIAL and
DISPLACEMENT exactly as in the other templates.

A worked NACA 0012 airfoil example is included — delete it and build whatever
geometry you need in its place. Run:
    python template_custom.py
"""

import gmsh
import numpy as np

from displacement import DisplacementGroup
from materials import Material
from mesh_export import export_coupled_mesh

# -----------------------------------------------------------------------------
# GEOMETRY
# -----------------------------------------------------------------------------
TITLE = "naca0012"
OUT_DIR = TITLE

# If your geometry needs very small physical coordinates, gmsh can lose
# numerical stability. Build the geometry at physical_units * INTERNAL_SCALE_FACTOR
# and export_coupled_mesh will divide back down to physical units. Leave at
# 1.0 if your coordinates are already a reasonable size (as in this example).
INTERNAL_SCALE_FACTOR = 1.0

gmsh.initialize()
gmsh.model.add(TITLE)

# --- Worked example: NACA 0012 airfoil at an angle of attack ---------------
# Replace everything in this block with your own geometry. You just need to
# end up with a meshed 2D plane surface tag assigned to `surface`.

CHORD = 1e-1
AOA_DEG = -6.0
LC = CHORD * 0.005  # mesh size control


def _naca0012(chord, npts=200):
    t = 0.12
    x = np.linspace(0, 1, npts)
    yt = 5 * t * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1015 * x**4
    )
    # upper/lower surfaces, combined into one closed loop
    xu, yu = x[::-1], yt[::-1]
    xl, yl = x[1:], -yt[1:]
    X = np.concatenate([xu, xl])
    Y = np.concatenate([yu, yl])
    return chord * X, chord * Y


def _rotate(x, y, aoa_deg):
    a = np.deg2rad(aoa_deg)
    return x * np.cos(a) - y * np.sin(a), x * np.sin(a) + y * np.cos(a)


x, y = _naca0012(chord=CHORD)
x = x - CHORD / 2  # center on the quarter-chord-ish origin
x, y = _rotate(x, y, AOA_DEG)

points = [gmsh.model.geo.addPoint(float(xi), float(yi), 0, LC) for xi, yi in zip(x, y)]
curve = gmsh.model.geo.addSpline(points + [points[0]])
curve_loop = gmsh.model.geo.addCurveLoop([curve])
surface = gmsh.model.geo.addPlaneSurface([curve_loop])

geometry_description = f"NACA0012 airfoil: chord={CHORD}, aoa={AOA_DEG} deg, lc={LC}"

# --- End example -------------------------------------------------------------

gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(2)

# -----------------------------------------------------------------------------
# MATERIAL
# -----------------------------------------------------------------------------
# MATERIAL id area youngs_modulus poisson_ratio density convection_coeff
#           conduction_coeff elem_thickness perimeter ref_temp cp
#           coeff_therm_expans ixx iyy izz ymin ymax zmin zmax
MATERIALS = [
    Material(
        id=1,
        area=0.0,
        youngs_modulus=15e9,
        poisson_ratio=0.3,
        density=3e3,
        convection_coeff=0.0,
        conduction_coeff=3.0,
        elem_thickness=1.0,
        perimeter=0.0,
        ref_temp=0.0,
        cp=1000.0,
        coeff_therm_expans=0.0,
    ),
    # Add more Material(...) entries here if the mesh needs multiple materials.
]
DEFAULT_MATERIAL_ID = 1  # material id referenced by ATTR for every element
DEFAULT_TEMPERATURE = 300.0

# -----------------------------------------------------------------------------
# DISPLACEMENT BOUNDARY CONDITIONS
# -----------------------------------------------------------------------------
# Each group pins the listed DOFs (1=x, 2=y) to zero for every node matching
# axis <op> value, e.g. "all nodes with y < -5". Set DO_DISP = False to skip
# writing a DISP section entirely.
DO_DISP = False
DISPLACEMENT_GROUPS = [
    DisplacementGroup(axis="y", op="==", value=0.0, dofs=[1, 2]),
    # DisplacementGroup(axis="x", op="==", value=0.0, dofs=[1, 2]),
    # DisplacementGroup(axis="y", op="<", value=-5.0, dofs=[1, 2]),
]

# -----------------------------------------------------------------------------
# EXPORT
# -----------------------------------------------------------------------------
generated_files = export_coupled_mesh(
    surface=surface,
    title=TITLE,
    out_dir=OUT_DIR,
    materials=MATERIALS,
    displacement_groups=DISPLACEMENT_GROUPS if DO_DISP else None,
    default_material_id=DEFAULT_MATERIAL_ID,
    default_temperature=DEFAULT_TEMPERATURE,
    internal_scale_factor=INTERNAL_SCALE_FACTOR,
    geometry_description=geometry_description,
)

gmsh.finalize()

for label, path in generated_files.items():
    print(f"{label}: {path}")
