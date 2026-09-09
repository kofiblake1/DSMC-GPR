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

from coupling import Coupling, NodeSelector
from displacement import DisplacementGroup
from materials import Material, RigidMaterial, is_rigid_struct_elem_code
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

# -----------------------------------------------------------------------------
# ELEMENT TYPE
# -----------------------------------------------------------------------------
# "tri"  - 3-node triangles (default AERO-S TOPO codes: struct=4, heat=46).
# "quad" - 4-node quadrilaterals (gmsh recombines the mesh into quads).
#          Verify HEAT_ELEM_CODE below against your AERO-S element library -
#          the quad heat element code varies with convection/radiation setup
#          (e.g. 46/48/58/4646).
ELEMENT_TYPE = "tri"  # "tri" or "quad" -- controls gmsh mesh generation only

if ELEMENT_TYPE == "tri":
    STRUCT_ELEM_CODE = 4
    HEAT_ELEM_CODE = 46
elif ELEMENT_TYPE == "quad":
    STRUCT_ELEM_CODE = 2
    HEAT_ELEM_CODE = 48  # TODO: confirm against your AERO-S heat element library
else:
    raise ValueError(f"unknown ELEMENT_TYPE: {ELEMENT_TYPE!r}")

# Set True to make every element a rigid element instead: overrides
# STRUCT_ELEM_CODE with RIGID_STRUCT_ELEM_CODE (AERO-S codes 65-76, excluding
# 72 and 75) and switches the MATERIAL section below to the simple "id
# CONMAT" line. The gmsh mesh itself (tri/quad, per ELEMENT_TYPE above) is
# unaffected.
USE_RIGID_ELEMENT = False
RIGID_STRUCT_ELEM_CODE = 65  # only used when USE_RIGID_ELEMENT is True

if USE_RIGID_ELEMENT:
    if not is_rigid_struct_elem_code(RIGID_STRUCT_ELEM_CODE):
        raise ValueError(
            f"RIGID_STRUCT_ELEM_CODE={RIGID_STRUCT_ELEM_CODE!r} is not a rigid "
            "element code (must be 65-76, excluding 72 and 75)."
        )
    STRUCT_ELEM_CODE = RIGID_STRUCT_ELEM_CODE

gmsh.model.geo.synchronize()
if ELEMENT_TYPE == "quad":
    gmsh.model.mesh.setRecombine(2, surface)
gmsh.model.mesh.generate(2)

# -----------------------------------------------------------------------------
# VISUALIZATION
# -----------------------------------------------------------------------------
# Set True to open the native gmsh GUI right after meshing, so you can inspect
# the geometry/mesh before export continues. Closing the window resumes the
# script.
VISUALIZE = False

# -----------------------------------------------------------------------------
# MATERIAL
# -----------------------------------------------------------------------------
# Rigid elements (USE_RIGID_ELEMENT above) only need an id and the CONMAT
# keyword; regular elements use the full AERO-S MATERIAL line:
# MATERIAL id area youngs_modulus poisson_ratio density convection_coeff
#           conduction_coeff elem_thickness perimeter ref_temp cp
#           coeff_therm_expans ixx iyy izz ymin ymax zmin zmax
if USE_RIGID_ELEMENT:
    MATERIALS = [
        RigidMaterial(id=1),
        # RigidMaterial(id=1, density=..., thickness=...),  # adds "MASS density thickness"
        # Add more RigidMaterial(...) entries here if needed.
    ]
else:
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
# COUPLING MODE
# -----------------------------------------------------------------------------
# How the AERO-S side relates to the SPARTA boundary. SPARTA always gets the
# full 2D perimeter of the geometry; only the AERO-S side/mapping changes:
#   "default"           - 1:1 correspondence, full interior AERO-S mesh.
#   "zero_dimensional"  - collapse AERO-S to a single node; every SPARTA
#                         boundary element's load sums onto that one node.
#   "partial"           - AERO-S is a subset of the SPARTA boundary nodes
#                         selected by simple coordinate conditions, e.g.
#                         NodeSelector(axis="x", op="<=", value=0.0).
COUPLING_MODE = "partial"  # "default", "zero_dimensional", or "partial"

if COUPLING_MODE == "default":
    COUPLING = Coupling.default()
elif COUPLING_MODE == "zero_dimensional":
    ZERO_D_AERO_NODE_ID = 1  # user-chosen AERO-S node id
    COUPLING = Coupling.zero_dimensional(aero_node_id=ZERO_D_AERO_NODE_ID)
elif COUPLING_MODE == "partial":
    COUPLING = Coupling.partial([
        NodeSelector(axis="x", op="<=", value=0.0),
    ])
else:
    raise ValueError(f"unknown COUPLING_MODE: {COUPLING_MODE!r}")

# -----------------------------------------------------------------------------
# EXPORT
# -----------------------------------------------------------------------------
if VISUALIZE:
    gmsh.fltk.run()

generated_files = export_coupled_mesh(
    surface=surface,
    title=TITLE,
    out_dir=OUT_DIR,
    materials=MATERIALS,
    displacement_groups=DISPLACEMENT_GROUPS if DO_DISP else None,
    default_material_id=DEFAULT_MATERIAL_ID,
    struct_elem_code=STRUCT_ELEM_CODE,
    heat_elem_code=HEAT_ELEM_CODE,
    default_temperature=DEFAULT_TEMPERATURE,
    internal_scale_factor=INTERNAL_SCALE_FACTOR,
    geometry_description=geometry_description,
    coupling=COUPLING,
)

gmsh.finalize()

for label, path in generated_files.items():
    print(f"{label}: {path}")
