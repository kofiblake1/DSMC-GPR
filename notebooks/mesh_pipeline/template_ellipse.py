"""Template: elliptical domain -> AERO-S + SPARTA + mapping files.

Edit the GEOMETRY, MATERIAL, and DISPLACEMENT sections below, then run:
    python template_ellipse.py
"""

import gmsh

from coupling import Coupling
from displacement import DisplacementGroup
from geometry_ellipse import build_ellipse
from materials import Material, RigidMaterial, is_rigid_struct_elem_code
from mesh_export import export_coupled_mesh

# -----------------------------------------------------------------------------
# GEOMETRY
# -----------------------------------------------------------------------------
TITLE = "ELLIPSE_TEST"  # used for output file names
OUT_DIR = TITLE

SEMI_MAJOR = 0.01  # semi-major axis length (must be >= SEMI_MINOR)
SEMI_MINOR = 0.005  # semi-minor axis length
CENTER = (0.0, 0.0)
ANGLE_OF_ATTACK_DEG = 45.0  # rotation of the major axis from +x, ccw, about CENTER
LC_FRACTION = 1e-1  # mesh size as a fraction of the semi-minor axis

# -----------------------------------------------------------------------------
# ELEMENT TYPE
# -----------------------------------------------------------------------------
# "tri"  - 3-node triangles (default AERO-S TOPO codes: struct=71, heat=46).
# "quad" - 4-node quadrilaterals (gmsh recombines the mesh into quads).
#          Verify HEAT_ELEM_CODE below against your AERO-S element library -
#          the quad heat element code varies with convection/radiation setup
#          (e.g. 46/48/58/4646).
ELEMENT_TYPE = "tri"  # "tri" or "quad" -- controls gmsh mesh generation only

if ELEMENT_TYPE == "tri":
    STRUCT_ELEM_CODE = 73
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
USE_RIGID_ELEMENT = True
RIGID_STRUCT_ELEM_CODE = 73  # only used when USE_RIGID_ELEMENT is True

if USE_RIGID_ELEMENT:
    if not is_rigid_struct_elem_code(RIGID_STRUCT_ELEM_CODE):
        raise ValueError(
            f"RIGID_STRUCT_ELEM_CODE={RIGID_STRUCT_ELEM_CODE!r} is not a rigid "
            "element code (must be 65-76, excluding 72 and 75)."
        )
    STRUCT_ELEM_CODE = RIGID_STRUCT_ELEM_CODE

# -----------------------------------------------------------------------------
# VISUALIZATION
# -----------------------------------------------------------------------------
# Set True to open the native gmsh GUI right after meshing, so you can inspect
# the geometry/mesh before export continues. Closing the window resumes the
# script.
VISUALIZE = True

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
        RigidMaterial(id=1, density=1,thickness=1),
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
# full 2D perimeter of the ellipse; only the AERO-S side/mapping changes:
#   "default"          - 1:1 correspondence, full interior AERO-S mesh.
#   "zero_dimensional"  - collapse AERO-S to a single node (e.g. a rigid
#                         body); every SPARTA boundary element's load
#                         sums onto that one node.
#   "partial"           - AERO-S is a subset of the SPARTA boundary nodes
#                         selected by simple coordinate conditions.
COUPLING_MODE = "default"

if COUPLING_MODE == "default":
    COUPLING = Coupling.default()
elif COUPLING_MODE == "zero_dimensional":
    ZERO_D_AERO_NODE_ID = 2  # user-chosen AERO-S node id for the rigid body
    COUPLING = Coupling.zero_dimensional(aero_node_id=ZERO_D_AERO_NODE_ID)
elif COUPLING_MODE == "partial":
    from coupling import NodeSelector

    COUPLING = Coupling.partial([
        NodeSelector(axis="x", op=">=", value=0.0),
    ])
else:
    raise ValueError(f"unknown COUPLING_MODE: {COUPLING_MODE!r}")

# -----------------------------------------------------------------------------
# BUILD + EXPORT
# -----------------------------------------------------------------------------
surface, internal_scale_factor = build_ellipse(
    TITLE,
    SEMI_MAJOR,
    SEMI_MINOR,
    center=CENTER,
    angle_of_attack_deg=ANGLE_OF_ATTACK_DEG,
    lc_fraction=LC_FRACTION,
    element_type=ELEMENT_TYPE,
)

geometry_description = (
    f"Ellipse: semi_major={SEMI_MAJOR}, semi_minor={SEMI_MINOR}, center={CENTER}, "
    f"angle_of_attack_deg={ANGLE_OF_ATTACK_DEG}, lc_fraction={LC_FRACTION}"
)

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
    internal_scale_factor=internal_scale_factor,
    geometry_description=geometry_description,
    coupling=COUPLING,
)

gmsh.finalize()

for label, path in generated_files.items():
    print(f"{label}: {path}")
