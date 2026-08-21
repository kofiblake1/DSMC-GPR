"""Template: rectangular domain -> AERO-S + SPARTA + mapping files.

Edit the GEOMETRY, MATERIAL, and DISPLACEMENT sections below, then run:
    python template_rectangle.py
"""

import gmsh

from coupling import Coupling, NodeSelector
from displacement import DisplacementGroup
from geometry_rectangle import build_rectangle
from materials import Material
from mesh_export import export_coupled_mesh

# -----------------------------------------------------------------------------
# GEOMETRY
# -----------------------------------------------------------------------------
TITLE = "BEAM_FINAL_QUAD_REV4"
OUT_DIR = TITLE

WIDTH = 2e-2
HEIGHT = 1
x = -WIDTH / 2
y = -HEIGHT / 2
ORIGIN = (x, y)
LC = 5e-3  # mesh characteristic length

# -----------------------------------------------------------------------------
# ELEMENT TYPE
# -----------------------------------------------------------------------------
# "tri"  - 3-node triangles (default AERO-S TOPO codes: struct=4, heat=46).
# "quad" - 4-node quadrilaterals (gmsh recombines the mesh into quads).
#          Verify HEAT_ELEM_CODE below against your AERO-S element library -
#          the quad heat element code varies with convection/radiation setup
#          (e.g. 46/48/58/4646).
ELEMENT_TYPE = "quad"  # "tri" or "quad"

if ELEMENT_TYPE == "tri":
    STRUCT_ELEM_CODE = 4
    HEAT_ELEM_CODE = 46
elif ELEMENT_TYPE == "quad":
    STRUCT_ELEM_CODE = 2
    HEAT_ELEM_CODE = 48  # TODO: confirm against your AERO-S heat element library
else:
    raise ValueError(f"unknown ELEMENT_TYPE: {ELEMENT_TYPE!r}")

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
# MATERIAL id area youngs_modulus poisson_ratio density convection_coeff
#           conduction_coeff elem_thickness perimeter ref_temp cp
#           coeff_therm_expans ixx iyy izz ymin ymax zmin zmax
MATERIALS = [
    Material(
        id=1,
        area=0.0,
        youngs_modulus=0.618E6,
        poisson_ratio=0.0,
        density=0.0058,
        convection_coeff=0.0,
        conduction_coeff=0.0,
        elem_thickness=1.0,
        perimeter=0.0,
        ref_temp=0.0,
        cp=0.0,
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
DO_DISP = True
DISPLACEMENT_GROUPS = [
    DisplacementGroup(axis="y", op="==", value=-HEIGHT / 2, dofs=[1, 2]),
    # DisplacementGroup(axis="x", op="==", value=0.0, dofs=[1, 2]),
    # DisplacementGroup(axis="y", op="<", value=-5.0, dofs=[1, 2]),
]

# -----------------------------------------------------------------------------
# COUPLING MODE
# -----------------------------------------------------------------------------
# How the AERO-S side relates to the SPARTA boundary. SPARTA always gets the
# full 2D perimeter of the box; only the AERO-S side/mapping changes:
#   "default"           - 1:1 correspondence, full interior AERO-S mesh.
#   "zero_dimensional"  - collapse AERO-S to a single node; every SPARTA
#                         boundary element's load sums onto that one node.
#   "partial"           - AERO-S is a subset of the SPARTA boundary nodes
#                         selected by simple coordinate conditions. This is
#                         the Euler-Bernoulli beam case: SPARTA has no native
#                         1D representation, so it gets the full 2D box, and
#                         only the top edge of that box (y == HEIGHT) is kept
#                         as the 1D AERO-S beam. Only the SPARTA boundary
#                         elements along that top edge are active.
COUPLING_MODE = "default"

if COUPLING_MODE == "default":
    COUPLING = Coupling.default()
elif COUPLING_MODE == "zero_dimensional":
    ZERO_D_AERO_NODE_ID = 1  # user-chosen AERO-S node id
    COUPLING = Coupling.zero_dimensional(aero_node_id=ZERO_D_AERO_NODE_ID)
elif COUPLING_MODE == "partial":
    COUPLING = Coupling.partial([
        NodeSelector(axis="x", op=">=", value=ORIGIN[0] + WIDTH),
        NodeSelector(axis="y", op=">=", value=ORIGIN[1] + HEIGHT),
        # NodeSelector(axis="x", op="==", value=ORIGIN[0]),  # OR'd with the above
    ])
else:
    raise ValueError(f"unknown COUPLING_MODE: {COUPLING_MODE!r}")

# -----------------------------------------------------------------------------
# BUILD + EXPORT
# -----------------------------------------------------------------------------
surface = build_rectangle(TITLE, WIDTH, HEIGHT, origin=ORIGIN, lc=LC, element_type=ELEMENT_TYPE)

geometry_description = (
    f"Rectangle: width={WIDTH}, height={HEIGHT}, origin={ORIGIN}, lc={LC}"
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
    internal_scale_factor=1.0,
    geometry_description=geometry_description,
    coupling=COUPLING,
)

gmsh.finalize()

for label, path in generated_files.items():
    print(f"{label}: {path}")
