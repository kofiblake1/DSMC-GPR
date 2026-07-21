"""Template: rectangular domain -> AERO-S + SPARTA + mapping files.

Edit the GEOMETRY, MATERIAL, and DISPLACEMENT sections below, then run:
    python template_rectangle.py
"""

import gmsh

from displacement import DisplacementGroup
from geometry_rectangle import build_rectangle
from materials import Material
from mesh_export import export_coupled_mesh

# -----------------------------------------------------------------------------
# GEOMETRY
# -----------------------------------------------------------------------------
TITLE = "beam"
OUT_DIR = TITLE

WIDTH = 2.49e-8
HEIGHT = 2.49e-6
ORIGIN = (0.0, 0.0)
LC = 1e-8  # mesh characteristic length

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
# BUILD + EXPORT
# -----------------------------------------------------------------------------
surface = build_rectangle(TITLE, WIDTH, HEIGHT, origin=ORIGIN, lc=LC)

geometry_description = (
    f"Rectangle: width={WIDTH}, height={HEIGHT}, origin={ORIGIN}, lc={LC}"
)

generated_files = export_coupled_mesh(
    surface=surface,
    title=TITLE,
    out_dir=OUT_DIR,
    materials=MATERIALS,
    displacement_groups=DISPLACEMENT_GROUPS if DO_DISP else None,
    default_material_id=DEFAULT_MATERIAL_ID,
    default_temperature=DEFAULT_TEMPERATURE,
    internal_scale_factor=1.0,
    geometry_description=geometry_description,
)

gmsh.finalize()

for label, path in generated_files.items():
    print(f"{label}: {path}")
