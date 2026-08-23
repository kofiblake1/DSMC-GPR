"""1D-beam representation of BEAM_FINAL_QUAD_REV4, for BEAM_LOCKIN/PHASE_1.

REV4 (template_rectangle.py, ELEMENT_TYPE="quad") is a 2D-quad-shell
structural mesh of a 1 m x 0.02 m cantilever, coupled to SPARTA via the
coincident-mesh path. It cannot use AERO-S's NONLINEAR (geometric
nonlinearity) keyword - 2D quads aren't in the supported element list, and
requesting nonlinear-path stress/strain output on one crashes outright
(confirmed in BEAM_LOCKIN/PHASE_1/log.out: segfault in
Domain::getStressStrain under NonLinDynamic::dynamOutput).

This script builds the 1D EulerBeam (type 6, NONLINEAR-capable) equivalent:
same span (1 m), same in-plane bending thickness (0.02 m), same out-of-plane
depth (1 m), same material (E=618000 Pa, density=0.0058, poisson=0), same
cantilever root BC - but projected onto REV4's *existing* SPARTA outline
(BEAM_FINAL_QUAD_REV4_sparta.txt) rather than a freshly generated one, so
the coupled flow/grid setup already validated around that exact outline in
PHASE_1 does not need to change at all.

Generates two node-count variants (coarse/fine) purely for the modal
convergence check documented in BEAM_LOCKIN/PHASE_1/README_BEAM.md - the
fine one is what actually gets used for the coupled run.

Run:
    python template_beam_lockin_phase1.py
"""

import os

from beam_export import export_beam_coupled_mesh
from materials import Material

SPARTA_SURFACE_FILE = "BEAM_FINAL_QUAD_REV4/BEAM_FINAL_QUAD_REV4_sparta.txt"

# -----------------------------------------------------------------------------
# GEOMETRY - identical to BEAM_FINAL_QUAD_REV4 (template_rectangle.py):
# width=0.02 (in-plane bending thickness), height=1 (span), origin=(-0.01,-0.5)
# -----------------------------------------------------------------------------
T = 0.02      # thickness (bending direction)
L = 1.0       # span
Y0 = -0.5     # root y-coordinate (matches REV4's DISP "y == -0.5" root)

# -----------------------------------------------------------------------------
# MATERIAL - identical to BEAM_FINAL_QUAD_REV4's MATERIAL line
# (E=618000, poisson=0, density=0.0058, out-of-plane depth=1.0)
# -----------------------------------------------------------------------------
DEPTH = 1.0
E = 618000.0
NU = 0.0
RHO = 0.0058

# AERO-S's EulerBeam resists in-plane (local-y) bending via Izz, not Iyy -
# see Element.d/Beam.d/modmstif6.f (ke(2,2) uses Iz) and the note in
# template_beam_example.py. ixx/iyy are irrelevant here (torsion and
# out-of-plane bending are both fully suppressed by the fixed_dofs below)
# and are left as small placeholders.
material = Material(
    id=1,
    area=T * DEPTH,
    youngs_modulus=E,
    poisson_ratio=NU,
    density=RHO,
    ixx=1e-9,
    iyy=1e-9,
    izz=DEPTH * T ** 3 / 12.0,
)


def make_fixed_dofs(n_nodes):
    fixed_dofs = {0: [1, 2, 3, 4, 5, 6]}  # cantilever root
    for i in range(1, n_nodes):
        fixed_dofs[i] = [3, 4, 5]  # pin out-of-plane dofs everywhere else
    return fixed_dofs


def build(title, n_nodes):
    beam_points = [(0.0, Y0 + L * i / (n_nodes - 1)) for i in range(n_nodes)]
    out_dir = title
    os.makedirs(out_dir, exist_ok=True)
    generated = export_beam_coupled_mesh(
        title=title,
        out_dir=out_dir,
        sparta_surface_file=SPARTA_SURFACE_FILE,
        beam_points=beam_points,
        materials=[material],
        default_material_id=1,
        fixed_dofs=make_fixed_dofs(n_nodes),
    )
    for label, path in generated.items():
        print(f"{label}: {path}")
    return generated


if __name__ == "__main__":
    # Coarse variant: modal-convergence check only (see PHASE_1's README).
    build("BEAM_FINAL_QUAD_REV4_BEAM_COARSE", n_nodes=11)
    # Fine variant: what actually gets used for the coupled run.
    build("BEAM_FINAL_QUAD_REV4_BEAM", n_nodes=41)
