"""Material property definitions for the AERO-S MATERIAL section.

Each Material instance writes one MATERIAL line in the field order AERO-S
expects:

MATERIAL
[id] [area] [youngs_modulus] [poisson_ratio] [density] [convection_coeff]
    [conduction_coeff] [elem_thickness] [perimeter] [ref_temp] [cp]
    [coeff_therm_expans] [ixx] [iyy] [izz] [ymin] [ymax] [zmin] [zmax]

Rigid elements (AERO-S struct element codes 65-76, excluding 72 and 75) use
RigidMaterial instead: their MATERIAL line is "[id] CONMAT", optionally
followed by "MASS [density] [thickness]" to give the rigid element mass
properties.
"""

from dataclasses import dataclass

RIGID_STRUCT_ELEM_CODES = frozenset(range(65, 77)) - {72, 75}


def is_rigid_struct_elem_code(struct_elem_code):
    """True if struct_elem_code is one of AERO-S's rigid element codes
    (65-76, excluding 72 and 75), which take a RigidMaterial instead of a
    full Material."""
    return struct_elem_code in RIGID_STRUCT_ELEM_CODES


@dataclass
class RigidMaterial:
    """MATERIAL line for a rigid element: an id and the CONMAT keyword,
    optionally followed by "MASS [density] [thickness]" -- set both
    density and thickness together to include it, or leave both None to
    omit it."""

    id: int
    density: float = None
    thickness: float = None

    def to_line(self):
        line = f"{self.id} CONMAT"
        if self.density is not None or self.thickness is not None:
            if self.density is None or self.thickness is None:
                raise ValueError(
                    "RigidMaterial needs both density and thickness for the "
                    "MASS keyword, or neither."
                )
            line += f" MASS {self.density} {self.thickness}"
        return line


@dataclass
class Material:
    id: int
    area: float = 0.0
    youngs_modulus: float = 0.0
    poisson_ratio: float = 0.0
    density: float = 0.0
    convection_coeff: float = 0.0
    conduction_coeff: float = 0.0
    elem_thickness: float = 0.0
    perimeter: float = 0.0
    ref_temp: float = 0.0
    cp: float = 0.0
    coeff_therm_expans: float = 0.0
    ixx: float = 0.0
    iyy: float = 0.0
    izz: float = 0.0
    ymin: float = 0.0
    ymax: float = 0.0
    zmin: float = 0.0
    zmax: float = 0.0

    def to_line(self):
        fields = [
            self.id,
            self.area,
            self.youngs_modulus,
            self.poisson_ratio,
            self.density,
            self.convection_coeff,
            self.conduction_coeff,
            self.elem_thickness,
            self.perimeter,
            self.ref_temp,
            self.cp,
            self.coeff_therm_expans,
            self.ixx,
            self.iyy,
            self.izz,
            self.ymin,
            self.ymax,
            self.zmin,
            self.zmax,
        ]
        return " ".join(str(field) for field in fields)
