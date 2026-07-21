"""Material property definitions for the AERO-S MATERIAL section.

Each Material instance writes one MATERIAL line in the field order AERO-S
expects:

MATERIAL
[id] [area] [youngs_modulus] [poisson_ratio] [density] [convection_coeff]
    [conduction_coeff] [elem_thickness] [perimeter] [ref_temp] [cp]
    [coeff_therm_expans] [ixx] [iyy] [izz] [ymin] [ymax] [zmin] [zmax]
"""

from dataclasses import dataclass


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
