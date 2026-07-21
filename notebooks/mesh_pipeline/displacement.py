"""Displacement boundary condition groups for the AERO-S DISP section.

A DisplacementGroup pins a set of DOFs to zero displacement for every node
whose coordinate on a given axis satisfies a simple comparison, e.g.:

    DisplacementGroup(axis="x", op="==", value=0.0, dofs=[1, 2])   # x == 0
    DisplacementGroup(axis="y", op="<", value=-5.0, dofs=[1, 2])   # y < -5
"""

from dataclasses import dataclass

import numpy as np

_OPS = {
    "==": lambda col, value, tol: np.isclose(col, value, atol=tol),
    "<": lambda col, value, tol: col < value,
    "<=": lambda col, value, tol: col <= value,
    ">": lambda col, value, tol: col > value,
    ">=": lambda col, value, tol: col >= value,
}


@dataclass
class DisplacementGroup:
    axis: str  # "x" or "y"
    op: str  # one of "==", "<", "<=", ">", ">="
    value: float
    dofs: list
    tol: float = 1e-9

    def select(self, coords_2d):
        if self.axis not in ("x", "y"):
            raise ValueError(f"axis must be 'x' or 'y', got {self.axis!r}")
        if self.op not in _OPS:
            raise ValueError(f"op must be one of {sorted(_OPS)}, got {self.op!r}")
        col = coords_2d[:, 0] if self.axis == "x" else coords_2d[:, 1]
        return _OPS[self.op](col, self.value, self.tol)

    def describe(self):
        return f"{self.axis} {self.op} {self.value} -> pin dofs {self.dofs}"
