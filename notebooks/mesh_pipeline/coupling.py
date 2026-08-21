"""Coupling-mode configuration for export_coupled_mesh.

Three modes control how the FEM (AERO-S) side relates to the DSMC (SPARTA)
boundary mesh. In every mode SPARTA always gets the full 2D perimeter
boundary; only the AERO-S side and the mapping between the two changes:

  default (Coupling.default())
      One-to-one: every SPARTA boundary node has its own AERO-S node, and the
      AERO-S mesh is the full interior mesh. Original behavior.

  zero_dimensional (Coupling.zero_dimensional(aero_node_id))
      The AERO-S side collapses to a single user-supplied node. Every SPARTA
      boundary element routes its load to that one node (e.g. a rigid body
      receiving the net aerodynamic load).

  partial (Coupling.partial(node_selectors))
      The AERO-S side is a subset of the SPARTA boundary nodes chosen by
      simple coordinate conditions (e.g. the top edge of a box standing in
      for a 1D beam, since SPARTA has no native 1D representation). Only
      SPARTA elements whose two nodes are both in the subset are active.
"""

from dataclasses import dataclass, field

import numpy as np

_OPS = {
    "==": lambda col, value, tol: np.isclose(col, value, atol=tol),
    "<": lambda col, value, tol: col < value,
    "<=": lambda col, value, tol: col <= value,
    ">": lambda col, value, tol: col > value,
    ">=": lambda col, value, tol: col >= value,
}


@dataclass
class NodeSelector:
    """Selects nodes whose coordinate on `axis` satisfies `axis op value`."""

    axis: str  # "x" or "y"
    op: str  # one of "==", "<", "<=", ">", ">="
    value: float
    tol: float = 1e-9

    def select(self, coords_2d):
        if self.axis not in ("x", "y"):
            raise ValueError(f"axis must be 'x' or 'y', got {self.axis!r}")
        if self.op not in _OPS:
            raise ValueError(f"op must be one of {sorted(_OPS)}, got {self.op!r}")
        col = coords_2d[:, 0] if self.axis == "x" else coords_2d[:, 1]
        return _OPS[self.op](col, self.value, self.tol)

    def describe(self):
        return f"{self.axis} {self.op} {self.value}"


@dataclass
class Coupling:
    """Coupling-mode configuration. Build with the classmethods below rather
    than the constructor directly."""

    mode: str
    aero_node_id: int = None
    aero_node_coords: tuple = None
    node_selectors: list = field(default_factory=list)

    @classmethod
    def default(cls):
        return cls(mode="default")

    @classmethod
    def zero_dimensional(cls, aero_node_id, aero_node_coords=None):
        """aero_node_coords: optional (x, y) for the single AERO-S node;
        defaults to the centroid of the SPARTA boundary."""
        return cls(
            mode="zero_dimensional",
            aero_node_id=aero_node_id,
            aero_node_coords=aero_node_coords,
        )

    @classmethod
    def partial(cls, node_selectors):
        """node_selectors: list[NodeSelector], OR-combined - a boundary node
        is part of the AERO-S subset if it satisfies ANY selector."""
        if not node_selectors:
            raise ValueError("partial coupling needs at least one NodeSelector")
        return cls(mode="partial", node_selectors=list(node_selectors))

    def select_subset(self, coords_2d):
        """OR-combine all node_selectors into one boolean mask over coords_2d."""
        mask = np.zeros(len(coords_2d), dtype=bool)
        for selector in self.node_selectors:
            mask |= selector.select(coords_2d)
        return mask

    def describe(self):
        if self.mode == "default":
            return "default: 1:1 SPARTA<->AERO-S node correspondence, full interior AERO-S mesh"
        if self.mode == "zero_dimensional":
            return (
                f"zero_dimensional: all SPARTA boundary elements summed to a single "
                f"AERO-S node (id={self.aero_node_id})"
            )
        if self.mode == "partial":
            conditions = " OR ".join(s.describe() for s in self.node_selectors)
            return (
                f"partial: AERO-S subset = SPARTA boundary nodes where {conditions}; "
                "only SPARTA elements with both nodes in the subset are active"
            )
        raise ValueError(f"unknown coupling mode: {self.mode!r}")
