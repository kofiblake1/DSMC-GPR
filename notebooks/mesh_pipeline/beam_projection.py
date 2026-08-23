"""Point-to-segment projection of a 2D SPARTA boundary onto a 1D beam polyline.

This is the geometric core of the beam-coupling path: for every SPARTA
boundary node (a point on the 2D thickness-t outline of a body), find which
beam element it should be pinned to, at what fractional position along that
element (`xi`, 0..1), and what its fixed offset ("gap") from the beam axis
is in the reference configuration.

This replicates the projection formula used by the FRG `MATCHER` tool's
`Beam::match` routine (project onto the nearest segment, clamp `xi` to
[0, 1]), but is implemented directly here rather than shelling out to that
binary, since the SPARTA<->AERO-S conservative-transfer math this feeds
(EulerBeam::getFlLoad/computeDisp in AERO-S) only needs (elem, xi, gap) as
input - no other MATCHER machinery is required.

No gmsh/AERO-S-specific imports here - this module is pure geometry and is
unit-tested standalone (see tests/test_beam_projection.py).
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class BeamSegment:
    index: int  # 0-based, matches the AERO-S beam TOPO element index
    p0: np.ndarray  # (2,)
    p1: np.ndarray  # (2,)


@dataclass
class ProjectionResult:
    elem_num: int  # 0-based beam element index (best match)
    xi: float  # clamped to [0, 1]
    xi_raw: float  # unclamped natural coordinate, for diagnostics
    gap: np.ndarray  # (2,) point - projected_point, reference configuration
    perp_distance: float
    clamped: bool  # True if xi_raw was outside [0, 1]


def read_sparta_surface(sparta_filename):
    """Parse a {title}_sparta.txt file as written by
    mesh_export._write_sparta_surface(). Returns (node_ids, coords, lines):

      node_ids: np.ndarray[int], 1-based, shape (N,)
      coords:   np.ndarray[float], shape (N, 2)
      lines:    list[(line_id, node1, node2)], all 1-based
    """
    with open(sparta_filename) as f:
        raw_lines = [line.strip() for line in f]

    def _find(label):
        for i, line in enumerate(raw_lines):
            if line == label:
                return i
        raise ValueError(f"section {label!r} not found in {sparta_filename}")

    points_start = _find("Points") + 1
    lines_start = _find("Lines") + 1

    node_ids, coords = [], []
    for line in raw_lines[points_start:]:
        if not line:
            if node_ids:
                break
            continue
        parts = line.split()
        node_ids.append(int(parts[0]))
        coords.append((float(parts[1]), float(parts[2])))

    line_records = []
    for line in raw_lines[lines_start:]:
        if not line:
            if line_records:
                break
            continue
        parts = line.split()
        line_records.append((int(parts[0]), int(parts[1]), int(parts[2])))

    return np.array(node_ids, dtype=int), np.array(coords, dtype=float), line_records


def build_beam_polyline(points):
    """points: ordered list of (x, y) [or (x, y, z), z ignored] beam-node
    coordinates spanning the beam's centerline, len >= 2.

    Returns list[BeamSegment], element i connects beam node i to i+1 (0-based),
    matching the AERO-S TOPO element numbering convention used by
    beam_export.write_beam_struct_mesh().
    """
    if len(points) < 2:
        raise ValueError("a beam polyline needs at least 2 nodes")
    pts = np.array([(p[0], p[1]) for p in points], dtype=float)
    return [
        BeamSegment(index=i, p0=pts[i], p1=pts[i + 1])
        for i in range(len(pts) - 1)
    ]


def _project_to_segment(point, seg, tol):
    d = seg.p1 - seg.p0
    len2 = float(np.dot(d, d))
    if len2 < tol:
        xi_raw = 0.0
        closest = seg.p0
    else:
        xi_raw = float(np.dot(point - seg.p0, d) / len2)
        xi = min(1.0, max(0.0, xi_raw))
        closest = seg.p0 + xi * d
    dist = float(np.linalg.norm(point - closest))
    return xi_raw, closest, dist


def project_point_to_polyline(point, segments, tol=1e-9, on_out_of_span="clamp"):
    """Project a single 2D point onto the nearest segment of a beam polyline.

    on_out_of_span: "clamp" (default) - xi is clamped into [0, 1], gap grows
    to include the along-axis overshoot; "reject" - raises ValueError if the
    winning segment's xi_raw falls outside [0, 1]; any other value is treated
    as "clamp" (callers wanting a soft-warn policy should inspect the
    returned `clamped` flag themselves instead).

    Returns a ProjectionResult. point may be a length-2 or length-3 sequence
    (z, if present, is ignored - this pipeline is 2D-only).
    """
    point = np.asarray(point, dtype=float)[:2]

    best = None
    for seg in segments:
        xi_raw, closest, dist = _project_to_segment(point, seg, tol)
        if best is None or dist < best[2]:
            xi = min(1.0, max(0.0, xi_raw))
            best = (seg.index, xi, dist, xi_raw, closest)

    elem_num, xi, dist, xi_raw, closest = best
    clamped = xi_raw < -tol or xi_raw > 1.0 + tol

    if clamped and on_out_of_span == "reject":
        raise ValueError(
            f"point {tuple(point)} projects outside the beam's span "
            f"(xi_raw={xi_raw:.6g} on element {elem_num}); on_out_of_span='reject'"
        )

    gap = point - closest
    return ProjectionResult(
        elem_num=elem_num,
        xi=xi,
        xi_raw=xi_raw,
        gap=gap,
        perp_distance=dist,
        clamped=clamped,
    )


def project_boundary_to_beam(coords, segments, tol=1e-9, on_out_of_span="clamp"):
    """Vectorized convenience wrapper: coords is (N, 2), returns a list of
    ProjectionResult, one per row, in input order."""
    return [
        project_point_to_polyline(coords[i], segments, tol=tol, on_out_of_span=on_out_of_span)
        for i in range(len(coords))
    ]
