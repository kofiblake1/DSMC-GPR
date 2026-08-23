"""Unit tests for beam_projection.py - pure geometry, no gmsh/AERO-S deps.

Run with:
    /home/groups/cfarhat/kofib_envs/mesh_pipeline/bin/python -m pytest \
        notebooks/mesh_pipeline/tests/test_beam_projection.py -v
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from beam_projection import (
    build_beam_polyline,
    project_point_to_polyline,
    project_boundary_to_beam,
    read_sparta_surface,
)


# A single-segment beam along y, from (0,0) to (0,1).
STRAIGHT_BEAM = build_beam_polyline([(0.0, 0.0), (0.0, 1.0)])

W = 0.1  # half-width of the surrounding rectangle outline


def test_midspan_projection_analytic():
    r = project_point_to_polyline((W, 0.5), STRAIGHT_BEAM)
    assert r.elem_num == 0
    assert r.xi == pytest.approx(0.5)
    assert not r.clamped
    assert r.gap == pytest.approx([W, 0.0])
    assert r.perp_distance == pytest.approx(W)


def test_quarter_span_negative_side():
    r = project_point_to_polyline((-W, 0.25), STRAIGHT_BEAM)
    assert r.xi == pytest.approx(0.25)
    assert r.gap == pytest.approx([-W, 0.0])
    assert not r.clamped


def test_node_exactly_at_beam_endpoint():
    r = project_point_to_polyline((0.0, 0.0), STRAIGHT_BEAM)
    assert r.xi == pytest.approx(0.0)
    assert r.gap == pytest.approx([0.0, 0.0])
    assert not r.clamped

    r2 = project_point_to_polyline((0.0, 1.0), STRAIGHT_BEAM)
    assert r2.xi == pytest.approx(1.0)
    assert r2.gap == pytest.approx([0.0, 0.0])
    assert not r2.clamped


def test_node_beyond_span_is_clamped():
    r = project_point_to_polyline((0.0, 1.5), STRAIGHT_BEAM)
    assert r.clamped
    assert r.xi == pytest.approx(1.0)
    assert r.xi_raw == pytest.approx(1.5)
    # clamped: gap includes the along-axis overshoot (0.5) plus zero
    # perpendicular offset in this analytic case.
    assert r.gap == pytest.approx([0.0, 0.5])
    assert r.perp_distance == pytest.approx(0.5)


def test_node_beyond_span_reject_mode_raises():
    with pytest.raises(ValueError):
        project_point_to_polyline((0.0, 1.5), STRAIGHT_BEAM, on_out_of_span="reject")


def test_degenerate_zero_length_segment_does_not_crash():
    segs = build_beam_polyline([(0.0, 0.0), (0.0, 0.0), (0.0, 1.0)])
    # segment 0 is degenerate (zero length); segment 1 is the real one.
    r = project_point_to_polyline((W, 0.75), segs, tol=1e-9)
    assert r.elem_num == 1
    assert r.xi == pytest.approx(0.75)
    assert not np.isnan(r.gap).any()


def test_multi_segment_boundary_selects_nearest():
    # An "L"-shaped beam: (0,0)->(0,1)->(1,1). A point near the corner but
    # closer to the second segment must pick element 1, with a small xi.
    segs = build_beam_polyline([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
    r = project_point_to_polyline((0.05, 1.05), segs)
    assert r.elem_num == 1
    assert 0.0 <= r.xi <= 0.2


def test_project_boundary_to_beam_batch():
    coords = np.array([(W, 0.0), (W, 0.5), (W, 1.0), (-W, 0.5)])
    results = project_boundary_to_beam(coords, STRAIGHT_BEAM)
    assert len(results) == 4
    assert [r.xi for r in results] == pytest.approx([0.0, 0.5, 1.0, 0.5])


def _write_synthetic_sparta_file(path):
    # Mirrors mesh_export._write_sparta_surface's exact format.
    pts = [(W, 0.0), (W, 1.0), (-W, 1.0), (-W, 0.0)]
    with open(path, "w") as f:
        f.write("Refined Surface (Ordered)\n\n")
        f.write(f"{len(pts)} points\n")
        f.write(f"{len(pts)} lines\n\n")
        f.write("Points\n\n")
        for i, (x, y) in enumerate(pts, 1):
            f.write(f"{i} {x} {y}\n")
        f.write("\nLines\n\n")
        n = len(pts)
        for i in range(n):
            nxt = (i + 1) % n
            f.write(f"{i + 1} {i + 1} {nxt + 1}\n")


def test_read_sparta_surface_round_trip(tmp_path):
    sparta_file = tmp_path / "rect_sparta.txt"
    _write_synthetic_sparta_file(sparta_file)

    node_ids, coords, lines = read_sparta_surface(str(sparta_file))
    assert list(node_ids) == [1, 2, 3, 4]
    assert coords.shape == (4, 2)
    assert coords[0] == pytest.approx([W, 0.0])
    assert lines == [(1, 1, 2), (2, 2, 3), (3, 3, 4), (4, 4, 1)]


def test_mapping_round_trip_via_beam_export(tmp_path):
    # Full round-trip: write a synthetic sparta file, run it through
    # beam_export.export_beam_coupled_mesh, re-parse the mapping file, and
    # check the re-parsed (elem, xi, gap) matches the in-memory projection.
    from materials import Material
    from beam_export import export_beam_coupled_mesh

    sparta_file = tmp_path / "rect_sparta.txt"
    _write_synthetic_sparta_file(sparta_file)

    beam_points = [(0.0, 0.0), (0.0, 1.0)]
    material = Material(id=1, area=1e-4, youngs_modulus=1e9, poisson_ratio=0.0,
                         density=1000.0, ixx=1e-9, iyy=1e-9, izz=1e-9)

    generated = export_beam_coupled_mesh(
        title="unittest_beam",
        out_dir=str(tmp_path / "out"),
        sparta_surface_file=str(sparta_file),
        beam_points=beam_points,
        materials=[material],
        default_material_id=1,
        fixed_dofs={0: [1, 2, 3, 4, 5, 6]},
    )

    with open(generated["beam_mapping"]) as f:
        content = f.read()

    assert "SPARTA_BEAM_MAPPING" in content
    assert "NODE_PROJECTIONS" in content
    assert "SPARTA_LINE_TOPOLOGY" in content
    assert "NODE_ADJACENT_ELEMENTS" in content

    # node 1 is at (W, 0.0) -> xi=0, gap=(W, 0)
    lines_in_section = []
    in_section = False
    for line in content.splitlines():
        if line.startswith("# SECTION NODE_PROJECTIONS"):
            in_section = True
            continue
        if in_section and line.startswith("# SECTION"):
            break
        if in_section and line and not line.startswith("#"):
            lines_in_section.append(line.split())

    assert len(lines_in_section) == 4
    node1 = lines_in_section[0]
    assert int(node1[0]) == 1
    assert float(node1[2]) == pytest.approx(0.0, abs=1e-8)
    assert float(node1[3]) == pytest.approx(W, abs=1e-8)
    assert float(node1[4]) == pytest.approx(0.0, abs=1e-8)
    assert int(node1[5]) == 1  # active flag
