"""Export a 1D AERO-S beam + its per-node mapping onto an existing 2D SPARTA
boundary outline.

This is the beam-coupling counterpart to mesh_export.export_coupled_mesh(),
for the case where the AERO-S structural model is a 1D beam centerline that
does NOT share nodes with the SPARTA surface (unlike every coupling mode in
mesh_export.py/coupling.py, which all assume the AERO-S side is built from
the same gmsh mesh, or a lumped point derived from it).

Given an already-exported {title}_sparta.txt (from mesh_export.py, any
Coupling mode - SPARTA always gets the full 2D perimeter regardless of mode)
and a user-specified beam polyline, export_beam_coupled_mesh() writes:

  - {title}_beam_struct.txt   AERO-S 1D beam mesh (NODE/TOPO type-6 EulerBeam/
                              ATTR/MATERIAL/EFRAMES/ITEMPERATURES/DISP)
  - {title}_beam_mapping.txt  3-section per-node SPARTA<->beam correspondence
  - {title}_beam_summary.txt  Human-readable record (xi/gap stats, clamped count)

This module has no gmsh dependency - it only needs the SPARTA boundary
points, which are already plain text by the time this runs.
"""

import os
from datetime import datetime

import numpy as np

from beam_projection import build_beam_polyline, project_boundary_to_beam, read_sparta_surface

EULERBEAM_ELEM_CODE = 6


def _axial_and_seed(p0, p1):
    axis = p1 - p0
    length = np.linalg.norm(axis)
    if length < 1e-14:
        raise ValueError(f"degenerate beam element with coincident nodes {p0}, {p1}")
    bx, by = axis / length
    seed = (-by, bx, 0.0)  # in-plane 90-degree rotation of the axial direction;
    # EulerBeam::buildFrame() always recomputes local-x from the node
    # positions and Gram-Schmidt-orthogonalizes local-z from this seed, so a
    # globally-fixed seed (e.g. (0,1,0)) would degenerate whenever a beam
    # segment runs parallel to it. (-by, bx, 0) guarantees local-z stays
    # parallel to global z for every element regardless of orientation.
    return length, seed


def write_beam_struct_mesh(
    mesh_filename,
    beam_nodes,
    materials,
    default_material_id,
    elem_code=EULERBEAM_ELEM_CODE,
    fixed_dofs=None,
    default_temperature=300.0,
):
    """Writes NODE / TOPO (elem_code, 2 nodes/elem) / ATTR / MATERIAL /
    EFRAMES / ITEMPERATURES / DISP for a 1D beam.

    beam_nodes: ordered list of (x, y) [or (x, y, z), z ignored], node tag =
        index + 1 (1-based), matching AERO-S convention.
    fixed_dofs: optional {node_index (0-based): [dof, ...]} - dof numbers are
        AERO-S's 1-6 (Xdisp,Ydisp,Zdisp,Xrot,Yrot,Zrot) convention.
    """
    pts = np.array([(p[0], p[1]) for p in beam_nodes], dtype=float)
    n_nodes = len(pts)
    n_elem = n_nodes - 1
    if n_elem < 1:
        raise ValueError("a beam needs at least 2 nodes (1 element)")

    with open(mesh_filename, "w") as f:
        f.write("NODE\n")
        for i, (x, y) in enumerate(pts, 1):
            f.write(f"{i} {x} {y} 0.0\n")

        f.write("*\n")
        f.write("TOPO\n")
        for e in range(n_elem):
            f.write(f"{e + 1} {elem_code} {e + 1} {e + 2}\n")

        f.write("*\n")
        f.write("ATTR\n")
        f.write(f"1 {n_elem} {default_material_id}\n")

        f.write("*\n")
        f.write("MATERIAL\n")
        for material in materials:
            f.write(material.to_line() + "\n")

        f.write("*\n")
        f.write("EFRAMES\n")
        for e in range(n_elem):
            _, seed = _axial_and_seed(pts[e], pts[e + 1])
            f.write(
                f"{e + 1} 0 0 0 {seed[0]} {seed[1]} {seed[2]} 0 0 0\n"
            )

        f.write("*\n")
        f.write("ITEMPERATURES\n")
        for i in range(1, n_nodes + 1):
            f.write(f"{i} {default_temperature}\n")

        if fixed_dofs:
            f.write("*\n")
            f.write("DISP\n")
            for node_index, dofs in sorted(fixed_dofs.items()):
                for dof in dofs:
                    f.write(f"{node_index + 1} {dof} 0\n")


def _build_node_adjacency(line_records):
    """line_records: list[(elem_id, node1, node2)], both 1-based.
    Returns {node_id: (prev_elem, next_elem)} - prev_elem is the line ending
    at node_id, next_elem is the line starting at node_id (both 1-based,
    0 if none found, which should only happen on a non-closed boundary)."""
    next_of, prev_of = {}, {}
    for elem_id, n1, n2 in line_records:
        next_of[n1] = elem_id
        prev_of[n2] = elem_id
    all_nodes = {n for _, n1, n2 in line_records for n in (n1, n2)}
    return {n: (prev_of.get(n, 0), next_of.get(n, 0)) for n in all_nodes}


def _write_beam_mapping(mapping_filename, title, node_ids, projections, line_records, adjacency):
    n = len(node_ids)
    with open(mapping_filename, "w") as f:
        f.write("# SPARTA_BEAM_MAPPING v1\n")
        f.write(f"# title: {title}\n")
        f.write(f"# sparta_boundary_nodes: {n}\n\n")

        f.write("# SECTION NODE_PROJECTIONS\n")
        f.write("# sparta_node_id  beam_elem  xi  gap_x  gap_y  active_flag\n")
        for node_id, proj in zip(node_ids, projections):
            f.write(
                f"{int(node_id)} {proj.elem_num} {proj.xi:.10f} "
                f"{proj.gap[0]:.10e} {proj.gap[1]:.10e} 1\n"
            )
        f.write("\n")

        f.write("# SECTION SPARTA_LINE_TOPOLOGY\n")
        f.write("# sparta_elem  sparta_node1  sparta_node2\n")
        for elem_id, n1, n2 in line_records:
            f.write(f"{elem_id} {n1} {n2}\n")
        f.write("\n")

        f.write("# SECTION NODE_ADJACENT_ELEMENTS\n")
        f.write("# sparta_node_id  sparta_elem_prev  sparta_elem_next\n")
        for node_id in node_ids:
            prev_elem, next_elem = adjacency[int(node_id)]
            f.write(f"{int(node_id)} {prev_elem} {next_elem}\n")


def _write_beam_summary(summary_filename, title, beam_points, materials, default_material_id,
                         fixed_dofs, projections, generated_files):
    xis = np.array([p.xi for p in projections])
    gaps = np.array([np.linalg.norm(p.gap) for p in projections])
    n_clamped = sum(1 for p in projections if p.clamped)

    with open(summary_filename, "w") as f:
        f.write(f"Beam mesh export summary: {title}\n")
        f.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n\n")

        f.write("Beam geometry\n-------------\n")
        f.write(f"Beam nodes: {len(beam_points)}\n")
        for i, p in enumerate(beam_points, 1):
            f.write(f"  {i}: {tuple(p)}\n")
        f.write("\n")

        f.write("Materials\n---------\n")
        for material in materials:
            f.write(f"{material.to_line()}\n")
        f.write(f"Default material id (ATTR): {default_material_id}\n\n")

        f.write("Fixed DOFs\n----------\n")
        if fixed_dofs:
            for node_index, dofs in sorted(fixed_dofs.items()):
                f.write(f"  beam node {node_index + 1}: dofs {dofs}\n")
        else:
            f.write("  (none)\n")
        f.write("\n")

        f.write("Projection stats\n----------------\n")
        f.write(f"SPARTA boundary nodes: {len(projections)}\n")
        f.write(f"xi range: [{xis.min():.6f}, {xis.max():.6f}]\n")
        f.write(f"gap magnitude: min={gaps.min():.6e} max={gaps.max():.6e} mean={gaps.mean():.6e}\n")
        f.write(f"Clamped (out-of-span) nodes: {n_clamped}/{len(projections)}\n\n")

        f.write("Generated files\n---------------\n")
        for label, path in generated_files.items():
            f.write(f"{label}: {path}\n")


def export_beam_coupled_mesh(
    title,
    out_dir,
    sparta_surface_file,
    beam_points,
    materials,
    default_material_id,
    fixed_dofs=None,
    max_gap_ratio=0.5,
    on_out_of_span="clamp",
    default_temperature=300.0,
):
    """Writes {title}_beam_struct.txt, {title}_beam_mapping.txt,
    {title}_beam_summary.txt. Returns dict of generated file paths.

    sparta_surface_file: path to an existing {title}_sparta.txt (produced by
        mesh_export.export_coupled_mesh, any Coupling mode - the SPARTA side
        is always the full 2D perimeter regardless of mode).
    beam_points: ordered list of (x, y) spanning the beam's centerline.
    max_gap_ratio: if a node's gap magnitude exceeds this fraction of its
        matched element's length, a warning is printed (not fatal) - catches
        "this beam isn't actually near this outline" mistakes early.
    on_out_of_span: passed through to beam_projection.project_point_to_polyline
        for every SPARTA node ("clamp" or "reject").
    """
    if not materials:
        raise ValueError("At least one Material must be provided.")

    os.makedirs(out_dir, exist_ok=True)

    node_ids, coords, line_records = read_sparta_surface(sparta_surface_file)
    segments = build_beam_polyline(beam_points)
    projections = project_boundary_to_beam(coords, segments, on_out_of_span=on_out_of_span)

    seg_lengths = [float(np.linalg.norm(s.p1 - s.p0)) for s in segments]
    for node_id, proj in zip(node_ids, projections):
        seg_len = seg_lengths[proj.elem_num]
        if seg_len > 0 and proj.perp_distance > max_gap_ratio * seg_len:
            print(
                f"*** WARNING: SPARTA node {node_id} has gap magnitude "
                f"{proj.perp_distance:.6e} > {max_gap_ratio} * element length "
                f"({seg_len:.6e}) on beam element {proj.elem_num} - check that "
                f"the beam is actually positioned inside/near this outline."
            )

    adjacency = _build_node_adjacency(line_records)

    struct_filename = os.path.join(out_dir, f"{title}_beam_struct.txt")
    mapping_filename = os.path.join(out_dir, f"{title}_beam_mapping.txt")
    summary_filename = os.path.join(out_dir, f"{title}_beam_summary.txt")

    write_beam_struct_mesh(
        struct_filename, beam_points, materials, default_material_id,
        fixed_dofs=fixed_dofs, default_temperature=default_temperature,
    )
    _write_beam_mapping(mapping_filename, title, node_ids, projections, line_records, adjacency)

    generated_files = {
        "beam_struct": struct_filename,
        "beam_mapping": mapping_filename,
    }
    _write_beam_summary(
        summary_filename, title, beam_points, materials, default_material_id,
        fixed_dofs, projections, generated_files,
    )
    generated_files["beam_summary"] = summary_filename

    return generated_files
