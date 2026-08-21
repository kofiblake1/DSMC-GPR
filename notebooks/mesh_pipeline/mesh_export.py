"""Export a meshed gmsh surface into the coupled SPARTA/AERO-S file set.

Given a gmsh plane surface (already meshed) plus material and displacement
config, export_coupled_mesh() writes:

  - {title}_aero_struct.txt  AERO-S structural mesh (NODE/TOPO/ATTR/MATERIAL/
                             ITEMPERATURES/DISP)
  - {title}_aero_heat.txt    Same mesh with the heat-problem element code
  - {title}_sparta.txt       Ordered boundary surface for SPARTA
  - {title}_mapping.txt      SPARTA <-> AERO-S element correspondence
  - {title}_surface.stl      STL of the boundary surface (for visualization)
  - {title}_coupling.png     SPARTA vs AERO-S comparison, active/inactive
  - {title}_summary.txt      Human-readable record of every input used

Node ordering matches the original notebook exactly: NODE/ITEMPERATURES use
gmsh's native node-tag order, and the boundary (SPARTA/mapping/STL) is sorted
clockwise by angle from the boundary centroid.

The mapping file has a single section, one line per SPARTA boundary element:

    # SPARTA_ELEMENT_ID SPARTA_NODE_ID_1 SPARTA_NODE_ID_2 AERO-S_NODE_ID... ACTIVE_FLAG

The AERO-S node id list is variable-length (0, 1, or 2 ids) so this one
format covers every coupling mode:
  - default: always 2 AERO-S ids, flag always 1.
  - zero_dimensional: always 1 AERO-S id (the single node), flag always 1.
  - partial: 2 AERO-S ids and flag 1 when both of the element's nodes are in
    the AERO-S subset; no AERO-S ids and flag 0 otherwise.

By default (coupling=None, i.e. Coupling.default()) SPARTA and AERO-S share
the exact same geometry and every boundary element is active. Pass a
Coupling.zero_dimensional(...) or Coupling.partial(...) (see coupling.py) to
relax that: SPARTA always keeps the full 2D perimeter, but the AERO-S side
and the mapping file's SPARTA<->AERO-S correspondence change to match.
"""

import os
from datetime import datetime

import gmsh
import numpy as np

from coupling import Coupling
from coupling_plot import write_coupling_plot


def _get_all_nodes(internal_scale_factor):
    all_node_tags, all_node_coords, _ = gmsh.model.mesh.getNodes()
    all_node_coords = np.array(all_node_coords).reshape(-1, 3)
    physical_all_node_coords = all_node_coords / internal_scale_factor
    return all_node_tags, physical_all_node_coords


def _get_boundary_nodes(surface, all_node_tags, physical_all_node_coords):
    boundary_entities = gmsh.model.getBoundary([(2, surface)], oriented=False)

    dim = 1  # 1D elements
    element_types, element_tags, node_tags = gmsh.model.mesh.getElements(dim)

    boundary_elements = []
    boundary_node_tags = set()
    for etype, etags, enodes in zip(element_types, element_tags, node_tags):
        for i, e in enumerate(etags):
            node_pair = enodes[i * 2: (i + 1) * 2]  # each line has 2 nodes
            boundary_elements.append((e, node_pair))
            boundary_node_tags.update(node_pair)
    boundary_node_tags = np.array(list(boundary_node_tags))

    boundary_indices = np.isin(all_node_tags, boundary_node_tags)
    boundary_nodes = physical_all_node_coords[boundary_indices]
    boundary_tags = boundary_node_tags

    coords_2d = boundary_nodes[:, :2]
    centroid = np.mean(coords_2d, axis=0)
    angles = np.arctan2(coords_2d[:, 1] - centroid[1], coords_2d[:, 0] - centroid[0])
    sorted_indices = np.argsort(-angles)  # clockwise order

    sorted_boundary_nodes = boundary_nodes[sorted_indices]
    sorted_boundary_tags = boundary_tags[sorted_indices]

    return boundary_elements, sorted_boundary_nodes, sorted_boundary_tags


def _get_topology():
    element_types_2d, element_tags_2d, node_tags_2d = gmsh.model.mesh.getElements(2)
    topology = []
    for etype, etags, enodes in zip(element_types_2d, element_tags_2d, node_tags_2d):
        if etype == 2:  # triangle
            num_nodes_per_elem = 3
        elif etype == 3:  # quadrangle
            num_nodes_per_elem = 4
        else:
            continue
        for i, tag in enumerate(etags):
            nodes = enodes[i * num_nodes_per_elem: (i + 1) * num_nodes_per_elem]
            topology.append((tag, nodes))
    return topology


def _resolve_zero_d_coords(coupling, sorted_boundary_nodes):
    if coupling.aero_node_coords is not None:
        return (float(coupling.aero_node_coords[0]), float(coupling.aero_node_coords[1]))
    centroid = np.mean(sorted_boundary_nodes[:, :2], axis=0)
    return (float(centroid[0]), float(centroid[1]))


def _build_records(coupling, sorted_boundary_tags, sorted_boundary_nodes, zero_d_coords=None):
    """One record per SPARTA boundary element (index i -> i+1, wrapping),
    describing its AERO-S correspondence and whether it is active. This is
    the single source of truth for the mapping file and the coupling plot."""
    n = len(sorted_boundary_tags)
    coords_2d = sorted_boundary_nodes[:, :2]
    mask = coupling.select_subset(coords_2d) if coupling.mode == "partial" else None

    records = []
    for i in range(n):
        j = (i + 1) % n
        p1, p2 = tuple(coords_2d[i]), tuple(coords_2d[j])
        tag1, tag2 = int(sorted_boundary_tags[i]), int(sorted_boundary_tags[j])

        if coupling.mode == "default":
            active = True
            aero_node_ids, aero_coords = [tag1, tag2], [p1, p2]
        elif coupling.mode == "zero_dimensional":
            active = True
            aero_node_ids, aero_coords = [coupling.aero_node_id], [zero_d_coords]
        elif coupling.mode == "partial":
            active = bool(mask[i] and mask[j])
            aero_node_ids = [tag1, tag2] if active else []
            aero_coords = [p1, p2] if active else []
        else:
            raise ValueError(f"unknown coupling mode: {coupling.mode!r}")

        records.append({
            "sparta_elem_id": i + 1,
            "sparta_node_ids": (i + 1, j + 1),
            "sparta_coords": (p1, p2),
            "aero_node_ids": aero_node_ids,
            "aero_coords": aero_coords,
            "active": active,
        })
    return records


def _resolve_aero_export_data(coupling, records, all_node_tags, physical_all_node_coords, topology, zero_d_coords):
    """AERO-S NODE/TOPO data to write, per coupling mode."""
    if coupling.mode == "default":
        return all_node_tags, physical_all_node_coords, topology

    if coupling.mode == "zero_dimensional":
        aero_node_tags = np.array([coupling.aero_node_id])
        aero_node_coords = np.array([[zero_d_coords[0], zero_d_coords[1], 0.0]])
        return aero_node_tags, aero_node_coords, []

    if coupling.mode == "partial":
        active_records = [r for r in records if r["active"]]
        tags, coords, seen = [], [], set()
        for r in active_records:
            for tag, xy in zip(r["aero_node_ids"], r["aero_coords"]):
                if tag not in seen:
                    seen.add(tag)
                    tags.append(tag)
                    coords.append((xy[0], xy[1], 0.0))
        if not tags:
            raise ValueError(
                "partial coupling matched zero active SPARTA elements; "
                "check that node_selectors actually select boundary nodes"
            )
        aero_topology = [
            (elem_id, tuple(r["aero_node_ids"]))
            for elem_id, r in enumerate(active_records, start=1)
        ]
        return np.array(tags), np.array(coords), aero_topology

    raise ValueError(f"unknown coupling mode: {coupling.mode!r}")


def _write_aero_mesh(
    mesh_filename,
    elem_code,
    all_node_tags,
    physical_all_node_coords,
    topology,
    materials,
    default_material_id,
    displacement_groups,
    default_temperature,
):
    tag_to_index = {tag: i for i, tag in enumerate(all_node_tags)}

    with open(mesh_filename, "w") as f:
        f.write("NODE\n")
        for tag, (x, y, _) in zip(all_node_tags, physical_all_node_coords):
            f.write(f"{tag} {x} {y}\n")

        f.write("*\n")
        f.write("TOPO\n")
        for tag, nodes in topology:
            nodes = list(nodes)

            node_coords = [physical_all_node_coords[tag_to_index[n], :2] for n in nodes]

            # Positive signed area = counter-clockwise; flip winding if clockwise.
            signed_area = 0.5 * sum(
                node_coords[i][0] * node_coords[(i + 1) % len(node_coords)][1]
                - node_coords[(i + 1) % len(node_coords)][0] * node_coords[i][1]
                for i in range(len(node_coords))
            )
            if signed_area < 0:
                nodes = list(reversed(nodes))

            node_str = " ".join(map(str, nodes))
            f.write(f"{tag} {elem_code} {node_str}\n")

        f.write("*\n")
        if topology:
            first_elem_tag = topology[0][0]
            last_elem_tag = topology[-1][0]
            f.write("ATTR\n")
            f.write(f"{first_elem_tag} {last_elem_tag} {default_material_id}\n")
            f.write("*\n")

        f.write("MATERIAL\n")
        for material in materials:
            f.write(material.to_line() + "\n")

        f.write("*\n")
        f.write("ITEMPERATURES\n")
        for tag in all_node_tags:
            f.write(f"{tag} {default_temperature}\n")

        if displacement_groups:
            f.write("*\n")
            f.write("DISP\n")
            coords_2d = physical_all_node_coords[:, :2]
            seen = set()
            for group in displacement_groups:
                mask = group.select(coords_2d)
                for tag in all_node_tags[mask]:
                    for dof in group.dofs:
                        key = (tag, dof)
                        if key in seen:
                            continue
                        seen.add(key)
                        f.write(f"{tag} {dof} 0\n")


def _write_sparta_surface(sparta_filename, sorted_boundary_nodes):
    num_boundary_nodes = len(sorted_boundary_nodes)
    with open(sparta_filename, "w") as f:
        f.write("Refined Surface (Ordered)\n\n")
        f.write(f"{num_boundary_nodes} points\n")
        f.write(f"{num_boundary_nodes} lines\n\n")
        f.write("Points\n\n")
        for i, (x, y, _) in enumerate(sorted_boundary_nodes, 1):
            f.write(f"{i} {x} {y}\n")

        f.write("\nLines\n\n")
        for i in range(num_boundary_nodes):
            next_i = (i + 1) % num_boundary_nodes
            f.write(f"{i + 1} {i + 1} {next_i + 1}\n")


def _write_mapping(mapping_filename, records):
    """Single-section mapping: one line per SPARTA boundary element with its
    two SPARTA node ids, its variable-length list of AERO-S node ids (0, 1,
    or 2 depending on coupling mode and activity), and a trailing
    ACTIVE_FLAG marking whether it participates in the SPARTA<->AERO-S
    information exchange."""
    with open(mapping_filename, "w") as f:
        f.write(
            "# SPARTA_ELEMENT_ID SPARTA_NODE_ID_1 SPARTA_NODE_ID_2 "
            "AERO-S_NODE_ID... ACTIVE_FLAG\n"
        )
        for r in records:
            n1, n2 = r["sparta_node_ids"]
            flag = 1 if r["active"] else 0
            aero_ids = " ".join(str(a) for a in r["aero_node_ids"])
            prefix = f"{r['sparta_elem_id']} {n1} {n2}"
            if aero_ids:
                f.write(f"{prefix} {aero_ids} {flag}\n")
            else:
                f.write(f"{prefix} {flag}\n")


def _write_stl(stl_filename, sorted_boundary_nodes):
    num_boundary_nodes = len(sorted_boundary_nodes)
    centroid_2d = np.mean(sorted_boundary_nodes[:, :2], axis=0)
    centroid = np.array([centroid_2d[0], centroid_2d[1], 0])

    with open(stl_filename, "w") as f:
        f.write("solid surface\n")
        for i in range(num_boundary_nodes):
            p1 = sorted_boundary_nodes[i]
            p2 = sorted_boundary_nodes[(i + 1) % num_boundary_nodes]

            v1 = p2 - p1
            v2 = centroid - p1
            normal = np.cross(v1, v2)
            normal = normal / np.linalg.norm(normal)

            f.write(f"  facet normal {normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}\n")
            f.write("    outer loop\n")
            f.write(f"      vertex {p1[0]:.6e} {p1[1]:.6e} {p1[2]:.6e}\n")
            f.write(f"      vertex {p2[0]:.6e} {p2[1]:.6e} {p2[2]:.6e}\n")
            f.write(f"      vertex {centroid[0]:.6e} {centroid[1]:.6e} {centroid[2]:.6e}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write("endsolid surface\n")


def _write_summary(
    summary_filename,
    title,
    geometry_description,
    materials,
    default_material_id,
    displacement_groups,
    struct_elem_code,
    heat_elem_code,
    default_temperature,
    coupling,
    num_nodes,
    num_elements,
    num_boundary_nodes,
    num_active_elements,
    generated_files,
):
    with open(summary_filename, "w") as f:
        f.write(f"Mesh export summary: {title}\n")
        f.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n\n")

        f.write("Geometry\n--------\n")
        f.write(f"{geometry_description}\n\n")

        f.write("Coupling\n--------\n")
        f.write(f"{coupling.describe()}\n\n")

        f.write("Mesh stats\n----------\n")
        f.write(f"Total AERO-S nodes: {num_nodes}\n")
        f.write(f"Total AERO-S elements: {num_elements}\n")
        f.write(f"SPARTA boundary nodes: {num_boundary_nodes}\n")
        f.write(f"Active SPARTA boundary elements: {num_active_elements}/{num_boundary_nodes}\n\n")

        f.write("Materials\n---------\n")
        f.write("id area E poisson density conv cond thickness perimeter ref_temp cp therm_exp ixx iyy izz ymin ymax zmin zmax\n")
        for material in materials:
            f.write(f"{material.to_line()}\n")
        f.write(f"Default material id (ATTR): {default_material_id}\n\n")

        f.write("Element codes\n-------------\n")
        f.write(f"Structural: {struct_elem_code}\n")
        f.write(f"Heat: {heat_elem_code}\n\n")

        f.write("Default nodal temperature\n-------------------------\n")
        f.write(f"{default_temperature}\n\n")

        f.write("Displacement boundary conditions\n---------------------------------\n")
        if displacement_groups:
            for group in displacement_groups:
                f.write(f"{group.describe()}\n")
        else:
            f.write("(none)\n")
        f.write("\n")

        f.write("Generated files\n---------------\n")
        for label, path in generated_files.items():
            f.write(f"{label}: {path}\n")


def export_coupled_mesh(
    surface,
    title,
    out_dir,
    materials,
    displacement_groups=None,
    default_material_id=None,
    struct_elem_code=4,
    heat_elem_code=46,
    default_temperature=300.0,
    internal_scale_factor=1.0,
    geometry_description="",
    coupling=None,
):
    """Write the AERO-S struct/heat meshes, SPARTA surface, mapping, STL,
    coupling plot, and summary.

    surface: gmsh plane-surface tag for the meshed 2D domain.
    materials: list[Material], written verbatim to the MATERIAL section.
    displacement_groups: list[DisplacementGroup] or None to skip the DISP section.
    default_material_id: material id referenced by the ATTR line for every
        element; defaults to the first material's id.
    internal_scale_factor: divide gmsh node coordinates by this to recover
        physical units (use whatever the geometry builder used, e.g. 1/radius).
    coupling: a coupling.Coupling instance controlling how the AERO-S side
        relates to the SPARTA boundary (see coupling.py). Defaults to
        Coupling.default() - the original 1:1, fully-connected behavior.
    """
    if not materials:
        raise ValueError("At least one Material must be provided.")
    if default_material_id is None:
        default_material_id = materials[0].id
    if coupling is None:
        coupling = Coupling.default()

    os.makedirs(out_dir, exist_ok=True)

    all_node_tags, physical_all_node_coords = _get_all_nodes(internal_scale_factor)
    _, sorted_boundary_nodes, sorted_boundary_tags = _get_boundary_nodes(
        surface, all_node_tags, physical_all_node_coords
    )
    topology = _get_topology()

    zero_d_coords = None
    if coupling.mode == "zero_dimensional":
        zero_d_coords = _resolve_zero_d_coords(coupling, sorted_boundary_nodes)

    records = _build_records(coupling, sorted_boundary_tags, sorted_boundary_nodes, zero_d_coords)
    aero_node_tags, aero_node_coords, aero_topology = _resolve_aero_export_data(
        coupling, records, all_node_tags, physical_all_node_coords, topology, zero_d_coords
    )

    struct_mesh_filename = os.path.join(out_dir, f"{title}_aero_struct.txt")
    heat_mesh_filename = os.path.join(out_dir, f"{title}_aero_heat.txt")
    sparta_filename = os.path.join(out_dir, f"{title}_sparta.txt")
    mapping_filename = os.path.join(out_dir, f"{title}_mapping.txt")
    stl_filename = os.path.join(out_dir, f"{title}_surface.stl")
    coupling_plot_filename = os.path.join(out_dir, f"{title}_coupling.png")
    summary_filename = os.path.join(out_dir, f"{title}_summary.txt")

    _write_aero_mesh(
        struct_mesh_filename, struct_elem_code, aero_node_tags, aero_node_coords,
        aero_topology, materials, default_material_id, displacement_groups, default_temperature,
    )
    _write_aero_mesh(
        heat_mesh_filename, heat_elem_code, aero_node_tags, aero_node_coords,
        aero_topology, materials, default_material_id, displacement_groups, default_temperature,
    )
    _write_sparta_surface(sparta_filename, sorted_boundary_nodes)
    _write_mapping(mapping_filename, records)
    _write_stl(stl_filename, sorted_boundary_nodes)
    write_coupling_plot(coupling_plot_filename, title, coupling, records, aero_node_coords)

    generated_files = {
        "struct_mesh": struct_mesh_filename,
        "heat_mesh": heat_mesh_filename,
        "sparta": sparta_filename,
        "mapping": mapping_filename,
        "stl": stl_filename,
        "coupling_plot": coupling_plot_filename,
    }

    _write_summary(
        summary_filename,
        title,
        geometry_description,
        materials,
        default_material_id,
        displacement_groups,
        struct_elem_code,
        heat_elem_code,
        default_temperature,
        coupling,
        num_nodes=len(aero_node_tags),
        num_elements=len(aero_topology),
        num_boundary_nodes=len(sorted_boundary_tags),
        num_active_elements=sum(1 for r in records if r["active"]),
        generated_files=generated_files,
    )
    generated_files["summary"] = summary_filename

    return generated_files
