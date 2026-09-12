"""Minimal reader for gmsh MSH 2.2, plus a VTU writer for ParaView.

Only MSH 2.2 is handled, deliberately: it is the format gmsh2top can actually
read (it fails on 4.1, and its version-4 branch was written against 4.0), so
it is the only format this pipeline ever produces.

MSH 2.2 element records are:
    id  type  n_tags  tag1 ... tagN  node1 node2 ...
where tag1 is the physical id and tag2 the geometrical id. We keep only the
element types gmsh2top accepts, 2 (3-node triangle) and 4 (4-node tet), and
assert that nothing else volumetric slipped in.
"""

import numpy as np

TYPE_TRI = 2
TYPE_TET = 4
_NODES_PER_TYPE = {15: 1, 1: 2, TYPE_TRI: 3, TYPE_TET: 4, 6: 6, 5: 8}


class Mesh:
    """Nodes, tets and boundary triangles with their physical ids."""

    def __init__(self, nodes, node_ids, tets, tet_phys, tris, tri_phys, phys_names):
        self.nodes = nodes            # (N, 3) float, compacted 0-based
        self.node_ids = node_ids      # (N,)  original gmsh node ids
        self.tets = tets              # (T, 4) int, 0-based into nodes
        self.tet_phys = tet_phys      # (T,)  physical id
        self.tris = tris              # (F, 3) int, 0-based
        self.tri_phys = tri_phys      # (F,)  physical id
        self.phys_names = phys_names  # {phys_id: (dim, name)}

    # -- derived quantities -------------------------------------------------

    def tet_volumes(self):
        """Signed volumes; positive means right-handed node ordering."""
        p = self.nodes[self.tets]
        v1 = p[:, 1] - p[:, 0]
        v2 = p[:, 2] - p[:, 0]
        v3 = p[:, 3] - p[:, 0]
        return np.einsum("ij,ij->i", v3, np.cross(v1, v2)) / 6.0

    def tri_areas(self):
        p = self.nodes[self.tris]
        return 0.5 * np.linalg.norm(
            np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)

    def tet_edge_lengths(self):
        """(T, 6) edge lengths of every tet."""
        p = self.nodes[self.tets]
        pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        return np.stack(
            [np.linalg.norm(p[:, b] - p[:, a], axis=1) for a, b in pairs], axis=1)

    def tet_aspect_ratio(self):
        """Longest edge over inscribed-sphere diameter.

        Equals ~2.45 for an equilateral tet; larger is worse. Reported rather
        than gated on, because the one-cell extrusion makes some anisotropy
        unavoidable by construction.
        """
        vol = np.abs(self.tet_volumes())
        p = self.nodes[self.tets]
        faces = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
        area = np.zeros(len(self.tets))
        for a, b, c in faces:
            area += 0.5 * np.linalg.norm(
                np.cross(p[:, b] - p[:, a], p[:, c] - p[:, a]), axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            r_in = 3.0 * vol / area           # inscribed sphere radius
            return self.tet_edge_lengths().max(axis=1) / (2.0 * r_in)

    def phys_id(self, name):
        for pid, (_dim, nm) in self.phys_names.items():
            if nm == name:
                return pid
        raise KeyError("no physical group named %r (have %s)"
                       % (name, sorted(nm for _d, nm in self.phys_names.values())))

    def tris_of(self, name):
        return self.tris[self.tri_phys == self.phys_id(name)]

    def tets_of(self, name):
        return self.tets[self.tet_phys == self.phys_id(name)]

    def nodes_of_group(self, name):
        """Unique node indices used by a surface group."""
        return np.unique(self.tris_of(name).ravel())


def _read_block(f, terminator):
    lines = []
    for raw in f:
        s = raw.strip()
        if s == terminator:
            return lines
        if s:
            lines.append(s)
    raise ValueError("unterminated block, expected %s" % terminator)


def read_msh22(path):
    """Parse an MSH 2.2 file into a Mesh."""
    phys_names, node_ids, coords = {}, [], []
    elem_lines = []

    with open(path) as f:
        for raw in f:
            s = raw.strip()
            if s == "$MeshFormat":
                ver = _read_block(f, "$EndMeshFormat")[0].split()[0]
                if not ver.startswith("2"):
                    raise ValueError(
                        "%s is MSH %s; this pipeline only handles 2.2 because "
                        "gmsh2top cannot read newer versions" % (path, ver))
            elif s == "$PhysicalNames":
                blk = _read_block(f, "$EndPhysicalNames")
                for ln in blk[1:]:
                    parts = ln.split(None, 2)
                    phys_names[int(parts[1])] = (int(parts[0]),
                                                 parts[2].strip().strip('"'))
            elif s == "$Nodes":
                blk = _read_block(f, "$EndNodes")
                for ln in blk[1:]:
                    p = ln.split()
                    node_ids.append(int(p[0]))
                    coords.append((float(p[1]), float(p[2]), float(p[3])))
            elif s == "$Elements":
                elem_lines = _read_block(f, "$EndElements")[1:]

    node_ids = np.asarray(node_ids, dtype=np.int64)
    nodes = np.asarray(coords, dtype=float)

    # gmsh node ids are 1..N and contiguous in practice, but remap anyway.
    lookup = np.full(node_ids.max() + 1, -1, dtype=np.int64)
    lookup[node_ids] = np.arange(len(node_ids))

    tets, tet_phys, tris, tri_phys = [], [], [], []
    seen_types = set()
    for ln in elem_lines:
        p = ln.split()
        etype = int(p[1])
        seen_types.add(etype)
        ntags = int(p[2])
        phys = int(p[3]) if ntags >= 1 else 0
        conn = [int(x) for x in p[3 + ntags:]]
        if etype == TYPE_TET:
            tets.append(conn)
            tet_phys.append(phys)
        elif etype == TYPE_TRI:
            tris.append(conn)
            tri_phys.append(phys)

    bad = seen_types - {15, 1, TYPE_TRI, TYPE_TET}
    if bad:
        names = {6: "prism", 5: "hexahedron", 9: "tri6", 11: "tet10"}
        raise ValueError(
            "mesh contains element types %s (%s); gmsh2top accepts only "
            "triangles and tetrahedra. Check that Recombine is not set and "
            "Mesh.ElementOrder = 1."
            % (sorted(bad), ", ".join(names.get(t, str(t)) for t in sorted(bad))))

    return Mesh(
        nodes=nodes,
        node_ids=node_ids,
        tets=lookup[np.asarray(tets, dtype=np.int64)] if tets else np.zeros((0, 4), np.int64),
        tet_phys=np.asarray(tet_phys, dtype=np.int64),
        tris=lookup[np.asarray(tris, dtype=np.int64)] if tris else np.zeros((0, 3), np.int64),
        tri_phys=np.asarray(tri_phys, dtype=np.int64),
        phys_names=phys_names,
    )


# --------------------------------------------------------------------------
# VTU output (ASCII XML unstructured grid) -- for local ParaView inspection
# --------------------------------------------------------------------------

def write_vtu(path, nodes, cells, cell_data=None, point_data=None):
    """Write tetrahedra (or triangles) to a legacy-free ASCII .vtu.

    Written by hand rather than via meshio to keep the dependency list to
    numpy/scipy/matplotlib, all of which are available as cluster modules.
    """
    cells = np.asarray(cells)
    n_pts, n_cells = len(nodes), len(cells)
    npc = cells.shape[1]
    vtk_type = {4: 10, 3: 5}[npc]  # 10 = VTK_TETRA, 5 = VTK_TRIANGLE

    def arr(name, a, fmt="%.9g", ncomp=1):
        body = " ".join(fmt % v for v in np.asarray(a).ravel())
        return ('        <DataArray type="%s" Name="%s" '
                'NumberOfComponents="%d" format="ascii">\n          %s\n'
                '        </DataArray>\n'
                % ("Float64" if a.dtype.kind == "f" else "Int32",
                   name, ncomp, body))

    with open(path, "w") as f:
        f.write('<?xml version="1.0"?>\n'
                '<VTKFile type="UnstructuredGrid" version="0.1" '
                'byte_order="LittleEndian">\n  <UnstructuredGrid>\n')
        f.write('    <Piece NumberOfPoints="%d" NumberOfCells="%d">\n'
                % (n_pts, n_cells))

        f.write('      <Points>\n')
        f.write(arr("Points", np.asarray(nodes, dtype=float), ncomp=3))
        f.write('      </Points>\n')

        f.write('      <Cells>\n')
        f.write(arr("connectivity", cells.astype(np.int32), fmt="%d"))
        f.write(arr("offsets",
                    (np.arange(1, n_cells + 1) * npc).astype(np.int32), fmt="%d"))
        f.write(arr("types",
                    np.full(n_cells, vtk_type, dtype=np.int32), fmt="%d"))
        f.write('      </Cells>\n')

        if cell_data:
            first = next(iter(cell_data))
            f.write('      <CellData Scalars="%s">\n' % first)
            for name, vals in cell_data.items():
                f.write(arr(name, np.asarray(vals),
                            fmt="%d" if np.asarray(vals).dtype.kind in "iu" else "%.9g"))
            f.write('      </CellData>\n')

        if point_data:
            first = next(iter(point_data))
            f.write('      <PointData Scalars="%s">\n' % first)
            for name, vals in point_data.items():
                f.write(arr(name, np.asarray(vals),
                            fmt="%d" if np.asarray(vals).dtype.kind in "iu" else "%.9g"))
            f.write('      </PointData>\n')

        f.write('    </Piece>\n  </UnstructuredGrid>\n</VTKFile>\n')
