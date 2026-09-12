"""Visual artefacts: a triage PNG rendered on the cluster, and VTUs for
local ParaView inspection.

The PNG exists so a few hundred meshes can be eyeballed without downloading
anything. The VTUs are for the cases worth looking at properly.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from rarefied.aerof import geo as _geo
from rarefied.aerof import mshio as _mshio


def _plane_triangles(mesh, resolved):
    """The lower side plane, as 2D triangles in the shape's own two axes."""
    ax = list(resolved.get("plane_axes", (0, 2)))
    tris = mesh.tris_of(_geo.GROUP_SIDE_LO[0])
    return mesh.nodes[tris][:, :, ax]


def _edges_of(tri2d):
    e = np.concatenate([tri2d[:, [0, 1]], tri2d[:, [1, 2]], tri2d[:, [2, 0]]], axis=0)
    return e


def render_png(path, mesh, curve, resolved, title=""):
    """Three panels: whole domain, near field, and the wall itself."""
    tri2d = _plane_triangles(mesh, resolved)
    D = resolved["characteristic_diameter"]
    R = resolved["farfield_radius"]
    cl = np.vstack((curve, curve[:1]))
    lab = ["xyz"[i] for i in resolved.get("plane_axes", (0, 2))]

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6))
    spans = [(R * 1.05, "full domain (R = %.3g)" % R),
             (D * 2.0, "near field (2 D)"),
             (D * 0.75, "wall (0.75 D)")]

    for ax, (half, label) in zip(axes, spans):
        keep = np.all(np.abs(tri2d).max(axis=1) < half * 1.6, axis=1)
        sub = tri2d[keep] if keep.any() else tri2d
        lw = 0.08 if half > D * 5 else (0.25 if half > D else 0.5)
        ax.add_collection(LineCollection(_edges_of(sub), linewidths=lw,
                                         colors="0.45", zorder=1))
        ax.plot(cl[:, 0], cl[:, 1], "-", color="crimson", lw=1.4, zorder=3)
        ax.set_xlim(-half, half)
        ax.set_ylim(-half, half)
        ax.set_aspect("equal")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("%s [m]" % lab[0])
        ax.set_ylabel("%s [m]" % lab[1])
        ax.tick_params(labelsize=8)

    n = resolved["n_surf"]
    fig.suptitle("%s   |   %d nodes, %d tets, n_surf %d, size %.3g - %.3g"
                 % (title, len(mesh.nodes), len(mesh.tets), n,
                    resolved["size_min"], resolved["size_max"]),
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_volume_vtu(path, mesh, parts=None):
    """Volume tets, optionally coloured by subdomain id."""
    cd = {}
    if parts is not None:
        cd["subdomain"] = np.asarray(parts, dtype=np.int32)
    cd["tet_volume"] = mesh.tet_volumes()
    cd["aspect_ratio"] = mesh.tet_aspect_ratio()
    _mshio.write_vtu(path, mesh.nodes, mesh.tets, cell_data=cd)
    return path


def write_surface_vtu(path, mesh):
    """Boundary triangles coloured by which physical group they belong to,
    so the BC assignment can be checked visually rather than only by area."""
    name_of = {pid: nm for pid, (_d, nm) in mesh.phys_names.items()}
    groups = [nm for nm, _tag in _geo.SURFACE_GROUPS]
    code = np.array([groups.index(name_of[p]) if name_of.get(p) in groups else -1
                     for p in mesh.tri_phys], dtype=np.int32)
    _mshio.write_vtu(path, mesh.nodes, mesh.tris,
                     cell_data={"group": code, "physical_id": mesh.tri_phys.astype(np.int32)})
    return path


def group_legend():
    """Mapping from the integer in the VTU 'group' field to a name."""
    return {i: nm for i, (nm, _tag) in enumerate(_geo.SURFACE_GROUPS)}
