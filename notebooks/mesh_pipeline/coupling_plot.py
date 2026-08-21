"""Two-panel sanity-check plot: SPARTA boundary vs AERO-S side, colored by
whether each SPARTA boundary element is active in the coupling mapping.
"""

import matplotlib.pyplot as plt

_SURFACE = "#fcfcfb"
_GRID = "#e1e0d9"
_PRIMARY_INK = "#0b0b0b"
_SECONDARY_INK = "#52514e"
_NODE_COLOR = "#0b0b0b"
_ACTIVE_COLOR = "#0ca30c"  # status: good
_INACTIVE_COLOR = "#898781"  # muted ink


def _style_axes(ax, title):
    ax.set_facecolor(_SURFACE)
    ax.set_aspect("equal")
    ax.grid(True, color=_GRID, linewidth=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_color(_GRID)
    ax.tick_params(colors=_SECONDARY_INK, labelsize=8)
    ax.set_title(title, color=_PRIMARY_INK, fontsize=11, pad=10)


def write_coupling_plot(png_filename, title, coupling, records, aero_node_coords):
    """records: per-boundary-element dicts from mesh_export._build_records.
    aero_node_coords: (N, 3) array of the AERO-S node coordinates actually
    exported (1 row for zero_dimensional, the active subset for partial, the
    full interior mesh for default)."""
    fig, (ax_sparta, ax_aero) = plt.subplots(1, 2, figsize=(11, 5), facecolor=_SURFACE)
    _style_axes(ax_sparta, "SPARTA (DSMC boundary)")
    _style_axes(ax_aero, "AERO-S (FEM side)")

    sparta_x, sparta_y = [], []
    for r in records:
        (x1, y1), (x2, y2) = r["sparta_coords"]
        color = _ACTIVE_COLOR if r["active"] else _INACTIVE_COLOR
        ax_sparta.plot([x1, x2], [y1, y2], color=color, linewidth=2, zorder=2)
        sparta_x += [x1, x2]
        sparta_y += [y1, y2]
    ax_sparta.scatter(sparta_x, sparta_y, s=10, color=_NODE_COLOR, zorder=3)

    active_records = [r for r in records if r["active"]]

    if coupling.mode == "zero_dimensional":
        x0, y0 = aero_node_coords[0, 0], aero_node_coords[0, 1]
        for r in active_records:
            (x1, y1), (x2, y2) = r["sparta_coords"]
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax_aero.plot([mx, x0], [my, y0], color=_ACTIVE_COLOR, linewidth=0.5, alpha=0.3, zorder=1)
        ax_aero.scatter([x0], [y0], s=220, color=_ACTIVE_COLOR, zorder=3)
        ax_aero.annotate(
            f"node {coupling.aero_node_id}\n(all loads summed here)",
            (x0, y0),
            textcoords="offset points",
            xytext=(10, 10),
            fontsize=9,
            color=_PRIMARY_INK,
        )
    else:
        # Draw only the boundary/subset nodes actually involved in active
        # elements - aero_node_coords may hold the full interior AERO-S mesh
        # (default mode), which would otherwise flood the plot.
        aero_x, aero_y, seen = [], [], set()
        for r in active_records:
            coords = r["aero_coords"]
            if len(coords) == 2:
                (x1, y1), (x2, y2) = coords
                ax_aero.plot([x1, x2], [y1, y2], color=_ACTIVE_COLOR, linewidth=2, zorder=2)
            for x, y in coords:
                key = (x, y)
                if key not in seen:
                    seen.add(key)
                    aero_x.append(x)
                    aero_y.append(y)
        ax_aero.scatter(aero_x, aero_y, s=10, color=_NODE_COLOR, zorder=3)

    legend_handles = [
        plt.Line2D([0], [0], color=_ACTIVE_COLOR, linewidth=2, label="Active"),
        plt.Line2D([0], [0], color=_INACTIVE_COLOR, linewidth=2, label="Inactive"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        labelcolor=_PRIMARY_INK,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.suptitle(f"{title} — coupling mode: {coupling.mode}", color=_PRIMARY_INK, fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(png_filename, dpi=150, facecolor=_SURFACE)
    plt.close(fig)
