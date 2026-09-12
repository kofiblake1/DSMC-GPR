"""Assemble a grid of shape outlines for quick visual triage of a batch.

Deliberately plots the wall curves rather than the meshes: at thumbnail size
mesh lines are illegible, and what you want to scan for is whether the shape
population looks sane (no slivers, no near-degenerate lobes, sensible spread
of concavity).
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build(run_root, records, out_path, ncols=6, max_cases=48):
    recs = [r for r in records if r.get("ok")][:max_cases]
    if not recs:
        raise ValueError("no successful cases to plot")

    nrows = int(np.ceil(len(recs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.0 * ncols, 2.0 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax in axes:
        ax.set_axis_off()

    for ax, rec in zip(axes, recs):
        path = rec["files"].get("curve")
        if not path or not os.path.exists(path):
            continue
        c = np.loadtxt(path)
        cl = np.vstack((c, c[:1]))
        ax.plot(cl[:, 0], cl[:, 1], "-", color="crimson", lw=1.0)
        ax.fill(cl[:, 0], cl[:, 1], color="crimson", alpha=0.12)
        ax.set_aspect("equal")
        half = 0.6 * max(np.ptp(c[:, 0]), np.ptp(c[:, 1]))
        ax.set_xlim(-half, half)
        ax.set_ylim(-half, half)
        d = rec.get("geometry", {}).get("characteristic_diameter", float("nan"))
        n = rec.get("quality", {}).get("counts", {}).get("tets", 0)
        ax.set_title("%s\nD %.3g  %d tets"
                     % (rec["name"].replace("shape_", "s"), d, n), fontsize=7)

    fig.suptitle("%s  --  %d cases" % (os.path.basename(run_root), len(recs)),
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path
