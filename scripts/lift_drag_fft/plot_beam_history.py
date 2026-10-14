#!/usr/bin/env python3
"""Plot every deformed configuration of the beam from one run on one figure.

Given a single run directory, this reads results_beam/gdisplac -- which holds
the displacement of every node at every output step -- adds it to the
undeformed nodal coordinates, and draws the whole beam at each timestep,
overlaid, colored by time. The result is the beam's entire motion history in a
single static image: where it swept, how far, and whether the envelope is
growing, decaying, or steady.

Note that gdisplac stores DISPLACEMENTS, not positions, so the undeformed
coordinates are needed too. They come from the AERO-S input deck's NODE
section (<RUN_DIR>/*_beam_struct.txt), which is located automatically.

Setup (Sherlock modules):
    ml load math py-numpy/1.26.3_py312 viz py-matplotlib/3.10.3_py312

Usage: edit RUN_DIR below, then:
    python3 plot_beam_history.py
"""

import glob
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

# ============================================================================
# CONFIG
# ============================================================================
RUN_DIR = ("/Volumes/SherlockScratch/DSMC_FEM_VALIDATION_P1/BEAM_LOCKIN/"
           "PHASE_5_FINAL/SWEEP/f1_1p00")

OUTPUT_DIR = os.getcwd()
OUTPUT_NAME = "beam_history.png"

# Path to the AERO-S input deck holding the NODE section. None = look for
# <RUN_DIR>/*_beam_struct.txt.
STRUCT_FILE = None

# Skip configurations before this time (gdisplac's own clock, which starts at
# 0.0 when structural coupling begins -- no restart offset is applied here
# because this script never compares against SPARTA's absolute clock).
ST = 1.0

# At most this many configurations get drawn, evenly strided through the
# record. 8000 timesteps of a 41-node beam is ~330k segments: slow to
# rasterize and so dense it becomes a solid block. A few hundred curves is
# enough to read the envelope and still see individual sweeps.
MAX_CURVES = 400

# Per-curve transparency and width. Low alpha is what makes the density
# visible -- regions the beam lingers in darken, fast transits stay faint.
ALPHA = 0.25
LINEWIDTH = 0.6
CMAP = 'viridis'

# Multiply displacements by this before plotting. 1.0 is physical/undistorted.
# Raise it only if a run's deflection is too small to see; the figure is
# annotated whenever it isn't 1.0 so an exaggerated plot can't be mistaken for
# a real one.
DEFORM_SCALE = 1.0

# True keeps the two axes at the same scale, so the plotted shape is the
# beam's real shape. Turn it off only to stretch a very small deflection.
EQUAL_ASPECT = True

# Which coordinate direction is along the beam and which is the bending
# direction, as 0/1/2 for x/y/z. None auto-detects: the axial axis is the one
# the undeformed nodes span, and the bending axis is the one that moves most.
AXIAL_AXIS = None
BEND_AXIS = None

AXIS_NAMES = ('x', 'y', 'z')


def find_struct_file(run_dir):
    """Locate the AERO-S deck with the NODE section."""
    if STRUCT_FILE is not None:
        return STRUCT_FILE
    matches = sorted(glob.glob(os.path.join(run_dir, "*_beam_struct.txt")))
    if not matches:
        sys.exit(f"No *_beam_struct.txt in {run_dir} -- set STRUCT_FILE to "
                 f"the deck holding the NODE section.")
    if len(matches) > 1:
        print(f"NOTE: several candidate decks, using {matches[0]}")
    return matches[0]


def load_reference_nodes(path):
    """Parse the NODE section of an AERO-S input deck.

    Format is a bare "NODE" line, then "<id> <x> <y> <z>" per node, ending at
    the "*" that separates sections. Returns coordinates ordered by node id,
    which is the order gdisplac writes its node lines in."""
    coords = {}
    in_node = False
    with open(path) as f:
        for line in f:
            token = line.strip()
            if not token:
                continue
            if token.upper().startswith("NODE"):
                in_node = True
                continue
            if in_node:
                # "*" ends the section; any other keyword line means the deck
                # isn't laid out the way this parser assumes.
                if token.startswith("*"):
                    break
                parts = token.split()
                if len(parts) < 4:
                    break
                coords[int(parts[0])] = [float(v) for v in parts[1:4]]

    if not coords:
        sys.exit(f"Found no NODE entries in {path}.")
    return np.array([coords[k] for k in sorted(coords)])


def load_displacement_field(path, st):
    """Parse gdisplac into the full nodal displacement history.

    Same file format plot_lift_drag.py's load_tip_displacement() reads -- a
    2-line header, then repeating [time line] + [nnodes x "dx dy dz"] blocks --
    but this keeps every node instead of just the tip.

    Returns (times, disp) where disp has shape (n_blocks, nnodes, 3)."""
    with open(path) as f:
        lines = f.readlines()

    nnodes = int(lines[1])
    block_size = 1 + nnodes
    body = lines[2:]
    n_blocks = len(body) // block_size

    times = np.empty(n_blocks)
    disp = np.empty((n_blocks, nnodes, 3))
    for b in range(n_blocks):
        block = body[b * block_size:(b + 1) * block_size]
        times[b] = float(block[0])
        disp[b] = [[float(v) for v in line.split()] for line in block[1:]]

    mask = times >= st
    return times[mask], disp[mask]


def pick_axes(ref, disp):
    """Choose the axial and bending directions from the data itself.

    Axial = the axis the undeformed nodes actually span (the beam's length).
    Bending = the axis with the most movement, excluding the axial one: axial
    displacement here is second-order foreshortening from large-deflection
    bending, so it must not be allowed to win."""
    axial = AXIAL_AXIS
    if axial is None:
        axial = int(np.argmax(ref.max(axis=0) - ref.min(axis=0)))

    bend = BEND_AXIS
    if bend is None:
        spread = disp.reshape(-1, 3).std(axis=0)
        spread[axial] = -1.0
        bend = int(np.argmax(spread))
    return axial, bend


def main():
    gd_path = os.path.join(RUN_DIR, "results_beam", "gdisplac")
    if not os.path.isfile(gd_path):
        sys.exit(f"Missing {gd_path}")
    struct_path = find_struct_file(RUN_DIR)

    ref = load_reference_nodes(struct_path)
    times, disp = load_displacement_field(gd_path, ST)

    if len(ref) != disp.shape[1]:
        sys.exit(f"{len(ref)} nodes in {struct_path} but {disp.shape[1]} per "
                 f"block in gdisplac -- these aren't the same model.")

    axial, bend = pick_axes(ref, disp)
    print(f"{os.path.basename(os.path.normpath(RUN_DIR))}: "
          f"{len(ref)} nodes, {len(times)} configurations, "
          f"t = {times[0]:.4f}-{times[-1]:.4f} s")
    print(f"beam axis = {AXIS_NAMES[axial]}, bending = {AXIS_NAMES[bend]}")

    # Deformed position of every node at every step, in the 2D plane spanned
    # by the bending (horizontal) and axial (vertical) directions.
    pos = ref[None, :, :] + DEFORM_SCALE * disp
    x = pos[:, :, bend]
    y = pos[:, :, axial]

    # Envelope over EVERY step, not just the drawn subset, so decimation can't
    # make the extremes look smaller than they were.
    x_min = x.min(axis=0)
    x_max = x.max(axis=0)
    tip_sweep = x[:, -1].max() - x[:, -1].min()
    print(f"tip sweep in {AXIS_NAMES[bend]}: {tip_sweep:.5f} "
          f"({100 * tip_sweep / np.ptp(ref[:, axial]):.1f}% of beam length)")

    stride = max(1, len(times) // MAX_CURVES)
    idx = np.arange(0, len(times), stride)

    fig, ax = plt.subplots(figsize=(7, 9))

    # One LineCollection instead of thousands of plot() calls -- same picture,
    # far less time and a much smaller PNG.
    segments = [np.column_stack([x[i], y[i]]) for i in idx]
    lc = LineCollection(segments, array=times[idx], cmap=CMAP,
                        linewidths=LINEWIDTH, alpha=ALPHA)
    ax.add_collection(lc)
    fig.colorbar(lc, ax=ax, label="Time [s]", pad=0.02)

    ax.plot(x_min, ref[:, axial], color='red', lw=1.4, label='Envelope')
    ax.plot(x_max, ref[:, axial], color='red', lw=1.4)
    ax.plot(ref[:, bend], ref[:, axial], color='black', lw=1.2, ls='--',
            label='Undeformed')

    ax.set_xlabel(f"{AXIS_NAMES[bend]} (bending direction)")
    ax.set_ylabel(f"{AXIS_NAMES[axial]} (along beam)")
    title = (f"{os.path.basename(os.path.normpath(RUN_DIR))} -- "
             f"{len(idx)} of {len(times)} configurations")
    if DEFORM_SCALE != 1.0:
        title += f"\nDISPLACEMENTS EXAGGERATED {DEFORM_SCALE}x"
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc='best')
    if EQUAL_ASPECT:
        ax.set_aspect('equal')
    ax.autoscale_view()
    ax.grid(alpha=0.2)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
    plt.savefig(out, dpi=300)
    plt.close(fig)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
