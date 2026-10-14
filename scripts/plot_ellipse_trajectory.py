#!/usr/bin/env python3
"""Trajectory of an ellipse's center and the drag on it, per DSMC case.

Grown out of the smoke-test cell at the end of examples/plot_lift_drag.ipynb.
For each case directory it writes, into its own output folder:
    trajectory_center.png   the center node's path in the x-y plane,
                            colored by time
    drag_vs_time.png        the x force (drag) on the body

Each case directory is laid out as:
    <CASE>/circle_beam.top          undeformed nodal coordinates
    <CASE>/results_ellipse/gdisplac nodal displacements at every output step
    <CASE>/EXAMPLE/lift_drag.dat    step / time / drag / lift

gdisplac stores DISPLACEMENTS, so the .top coordinates are needed to turn them
into positions.

Setup (Sherlock modules):
    ml load math py-numpy/1.26.3_py312 viz py-matplotlib/3.10.3_py312

Usage: edit CASE_DIRS below, then:
    python3 plot_ellipse_trajectory.py
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# CONFIG
# ============================================================================
BASE = "/Volumes/SherlockScratch/DSMC_FEM_VALIDATION_P1/ELLIPSE"
CASE_DIRS = [
    f"{BASE}/KN_1.0",
    f"{BASE}/KN_0.1",
]

# Each case gets its own subfolder under OUTPUT_DIR/OUT_ROOT, named after the
# case directory (e.g. ellipse_plots/KN_1.0/).
OUTPUT_DIR = os.getcwd()
OUT_ROOT = "ellipse_plots_nn"

# File layout inside a case directory.
TOP_NAME = "circle_beam.top"
DISP_RELPATH = os.path.join("results_ellipse", "gdisplac")
LIFT_DRAG_RELPATH = os.path.join("EXAMPLE", "lift_drag.dat")

# Which node to track. None = pick automatically (see find_center_node): the
# node nearest the origin that actually moves. The notebook hardcoded index
# 166 for this mesh; the automatic rule reproduces that without the magic
# number. Set an integer here to force a specific 0-based index.
CENTER_NODE_INDEX = None

# Only plot up to this time [s]; None uses the whole record. The notebook
# truncated at 3e-3.
TIME_MAX = None

# -- Drag normalization ------------------------------------------------------
# True divides the drag by rho_inf * U_0^2 * L, giving a dimensionless
# coefficient that is comparable across cases with different densities (the
# whole point when comparing KN_1.0 against KN_0.1, whose raw drags differ by
# ~8x purely from nrho). Each factor is read from that case's own input files:
#   rho_inf = m_Ar * nrho   m_Ar from argon.species, nrho from sparta_ellipse
#   U_0     = sqrt(2*R_g*T_inf)   with R_g = k_B/m_Ar, T_inf from
#             config_ellipse.txt -- this is the most probable thermal speed,
#             NOT the freestream velocity u_inf (1538.73 m/s here)
#   L       = LENGTH_SCALE below
# A case whose parameters can't be read is plotted unnormalized with a
# warning rather than being skipped.
NORMALIZE_DRAG = False

# Semi-major axis [m] used as L in the normalization. Note the mesh's own
# outer radius is 0.01 m, so set this to 0.01 if you want the geometric
# length rather than a unit reference length.
LENGTH_SCALE = 0.01

# Files inside a case directory that carry the freestream parameters.
SPECIES_NAME = "argon.species"
SPARTA_INPUT_NAME = "sparta_ellipse"
CONFIG_NAME = "config_ellipse.txt"

BOLTZMANN = 1.380649e-23  # J/K

# Every figure is square (same width and height).
FIGSIZE = (7.0, 7.0)

CMAP = 'viridis'


def load_reference_nodes(path, n_nodes):
    """Read the first n_nodes rows of a .top file's node block.

    Line 1 is the "Nodes nodeset" header and the node rows are followed by
    "Elements ..." sections, so the read has to be capped at the node count
    that gdisplac reports rather than running to EOF."""
    data = np.loadtxt(path, skiprows=1, max_rows=n_nodes)
    return data[:, 0].astype(int), data[:, 1:4]


def load_displacement_field(path):
    """Parse gdisplac into (times, disp) with disp shape (n_steps, n_nodes, 3).

    Format: a 2-line header (title, node count), then repeating blocks of one
    time line followed by n_nodes "dx dy dz" lines."""
    with open(path) as f:
        lines = f.readlines()

    n_nodes = int(lines[1])
    block_size = 1 + n_nodes
    body = lines[2:]
    n_blocks = len(body) // block_size

    times = np.empty(n_blocks)
    disp = np.empty((n_blocks, n_nodes, 3))
    for b in range(n_blocks):
        block = body[b * block_size:(b + 1) * block_size]
        times[b] = float(block[0])
        disp[b] = [[float(v) for v in line.split()] for line in block[1:]]
    return times, disp


def find_center_node(node_xyz, disp):
    """Index of the node at the ellipse's center.

    The nearest-to-origin node is NOT simply argmin(radius): node 1 sits at
    exactly (0,0,0) but is the single pinned node, with zero displacement at
    every step, so tracking it would draw a stationary dot. Nodes that never
    move are excluded, and the nearest of the rest is taken -- which is the
    node the notebook hardcoded as index 166."""
    if CENTER_NODE_INDEX is not None:
        return CENTER_NODE_INDEX

    radius = np.linalg.norm(node_xyz[:, :2], axis=1)
    moves = np.abs(disp).max(axis=(0, 2)) > 0.0
    if not moves.any():
        sys.exit("Every node is stationary in gdisplac -- nothing to track.")

    candidates = np.where(moves)[0]
    return int(candidates[np.argmin(radius[candidates])])


def read_species_mass(path):
    """Particle mass [kg] from a SPARTA species file.

    Columns are: ID, Molwt (amu), Molmass (kg), ... -- the third field is the
    per-particle mass in kg, which is what's wanted here (not the amu value).
    Returns the first species listed, or None."""
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        for line in f:
            line = line.split('#')[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    return float(parts[2])
                except ValueError:
                    return None
    return None


def read_nrho(path):
    """Number density [m^-3] from the "global ... nrho <value>" line.

    Comments are stripped first: these decks carry commented-out alternative
    nrho values (and explanatory "# nrho = rho_inf/m = ..." lines) that must
    not be mistaken for the active setting."""
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        for line in f:
            line = line.split('#')[0].strip()
            parts = line.split()
            if not parts or parts[0] != 'global' or 'nrho' not in parts:
                continue
            try:
                return float(parts[parts.index('nrho') + 1])
            except (IndexError, ValueError):
                continue
    return None


def read_config_value(path, key):
    """Value of a "<key>  <value>" line in config_ellipse.txt."""
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        for line in f:
            parts = line.split('#')[0].split()
            if len(parts) >= 2 and parts[0] == key:
                try:
                    return float(parts[1])
                except ValueError:
                    return None
    return None


def drag_scale(case_dir, label):
    """rho_inf * U_0^2 * L for a case, or None if a parameter is missing.

    Also returns the pieces so they can be printed and checked by eye."""
    mass = read_species_mass(os.path.join(case_dir, SPECIES_NAME))
    nrho = read_nrho(os.path.join(case_dir, SPARTA_INPUT_NAME))
    t_inf = read_config_value(os.path.join(case_dir, CONFIG_NAME), 't_inf')

    if not mass or not nrho or not t_inf:
        print(f"WARNING: {label}: could not read "
              f"mass={mass} nrho={nrho} t_inf={t_inf} -- "
              f"plotting drag unnormalized.")
        return None

    rho_inf = mass * nrho
    R_g = BOLTZMANN / mass
    u_0 = np.sqrt(2.0 * R_g * t_inf)
    scale = rho_inf * u_0**2 * LENGTH_SCALE
    return {'mass': mass, 'nrho': nrho, 't_inf': t_inf, 'rho_inf': rho_inf,
            'R_g': R_g, 'U_0': u_0, 'L': LENGTH_SCALE, 'scale': scale}


def load_lift_drag(path):
    """Read step / time / drag / lift columns from lift_drag.dat."""
    _, time, drag, lift = np.loadtxt(path, skiprows=1, unpack=True)
    return time, drag, lift


def plot_trajectory(path, traj_x, traj_y, times, title, time_label,
                    x_label, y_label):
    """Center-node path in the x-y plane, colored by time.

    Equal aspect throughout: an orbit is only meaningful if the two axes
    share a scale, otherwise a circular path reads as an ellipse. To get a
    square plot box anyway, the SHORTER axis is padded out to the longer
    one's span rather than the aspect ratio being relaxed -- the path keeps
    its true shape and still fills a square."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(traj_x, traj_y, color='0.75', lw=0.8, zorder=1)
    pts = ax.scatter(traj_x, traj_y, c=times, cmap=CMAP, s=26, zorder=2,
                     edgecolor='black', linewidth=0.3)
    fig.colorbar(pts, ax=ax, label=time_label)

    ax.scatter(traj_x[0], traj_y[0], marker='o', s=110, facecolor='none',
               edgecolor='tab:green', linewidth=1.6, zorder=3, label='Start')
    ax.scatter(traj_x[-1], traj_y[-1], marker='s', s=110, facecolor='none',
               edgecolor='tab:red', linewidth=1.6, zorder=3, label='End')

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, fontsize=10)

    # Equal data spans on both axes -> square box under set_aspect('equal').
    half = 0.5 * 1.1 * max(np.ptp(traj_x), np.ptp(traj_y), 1e-12)
    cx = 0.5 * (traj_x.max() + traj_x.min())
    cy = 0.5 * (traj_y.max() + traj_y.min())
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect('equal')
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_drag(path, time, drag, title, ylabel, xlabel):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(time, drag, lw=0.9, color='tab:blue')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def process_case(case_dir):
    """Load one case and write its two figures. Returns True on success."""
    label = os.path.basename(os.path.normpath(case_dir))
    top_path = os.path.join(case_dir, TOP_NAME)
    disp_path = os.path.join(case_dir, DISP_RELPATH)
    ld_path = os.path.join(case_dir, LIFT_DRAG_RELPATH)

    missing = [p for p in (top_path, disp_path, ld_path)
               if not os.path.isfile(p)]
    if missing:
        for p in missing:
            print(f"WARNING: {label}: missing {p} -- skipping this case.")
        return False

    times, disp = load_displacement_field(disp_path)
    node_ids, node_xyz = load_reference_nodes(top_path, disp.shape[1])
    if len(node_ids) != disp.shape[1]:
        print(f"WARNING: {label}: {len(node_ids)} nodes in {TOP_NAME} but "
              f"{disp.shape[1]} in gdisplac -- skipping.")
        return False

    idx = find_center_node(node_xyz, disp)
    radius = np.linalg.norm(node_xyz[idx, :2])
    print(f"\n{label}: {disp.shape[1]} nodes, {len(times)} steps, "
          f"t = {times[0]:.3e}-{times[-1]:.3e} s")
    print(f"  center node: index {idx}, id {node_ids[idx]}, "
          f"r = {radius:.3e} m from origin")

    # Resolve the normalization before anything is plotted: the same factors
    # set the drag scale AND the time scale, and both figures have to agree on
    # which clock they're using.
    params = drag_scale(case_dir, label) if NORMALIZE_DRAG else None
    if params:
        print(f"  normalization: m = {params['mass']:.4g} kg, "
              f"nrho = {params['nrho']:.4g} m^-3, "
              f"T_inf = {params['t_inf']:g} K")
        print(f"    rho_inf = {params['rho_inf']:.4g} kg/m^3, "
              f"R_g = {params['R_g']:.4g} J/(kg K), "
              f"U_0 = {params['U_0']:.4g} m/s, L = {params['L']:g} m")
        print(f"    rho_inf*U_0^2*L = {params['scale']:.5g}, "
              f"U_0/L = {params['U_0'] / params['L']:.5g} 1/s")

    # Time in units of L/U_0 when normalizing (t* = U_0 t / L), applied to
    # both the drag x axis and the trajectory colorbar so the two figures
    # share one clock. Raw seconds are kept for the printed ranges.
    t_factor = params['U_0'] / params['L'] if params else 1.0
    time_label = r"$U_0 t / L$" if params else "Time [s]"

    # Positions in units of L on the same switch, so the trajectory is in
    # body lengths rather than metres. Dividing both axes by the same L
    # leaves the path's shape (and the equal aspect) untouched.
    pos_factor = 1.0 / params['L'] if params else 1.0
    x_label = r"$x / L$" if params else "x [m]"
    y_label = r"$y / L$" if params else "y [m]"

    traj_x = node_xyz[idx, 0] + disp[:, idx, 0]
    traj_y = node_xyz[idx, 1] + disp[:, idx, 1]

    keep = slice(None) if TIME_MAX is None else times <= TIME_MAX
    t_keep, traj_x, traj_y = times[keep], traj_x[keep], traj_y[keep]
    print(f"  trajectory: x {traj_x.min():.5g}..{traj_x.max():.5g}, "
          f"y {traj_y.min():.5g}..{traj_y.max():.5g} "
          f"({len(t_keep)} points)")

    ld_time, drag, _ = load_lift_drag(ld_path)
    if TIME_MAX is not None:
        mask = ld_time <= TIME_MAX
        ld_time, drag = ld_time[mask], drag[mask]

    drag_label = "Drag (x force)"
    if params:
        drag = drag / params['scale']
        drag_label = r"Drag / ($\rho_\infty U_0^2 L$)"

    print(f"  drag: mean {drag.mean():.5g}, min {drag.min():.5g}, "
          f"max {drag.max():.5g} ({len(drag)} samples, "
          f"t = {ld_time[0]:.3e}-{ld_time[-1]:.3e} s)")

    out_dir = os.path.join(OUTPUT_DIR, OUT_ROOT, label)
    os.makedirs(out_dir, exist_ok=True)

    traj_png = os.path.join(out_dir, "trajectory_center.png")
    drag_png = os.path.join(out_dir, "drag_vs_time.png")
    plot_trajectory(traj_png, traj_x * pos_factor, traj_y * pos_factor,
                    t_keep * t_factor,
                    f"{label} -- trajectory of center node "
                    f"{node_ids[idx]}", time_label, x_label, y_label)
    plot_drag(drag_png, ld_time * t_factor, drag,
              f"{label} -- drag (x force)", drag_label, time_label)
    print(f"  wrote {traj_png}")
    print(f"  wrote {drag_png}")
    return True


def main():
    done = sum(process_case(d) for d in CASE_DIRS)
    if not done:
        sys.exit("No cases could be processed.")
    print(f"\nProcessed {done} of {len(CASE_DIRS)} cases into "
          f"{os.path.join(OUTPUT_DIR, OUT_ROOT)}/")


if __name__ == "__main__":
    main()
