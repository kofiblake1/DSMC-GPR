#!/usr/bin/env python3
"""Plot AERO-F surface distributions for a 2D-in-3D cylinder: skin friction,
pressure coefficient and heat flux, each split into a top-surface and a
bottom-surface line, stacked as one panel per field over a shared x axis.

Which fields get a panel is the PANELS list in CONFIG; everything else here is
indifferent to how many there are.

Inputs are what AERO-F already writes out:

  * the .top mesh file -- its leading "Nodes FluidNodes" block gives every
    node's (x, y, z). This is the whole fluid mesh, not just the wetted
    surface, so geometry alone can't tell us which nodes are on the body.
  * SkinFriction.xpost -- one scalar per node per output time, nonzero ONLY
    on the wetted surface nodes (AERO-F leaves the interior and far-field
    nodes at exactly 0.0). So the nonzero mask IS the surface selection.
  * HFpUS.xpost -- heat flux per unit surface, also surface-only, and on
    exactly the same nodes (its nonzero set was verified identical to the
    skin friction's for this run).
  * PressureCoefficient.xpost -- same per-node layout, but Cp is a volume
    field: it's nonzero on all 66048 nodes, interior included. Nothing in
    that file marks the surface, so the Cp panel reuses the node selection
    derived from the skin friction, which is why SKIN_FRICTION_FILE drives
    the selection for every panel.

The mesh is a 2D problem meshed in 3D: it's one cell thick in z, with the
identical surface node pattern extruded onto two z planes (z = 0 and
z = 0.01524 here). Keeping both planes would draw every curve twice, so we
keep only the z = Z_PLANE nodes. (The two planes' values are close but not
bit-identical -- a couple percent of the peak -- since each plane gets its
own DSMC sample, so this is a choice of one plane rather than an averaging of
the two. Set Z_PLANE to the other plane's z to spot-check.)

Top vs. bottom surface is then just the sign of y, with the two apex nodes at
y ~ 0 (within Y_TOL_REL of the chord line, i.e. the fore and aft points at
x = -R and x = +R) assigned to BOTH curves so each one spans the full chord
and the two meet at the ends instead of stopping a node short.

The horizontal axis is either the node's x coordinate or its surface angle
(X_AXIS). The angle is measured about the body center from the ANGLE_ZERO_APEX
apex, so each of the two curves sweeps 0 deg at that apex to 180 deg at the
opposite one -- the usual way to read a cylinder's surface distribution, and
it also spreads out the apex regions that an x axis crowds together.

Usage: edit the CONFIG block below, then:
    python3 plot_surf_distributions.py          # last output time (default)
    python3 plot_surf_distributions.py 3        # that timestep INDEX
    python3 plot_surf_distributions.py 2.9e-5   # nearest output TIME
    python3 plot_surf_distributions.py --angle   # angle axis, overriding X_AXIS
    python3 plot_surf_distributions.py --x       # x-coordinate axis, likewise
Writes a PNG to OUTPUT_DIR (the axis choice is in the filename, so the two
versions don't overwrite each other); headless 'Agg' backend so this runs over
ssh or from a batch script without needing a display.
"""

import os
import sys
from typing import NamedTuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# CONFIG -- point this at whichever run's output you want to look at.
# ============================================================================
AERO_F_TOP_FILE = "/Volumes/SherlockScratch/rs_argon_run24/sources/argon_cylinder_kn0p002.top"
# Note the postpro folder also holds a 0-byte "PressureCoefficieint.xpost"
# (misspelled); the real file is the one below.
SKIN_FRICTION_FILE = "/Volumes/SherlockScratch/rs_argon_run24/simulations/argon_cylinder_kn0p002/postpro/PPSkinFriction.xpost"
PRESSURE_FILE = "/Volumes/SherlockScratch/rs_argon_run24/simulations/argon_cylinder_kn0p002/postpro/PressureCoefficient.xpost"
HEATFLUX_FILE = "/Volumes/SherlockScratch/rs_argon_run24/simulations/argon_cylinder_kn0p002/postpro/PPHeatFlux.xpost"
OUTPUT_DIR = os.getcwd()  # PNG lands here; change if you'd rather it land elsewhere

# One panel per entry, top to bottom, as (y-axis label, .xpost file). Add
# another of that folder's nodal fields (Pressure, Temperature, Density, Mach)
# by appending a line -- the figure grows a panel and the selection, timestep
# matching and axis handling all carry over unchanged.
#
# Only skin friction and HFpUS are surface-only fields (nonzero exactly on the
# wetted nodes, and on the same ones -- verified equal for this run); Cp and
# the rest are volume fields, nonzero at the interior nodes too. That's fine
# because every panel is sampled through the one selection built from
# SKIN_FRICTION_FILE, which must stay first here: it's also the field whose
# time list drives the timestep choice.
#
# Units: AERO-F's .xpost files don't record any, so these labels don't assert
# them. Cp is dimensionless; HFpUS ("heat flux per unit surface") comes out
# O(1e4) here, consistent with W/m^2; skin friction's normalization depends on
# the input deck. Check the deck before putting units on an axis.
PANELS = [
    ("Skin friction", SKIN_FRICTION_FILE),
    (r"Pressure coefficient $C_p$", PRESSURE_FILE),
    (r"Heat flux, HFpUS", HEATFLUX_FILE),
]

# Which output time to plot. TIMESTEP_INDEX indexes the blocks of the SKIN
# FRICTION file in file order, so -1 is the last (and latest) one -- the
# default, since that's the most converged sample. TIMESTEP_TIME, if not None,
# wins over the index and selects the block whose time is nearest to it
# instead. Either can also be overridden from the command line (see the module
# docstring): an integer argument is read as an index, anything else as a time.
# The other fields are then matched to the chosen time, not to its index --
# see match_time().
TIMESTEP_INDEX = -1
TIMESTEP_TIME = None

# How far another field's output time may sit from the skin friction's chosen
# time, as a fraction of the run's total output span, before it's worth
# flagging. These files are written by the same solver run and normally carry
# identical time lists; they only drift apart when one file is mid-rsync or
# came from a different run, which is exactly the mix-up worth catching.
TIME_MATCH_TOL_REL = 1e-6

# What to put on the horizontal axis: "x" for the node's x coordinate [m], or
# "angle" for its surface angle [deg]. Overridable with --x / --angle on the
# command line.
X_AXIS = "angle"

# Only used when X_AXIS == "angle". The angle is measured about BODY_CENTER
# (x, y) -- the angle is only meaningful about the body's own center, and
# nothing in the .top file says where that is, so it has to be stated here;
# this cylinder is centered on the origin. ANGLE_ZERO_APEX names the apex the
# sweep starts from: "-x" puts 0 deg at the x = -R apex and 180 deg at x = +R,
# "+x" is the reverse. Both curves then sweep the same 0..180 range, one over
# the top of the body and one under the bottom.
#
# "-x" is the default because that's the windward apex for this run -- Cp
# peaks there (~1.6, the stagnation value) and falls to ~0 at x = +R, so the
# freestream runs in +x and theta = 0 lands on the stagnation point, which is
# the convention cylinder Cp/cf distributions are normally plotted in. Flip
# this to "+x" for a run whose flow comes the other way.
BODY_CENTER = (0.0, 0.0)
ANGLE_ZERO_APEX = "-x"

# The z plane to keep out of the two extruded copies of the surface. See the
# module docstring -- 0.0 is the plane the problem is nominally defined on.
# Z_TOL is absolute, in mesh length units, and only has to be tight enough to
# separate the two planes (they're ~0.015 apart here).
Z_PLANE = 0.0
Z_TOL = 1e-9

# A surface node counts as "on the chord line" (and so belongs to both the top
# and bottom curves) when |y| <= Y_TOL_REL * max|y| over the surface nodes.
# Relative because the apex nodes don't land at exactly y = 0 -- they come out
# of the mesher at y ~ 1e-16, i.e. float noise on a ~0.15 radius -- and a
# strict y > 0 / y < 0 split would hand each apex to whichever side its noise
# happened to fall on. Keep this many orders of magnitude below any real
# surface spacing so it never swallows a genuine near-axis node.
Y_TOL_REL = 1e-9

# Series colors: categorical slots 1 and 2 (blue, orange) of the project's
# default palette, a pre-validated pair for colorblind separation. Both panels
# reuse them for the same two series, so one legend covers the whole figure.
COLOR_TOP = "#2a78d6"
COLOR_BOTTOM = "#eb6834"


def load_nodes(path):
    """Read the leading node block of an AERO-F .top file.

    Returns (ids, xyz) with ids an (N,) int array of the file's own 1-based
    node numbers and xyz an (N, 3) float array, both in file order. Stops at
    the first line that isn't a 4-field "id x y z" record, which is the
    "Elements ... using FluidNodes" header that follows the node block -- we
    don't need the element connectivity here.
    """
    ids = []
    xyz = []
    with open(path) as fh:
        header = fh.readline()
        if not header.lower().startswith("nodes"):
            raise ValueError(f"{path}: expected a 'Nodes' block header, got {header!r}")
        for line in fh:
            fields = line.split()
            if len(fields) != 4 or not fields[0].isdigit():
                break
            ids.append(int(fields[0]))
            xyz.append(fields[1:])
    return np.array(ids, dtype=int), np.array(xyz, dtype=float)


def load_xpost_scalar(path, n_nodes_expected=None):
    """Read a nodal scalar .xpost file written by AERO-F.

    Format is: a description line, the node count N, then one block per
    output time consisting of the time on its own line followed by N values
    (one per node, in .top node order).

    Returns (times, values, description) with times an (T,) array and values a
    (T, N) array. Parsed by flattening every token after the node count and
    reshaping into T x (1 + N) -- which is both faster than going line by line
    and indifferent to how the values are wrapped across lines.
    """
    with open(path) as fh:
        description = fh.readline()
        n_nodes = int(fh.readline())
        flat = np.fromstring(fh.read(), sep=" ")
    if n_nodes_expected is not None and n_nodes != n_nodes_expected:
        raise ValueError(
            f"{path}: {n_nodes} nodes, but the mesh has {n_nodes_expected} -- "
            f"this field isn't on the same mesh"
        )
    if flat.size % (n_nodes + 1) != 0:
        raise ValueError(
            f"{path}: {flat.size} values after the header isn't a whole number of "
            f"{n_nodes + 1}-value (time + {n_nodes} nodes) blocks -- a truncated "
            f"or still-being-written file looks like this"
        )
    blocks = flat.reshape(-1, n_nodes + 1)
    return blocks[:, 0].copy(), blocks[:, 1:], description.strip()


def select_timestep(times, index=None, time=None):
    """Resolve the configured timestep request to an index into `times`."""
    if time is not None:
        return int(np.argmin(np.abs(times - time)))
    try:
        return range(len(times))[index]  # normalizes -1 etc., and range-checks
    except IndexError:
        raise IndexError(
            f"timestep index {index} is out of range -- the file has "
            f"{len(times)} output times (valid indices 0..{len(times) - 1}, "
            f"or -1..-{len(times)} from the end)"
        ) from None


def match_time(times, target, label, reference_span):
    """Find the output in `times` closest to `target`, by time rather than index.

    Matching on time instead of index is what keeps the panels honest: the
    fields are separate files and can hold different numbers of outputs (a
    partially synced file, say), in which case index i is not the same instant
    in both. Prints a warning rather than raising if the match is loose -- a
    labeled near-miss is more useful than no plot.
    """
    i = int(np.argmin(np.abs(times - target)))
    mismatch = abs(times[i] - target)
    if mismatch > TIME_MATCH_TOL_REL * reference_span:
        print(
            f"WARNING: {label} has no output at t = {target:.6g} s; using its "
            f"nearest, t = {times[i]:.6g} s (off by {mismatch:.3g} s). The "
            f"panels are then not at the same instant."
        )
    return i


def abscissa(x, y, x_axis):
    """The horizontal coordinate to plot these surface nodes against.

    Returns (values, axis label). For "angle": the polar angle about
    BODY_CENTER, folded onto 0..180 deg measured from the ANGLE_ZERO_APEX apex,
    so the top and bottom halves each sweep the full 0..180 rather than
    splitting a 360 deg circle between them. Folding |phi| this way needs no
    radius, so it stays correct for a non-circular body (an ellipse, say) as
    long as it's convex about its center.
    """
    if x_axis == "x":
        return x, "x [m]"
    if x_axis == "angle":
        phi = np.abs(np.degrees(np.arctan2(y - BODY_CENTER[1], x - BODY_CENTER[0])))
        if ANGLE_ZERO_APEX == "-x":
            return 180.0 - phi, r"$\theta$ from $-x$ apex [deg]"
        if ANGLE_ZERO_APEX == "+x":
            return phi, r"$\theta$ from $+x$ apex [deg]"
        raise ValueError(
            f"ANGLE_ZERO_APEX must be '-x' or '+x', got {ANGLE_ZERO_APEX!r}"
        )
    raise ValueError(f"X_AXIS must be 'x' or 'angle', got {x_axis!r}")


class SurfaceSelection(NamedTuple):
    """The wetted-surface nodes on one z plane, split into top and bottom.

    i_top / i_bottom index into the full per-node arrays (so any field on the
    same mesh can be pulled through this same selection), each already sorted
    by the matching abscissa. a_top / a_bottom are those nodes' abscissa
    values -- x or angle, per the x_axis the selection was built with -- and
    xlabel is what that axis is called.
    """

    a_top: np.ndarray
    i_top: np.ndarray
    a_bottom: np.ndarray
    i_bottom: np.ndarray
    xlabel: str

    def top(self, scalar):
        return self.a_top, scalar[self.i_top]

    def bottom(self, scalar):
        return self.a_bottom, scalar[self.i_bottom]


def surface_selection(xyz, skin_friction, x_axis=None):
    """Build the top/bottom surface selection from the skin-friction field.

    See the module docstring for the three selection rules (nonzero => on the
    surface, one z plane out of the two extruded copies, sign of y with the
    apex nodes shared). Derived from skin friction alone, then reused for
    every other field -- Cp included, since Cp is nonzero in the volume too
    and so can't identify the surface by itself.

    Each half is sorted by the abscissa it will actually be plotted against,
    not always by x, so the polyline joins neighbors in the plot regardless of
    which axis is in use.
    """
    x_axis = X_AXIS if x_axis is None else x_axis
    on_surface = skin_friction != 0.0
    on_plane = np.abs(xyz[:, 2] - Z_PLANE) <= Z_TOL
    (keep,) = np.nonzero(on_surface & on_plane)
    if keep.size == 0:
        raise ValueError(
            f"no nonzero skin-friction values at z = {Z_PLANE} (+/- {Z_TOL}); "
            f"z planes present in the mesh: {np.unique(xyz[:, 2])}"
        )

    y = xyz[keep, 1]
    y_tol = Y_TOL_REL * np.abs(y).max()
    apex = np.abs(y) <= y_tol

    def sorted_by_abscissa(mask):
        idx = keep[mask]
        a, xlabel = abscissa(xyz[idx, 0], xyz[idx, 1], x_axis)
        order = np.argsort(a)
        return a[order], idx[order], xlabel

    a_top, i_top, xlabel = sorted_by_abscissa((y > 0) | apex)
    a_bottom, i_bottom, _ = sorted_by_abscissa((y < 0) | apex)
    return SurfaceSelection(a_top, i_top, a_bottom, i_bottom, xlabel)


def plot_surface_distributions(selection, panels, time, index, n_times, out_path):
    """Stack one panel per field, top- and bottom-surface line in each.

    `panels` is a list of (ylabel, per-node value array) pairs, drawn top to
    bottom over a shared x axis -- shared because every panel is the same set
    of surface nodes plotted against the same abscissa, so one axis and one
    legend serve them all and the panels stay directly comparable column by
    column. The abscissa (x or angle) comes from `selection`.
    """
    fig, axes = plt.subplots(
        len(panels), 1, figsize=(8.0, 1.0 + 2.7 * len(panels)),
        sharex=True, constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    for ax, (ylabel, values) in zip(axes, panels):
        ax.plot(*selection.top(values), lw=2.0, color=COLOR_TOP,
                label="Top surface (y > 0)")
        ax.plot(*selection.bottom(values), lw=2.0, color=COLOR_BOTTOM,
                label="Bottom surface (y < 0)")
        # Recessive reference line at zero -- both of these quantities change
        # sign (skin friction at separation/reattachment, Cp where the surface
        # pressure crosses freestream), so where the curves cross it matters.
        ax.axhline(0.0, lw=0.8, color="0.6", zorder=0)
        ax.set_ylabel(ylabel)
        ax.grid(True, lw=0.5, color="0.9")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    axes[-1].set_xlabel(selection.xlabel)
    if selection.xlabel.endswith("[deg]"):
        # The sweep is 0..180 by construction, so pin the axis to it and tick
        # it at the 30 deg marks people read these distributions by.
        axes[-1].set_xlim(0.0, 180.0)
        axes[-1].set_xticks(np.arange(0.0, 181.0, 30.0))
    # One legend for the figure: the two series mean the same thing, and wear
    # the same colors, in every panel.
    axes[0].legend(frameon=False)
    fig.suptitle(
        f"Cylinder surface distributions at t = {time:.6g} s "
        f"(output {index + 1} of {n_times}, z = {Z_PLANE:g})"
    )

    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def parse_args(argv):
    """Read the optional axis flag and timestep argument off the command line.

    Returns (index, time, x_axis), each falling back to its CONFIG default.
    Hand-rolled rather than argparse to stay in the same edit-the-config style
    as the rest of this script -- the flags are a convenience for flipping the
    axis without touching the file, not a full CLI.
    """
    args = list(argv)
    x_axis = X_AXIS
    for flag, value in (("--angle", "angle"), ("--x", "x")):
        while flag in args:
            args.remove(flag)
            x_axis = value

    index, time = TIMESTEP_INDEX, TIMESTEP_TIME
    if args:
        try:
            index, time = int(args[0]), None
        except ValueError:
            index, time = None, float(args[0])
    return index, time, x_axis


def main():
    index, time, x_axis = parse_args(sys.argv[1:])

    if not PANELS or PANELS[0][1] != SKIN_FRICTION_FILE:
        raise ValueError(
            "PANELS[0] must be the SKIN_FRICTION_FILE entry -- it's the field "
            "that defines the surface selection and the timestep"
        )

    ids, xyz = load_nodes(AERO_F_TOP_FILE)
    print(f"{len(ids)} nodes")

    fields = []  # (label, times, values) in PANELS order
    for label, path in PANELS:
        times, values, description = load_xpost_scalar(path, len(ids))
        print(f"{os.path.basename(path)}: {description}, "
              f"{len(times)} output times")
        fields.append((label, times, values))

    # Skin friction is the reference field: it picks the timestep (and, below,
    # the surface nodes), and every other field is matched to its time rather
    # than its index, since the files are written separately and need not hold
    # the same number of outputs.
    _, cf_times, cf = fields[0]
    i = select_timestep(cf_times, index=index, time=time)
    span = cf_times[-1] - cf_times[0] if len(cf_times) > 1 else 1.0
    print(f"output times: {cf_times}")
    print(f"plotting index {i}, t = {cf_times[i]:.6g} s")

    selection = surface_selection(xyz, cf[i], x_axis)
    print(f"{selection.i_top.size} top-surface nodes, "
          f"{selection.i_bottom.size} bottom-surface nodes at z = {Z_PLANE:g}")
    print(f"horizontal axis: {selection.xlabel} "
          f"(top {selection.a_top.min():.4g} to {selection.a_top.max():.4g}, "
          f"bottom {selection.a_bottom.min():.4g} to {selection.a_bottom.max():.4g})")

    panels = []
    for (label, times, values), (_, path) in zip(fields, PANELS):
        k = match_time(times, cf_times[i], os.path.basename(path), span)
        panels.append((label, values[k]))
        print(f"  {label}: {os.path.basename(path)} index {k}, "
              f"t = {times[k]:.6g} s")

    out_path = os.path.join(
        OUTPUT_DIR, f"surface_distributions_t{i}_{x_axis}.png"
    )
    plot_surface_distributions(
        selection, panels, cf_times[i], i, len(cf_times), out_path
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
