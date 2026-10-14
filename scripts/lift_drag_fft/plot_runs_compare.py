#!/usr/bin/env python3
"""Overlay time series + FFT + RMS for SEVERAL coupled AERO-S/SPARTA runs.

Companion to plot_lift_drag.py, which analyzes one run in depth (time series,
FFT, and wavelet transforms). This script does time series, FFT and RMS ONLY
-- no CWT -- but takes a list of run directories and overlays them, so you can
compare runs across a structural sweep directly.

Every figure is written as its own PNG into PLOT_SUBDIR (created if needed),
one file per signal per plot type, plus two combined overview sheets.

Signals are generated automatically for tip displacement x, y AND magnitude
(TIP_COMPONENTS) -- no need to re-run with a different component selected.

RUN ORDER: runs are always sorted by ascending Young's modulus, read from each
run's branch_params.txt, regardless of the order they appear in RUN_DIRS. The
RMS plots use f1/fs as a numeric x axis (also from branch_params.txt), so they
read as a lock-in response curve rather than a categorical bar chart.

You give run directories; the file layout inside each is:
    <RUN_DIR>/lift_drag.dat               <- or <RUN_DIR>/EXAMPLE_BEAM/
    <RUN_DIR>/results_beam/gdisplac
    <RUN_DIR>/branch_params.txt           <- E_Pa, f1_Hz, fs_measured_Hz
lift_drag.dat is looked for in the run directory itself AND in an
EXAMPLE_BEAM/ subdirectory, because runs differ on this depending on how they
were launched. Point RUN_DIRS at the run directory either way -- gdisplac is
always at <RUN_DIR>/results_beam/, so pointing at EXAMPLE_BEAM/ itself would
find the forces but lose the displacement.

A run missing either data file is skipped with a warning rather than killing
the whole plot, so a sweep that's still filling in is fine to point at.

The loaders and the low-pass filter are imported from plot_lift_drag.py rather
than duplicated -- fix a parsing bug once, both scripts get it.

Setup (Sherlock modules):
    ml load math py-numpy/1.26.3_py312 py-scipy/1.16.0_py312 \
        py-pywavelets/1.6.0_py312 viz py-matplotlib/3.10.3_py312
(pywavelets is only needed because plot_lift_drag.py imports it at module
level; this script itself does no wavelet work.)

Usage: edit RUN_DIRS below, then:
    python3 plot_runs_compare.py
"""

import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Import the shared loaders from the sibling script. Resolved off __file__ so
# this works no matter what directory you invoke it from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_lift_drag import load_lift_drag, load_tip_displacement, lowpass

# ============================================================================
# CONFIG
# ============================================================================
# The runs to overlay. Order here does NOT matter -- runs are re-sorted by
# ascending Young's modulus before anything is plotted.
BASE = "/Volumes/SherlockScratch/DSMC_FEM_VALIDATION_P1/BEAM_LOCKIN"
RUN_DIRS = [
    # f1_0p70         f1_0p80         f1_0p85         f1_0p90         f1_0p95         f1_1p00         f1_1p05         f1_1p10         f1_1p20         f1_1p30
    f"{BASE}/PHASE_2R/PHASE_2R_SWEEP/f1_0p70",
    f"{BASE}/PHASE_2R/PHASE_2R_SWEEP/f1_0p80",
    f"{BASE}/PHASE_2R/PHASE_2R_SWEEP/f1_0p85",
    f"{BASE}/PHASE_2R/PHASE_2R_SWEEP/f1_0p90",
    f"{BASE}/PHASE_2R/PHASE_2R_SWEEP/f1_0p95",
    f"{BASE}/PHASE_2R/PHASE_2R_SWEEP/f1_1p00",
    f"{BASE}/PHASE_2R/PHASE_2R_SWEEP/f1_1p05",
    f"{BASE}/PHASE_2R/PHASE_2R_SWEEP/f1_1p10",
    f"{BASE}/PHASE_2R/PHASE_2R_SWEEP/f1_1p20",
    f"{BASE}/PHASE_2R/PHASE_2R_SWEEP/f1_1p30",
]

# Where to look for lift_drag.dat inside a run directory, in order. "" means
# the run directory itself.
LIFT_DRAG_SUBDIRS = ("", "EXAMPLE_BEAM")

# Every PNG is written into OUTPUT_DIR/PLOT_SUBDIR, which is created if it
# doesn't exist.
OUTPUT_DIR = os.getcwd()
PLOT_SUBDIR = "runs_compare_plots_stable"

# Also write the two multi-panel overview sheets alongside the individual
# files. They're the same content, just easier to scan at a glance.
ALSO_SAVE_COMBINED = True

# Skip data before this SPARTA absolute time, for every run.
ST = 2.75

# gdisplac's time column restarts at 0.0 at the start of structural coupling,
# so it needs restart_step * SPARTA_dt added to land on SPARTA's absolute
# clock (see plot_lift_drag.py for the full explanation). Runs restarted from
# DIFFERENT checkpoints need different offsets -- otherwise their tip traces
# are shifted relative to each other and to their own lift/drag. Add an entry
# keyed by the run's label (the directory basename) to override the default.
# 2.5 s for the PHASE_2R sweep: those runs restart at SPARTA step 500000 with
# sparta_timestep_s = 5e-6 (both in branch_params.txt), and their lift_drag.dat
# indeed starts at 2.5 s while gdisplac starts at 0. Set this to 0 for runs
# that were not restarted; the alignment check below will tell you if it's
# wrong.
DEFAULT_RESTART_TIME_OFFSET = 500000 * 5e-6  # = 2.5 s
RESTART_TIME_OFFSETS = {
    # "RUN2_rho075_early_restart": 100000 * 1e-5,
}

# Tip displacement components to generate plots for. 'z' is omitted because
# the beam model constrains it to exactly zero; 'magnitude' is
# sqrt(x^2+y^2+z^2).
TIP_COMPONENTS = ('x', 'y', 'magnitude')

# Optional zero-phase low-pass applied before plotting, in Hz. Set to None to
# plot the raw signals. Filtering mainly helps the FFT panels: the broadband
# DSMC particle-noise floor runs all the way to Nyquist, and with several runs
# overlaid those floors stack into a haze that buries the peaks.
LOWPASS_CUTOFF = None

# X limit for the FFT panels [Hz]. All the physics here sits below ~100 Hz.
FFT_XMAX = 200.0

# Restrict the RMS to the time interval EVERY run has in common. Records in a
# sweep rarely end together (some runs are still going, some died early), and
# an RMS taken over a longer window -- which therefore includes a different
# share of the startup transient -- isn't comparable to one taken over a
# settled stretch. Set False to use each run's full record instead.
RMS_COMMON_WINDOW = True

# Provenance file inside each run dir holding E_Pa / f1_Hz / fs_measured_Hz.
BRANCH_PARAMS_NAME = "branch_params.txt"


def signal_label(key):
    """Axis label for a signal key ('lift', 'drag', 'tip_x', ...)."""
    if key.startswith('tip_'):
        return f"Tip disp. {key[4:]} [m]"
    return key.capitalize()


def run_label(run_dir, all_dirs):
    """Short legend label for a run: the directory basename, or
    "<parent>/<basename>" when basenames collide across the list (sweeps love
    to reuse names like 'run' or 'f1_1p00' under different parents)."""
    base = os.path.basename(os.path.normpath(run_dir))
    bases = [os.path.basename(os.path.normpath(d)) for d in all_dirs]
    if bases.count(base) > 1:
        parent = os.path.basename(os.path.dirname(os.path.normpath(run_dir)))
        return f"{parent}/{base}"
    return base


def find_lift_drag(run_dir):
    """Locate lift_drag.dat for a run: directly in run_dir, or one level down
    in EXAMPLE_BEAM/ (see LIFT_DRAG_SUBDIRS). Returns the path, or None if it
    isn't in either place."""
    for sub in LIFT_DRAG_SUBDIRS:
        path = os.path.join(run_dir, sub, "lift_drag.dat")
        if os.path.isfile(path):
            return path
    return None


def read_branch_params(run_dir):
    """Parse a run's branch_params.txt into {key: float}.

    Format is "key = value" lines with '#' comments, written by
    generate_structural_sweep.py. Non-numeric values are skipped -- only the
    numeric provenance (E_Pa, f1_Hz, fs_measured_Hz, ...) is of interest
    here. Returns {} if the file doesn't exist."""
    path = os.path.join(run_dir, BRANCH_PARAMS_NAME)
    params = {}
    if not os.path.isfile(path):
        return params
    with open(path) as f:
        for line in f:
            line = line.split('#')[0].strip()
            if '=' not in line:
                continue
            key, _, value = line.partition('=')
            try:
                params[key.strip()] = float(value.strip())
            except ValueError:
                pass
    return params


def sweep_metadata(run_dir, label):
    """Young's modulus [Pa] and the f1/fs ratio for a run.

    Both come from branch_params.txt. The ratio prefers the measured
    f1_Hz/fs_measured_Hz over the requested f1_over_fs_target so the axis
    reflects what was actually simulated; if branch_params.txt is missing
    entirely it falls back to parsing the directory name ("f1_0p85" -> 0.85),
    and E is left as None."""
    params = read_branch_params(run_dir)
    E = params.get('E_Pa')

    f1, fs = params.get('f1_Hz'), params.get('fs_measured_Hz')
    if f1 is not None and fs:
        ratio = f1 / fs
    elif 'f1_over_fs_target' in params:
        ratio = params['f1_over_fs_target']
    else:
        # "f1_0p85" -> 0.85. Only a fallback; branch_params.txt is canonical.
        m = re.search(r'f1_(\d+)p(\d+)', label)
        ratio = float(f"{m.group(1)}.{m.group(2)}") if m else None
        if ratio is not None:
            print(f"NOTE: {label}: no {BRANCH_PARAMS_NAME}, took f1/fs = "
                  f"{ratio:g} from the directory name.")
    return E, ratio


def sort_runs_by_modulus(runs):
    """Sort runs by ascending Young's modulus, in place.

    Falls back to f1/fs if any run is missing E (the two are monotonically
    related -- f1 scales as sqrt(E) at fixed geometry and density -- so the
    resulting order is the same), and leaves the list alone if neither is
    available for every run. One key is chosen for the whole list rather than
    per run, since mixing E (~1e5 Pa) and a ratio (~1) as sort keys would
    interleave them nonsensically."""
    for key, name in (('E_Pa', "Young's modulus"), ('f1_over_fs', "f1/fs")):
        if all(r[key] is not None for r in runs):
            runs.sort(key=lambda r: r[key])
            print(f"Runs ordered by ascending {name}.")
            return
    print("WARNING: neither E_Pa nor f1/fs available for every run -- "
          "keeping the order given in RUN_DIRS.")


def amplitude_spectrum(signal, dt):
    """Single-sided amplitude spectrum of signal, mean removed.

    Normalized by record length (2*|rfft|/N) rather than returning raw
    |rfft| like plot_lift_drag.py's fft_and_peaks(). This matters here and not
    there: raw FFT magnitude scales with the number of samples, so overlaying
    a 5 s run and a 7.5 s run unnormalized would show the longer run as
    uniformly "stronger" purely from having more data. Normalized, the peak
    height is the physical amplitude of that sinusoid and the runs are
    directly comparable."""
    n = len(signal)
    freq = np.fft.rfftfreq(n, d=dt)
    amp = 2.0 * np.abs(np.fft.rfft(signal - signal.mean())) / n
    return freq, amp


def fluctuating_rms(signal):
    """RMS of the signal about its own mean (i.e. its standard deviation).

    The mean is removed so this measures oscillation strength rather than
    operating point -- otherwise drag's RMS would be dominated by its large
    steady value (~0.3) and would barely move between runs no matter how the
    unsteady response changed. The mean is reported separately in the printed
    table. This is also consistent with the FFT panels, which mean-remove
    before transforming."""
    return float(np.std(signal))


def common_window(runs):
    """Time interval covered by every run, across all of its signals.

    Exits rather than returning an empty interval: an empty window silently
    produces NaN RMS values for every run, which plots as a blank figure. The
    usual cause is a wrong RESTART_TIME_OFFSETS entry leaving a run's tip
    trace on a different stretch of the clock from its forces."""
    t0 = max(max(t[0] for t, _, _ in r['signals'].values()) for r in runs)
    t1 = min(min(t[-1] for t, _, _ in r['signals'].values()) for r in runs)
    if t1 <= t0:
        spans = "\n".join(
            f"  {r['label']:16s} " + "  ".join(
                f"{k}: {t[0]:.3f}-{t[-1]:.3f}"
                for k, (t, _, _) in r['signals'].items())
            for r in runs)
        sys.exit(f"No time window is common to all runs (latest start "
                 f"{t0:.3f} s is after earliest end {t1:.3f} s), so the RMS "
                 f"would be NaN. Check DEFAULT_RESTART_TIME_OFFSET / "
                 f"RESTART_TIME_OFFSETS.\nPer-run spans:\n{spans}")
    return t0, t1


def load_run(run_dir, all_dirs):
    """Load one run. Returns a dict of traces, or None if data is missing.

    run['signals'] maps a signal key to (time, dt, values) so every plotting
    routine can treat lift, drag and each tip component identically despite
    lift/drag and the tip being sampled at different rates."""
    label = run_label(run_dir, all_dirs)
    ld_path = find_lift_drag(run_dir)
    gd_path = os.path.join(run_dir, "results_beam", "gdisplac")

    missing = []
    if ld_path is None:
        searched = " or ".join(os.path.join(run_dir, sub) or run_dir
                               for sub in LIFT_DRAG_SUBDIRS)
        missing.append(f"lift_drag.dat (looked in {searched})")
    if not os.path.isfile(gd_path):
        missing.append(gd_path)
    if missing:
        for m in missing:
            print(f"WARNING: {label}: missing {m} -- skipping this run.")
        return None

    offset = RESTART_TIME_OFFSETS.get(label, DEFAULT_RESTART_TIME_OFFSET)
    time_ld, lift, drag = load_lift_drag(ld_path, ST)
    time_tip, tip_x, tip_y, tip_z = load_tip_displacement(gd_path, offset, ST)

    tip_all = {
        'x': tip_x,
        'y': tip_y,
        'z': tip_z,
        'magnitude': np.sqrt(tip_x**2 + tip_y**2 + tip_z**2),
    }

    dt_ld = time_ld[1] - time_ld[0]
    dt_tip = time_tip[1] - time_tip[0]

    # Catch a wrong restart offset before it silently corrupts the
    # comparison: lift/drag and the tip come from the same simulated
    # interval, so their time ranges should essentially coincide. If they
    # don't, this run's RESTART_TIME_OFFSETS entry is wrong and its tip trace
    # is shifted relative to both its own forces and every other run.
    overlap = (min(time_ld[-1], time_tip[-1]) - max(time_ld[0], time_tip[0]))
    span = max(time_ld[-1] - time_ld[0], time_tip[-1] - time_tip[0])
    if overlap < 0.5 * span:
        print(f"WARNING: {label}: lift/drag covers "
              f"{time_ld[0]:.3f}-{time_ld[-1]:.3f} s but the tip covers "
              f"{time_tip[0]:.3f}-{time_tip[-1]:.3f} s (offset applied: "
              f"{offset:g} s). These should overlap -- set "
              f"RESTART_TIME_OFFSETS['{label}'] to this run's "
              f"restart_step * SPARTA dt.")

    signals = {
        'lift': (time_ld, dt_ld, lift),
        'drag': (time_ld, dt_ld, drag),
    }
    for comp in TIP_COMPONENTS:
        signals[f"tip_{comp}"] = (time_tip, dt_tip, tip_all[comp])

    if LOWPASS_CUTOFF is not None:
        for key, (t, dt, values) in signals.items():
            signals[key] = (t, dt, lowpass(values, dt, LOWPASS_CUTOFF,
                                           label=f"{label} {key}"))

    E, ratio = sweep_metadata(run_dir, label)
    return {'label': label, 'signals': signals, 'E_Pa': E,
            'f1_over_fs': ratio}


# ---------------------------------------------------------------------------
# Panel painters. Each draws one signal onto a given axes, so the same code
# backs both the individual PNGs and the combined overview sheets.
# ---------------------------------------------------------------------------
def draw_timeseries(ax, runs, key):
    for run in runs:
        t, _, values = run['signals'][key]
        ax.plot(t, values, lw=0.6, color=run['color'], label=run['label'])
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(signal_label(key))


def draw_fft(ax, runs, key):
    for run in runs:
        _, dt, values = run['signals'][key]
        freq, amp = amplitude_spectrum(values, dt)
        ax.plot(freq, amp, lw=0.8, color=run['color'], label=run['label'])
    ax.set_xlim(0, FFT_XMAX)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel(f"{signal_label(key)} amplitude")


def draw_rms(ax, runs, key, rms_values):
    """RMS against f1/fs -- a lock-in response curve.

    A numeric x axis (rather than one categorical tick per run) puts the runs
    at their true spacing, so an unevenly sampled sweep doesn't read as
    evenly spaced, and marks where f1/fs = 1."""
    ratios = [r['f1_over_fs'] for r in runs]
    values = rms_values[key]

    if any(r is None for r in ratios):
        # No usable ratio for some run: fall back to evenly spaced positions
        # labeled by run name, so the plot still renders.
        ratios = list(range(len(runs)))
        ax.set_xticks(ratios)
        ax.set_xticklabels([r['label'] for r in runs], rotation=45,
                           ha='right', fontsize=8)
        ax.set_xlabel("Run (ascending Young's modulus)")
    else:
        ax.set_xlabel("$f_1 / f_s$")
        ax.axvline(1.0, color='black', ls=':', lw=1.0, zorder=1)

    ax.plot(ratios, values, '-', color='0.7', lw=1.0, zorder=2)
    ax.scatter(ratios, values, c=[r['color'] for r in runs], s=45,
               edgecolor='black', linewidth=0.4, zorder=3)
    ax.set_ylabel(f"RMS about mean -- {signal_label(key)}")
    ax.margins(y=0.15)
    ax.grid(alpha=0.25)


def save_single(painter, path, title, legend=False, figsize=(7.5, 4.5)):
    """Render one painter onto its own figure and write it to path."""
    fig, ax = plt.subplots(figsize=figsize)
    painter(ax)
    ax.set_title(title, fontsize=10)
    if legend:
        ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    runs = [r for r in (load_run(d, RUN_DIRS) for d in RUN_DIRS) if r]
    if not runs:
        sys.exit("No runs had both data files -- nothing to plot.")

    sort_runs_by_modulus(runs)

    # Color by sweep position rather than the default cycle: with ~10 runs the
    # cycle wraps and repeats, whereas a continuous map also makes the
    # ordering itself readable.
    shades = plt.cm.viridis(np.linspace(0.0, 0.92, len(runs)))
    for run, shade in zip(runs, shades):
        run['color'] = shade

    keys = ['lift', 'drag'] + [f"tip_{c}" for c in TIP_COMPONENTS]

    print(f"\n{'run':16s} {'E [Pa]':>10s} {'f1/fs':>7s}")
    for run in runs:
        E = f"{run['E_Pa']:10.0f}" if run['E_Pa'] is not None else " " * 10
        ratio = (f"{run['f1_over_fs']:7.3f}"
                 if run['f1_over_fs'] is not None else " " * 7)
        print(f"{run['label']:16s} {E} {ratio}")

    # -- Dominant frequency per run per signal -------------------------------
    print()
    for run in runs:
        for key in keys:
            t, dt, values = run['signals'][key]
            freq, amp = amplitude_spectrum(values, dt)
            in_band = freq <= FFT_XMAX
            peak = freq[in_band][np.argmax(amp[in_band])]
            print(f"{run['label']:16s} {key:13s} peak: {peak:7.2f} Hz  "
                  f"(fs = {1/dt:.0f} Hz, {len(values)} samples, "
                  f"t = {t[0]:.3f}-{t[-1]:.3f} s)")

    # -- RMS table -----------------------------------------------------------
    if RMS_COMMON_WINDOW:
        t0, t1 = common_window(runs)
        window = f"{t0:.3f}-{t1:.3f} s"
        print(f"\nRMS over the window common to all runs: {window}")
    else:
        t0, t1 = -np.inf, np.inf
        window = "each run's full record"
        print("\nRMS over each run's full record (RMS_COMMON_WINDOW = False)")

    print(f"{'run':16s} {'signal':13s} {'mean':>12s} {'RMS (fluct)':>12s} "
          f"{'samples':>8s}")
    rms_values = {key: [] for key in keys}
    for run in runs:
        for key in keys:
            t, _, values = run['signals'][key]
            seg = values[(t >= t0) & (t <= t1)]
            rms_values[key].append(fluctuating_rms(seg))
            print(f"{run['label']:16s} {key:13s} {seg.mean():12.5g} "
                  f"{rms_values[key][-1]:12.5g} {len(seg):8d}")

    # -- Individual figures --------------------------------------------------
    plot_dir = os.path.join(OUTPUT_DIR, PLOT_SUBDIR)
    os.makedirs(plot_dir, exist_ok=True)
    filt = (f" (low-pass {LOWPASS_CUTOFF:g} Hz)"
            if LOWPASS_CUTOFF is not None else "")
    written = []

    for key in keys:
        name = signal_label(key)
        for kind, painter, title, legend in (
            ("timeseries", lambda ax, k=key: draw_timeseries(ax, runs, k),
             f"{name} -- time series{filt}", True),
            ("fft", lambda ax, k=key: draw_fft(ax, runs, k),
             f"{name} -- amplitude spectrum{filt}", True),
            ("rms", lambda ax, k=key: draw_rms(ax, runs, k, rms_values),
             f"{name} -- fluctuating RMS ({window}){filt}", False),
        ):
            path = os.path.join(plot_dir, f"{kind}_{key}.png")
            save_single(painter, path, title, legend=legend)
            written.append(path)

    # -- Combined overview sheets -------------------------------------------
    if ALSO_SAVE_COMBINED:
        fig, axes = plt.subplots(len(keys), 2, figsize=(14, 3.2 * len(keys)),
                                 squeeze=False)
        for row, key in enumerate(keys):
            draw_timeseries(axes[row][0], runs, key)
            draw_fft(axes[row][1], runs, key)
        axes[0][0].legend(fontsize=6, ncol=2)
        axes[0][0].set_title(f"Time series{filt}", fontsize=10)
        axes[0][1].set_title(f"Amplitude spectrum{filt}", fontsize=10)
        plt.tight_layout()
        path = os.path.join(plot_dir, "overview_timeseries_fft.png")
        fig.savefig(path, dpi=300)
        plt.close(fig)
        written.append(path)

        fig, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 4.2),
                                 squeeze=False)
        for col, key in enumerate(keys):
            draw_rms(axes[0][col], runs, key, rms_values)
            axes[0][col].set_title(signal_label(key), fontsize=10)
        fig.suptitle(f"Fluctuating RMS vs $f_1/f_s$ ({window}){filt}",
                     fontsize=11)
        plt.tight_layout()
        path = os.path.join(plot_dir, "overview_rms.png")
        fig.savefig(path, dpi=300)
        plt.close(fig)
        written.append(path)

    print(f"\nWrote {len(written)} figures to {plot_dir}/ for "
          f"{len(runs)} runs:")
    for path in written:
        print(f"  {os.path.basename(path)}")


if __name__ == "__main__":
    main()
