#!/usr/bin/env python3
"""Plot and FFT lift, drag, and beam tip displacement for a coupled
AERO-S/SPARTA beam-in-crossflow run.

Converted from the original examples/plot_lift_drag.ipynb notebook (moved
here so it's a plain script instead of a notebook), extended to also
read/plot/FFT the beam tip's transverse displacement from
results_beam/gdisplac alongside the original lift/drag from
EXAMPLE_BEAM/lift_drag.dat.

Setup (Sherlock modules -- this repo doesn't otherwise pin a venv for this):
    ml load math py-numpy/1.26.3_py312 py-scipy/1.16.0_py312 \
        py-pywavelets/1.6.0_py312 viz py-matplotlib/3.10.3_py312
(Lmod may warn that py-pywavelets wants py-numpy/1.26.3_py312 specifically
even though a newer numpy ends up loaded -- harmless, all four imports work.)

Usage: edit RUN_DIR (and TIP_COMPONENT/ST if needed) below, then:
    python3 plot_lift_drag.py
Figures are saved as PNGs next to this script's inputs (see OUTPUT_DIR) --
this is meant to run headlessly (e.g. over ssh, or from an sbatch script),
not to pop up interactive windows, hence the non-interactive 'Agg' backend.
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pywt
from scipy.signal import find_peaks

# ============================================================================
# CONFIG -- point this at whichever run's output you want to analyze.
# ============================================================================
RUN_DIR = "/scratch/users/kofib/DSMC_FEM_VALIDATION_P1/BEAM_LOCKIN/PHASE_1"
# RUN_DIR = "/scratch/users/kofib/DSMC_FEM_VALIDATION_P1/BEAM_LOCKIN/PHASE_2/E_0.396MPa"

LIFT_DRAG_FILE = os.path.join(RUN_DIR, "EXAMPLE_BEAM", "lift_drag.dat")
GDISPLAC_FILE = os.path.join(RUN_DIR, "results_beam", "gdisplac")
OUTPUT_DIR = RUN_DIR  # PNGs land here; change if you'd rather they land elsewhere

# Only look at data after this SPARTA absolute time (matches the original
# notebook's st=0.3) -- skips the initial transient before the flow and
# structural response settle into their statistically-steady regime.
ST = 0.3

# gdisplac's own time column resets to 0.0 at the start of structural
# coupling (AERO-S's internal DYNAMICS clock), while lift_drag.dat's time
# column is SPARTA's absolute time, which already includes the steps before
# this run's restart checkpoint. Add this offset to gdisplac's time so both
# datasets share one absolute time axis: restart_step * SPARTA dt (1e-6).
# = 0.1 s for PHASE_1/PHASE_2's restart.100000 -- update if a future run
# restarts from a different checkpoint.
RESTART_TIME_OFFSET = 100000 * 1e-6

# Which tip displacement quantity to analyze: 'y' (transverse -- the
# vortex-shedding-driven bending direction, analogous to lift), 'x' (axial
# -- mostly geometric foreshortening from large-deflection bending,
# analogous to drag in that it's a secondary/rectified response), or
# 'magnitude' (sqrt(x^2+y^2+z^2) -- always >= 0). Note magnitude only
# folds/doubles the apparent frequency if the underlying signal actually
# crosses zero; for this cantilever the tip typically bends to one side and
# oscillates around a nonzero mean rather than crossing y=0, so magnitude's
# FFT peak usually lands at the same frequency as 'y', just with the DC/mean
# offset baked in differently -- verify against 'y'/'x' rather than assuming
# either behavior.
# TIP_COMPONENT = 'y'
# TIP_COMPONENT = 'x'
TIP_COMPONENT = 'magnitude'

# Peak-detection threshold in fft_and_peaks(), as a fraction of that
# signal's own peak FFT magnitude (not a fixed absolute value -- lift, drag,
# and tip displacement live on completely different physical scales, so a
# single hardcoded threshold, as the original notebook used, can't work for
# all three at once).
RELATIVE_PEAK_HEIGHT = 0.05

WINDOW = 10  # samples, for the moving-average overlay


def load_lift_drag(path, st):
    step, time, drag, lift = np.loadtxt(path, skiprows=1, unpack=True)
    mask = time > st
    return time[mask], lift[mask], drag[mask]


def load_tip_displacement(path, time_offset, st):
    """Parse an AERO-S gdisplac file: a 2-line header (title, node count),
    then repeating blocks of [1 time line] + [nnodes x "dx dy dz" lines].
    Returns (time, tip_x, tip_y, tip_z) using the LAST node in each block --
    the free end of the cantilever (node 1, the clamped root, is always the
    first node line and is always exactly zero)."""
    with open(path) as f:
        lines = f.readlines()

    nnodes = int(lines[1])
    block_size = 1 + nnodes
    body = lines[2:]
    n_blocks = len(body) // block_size

    times = np.empty(n_blocks)
    tip = np.empty((n_blocks, 3))
    for b in range(n_blocks):
        block = body[b * block_size:(b + 1) * block_size]
        times[b] = float(block[0])
        tip[b] = [float(v) for v in block[-1].split()]

    times = times + time_offset
    mask = times > st
    return times[mask], tip[mask, 0], tip[mask, 1], tip[mask, 2]


def fft_and_peaks(signal, dt, label, relative_height=RELATIVE_PEAK_HEIGHT):
    freq = np.fft.rfftfreq(len(signal), d=dt)
    mag = np.abs(np.fft.rfft(signal - signal.mean()))
    peak_idx, _ = find_peaks(mag, height=relative_height * mag.max(), distance=5)
    order = np.argsort(mag[peak_idx])[::-1]
    print(f"Detected Dominant Frequencies ({label}):")
    for f, a in zip(freq[peak_idx][order], mag[peak_idx][order]):
        print(f"-> Frequency: {f:.2f} Hz | Amplitude: {a:.2f}")
    return freq, mag, freq[peak_idx][order], mag[peak_idx][order]


def moving_average(x, window=WINDOW):
    return np.convolve(x, np.ones(window) / window, mode='same')


def main():
    time_ld, lift, drag = load_lift_drag(LIFT_DRAG_FILE, ST)
    time_tip, tip_x, tip_y, tip_z = load_tip_displacement(
        GDISPLAC_FILE, RESTART_TIME_OFFSET, ST)
    tip_options = {
        'x': tip_x,
        'y': tip_y,
        'z': tip_z,
        'magnitude': np.sqrt(tip_x**2 + tip_y**2 + tip_z**2),
    }
    tip_disp = tip_options[TIP_COMPONENT]
    tip_label = f"Tip disp. {TIP_COMPONENT} [m]"

    dt_ld = time_ld[1] - time_ld[0]
    dt_tip = time_tip[1] - time_tip[0]

    # -- Time series + windowed average -------------------------------------
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=False, figsize=(8, 9))

    ax1.plot(time_ld, lift)
    ax1.plot(time_ld, moving_average(lift), color='red', label='Windowed Average')
    ax1.set_ylabel("Lift")

    ax2.plot(time_ld, drag)
    ax2.plot(time_ld, moving_average(drag), color='red', label='Windowed Average')
    ax2.set_ylabel("Drag")
    ax2.set_xlabel("Time")

    ax3.plot(time_tip, tip_disp)
    ax3.plot(time_tip, moving_average(tip_disp), color='red', label='Windowed Average')
    ax3.set_ylabel(tip_label)
    ax3.set_xlabel("Time")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "timeseries.png"), dpi=150)
    plt.close(fig)

    # -- FFT + dominant peak detection ---------------------------------------
    freq_lift, lift_fft, lift_pf, lift_pm = fft_and_peaks(lift, dt_ld, "Lift")
    freq_drag, drag_fft, drag_pf, drag_pm = fft_and_peaks(drag, dt_ld, "Drag")
    freq_tip, tip_fft, tip_pf, tip_pm = fft_and_peaks(tip_disp, dt_tip, f"Tip disp. ({TIP_COMPONENT})")

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=False, figsize=(8, 9))

    for ax, freq, mag, pf, pm, ylabel in (
        (ax1, freq_lift, lift_fft, lift_pf, lift_pm, "Lift FFT"),
        (ax2, freq_drag, drag_fft, drag_pf, drag_pm, "Drag FFT"),
        (ax3, freq_tip, tip_fft, tip_pf, tip_pm, f"{tip_label} FFT"),
    ):
        ax.plot(freq, mag)
        ax.set_xlim(0, 200)
        ax.set_ylabel(ylabel)
        for f, a in zip(pf, pm):
            ax.vlines(f, 0, a, color='red', linestyle='--', alpha=0.7)
    ax3.set_xlabel("Frequency [Hz]")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fft.png"), dpi=150)
    plt.close(fig)

    # -- Continuous wavelet transform (time-frequency) -----------------------
    scales = np.arange(1, 128)
    lift_cwt, cwt_freqs_ld = pywt.cwt(lift - lift.mean(), scales, "morl", sampling_period=dt_ld)
    drag_cwt, _ = pywt.cwt(drag - drag.mean(), scales, "morl", sampling_period=dt_ld)
    tip_cwt, cwt_freqs_tip = pywt.cwt(tip_disp - tip_disp.mean(), scales, "morl", sampling_period=dt_tip)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=False, figsize=(8, 9))

    ax1.pcolormesh(time_ld, cwt_freqs_ld, np.abs(lift_cwt), shading="auto")
    ax1.set_ylabel("Lift Freq")
    ax1.set_yscale("log")

    ax2.pcolormesh(time_ld, cwt_freqs_ld, np.abs(drag_cwt), shading="auto")
    ax2.set_ylabel("Drag Freq")
    ax2.set_yscale("log")

    ax3.pcolormesh(time_tip, cwt_freqs_tip, np.abs(tip_cwt), shading="auto")
    ax3.set_ylabel(f"{tip_label} Freq")
    ax3.set_xlabel("Time")
    ax3.set_yscale("log")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "cwt.png"), dpi=150)
    plt.close(fig)

    print(f"\nSaved timeseries.png, fft.png, cwt.png to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
