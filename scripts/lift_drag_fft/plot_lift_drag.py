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
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import find_peaks

# ============================================================================
# CONFIG -- point this at whichever run's output you want to analyze.
# ============================================================================
#RUN_DIR = "/Volumes/SherlockScratch/DSMC_FEM_VALIDATION_P1/BEAM_LOCKIN/PHASE_1"
RUN_DIR = "/Volumes/SherlockScratch/DSMC_FEM_VALIDATION_P1/BEAM_LOCKIN/PHASE_4R_DIAGNOSTIC/RUN1_rho08_long/"

LIFT_DRAG_FILE = os.path.join(RUN_DIR, "EXAMPLE_BEAM", "lift_drag.dat")
GDISPLAC_FILE = os.path.join(RUN_DIR, "results_beam", "gdisplac")
OUTPUT_DIR = os.getcwd() #RUN_DIR  # PNGs land here; change if you'd rather they land elsewhere

# Only look at data after this SPARTA absolute time (matches the original
# notebook's st=0.3) -- skips the initial transient before the flow and
# structural response settle into their statistically-steady regime.
ST = 0.0

# gdisplac's own time column resets to 0.0 at the start of structural
# coupling (AERO-S's internal DYNAMICS clock), while lift_drag.dat's time
# column is SPARTA's absolute time, which already includes the steps before
# this run's restart checkpoint. Add this offset to gdisplac's time so both
# datasets share one absolute time axis: restart_step * SPARTA dt (1e-6).
# = 0.1 s for PHASE_1/PHASE_2's restart.100000 -- update if a future run
# restarts from a different checkpoint.
RESTART_TIME_OFFSET = 250000 * 1e-5

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
TIP_COMPONENT = 'y'

# Peak-detection threshold in fft_and_peaks(), as a fraction of that
# signal's own peak FFT magnitude (not a fixed absolute value -- lift, drag,
# and tip displacement live on completely different physical scales, so a
# single hardcoded threshold, as the original notebook used, can't work for
# all three at once).
RELATIVE_PEAK_HEIGHT = 0.05

WINDOW = 10  # samples, for the moving-average overlay

# -- 3D CWT surface settings --------------------------------------------------
# Frequency band to show in the 3D surfaces. The CWT's own band here is
# ~25.6-3250 Hz (scales 1..127 at these sampling rates), but all the energy in
# this problem sits below ~110 Hz, so rendering the full band would bury the
# structure in a flat plain. Raise this if you expect higher harmonics.
CWT3D_FMAX = 200.0
# Surfaces get decimated in time to this many columns before rendering --
# 10k-20k x 127 quads per panel is both unreadable (moire) and very slow to
# rasterize. The CWT envelope is smooth in time, so this loses nothing visible.
CWT3D_MAX_COLS = 500
# Viewing angle: elevation above the time-frequency plane, and azimuth. The
# default looks down the time axis from the front-right so ridges running
# left-to-right read as "constant frequency over time".
CWT3D_ELEV = 32
CWT3D_AZIM = 50


def load_lift_drag(path, st):
    step, time, drag, lift = np.loadtxt(path, skiprows=1, unpack=True)
    mask = (time > st) #& (time < st + 1)
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
    mask = (times > st) #& (times < st + 1)
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


def plot_cwt_surface(ax, time, freqs, coefs, zlabel, ridge_freq=None,
                     fmax=CWT3D_FMAX, max_cols=CWT3D_MAX_COLS):
    """Draw |CWT| as a 3D surface -- x = time, y = frequency, z = magnitude.

    Same data as the 2D pcolormesh panels, but with the quantity that was
    encoded as color now encoded as height, viewed from an angle so a
    persistent spectral peak reads as a ridge running along the time axis.

    pywt returns frequencies in DESCENDING order (scale 1 = highest freq), so
    both the frequency vector and the coefficient rows are reversed here to get
    an ascending y axis.

    If ridge_freq is given (one frequency per input time sample) it is traced
    as a red line on the *ceiling* of the axes rather than on the surface
    itself: mplot3d has no true per-face depth sorting, so a line lying on the
    crest gets swallowed by the surface it sits on. Floating it above
    everything keeps it readable from any angle -- read it as a top-down
    shadow of where the peak is, not as a height.

    Returns (t, f, z) of the decimated surface actually drawn."""
    band = freqs <= fmax
    f = freqs[band][::-1]
    z = np.abs(coefs[band][::-1])

    stride = max(1, len(time) // max_cols)
    t = time[::stride]
    z = z[:, ::stride]

    T, F = np.meshgrid(t, f)
    ax.plot_surface(T, F, z, cmap='viridis', rstride=1, cstride=1,
                    linewidth=0, antialiased=False)

    zmax = z.max()
    if ridge_freq is not None:
        ax.plot(t, ridge_freq[::stride], np.full(len(t), zmax * 1.08),
                color='red', lw=1.3, label='Dominant freq. (CWT ridge)')
    ax.set_zlim(0, zmax * 1.12)

    ax.set_xlabel("Time [s]", labelpad=6)
    ax.set_ylabel("Freq [Hz]", labelpad=6)
    ax.set_zlabel(zlabel, labelpad=6)
    ax.view_init(elev=CWT3D_ELEV, azim=CWT3D_AZIM)
    return t, f, z


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
    # ax3.plot(time_tip, moving_average(tip_disp), color='red', label='Windowed Average')
    ax3.set_ylabel(tip_label)
    ax3.set_xlabel("Time")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "timeseries.png"), dpi=600)
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
    plt.savefig(os.path.join(OUTPUT_DIR, "fft.png"), dpi=600)
    plt.close(fig)

    # -- Continuous wavelet transform (time-frequency) -----------------------
    scales = np.arange(1, 128)
    lift_cwt, cwt_freqs_ld = pywt.cwt(lift - lift.mean(), scales, "morl", sampling_period=dt_ld)
    drag_cwt, _ = pywt.cwt(drag - drag.mean(), scales, "morl", sampling_period=dt_ld)
    tip_cwt, cwt_freqs_tip = pywt.cwt(tip_disp - tip_disp.mean(), scales, "morl", sampling_period=dt_tip)

    # Instantaneous dominant frequency of the tip displacement: at each time,
    # the frequency whose wavelet coefficient has the largest magnitude (the
    # CWT "ridge"). Unlike the single FFT peak this tracks how the dominant
    # frequency drifts in time -- e.g. the transition into lock-in.
    #
    # A raw per-sample argmax is unusable here: twice per cycle the
    # fundamental's coefficient passes through a local minimum and the 2nd
    # harmonic (~2*f) momentarily wins, so the ridge flips 34.6 <-> 69 Hz
    # every few samples and the overlay becomes a solid band. Smoothing the
    # |CWT| envelope over ~1 dominant period before the argmax (then median-
    # filtering the result) removes that intra-cycle flicker while leaving
    # genuine multi-cycle frequency drift intact.
    if len(tip_pf) and tip_pf[0] > 0:
        nsmooth = max(3, int(round(1.0 / (tip_pf[0] * dt_tip))))
    else:
        nsmooth = 3
    tip_env = uniform_filter1d(np.abs(tip_cwt), size=nsmooth, axis=1, mode='nearest')
    tip_ridge_freq = cwt_freqs_tip[np.argmax(tip_env, axis=0)]
    tip_ridge_freq = median_filter(tip_ridge_freq, size=nsmooth, mode='nearest')
    # The first/last ~1 cycle of the low-frequency scales sits inside the cone
    # of influence (edge effects), so treat the ridge near t_start/t_end as
    # unreliable.
    print(f"\nTip disp. ({TIP_COMPONENT}) CWT ridge -- dominant frequency:")
    print(f"-> median over the record: {np.median(tip_ridge_freq):.2f} Hz")
    print(f"-> final value (t = {time_tip[-1]:.4f} s): {tip_ridge_freq[-1]:.2f} Hz")

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=False, figsize=(8, 9))

    ax1.pcolormesh(time_ld, cwt_freqs_ld, np.abs(lift_cwt), shading="auto")
    ax1.set_ylabel("Lift Freq")
    ax1.set_yscale("log")

    ax2.pcolormesh(time_ld, cwt_freqs_ld, np.abs(drag_cwt), shading="auto")
    ax2.set_ylabel("Drag Freq")
    ax2.set_yscale("log")

    ax3.pcolormesh(time_tip, cwt_freqs_tip, np.abs(tip_cwt), shading="auto")
    ax3.plot(time_tip, tip_ridge_freq, color='red', lw=1.2,
             label='Dominant freq. (CWT ridge)')
    if len(tip_pf):
        ax3.axhline(tip_pf[0], color='white', lw=0.9, linestyle='--',
                    label=f'FFT peak = {tip_pf[0]:.2f} Hz')
    ax3.set_ylabel(f"{tip_label} Freq")
    ax3.set_xlabel("Time")
    ax3.set_ylim(20,40)
    ax3.legend(loc='upper right', fontsize=8, framealpha=0.6)
    # ax3.set_yscale("log")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "cwt.png"), dpi=600)
    plt.close(fig)

    # -- Same CWTs as 3D surfaces (power as height) --------------------------
    fig = plt.figure(figsize=(9, 15))

    ax1 = fig.add_subplot(3, 1, 1, projection='3d')
    plot_cwt_surface(ax1, time_ld, cwt_freqs_ld, lift_cwt, "|CWT| Lift")
    ax1.set_title("Lift", pad=0)

    ax2 = fig.add_subplot(3, 1, 2, projection='3d')
    plot_cwt_surface(ax2, time_ld, cwt_freqs_ld, drag_cwt, "|CWT| Drag")
    ax2.set_title("Drag", pad=0)

    ax3 = fig.add_subplot(3, 1, 3, projection='3d')
    # Reuse the same smoothed ridge the 2D panel draws, so "dominant
    # frequency" means one thing across both figures.
    plot_cwt_surface(ax3, time_tip, cwt_freqs_tip, tip_cwt,
                     f"|CWT| {tip_label}", ridge_freq=tip_ridge_freq)
    ax3.legend(loc='upper left', fontsize=8)
    ax3.set_title(f"Tip displacement ({TIP_COMPONENT})", pad=0)

    # tight_layout misplaces 3D axes (it can't account for the projected
    # bounding box), so space the panels manually.
    fig.subplots_adjust(left=0.02, right=0.93, top=0.97, bottom=0.03,
                        hspace=0.18)
    plt.savefig(os.path.join(OUTPUT_DIR, "cwt3d.png"), dpi=300)
    plt.close(fig)

    print(f"\nSaved timeseries.png, fft.png, cwt.png, cwt3d.png to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
