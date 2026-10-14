#!/usr/bin/env python3
"""Generate a dataset of smooth 2D bodies from my_shapes.py -- N per family,
scaled to a physical size of order 1 m, with mesh files and a PNG for each.

Laid out like scripts/shapes/dataset.py's output:

    dataset/
      meshes/   <family>_<i>.surf   SPARTA 2D surface (the simulation input)
                <family>_<i>.geo    gmsh OpenCASCADE periodic spline
      images/   <family>_<i>.png    plain matplotlib outline plot
      manifest.csv                  one row per shape: family, parameters,
                                    size, and the QA metrics from my_shapes.qa
      gallery.png                   overview grid, first GALLERY_COLS of each family
      README.md                     what's in here and in what units

my_shapes.py builds every family on a unit chord (or unit mean radius for the
blobs). Each shape here is then scaled about its bounding-box center so its
largest extent is a target size drawn from SIZE_RANGE, with that center left at
the origin, and finally rotated about the origin -- by an angle of attack
within AOA_RANGE_DEG for the airfoil families, or anywhere in
ROTATION_RANGE_DEG for the shapes that have no nose to speak of.

Both transforms are similarities, so the G2 smoothness my_shapes works to
achieve is untouched: curvature scales as 1/L and is indifferent to rotation.
That is why the manifest carries both the unit-space nose radius the shape was
built with and its physical value r_nose * L, and why `size_m` stays the
body-axis size whatever the orientation.

Usage:
    python3 make_dataset.py            # 20 per family into ./dataset
    python3 make_dataset.py 5          # 5 per family (quick check)
Deterministic: the same SEED gives the same dataset. Writing into an existing
dataset/ overwrites same-named files and leaves others, so bump DATASET_DIR or
clear it if you want a clean set.
"""

import csv
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from my_shapes import FAMILIES, make, qa, to_geo, to_sparta

# ============================================================================
# CONFIG
# ============================================================================
N_PER_FAMILY = 20
SEED = 20261013  # any fixed value; the dataset is reproducible from it

# Target size of each body, in meters: the larger of its two bounding-box
# extents measured in BODY axes, i.e. before the orientation below is applied,
# so it's the chord of an airfoil or the major diameter of a blob however the
# shape is then turned. A range rather than exactly 1.0 so the set carries some
# scale variation while staying "of order 1 m".
SIZE_RANGE = (0.8, 1.25)

# Orientation, drawn per shape. The airfoil families get a realistic angle of
# attack; everything else is an arbitrary shape with no preferred heading, so
# it gets any orientation at all. Positive = nose-up (see rotate()).
AIRFOIL_FAMILIES = ("naca4", "biconvex", "diamond")
AOA_RANGE_DEG = (-15.0, 15.0)
ROTATION_RANGE_DEG = (0.0, 360.0)

# Points on the final closed outline. At 400 points a 1 m body gets ~2.5 mm
# segments; raise for a finer surface, at linear cost in file size.
N_OUT = 400

# Nose radius handed to my_shapes.make, sampled log-uniformly. This is in
# UNIT-CHORD space, so the physical radius is r_nose * size -- 1 to 10 mm for a
# 1 m body, the right order for a sharp hypersonic leading edge. The blends set
# peak curvature ~1/r_nose, so this is what bounds the finest surface feature.
R_NOSE_RANGE = (0.001, 0.01)

# Characteristic mesh length written into the .geo, as a fraction of the body's
# size. Only a hint to gmsh for a later volume mesh; it does not affect the
# surface written to the .surf file.
GEO_LC_FRACTION = 0.01

# SPARTA treats the left of each p1->p2 segment as the flow side, so an
# external-flow body needs the CCW outline reversed. to_sparta does that when
# flow_outside=True. See its docstring -- verify against your SPARTA build.
FLOW_OUTSIDE = True

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
GALLERY_COLS = 6  # how many of each family to show in gallery.png

# Series color: categorical slot 1 of the project's default palette.
COLOR_OUTLINE = "#2a78d6"
COLOR_FILL = "#dbe7f7"


def draw_shape(rng, family, max_tries=50):
    """Sample one valid shape of `family`.

    Same contract as my_shapes.sample() but for a single family, and it also
    retries past the families' own guards -- fourier_blob asserts when its
    amplitudes fold r(theta) too far, assemble raises when a nose radius
    doesn't fit a corner -- instead of letting one bad draw end the run. Both
    are rejections of a sampled parameter set, not bugs, so the right response
    is another draw. Returns (params, r_nose, X, tck, qa_report, n_rejected).
    """
    rejected = 0
    for _ in range(max_tries):
        params = FAMILIES[family](rng)
        r_nose = float(np.exp(rng.uniform(*np.log(R_NOSE_RANGE))))
        try:
            X, tck = make(family, r_nose=r_nose, n_out=N_OUT, **dict(params))
        except (AssertionError, ValueError):
            rejected += 1
            continue
        report = qa(X, tck)
        if report["self_intersection"]:
            rejected += 1
            continue
        return params, r_nose, X, tck, report, rejected
    raise RuntimeError(f"no valid {family} shape in {max_tries} tries")


def scale_to_size(X, target):
    """Scale about the bounding-box center so max extent == target, centered.

    Returns (X_scaled, scale). A similarity transform: shape and smoothness are
    unchanged, lengths scale by `scale` and curvatures by 1/scale.
    """
    center = 0.5 * (X.max(0) + X.min(0))
    scale = target / np.ptp(X, axis=0).max()
    return (X - center) * scale, float(scale)


def draw_rotation(rng, family):
    """Draw one shape's orientation in degrees, by family."""
    lo, hi = AOA_RANGE_DEG if family in AIRFOIL_FAMILIES else ROTATION_RANGE_DEG
    return float(rng.uniform(lo, hi))


def rotate(X, deg):
    """Rotate about the origin, where the shapes are already centered.

    Sign convention: the families are built nose-at-low-x with the chord along
    x, and these bodies are meant for a freestream running in +x (the AERO-F
    cylinder in this repo stagnates at its -x apex), so the nose points into
    the flow. A nose-up pitch is then a CLOCKWISE turn, and positive `deg` is
    made to mean exactly that -- for the airfoil families it reads as the
    aerodynamic angle of attack, positive nose-up. For the other families the
    body has no nose and the angle is just an arbitrary heading.

    Applied after scaling, about the origin, so it moves the body without
    changing its size, its center, or any of its intrinsic geometry.
    """
    a = np.radians(-deg)  # negate: positive deg -> clockwise -> nose up
    c, s = np.cos(a), np.sin(a)
    return X @ np.array([[c, s], [-s, c]])


def geometry_stats(X):
    """Closed-polyline area (shoelace), perimeter and bounding-box extents.

    Reference quantities for nondimensionalizing a run later, so they're worth
    recording alongside the geometry rather than recomputing per use. Called on
    the final rotated points, so area and perimeter are the body's own
    (rotation-invariant) while the extents are its footprint in flow axes --
    extent_y_m is the frontal height a +x freestream actually sees.
    """
    closed = np.vstack([X, X[:1]])
    d = np.diff(closed, axis=0)
    x, y = X[:, 0], X[:, 1]
    area = 0.5 * np.abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    center = 0.5 * (X.max(0) + X.min(0))
    return dict(
        area_m2=float(area),
        perimeter_m=float(np.sum(np.hypot(d[:, 0], d[:, 1]))),
        extent_x_m=float(np.ptp(x)),
        extent_y_m=float(np.ptp(y)),
        # Where the rotated footprint actually sits. The ROTATION PIVOT is the
        # origin, but an axis-aligned bounding box turns with the body, so its
        # center drifts off the origin by up to ~12% of the body size here.
        # Recorded so a domain can be built around the body without reparsing
        # the surface files.
        bbox_center_x_m=float(center[0]),
        bbox_center_y_m=float(center[1]),
    )


def save_image(X, name, size_m, angle_label, path):
    """Plain matplotlib PNG of one body: filled outline, true aspect, meters."""
    closed = np.vstack([X, X[:1]])
    fig, ax = plt.subplots(figsize=(4.0, 4.0), constrained_layout=True)
    ax.fill(closed[:, 0], closed[:, 1], color=COLOR_FILL, zorder=0)
    ax.plot(closed[:, 0], closed[:, 1], lw=2.0, color=COLOR_OUTLINE)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"{name}  ({size_m:.2f} m, {angle_label})", fontsize=10)
    ax.grid(True, lw=0.5, color="0.9")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_gallery(by_family, path):
    """One row per family, the first GALLERY_COLS shapes of each, shared scale.

    Every panel uses the same axis limits so the size variation across the set
    is visible rather than normalized away by per-panel autoscaling.
    """
    families = list(by_family)
    cols = min(GALLERY_COLS, max(len(v) for v in by_family.values()))
    fig, axes = plt.subplots(
        len(families), cols, figsize=(1.9 * cols, 2.0 * len(families)),
        constrained_layout=True, squeeze=False,
    )
    lim = 0.55 * max(np.ptp(X, axis=0).max() for v in by_family.values() for X in v)
    for i, fam in enumerate(families):
        for j in range(cols):
            ax = axes[i][j]
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect("equal")
            ax.axis("off")
            if j < len(by_family[fam]):
                X = np.vstack([by_family[fam][j], by_family[fam][j][:1]])
                ax.fill(X[:, 0], X[:, 1], color=COLOR_FILL, zorder=0)
                ax.plot(X[:, 0], X[:, 1], lw=1.4, color=COLOR_OUTLINE)
        # axis("off") hides set_ylabel, so label the row with text instead.
        axes[i][0].text(-0.08, 0.5, fam, transform=axes[i][0].transAxes,
                        rotation=90, va="center", ha="right", fontsize=9)
    fig.suptitle(f"my_shapes dataset -- all panels to the same {2*lim:.2f} m scale")
    fig.savefig(path, dpi=130)
    plt.close(fig)


README = """# my_shapes dataset

Generated by `make_dataset.py` (see its docstring for the generation rules).
{n_per_family} shapes per family, {n_total} in total.

- `meshes/<name>.surf` -- SPARTA 2D surface: `{n_out} points` / `{n_out} lines`,
  then 1-based point coordinates and segment connectivity. Written with
  `flow_outside={flow_outside}`, i.e. the outline is {orientation} so SPARTA's
  left-of-segment "outside" is the flow side for external flow.
- `meshes/<name>.geo` -- gmsh OpenCASCADE periodic C2 spline through the same
  points plus a `Curve Loop`, for generating a volume mesh around the body. The
  characteristic length is {geo_lc_fraction} x the body size.
- `images/<name>.png` -- matplotlib outline plot, axes in meters.
- `manifest.csv` -- one row per shape. Lengths in meters, curvatures in 1/m.
  `r_nose_unit` is the nose radius my_shapes built the shape with on a unit
  chord; `r_nose_m` is that times `size_m`. `params_json` holds the family's own
  sampled parameters, which stay in unit-chord terms (t/c and so on).
  `rotation_deg` is the orientation applied (see below); `extent_x_m`,
  `extent_y_m` and `bbox_center_*_m` describe the rotated footprint, while
  `area_m2` and `perimeter_m` are rotation-invariant.
- `gallery.png` -- overview grid, all panels to a common scale.

Coordinates are in meters. `size_m` is the larger bounding-box extent in BODY
axes -- the chord, or a blob's major diameter -- drawn from {size_range} m and
unaffected by the rotation.

Each shape is then rotated about the origin by `rotation_deg`, drawn per shape:

- airfoil families ({airfoil_families}): {aoa_range} deg, a realistic angle-of-attack
  range. The sign is the aerodynamic convention, positive = nose-up, for a
  freestream running in +x (the bodies are built nose-at-low-x).
- every other family: {rotation_range} deg, since these are arbitrary shapes with
  no preferred heading.

The rotation pivot -- the body-axis bounding-box center -- is at the origin for
every shape. The rotated body's own axis-aligned bounding box turns with it, so
`bbox_center_*_m` is generally NOT the origin: it reaches ~12% of `size_m` away
for a blob turned near 45 deg. Use those columns, not the origin, to place a
domain around a body.

Reproducible from SEED = {seed} in `make_dataset.py`.
"""


def main():
    n_per_family = N_PER_FAMILY
    if len(sys.argv) > 1:
        n_per_family = int(sys.argv[1])

    mesh_dir = os.path.join(DATASET_DIR, "meshes")
    img_dir = os.path.join(DATASET_DIR, "images")
    os.makedirs(mesh_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    rng = np.random.default_rng(SEED)
    rows = []
    by_family = {}
    rejected_total = 0
    t0 = time.time()

    for family in FAMILIES:
        by_family[family] = []
        for i in range(n_per_family):
            params, r_nose, X, tck, report, rejected = draw_shape(rng, family)
            rejected_total += rejected

            size_m = float(rng.uniform(*SIZE_RANGE))
            X, scale = scale_to_size(X, size_m)
            angle = draw_rotation(rng, family)
            X = rotate(X, angle)

            is_airfoil = family in AIRFOIL_FAMILIES
            angle_label = (f"AoA {angle:+.1f} deg" if is_airfoil
                           else f"rot {angle:.0f} deg")
            name = f"{family}_{i}"
            to_sparta(X, os.path.join(mesh_dir, name + ".surf"),
                      flow_outside=FLOW_OUTSIDE)
            to_geo(X, os.path.join(mesh_dir, name + ".geo"),
                   lc=GEO_LC_FRACTION * size_m)
            save_image(X, name, size_m, angle_label,
                       os.path.join(img_dir, name + ".png"))

            stats = geometry_stats(X)
            # qa()'s curvatures come from the unit-space spline; divide by the
            # scale to put them in 1/m alongside everything else here.
            kappa_max = report["kappa_max"] / scale
            rows.append(dict(
                name=name, family=family, size_m=round(size_m, 6),
                rotation_deg=round(angle, 4), is_airfoil=is_airfoil,
                n_points=len(X), r_nose_unit=r_nose, r_nose_m=r_nose * scale,
                **{k: round(v, 9) for k, v in stats.items()},
                kappa_max_per_m=kappa_max,
                min_radius_m=1.0 / kappa_max if kappa_max > 0 else np.inf,
                min_concave_radius_m=report["min_concave_radius"] * scale,
                convex=report["convex"],
                self_intersection=report["self_intersection"],
                params_json=json.dumps(
                    {k: (list(v) if isinstance(v, tuple) else v)
                     for k, v in params.items()}, default=float),
            ))
            if len(by_family[family]) < GALLERY_COLS:
                by_family[family].append(X)
        print(f"{family:14s} {n_per_family} shapes")

    manifest = os.path.join(DATASET_DIR, "manifest.csv")
    with open(manifest, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    save_gallery(by_family, os.path.join(DATASET_DIR, "gallery.png"))

    with open(os.path.join(DATASET_DIR, "README.md"), "w") as fh:
        fh.write(README.format(
            n_per_family=n_per_family, n_total=len(rows), n_out=N_OUT,
            flow_outside=FLOW_OUTSIDE,
            orientation="clockwise" if FLOW_OUTSIDE else "counter-clockwise",
            geo_lc_fraction=GEO_LC_FRACTION, size_range=list(SIZE_RANGE),
            airfoil_families=", ".join(AIRFOIL_FAMILIES),
            aoa_range=f"{AOA_RANGE_DEG[0]:+g} to {AOA_RANGE_DEG[1]:+g}",
            rotation_range=f"{ROTATION_RANGE_DEG[0]:g} to {ROTATION_RANGE_DEG[1]:g}",
            seed=SEED,
        ))

    sizes = np.array([r["size_m"] for r in rows])
    noses = np.array([r["r_nose_m"] for r in rows])
    print(f"\n{len(rows)} shapes in {time.time() - t0:.1f} s "
          f"({rejected_total} draws rejected and resampled)")
    print(f"size {sizes.min():.2f}-{sizes.max():.2f} m, "
          f"nose radius {1e3*noses.min():.1f}-{1e3*noses.max():.1f} mm, "
          f"{N_OUT} surface points each")
    print(f"wrote {DATASET_DIR}/")


if __name__ == "__main__":
    main()
