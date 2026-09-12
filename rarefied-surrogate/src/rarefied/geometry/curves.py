"""Closed 2D curves: sources, validation, arc-length resampling.

A curve is an (N, 2) float array holding an ordered loop in the plane. The
last point is NOT a repeat of the first -- closure is implicit.

ORIENTATION: curves are **clockwise** (decision M13), and centred on their
area centroid. Clockwise is not arbitrary -- it is forced from outside:

  * SPARTA requires clockwise surface-node definition (hard requirement).
  * The existing coupling path already does it: notebooks/mesh_pipeline/
    mesh_export.py:76 sorts the boundary with `np.argsort(-angles)`.
  * geometry/circle.py:51 already emits clockwise
    (`np.linspace(0, -2*np.pi, ...)`).
  * AERO-F does not care: it repairs orientation itself. GeoSource.C:340 calls
    Elem::checkVolume, which swaps nodes 1<->2 on any negative-volume tet, and
    the reference circle run reoriented 11004 of 21152 boundary faces while
    still producing correct physics.

One side has a hard requirement, one already complies, the third is free --
so clockwise wins on every count, and adopting it invalidated nothing.

Note the consequence for `signed_area()`: it is **negative** for a valid
curve here. Callers that want magnitude must use `abs()`.

STATION ORIGIN: index 0 is anchored where the +x ray from the area centroid
crosses the surface (`anchor_origin`), matching circle.py's start point at
(+r, 0). Without an anchor, station ids would not be reproducible across
shapes, which M6's registration layer depends on.

The mesher takes a curve and nothing else, so it does not depend on the
`shapes` package. `circle()` is analytic and therefore exact, which makes it a
real validation target rather than an approximation.
"""

import numpy as np


# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------

def circle(diameter, n_pts):
    """Analytic circle of the given diameter, uniform in angle, clockwise.

    Starts at (+r, 0) and sweeps negatively, matching
    geometry/circle.py:51's `np.linspace(0, -2*np.pi, ...)` exactly, so the
    AERO-F and SPARTA sides agree on station 0 for the reference case.
    """
    r = 0.5 * diameter
    ang = np.linspace(0.0, -2.0 * np.pi, n_pts, endpoint=False)
    return np.column_stack((r * np.cos(ang), r * np.sin(ang)))


def from_shape(shape):
    """Curve from a generated `shapes.Shape` (uses its x-y sampling points).

    `Shape.generate()` must have been called. Its `curve_pts` are (N, 3) with
    z == 0, CCW-sorted and centred, with duplicates already removed.
    """
    pts = np.asarray(shape.curve_pts, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 3:
        raise ValueError("shape.curve_pts is empty -- call Shape.generate() first")
    return pts[:, :2].copy()


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------

def _closed(curve):
    """Curve with the first point appended, for segment-wise operations."""
    return np.vstack((curve, curve[:1]))


def segment_lengths(curve):
    d = np.diff(_closed(curve), axis=0)
    return np.hypot(d[:, 0], d[:, 1])


def perimeter(curve):
    return float(segment_lengths(curve).sum())


def signed_area(curve):
    """Shoelace area. Positive for counter-clockwise."""
    c = _closed(curve)
    x, y = c[:, 0], c[:, 1]
    return float(0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


def centroid(curve):
    """Area centroid of the enclosed polygon (not the mean of the vertices)."""
    c = _closed(curve)
    x, y = c[:, 0], c[:, 1]
    cross = x[:-1] * y[1:] - x[1:] * y[:-1]
    a = 0.5 * np.sum(cross)
    if abs(a) < 1e-30:
        return np.asarray([float(np.mean(curve[:, 0])), float(np.mean(curve[:, 1]))])
    cx = np.sum((x[:-1] + x[1:]) * cross) / (6.0 * a)
    cy = np.sum((y[:-1] + y[1:]) * cross) / (6.0 * a)
    return np.asarray([cx, cy])


def as_cw(curve):
    """Return the curve oriented clockwise (the project convention, M13).

    Clockwise means a negative shoelace area, so this reverses anything with
    positive signed area.
    """
    return curve if signed_area(curve) < 0.0 else curve[::-1].copy()


def recentre(curve):
    """Translate so the area centroid sits at the origin."""
    return curve - centroid(curve)


def characteristic_length(curve):
    """Diameter of the circle with the same enclosed area.

    Used as the single length scale that every mesh parameter is expressed
    relative to, so the sizing rule is shape-scale invariant.
    """
    return float(2.0 * np.sqrt(abs(signed_area(curve)) / np.pi))


def bounding_box(curve):
    lo = curve.min(axis=0)
    hi = curve.max(axis=0)
    return float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])


def plus_x_ray_crossings(curve):
    """Parameters where the +x ray from the area centroid crosses the loop.

    Returns a list of (segment_index, t) with t in [0, 1) along that segment.
    The ray is {(cx + s, cy) : s > 0} where (cx, cy) is the area centroid.

    Used both to anchor station 0 (`anchor_origin`) and to reject shapes where
    that anchor would be ambiguous (`validate`).
    """
    c = centroid(curve)
    cl = _closed(curve)
    a, b = cl[:-1], cl[1:]
    ay, by = a[:, 1] - c[1], b[:, 1] - c[1]

    # Half-open crossing rule (ay <= 0 < by, or by <= 0 < ay) counts each
    # crossing exactly once even when a vertex lies on the ray.
    straddles = ((ay <= 0.0) & (by > 0.0)) | ((by <= 0.0) & (ay > 0.0))
    out = []
    for i in np.nonzero(straddles)[0]:
        t = ay[i] / (ay[i] - by[i])                      # ay + t*(by-ay) == 0
        x = a[i, 0] + t * (b[i, 0] - a[i, 0])
        if x - c[0] > 0.0:                               # +x side only
            out.append((int(i), float(t)))
    return out


def anchor_origin(curve):
    """Rotate the point order so index 0 is the +x-ray crossing (M13).

    Preserves orientation and the point set; only the starting index changes.
    Raises ValueError if the anchor is ambiguous or absent -- callers should
    have screened with `validate` first.
    """
    hits = plus_x_ray_crossings(curve)
    if len(hits) != 1:
        raise ValueError(
            "+x ray from the centroid crosses the surface %d times; station 0 "
            "is ambiguous (expected exactly 1)" % len(hits))
    i, t = hits[0]
    # Start at the vertex the crossing sits on or just before, so index 0 is an
    # actual curve point rather than an interpolated one.
    start = (i + 1) % len(curve) if t > 0.5 else i
    return np.roll(curve, -start, axis=0).copy()


def distance_to_curve(points, curve, chunk=4096):
    """Unsigned distance from each of `points` (M, 2) to the closed polygon.

    Exact point-to-segment distance, chunked over points to bound memory at
    M x N floats. Used to check the realised size grading away from the wall
    and to measure how far the meshed wall nodes sit from the input curve.
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    c = _closed(curve)
    a, b = c[:-1], c[1:]                     # (N, 2) segment endpoints
    ab = b - a
    denom = np.einsum("ij,ij->i", ab, ab)
    denom = np.where(denom > 0.0, denom, 1.0)

    out = np.empty(len(points))
    for lo in range(0, len(points), chunk):
        p = points[lo:lo + chunk]                       # (m, 2)
        ap = p[:, None, :] - a[None, :, :]              # (m, N, 2)
        t = np.clip(np.einsum("mnj,nj->mn", ap, ab) / denom, 0.0, 1.0)
        closest = a[None, :, :] + t[:, :, None] * ab[None, :, :]
        out[lo:lo + chunk] = np.linalg.norm(p[:, None, :] - closest, axis=2).min(axis=1)
    return out


# --------------------------------------------------------------------------
# Resampling
# --------------------------------------------------------------------------

def resample(curve, n_pts):
    """Resample to n_pts points equispaced in arc length around the loop.

    Piecewise-linear in arc length, so the resampled points lie exactly on the
    original polygon. For an analytic source such as `circle()` with a dense
    input this reproduces the underlying shape to the polygon's own accuracy;
    for the circle specifically, sampling uniformly in angle already gives
    uniform arc length, so a matched n_pts is a no-op up to round-off.

    Keeping n_pts fixed across a dataset is deliberate: it makes the wall
    discretisation, and therefore the extracted wall features, comparable
    from shape to shape.
    """
    if n_pts < 3:
        raise ValueError("n_pts must be >= 3")
    seg = segment_lengths(curve)
    s = np.concatenate(([0.0], np.cumsum(seg)))  # length n+1, s[-1] = perimeter
    total = s[-1]
    if total <= 0.0:
        raise ValueError("degenerate curve: zero perimeter")
    target = np.linspace(0.0, total, n_pts, endpoint=False)
    c = _closed(curve)
    x = np.interp(target, s, c[:, 0])
    y = np.interp(target, s, c[:, 1])
    return np.column_stack((x, y))


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def _segments_cross(p, p2, q, q2):
    """True if open segments p-p2 and q-q2 properly cross (vectorised over q)."""
    def cross(o, a, b):
        return (a[..., 0] - o[..., 0]) * (b[..., 1] - o[..., 1]) - \
               (a[..., 1] - o[..., 1]) * (b[..., 0] - o[..., 0])

    d1 = cross(p, p2, q)
    d2 = cross(p, p2, q2)
    d3 = cross(q, q2, p)
    d4 = cross(q, q2, p2)
    return ((d1 * d2) < 0.0) & ((d3 * d4) < 0.0)


def self_intersects(curve):
    """True if the closed polygon crosses itself.

    Adjacent segments share an endpoint and are skipped. O(N^2) but vectorised
    over one axis; at the few-hundred-point sizes used here it is negligible,
    and it runs before gmsh is ever invoked so bad geometry is rejected cheaply.
    """
    c = _closed(curve)
    a, b = c[:-1], c[1:]
    n = len(a)
    for i in range(n):
        # Compare segment i against all segments not sharing an endpoint.
        j = np.arange(n)
        mask = (j != i) & (j != (i + 1) % n) & (j != (i - 1) % n)
        if not mask.any():
            continue
        if _segments_cross(a[i], b[i], a[mask], b[mask]).any():
            return True
    return False


def validate(curve, min_seg_frac=1e-4):
    """Check a curve is usable and return metrics.

    Returns (ok, problems, metrics). `problems` is a list of strings; empty
    means the curve passed. Nothing here touches gmsh, so this is the cheap
    first gate in the dataset loop.
    """
    problems = []
    curve = np.asarray(curve, dtype=float)

    if curve.ndim != 2 or curve.shape[1] != 2:
        return False, ["curve must have shape (N, 2)"], {}
    if len(curve) < 3:
        return False, ["fewer than 3 points"], {}
    if not np.all(np.isfinite(curve)):
        return False, ["curve contains non-finite coordinates"], {}

    seg = segment_lengths(curve)
    per = float(seg.sum())
    area = signed_area(curve)
    dchar = characteristic_length(curve)

    metrics = {
        "n_pts": int(len(curve)),
        "perimeter": per,
        "signed_area": area,
        "characteristic_diameter": dchar,
        "seg_len_min": float(seg.min()),
        "seg_len_max": float(seg.max()),
        "seg_len_mean": float(seg.mean()),
        "bounding_box": bounding_box(curve),
    }

    if abs(area) <= 0.0:
        problems.append("zero enclosed area")
    if area > 0.0:
        problems.append("counter-clockwise orientation (call as_cw first); "
                        "the project convention is clockwise, see M13")

    # Station 0 is anchored to the +x ray from the centroid, so a shape whose
    # surface crosses that ray more than once has no reproducible station
    # origin. Rejecting them is consistent with M10's convex-body scope and
    # drops into the batch driver's existing reject/retry loop.
    n_hits = len(plus_x_ray_crossings(curve))
    metrics["plus_x_ray_crossings"] = n_hits
    if n_hits != 1:
        problems.append(
            "+x ray from the centroid crosses the surface %d times (expected "
            "1), so the station origin is ambiguous" % n_hits)

    if per > 0.0 and seg.min() < min_seg_frac * per:
        problems.append(
            "near-duplicate points: shortest segment is %.3g, "
            "below %.3g of the perimeter" % (seg.min(), min_seg_frac)
        )
    if self_intersects(curve):
        problems.append("curve is self-intersecting")

    return (len(problems) == 0), problems, metrics


def prepare(curve, n_pts):
    """Full front-end: orient CW, recentre, resample, anchor station 0, validate.

    Returns (curve, metrics) and raises ValueError if the curve is unusable.
    The anchor step runs after resampling so index 0 lands on a real resampled
    point, and before validation so the returned curve is the one checked.
    """
    curve = recentre(as_cw(np.asarray(curve, dtype=float)))
    curve = resample(curve, n_pts)
    curve = recentre(curve)
    if len(plus_x_ray_crossings(curve)) == 1:
        curve = anchor_origin(curve)
    # else: leave unanchored so validate() reports the ambiguity itself rather
    # than anchor_origin() raising a less informative error first.
    ok, problems, metrics = validate(curve)
    if not ok:
        raise ValueError("unusable curve: " + "; ".join(problems))
    return curve, metrics
