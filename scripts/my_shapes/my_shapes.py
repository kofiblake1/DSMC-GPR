"""
shape_families.py -- smooth (G2) 2D bodies for supersonic/hypersonic CFD and DSMC.

Every family is built as a list of internally smooth open "pieces" joined at
(possibly sharp) corners. Sharp corners are replaced by local quintic Hermite
blends that match position, tangent AND curvature on both sides (G2), with
`r_nose` (an effective nose radius) as an explicit, physical bluntness parameter. The result is
refit as a single periodic cubic spline (C2 everywhere, including closure) and
resampled with curvature-weighted spacing.

Families
  naca4       m, p, t           (round LE already; TE corner rounded)
  biconvex    t                 (LE + TE corners rounded)
  diamond     t, x_max          (4 corners rounded)
  ellipse     t (=b/a)          (C-infinity, no corners)
  superellipse t, n (even)      (C-infinity, no corners)
  fourier_blob amps, phases, t  (C-infinity, generally non-convex "bean" blobs)
Modifier
  bend(...)   theta             bends any family along a circular-arc centerline.
              A smooth shape under a smooth diffeomorphism stays smooth, so a
              bent ellipse is a smooth, non-convex "C blob"; a bent NACA is an
              extreme-camber airfoil.

Dependencies: numpy, scipy (matplotlib only for the demo).
"""
import numpy as np
from scipy.interpolate import CubicSpline, splprep, splev
from scipy.optimize import minimize_scalar

# ----------------------------------------------------------------------------
# Families: return (pieces, corner_after) with pieces ordered counter-clockwise.
# corner_after[i] is True if the junction between piece i and piece i+1
# (cyclically) is a sharp corner.
# ----------------------------------------------------------------------------

def naca4(m=0.02, p=0.4, t=0.12, n=600):
    b = np.linspace(0.0, np.pi, n)
    x = 0.5 * (1.0 - np.cos(b))                      # cosine clustering at LE/TE
    yt = 5 * t * (0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2
                  + 0.2843*x**3 - 0.1036*x**4)       # closed-TE coefficient
    if m == 0 or p == 0:
        yc = dyc = np.zeros_like(x)
    else:
        yc = np.where(x < p, m/p**2*(2*p*x - x**2),
                      m/(1-p)**2*((1 - 2*p) + 2*p*x - x**2))
        dyc = np.where(x < p, 2*m/p**2*(p - x), 2*m/(1-p)**2*(p - x))
    th = np.arctan(dyc)
    xu, yu = x - yt*np.sin(th), yc + yt*np.cos(th)
    xl, yl = x + yt*np.sin(th), yc - yt*np.cos(th)
    X = np.r_[xu[::-1], xl[1:]]                      # TE -> upper -> LE -> lower -> TE
    Y = np.r_[yu[::-1], yl[1:]]
    return [np.column_stack([X, Y])], [True]         # only the TE is a corner


def biconvex(t=0.08, n=400):
    h = t / 2
    R = (0.25 + h**2) / (2*h)                        # circular-arc radius, chord = 1
    a = np.arcsin(0.5 / R)                           # arc half-angle
    s = np.linspace(-a, a, n)
    upper = np.column_stack([0.5 + R*np.sin(-s), (h - R) + R*np.cos(s)])  # TE -> LE
    lower = np.column_stack([0.5 + R*np.sin(s), -(h - R) - R*np.cos(s)])  # LE -> TE
    return [upper, lower], [True, True]


def diamond(t=0.08, x_max=0.5, n=200):
    TE, top, LE, bot = (1, 0), (x_max, t/2), (0, 0), (x_max, -t/2)
    seg = lambda A, B: np.linspace(A, B, n)
    return [seg(TE, top), seg(top, LE), seg(LE, bot), seg(bot, TE)], [True]*4


def ellipse(t=0.5, n=800):
    phi = np.linspace(0, 2*np.pi, n, endpoint=False)
    return [np.column_stack([0.5 + 0.5*np.cos(phi), 0.5*t*np.sin(phi)])], [False]


def superellipse(t=0.5, n_exp=4, n=1200):
    """|x/a|^n + |y/b|^n = 1 with even n (smooth). Polar form avoids the
    clustering of the usual sign/abs parameterisation."""
    assert n_exp % 2 == 0, "use an even exponent for a smooth curve"
    th = np.linspace(0, 2*np.pi, n, endpoint=False)
    a, b = 0.5, 0.5*t
    r = (np.abs(np.cos(th)/a)**n_exp + np.abs(np.sin(th)/b)**n_exp)**(-1.0/n_exp)
    return [np.column_stack([0.5 + r*np.cos(th), r*np.sin(th)])], [False]


def fourier_blob(amps=(0.0, 0.15, 0.1, 0.05), phases=None, t=1.0, n=1200, rng=None):
    """Star-shaped C-infinity blob r(th) = 1 + sum_k a_k cos(k th + phi_k),
    k = 2.. (k=1 is a translation). Generally non-convex ("bean", "kidney").
    Stays a simple curve as long as r > 0."""
    rng = rng or np.random.default_rng()
    amps = np.asarray(amps); k = np.arange(2, 2 + len(amps))
    phases = rng.uniform(0, 2*np.pi, len(amps)) if phases is None else np.asarray(phases)
    th = np.linspace(0, 2*np.pi, n, endpoint=False)
    r = 1 + (amps[:, None]*np.cos(k[:, None]*th + phases[:, None])).sum(0)
    assert r.min() > 0.1, "amplitudes too large"
    return [np.column_stack([r*np.cos(th), t*r*np.sin(th)])], [False]


def bend(family_out, theta_deg):
    """Map chord coordinate x in [0,1] onto a circular-arc centerline that turns
    through theta_deg, preserving arc length; y becomes the signed normal
    offset. theta=0 is the identity. Valid while max|y| < R = 1/theta."""
    pieces, flags = family_out
    th = np.radians(theta_deg)
    if abs(th) < 1e-9:
        return pieces, flags
    R = 1.0 / th
    out = []
    for P in pieces:
        x, y = P[:, 0], P[:, 1]
        psi = (x - 0.5) / R                           # centerline angle
        out.append(np.column_stack([(R - y)*np.sin(psi), (R - y)*np.cos(psi) - R]))
    return out, flags

# ----------------------------------------------------------------------------
# Corner rounding (G2 quintic Hermite) and assembly
# ----------------------------------------------------------------------------

def _arclen(P):
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]

def _jet(cs, s):
    """position, unit tangent, curvature vector at arc length s."""
    p, d1, d2 = cs(s), cs(s, 1), cs(s, 2)
    sp = np.linalg.norm(d1)
    T = d1 / sp
    a = d2 / sp**2
    K = a - np.dot(a, T) * T                          # curvature vector, normal to T
    return p, T, K

def _quintic_blend(P0, T0, K0, P1, T1, K1, n=60, lam=None):
    """Quintic Hermite matching (P, P', P'') with P'=L*T, P''=L^2*K -> G2."""
    L = lam if lam is not None else np.linalg.norm(P1 - P0)
    V0, V1, A0, A1 = L*T0, L*T1, L**2*K0, L**2*K1
    t = np.linspace(0, 1, n)[:, None]
    h00 = 1 - 10*t**3 + 15*t**4 - 6*t**5
    h10 = t - 6*t**3 + 8*t**4 - 3*t**5
    h20 = 0.5*t**2 - 1.5*t**3 + 1.5*t**4 - 0.5*t**5
    h21 = 0.5*t**3 - t**4 + 0.5*t**5
    h11 = -4*t**3 + 7*t**4 - 3*t**5
    h01 = 10*t**3 - 15*t**4 + 6*t**5
    return h00*P0 + h10*V0 + h20*A0 + h21*A1 + h11*V1 + h01*P1

def _blend_peak_curvature(B):
    x1, y1 = np.gradient(B[:, 0]), np.gradient(B[:, 1])
    x2, y2 = np.gradient(x1), np.gradient(y1)
    k = ((x1*y2 - y1*x2) / (x1**2 + y1**2)**1.5)[3:-3]
    return k

def _best_blend(P0, T0, K0, P1, T1, K1, d, n=80):
    """Choose the tangent length that minimises the blend's peak curvature
    (no sign change allowed). Gives peak curvature ~1.15/r_nose for line-line
    corners at any turning angle, so r_nose behaves like a true nose radius."""
    def obj(f):
        k = _blend_peak_curvature(_quintic_blend(P0, T0, K0, P1, T1, K1, n=400, lam=f*2*d))
        sgn = np.sign(np.median(k))
        return np.max(np.abs(k)) + (1e6 if np.any(k*sgn < 0) else 0.0)
    f = minimize_scalar(obj, bounds=(0.05, 1.5), method="bounded",
                        options=dict(xatol=1e-3)).x
    return _quintic_blend(P0, T0, K0, P1, T1, K1, n=n, lam=f*2*d)

def _graded(s0, s1, h0, h1, hmax, growth=0.08, ngrid=20001):
    """Points on [s0, s1] whose spacing grows linearly (~geometrically) from h0
    at s0 and h1 at s1 up to hmax: avoids abrupt spacing jumps next to small
    blends, which would otherwise make the final interpolating spline wiggle."""
    s = np.linspace(s0, s1, ngrid)
    h = np.minimum.reduce([np.full_like(s, hmax), h0 + growth*(s - s0), h1 + growth*(s1 - s)])
    N = np.r_[0.0, np.cumsum(0.5*(1/h[1:] + 1/h[:-1])*np.diff(s))]
    m = max(int(np.ceil(N[-1])), 4)
    return np.interp(np.linspace(0, N[-1], m + 1), N, s)

def assemble(pieces, corner_after, r_nose=0.005, r_ridge=0.05, ridge_turn_deg=90.0,
             hmax=2e-3, n_blend=60):
    """Round every sharp corner with a G2 quintic blend.

    r_nose  -- radius used for sharp edges (turn > ridge_turn_deg): leading and
               trailing edges. Peak curvature of the blend is ~1.15/r_nose.
    r_ridge -- radius used for mild corners (turn <= ridge_turn_deg), e.g. the
               ridge/shoulder of a diamond airfoil.
    The inscribed circle of radius r would touch the sides at distance
    d = r*tan(turn/2) from the vertex; that is where each blend starts/ends.
    Returns one dense closed polyline (no repeated endpoint)."""
    n = len(pieces)
    if not any(corner_after):
        P = np.vstack(pieces)
        return P[:-1] if np.allclose(P[0], P[-1]) else P
    splines = [(CubicSpline(_arclen(P), P, axis=0), _arclen(P)[-1]) for P in pieces]
    trims = []
    for i in range(n):
        if not corner_after[i]:
            trims.append(0.0); continue
        (a, La), (b, Lb) = splines[i], splines[(i+1) % n]
        Ta, Tb = _jet(a, La)[1], _jet(b, 0.0)[1]
        turn = np.arccos(np.clip(Ta @ Tb, -1, 1))
        r = r_nose if np.degrees(turn) > ridge_turn_deg else r_ridge
        d = r * np.tan(turn/2)
        if d > 0.45*min(La, Lb):
            raise ValueError(f"radius {r} too large for corner {i} (turn {np.degrees(turn):.1f} deg)")
        trims.append(d)
    blends = []
    for i in range(n):
        if not corner_after[i]:
            blends.append(None); continue
        cs, L = splines[i]
        csn = splines[(i+1) % n][0]
        P0, T0, K0 = _jet(cs, L - trims[i])
        P1, T1, K1 = _jet(csn, trims[i])
        B = _best_blend(P0, T0, K0, P1, T1, K1, trims[i], n=n_blend)
        blends.append(B)
    hb = [None if B is None else np.linalg.norm(np.diff(B, axis=0), axis=1)[[0, -1]] for B in blends]
    out = []
    for i in range(n):
        cs, L = splines[i]
        h0 = hb[i-1][1] if hb[i-1] is not None else hmax
        h1 = hb[i][0] if hb[i] is not None else hmax
        seg = cs(_graded(trims[i-1], L - trims[i], h0, h1, hmax))
        out.append(seg[:-1])
        out.append(blends[i][:-1] if blends[i] is not None else seg[-1:])
    P = np.vstack(out)
    return P[:-1] if np.allclose(P[0], P[-1]) else P

# ----------------------------------------------------------------------------
# Final C2 periodic spline, resampling, QA
# ----------------------------------------------------------------------------

def _dedupe(P, tol=1e-12):
    keep = np.r_[True, np.linalg.norm(np.diff(P, axis=0), axis=1) > tol]
    return P[keep]

def finalize(P, n_out=400, curv_weight=0.5, normalize=False):
    """Periodic cubic spline through P (C2, incl. closure), CCW orientation,
    resampled with spacing ~ 1/(1 + c*|kappa|). normalize=True rescales to
    unit x-extent; leave it False for airfoils so that t/c etc. keep their
    meaning (rounding with the inscribed-circle rule removes a little chord:
    ~r_nose/sin(half-wedge-angle) at each sharp edge)."""
    P = _dedupe(P)
    area = 0.5*np.sum(P[:, 0]*np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1)*P[:, 1])
    if area < 0:
        P = P[::-1]
    if normalize:
        P = (P - P.min(0)) / np.ptp(P[:, 0])
        P[:, 1] -= 0.5*(P[:, 1].max() + P[:, 1].min())
    Q = np.vstack([P, P[:1]])
    u = _arclen(Q); u /= u[-1]
    tck, _ = splprep(Q.T, u=u, s=0, per=1, k=3)
    uf = np.linspace(0, 1, 20001)
    x1, y1 = splev(uf, tck, der=1); x2, y2 = splev(uf, tck, der=2)
    kap = (x1*y2 - y1*x2) / (x1**2 + y1**2)**1.5 / u[-1]  # (any scale, used for weights)
    sp = np.hypot(x1, y1)
    dens = sp * (1 + curv_weight*np.abs(kap)/np.mean(np.abs(kap)))
    cdf = np.r_[0, np.cumsum(0.5*(dens[1:] + dens[:-1]))]; cdf /= cdf[-1]
    us = np.interp(np.linspace(0, 1, n_out, endpoint=False), cdf, uf)
    X = np.column_stack(splev(us, tck))
    return X, tck

def curvature(tck, n=4000):
    """Signed curvature vs normalized arc length (positive = convex for CCW)."""
    u = np.linspace(0, 1, n, endpoint=False)
    x1, y1 = splev(u, tck, der=1); x2, y2 = splev(u, tck, der=2)
    k = (x1*y2 - y1*x2) / (x1**2 + y1**2)**1.5
    ds = np.hypot(x1, y1); s = np.r_[0, np.cumsum(0.5*(ds[1:]+ds[:-1]))]/n
    return s/s[-1], k

def self_intersects(X):
    """O(n^2) vectorised proper-crossing test for a closed polyline (round-off
    on collinear neighbours is ignored via a relative tolerance)."""
    A, B = X, np.roll(X, -1, axis=0)
    eps = 1e-12 * np.ptp(X, axis=0).max()**2
    def orient(p, q, r):
        v = ((q[..., 0]-p[..., 0])*(r[..., 1]-p[..., 1])
             - (q[..., 1]-p[..., 1])*(r[..., 0]-p[..., 0]))
        return np.where(np.abs(v) < eps, 0.0, np.sign(v))
    Ai, Bi, Aj, Bj = A[:, None], B[:, None], A[None], B[None]
    c = ((orient(Ai, Bi, Aj) * orient(Ai, Bi, Bj) < 0)
         & (orient(Aj, Bj, Ai) * orient(Aj, Bj, Bi) < 0))
    n = len(X); i, j = np.indices((n, n))
    adj = (np.abs(i - j) <= 1) | (np.abs(i - j) == n - 1)
    return bool(np.any(c & ~adj))

def qa(X, tck, min_radius=None):
    s, k = curvature(tck)
    rep = dict(self_intersection=self_intersects(X),
               convex=bool(np.all(k > -1e-3*np.max(np.abs(k)))),  # tolerance for spline round-off
               kappa_max=float(k.max()), kappa_min=float(k.min()),
               min_concave_radius=float(1/(-k.min())) if k.min() < 0 else np.inf,
               max_dkappa_ds=float(np.max(np.abs(np.diff(k))) * len(k)))
    if min_radius is not None:
        rep["ok_radius"] = 1/np.max(np.abs(k)) >= min_radius
    return rep

# ----------------------------------------------------------------------------
# Export
# ----------------------------------------------------------------------------

def to_geo(X, fname, lc=0.01, tag0=1000):
    """OpenCASCADE periodic C2 spline (first point repeated)."""
    with open(fname, "w") as f:
        f.write('SetFactory("OpenCASCADE");\n')
        for i, (x, y) in enumerate(X):
            f.write(f"Point({tag0+i}) = {{{x:.15e}, {y:.15e}, 0, {lc}}};\n")
        ids = ", ".join(str(tag0+i) for i in range(len(X)))
        f.write(f"Spline({tag0}) = {{{ids}, {tag0}}};\n")
        f.write(f"Curve Loop({tag0}) = {{{tag0}}};\n")

def to_sparta(X, fname, flow_outside=True):
    """SPARTA 2D surface. SPARTA treats the left side of each segment (p1->p2)
    as 'outside' (flow side); for a CCW body that is the interior, so reverse
    for external flow. Verify against your SPARTA build."""
    P = X[::-1] if flow_outside else X
    n = len(P)
    with open(fname, "w") as f:
        f.write(f"# shape_families body\n\n{n} points\n{n} lines\n\nPoints\n\n")
        f.writelines(f"{i} {x:.15e} {y:.15e}\n" for i, (x, y) in enumerate(P, 1))
        f.write("\nLines\n\n")
        f.writelines(f"{i} {i} {i % n + 1}\n" for i in range(1, n + 1))

# ----------------------------------------------------------------------------
# Sampler
# ----------------------------------------------------------------------------

FAMILIES = {
    "naca4":        lambda r: dict(m=r.uniform(0, 0.06), p=r.uniform(0.2, 0.6), t=r.uniform(0.06, 0.18)),
    "biconvex":     lambda r: dict(t=r.uniform(0.04, 0.12)),
    "diamond":      lambda r: dict(t=r.uniform(0.04, 0.12), x_max=r.uniform(0.3, 0.7)),
    "ellipse":      lambda r: dict(t=r.uniform(0.3, 1.0)),
    "superellipse": lambda r: dict(t=r.uniform(0.4, 1.0), n_exp=int(r.choice([4, 6]))),
    "c_blob":       lambda r: dict(t=r.uniform(0.15, 0.3), theta=r.uniform(120, 260)),
    "fourier_blob": lambda r: dict(amps=tuple(r.uniform(-1, 1, 4) * 0.25/np.arange(2, 6)**1.5),
                                   phases=tuple(r.uniform(0, 2*np.pi, 4)), t=r.uniform(0.6, 1.0)),
}

def make(family, r_nose=0.005, r_ridge=0.05, n_out=400, **kw):
    if family == "c_blob":
        out = bend(ellipse(t=kw["t"]), kw["theta"])
    else:
        theta = kw.pop("theta", 0.0)
        out = bend(globals()[family](**kw), theta)
    P = assemble(*out, r_nose=r_nose, r_ridge=r_ridge)
    normalize = family in ("c_blob", "fourier_blob")
    X, tck = finalize(P, n_out=n_out, normalize=normalize)
    return X, tck

def sample(rng=None, families=None, r_nose_range=(0.001, 0.01), max_tries=50):
    rng = rng or np.random.default_rng()
    families = families or list(FAMILIES)
    for _ in range(max_tries):
        fam = rng.choice(families)
        params = FAMILIES[fam](rng)
        rr = float(np.exp(rng.uniform(*np.log(r_nose_range))))
        X, tck = make(fam, r_nose=rr, **dict(params))
        rep = qa(X, tck)
        if not rep["self_intersection"]:
            return dict(family=str(fam), params=params, r_nose=rr, X=X, tck=tck, qa=rep)
    raise RuntimeError("no valid shape found")

# ----------------------------------------------------------------------------
# Demo: python shape_families.py  -> shape_gallery.png
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    def kappa_along(tck, n):
        u = np.linspace(0, 1, n, endpoint=False)
        x, y = splev(u, tck); x1, y1 = splev(u, tck, der=1); x2, y2 = splev(u, tck, der=2)
        k = (x1*y2 - y1*x2) / (x1**2 + y1**2)**1.5
        ds = np.hypot(x1, y1); s = np.r_[0, np.cumsum(0.5*(ds[1:] + ds[:-1]))] / n
        return np.asarray(x), np.asarray(y), k, s

    rng = np.random.default_rng(7)
    fams = list(FAMILIES)
    fig = plt.figure(figsize=(15, 9.5))
    gs = fig.add_gridspec(3, len(fams), height_ratios=[1, 1, 1.25], hspace=0.35, wspace=0.15)
    norm = Normalize(-2, 3)
    for j, fam in enumerate(fams):
        for i in range(2):
            d = sample(rng, families=[fam])
            x, y, k, _ = kappa_along(d["tck"], 3000)
            ax = fig.add_subplot(gs[i, j])
            pts = np.column_stack([x, y]).reshape(-1, 1, 2)
            lc = LineCollection(np.concatenate([pts[:-1], pts[1:]], 1), cmap="coolwarm", norm=norm, lw=3)
            lc.set_array((np.sign(k)*np.log10(1 + np.abs(k)))[:-1]); ax.add_collection(lc)
            ax.fill(x, y, color="0.93", zorder=0)
            ax.set_aspect("equal"); ax.autoscale(); ax.axis("off")
            if i == 0: ax.set_title(fam, fontsize=11)
    sm = plt.cm.ScalarMappable(norm=norm, cmap="coolwarm")
    cb = fig.colorbar(sm, ax=fig.axes[:], orientation="horizontal", fraction=0.03, pad=0.02, aspect=60)
    cb.set_label("sign(κ)·log10(1+|κ|)   (blue = concave, red = high convex curvature)")

    ax = fig.add_subplot(gs[2, 0:3]); ax2 = fig.add_subplot(gs[2, 4:])
    raw = np.vstack(diamond(t=0.08)[0])
    ax.plot(raw[:, 0], raw[:, 1], "k--", lw=1, label="sharp diamond")
    for rn, col in [(0.002, "C0"), (0.005, "C1"), (0.01, "C2")]:
        X, tck = make("diamond", r_nose=rn, t=0.08, n_out=2000)
        x, y, k, s = kappa_along(tck, 200000)
        ax.plot(x, y, col, lw=1.6, label=f"r_nose = {rn}")
        s = s - s[np.argmin(x)]; m = np.abs(s) < 0.05
        ax2.plot(s[m], k[m]*rn, col, label=f"r_nose = {rn}")
    ax.set_xlim(-0.005, 0.16); ax.set_ylim(-0.02, 0.02); ax.set_aspect("equal")
    ax.legend(fontsize=9, loc="upper left"); ax.set_xlabel("x/c")
    ax.set_title("Diamond (t/c = 0.08) leading edge, G2-rounded", fontsize=11)
    ax2.axhline(1, color="k", lw=0.6, ls=":")
    ax2.set_xlabel("arc length from leading edge, s/c"); ax2.set_ylabel("κ · r_nose")
    ax2.set_title("Curvature through the blend: continuous, peak ≈ 1/r_nose", fontsize=11)
    ax2.legend(fontsize=9)
    fig.savefig("shape_gallery.png", dpi=130, bbox_inches="tight")
    print("wrote shape_gallery.png")
