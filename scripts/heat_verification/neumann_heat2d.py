r"""
neumann_heat2d.py
=================

Analytical solver for the 2D unsteady heat equation on a rectangle
[0, a] x [0, b] with spatially-varying Neumann (flux) data on all four sides:

        dT/dt = alpha * laplacian(T)

    BCs (coordinate derivatives, matching the derivation):
        dT/dx (0, y, t) = p0(y)        dT/dx (a, y, t) = pa(y)
        dT/dy (x, 0, t) = r0(x)        dT/dy (x, b, t) = rb(x)

    IC:  T(x, y, 0) = T0(x, y)

The solution follows the decomposition derived in the companion notes:

        T(x, y, t) = gamma * t  +  psi(x, y)  +  theta(x, y, t)
                     \_______/    \________/    \____________/
                       drift        Poisson       transient
                                     field

with everything expanded in the Neumann cosine basis
        phi_mn = cos(m*pi*x/a) * cos(n*pi*y/b),   lambda_mn = (m*pi/a)^2 + (n*pi/b)^2.

Key relations implemented (see notes, Sec. 4-6):

    Q_net  = B[0,0]                                 (net boundary flux)
    gamma  = alpha * Q_net / (a*b)                  (forced drift rate)
    A_mn   = eps_m eps_n B_mn / (a b lambda_mn),  A_00 = 0  (zero-mean gauge)
    C_mn   = That0_mn - A_mn                        (transient coeffs; C_00 = mean T0)

    T(x,y,t) = gamma*t + sum_mn [A_mn + C_mn * exp(-alpha lambda_mn t)] phi_mn

where B_mn is the Green's-identity projection of the boundary data and
That0_mn are the cosine coefficients of the initial condition.

--------------------------------------------------------------------
SIGN / CONVENTION NOTE
--------------------------------------------------------------------
The boundary functions p0, pa, r0, rb are the *coordinate* derivatives
(dT/dx or dT/dy), NOT outward-normal derivatives. The outward-normal
derivative is  -p0 on x=0,  +pa on x=a,  -r0 on y=0,  +rb on y=b.

If you think in physical heat flux instead, use `NeumannHeat2D.from_flux`,
which takes the OUTWARD heat flux q_n (W/m^2, positive = heat leaving the
slab) on each edge and the conductivity k, via  dT/dn = -q_n / k.
--------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------
def _trap_weights(grid: np.ndarray) -> np.ndarray:
    """Composite-trapezoid quadrature weights for a 1D grid."""
    w = np.empty_like(grid, dtype=float)
    w[1:-1] = 0.5 * (grid[2:] - grid[:-2])
    w[0] = 0.5 * (grid[1] - grid[0])
    w[-1] = 0.5 * (grid[-1] - grid[-2])
    return w


def _wrap(f, coord_name="x"):
    """
    Turn a boundary/IC specification into a vectorized callable.

    Accepts: None (-> 0), a scalar (-> constant), or a callable.
    """
    if f is None:
        return lambda s: np.zeros_like(np.asarray(s, dtype=float))
    if np.isscalar(f):
        c = float(f)
        return lambda s: np.full_like(np.asarray(s, dtype=float), c)

    def g(s):
        s = np.asarray(s, dtype=float)
        out = np.asarray(f(s), dtype=float)
        if out.shape != s.shape:           # non-vectorized callable fallback
            out = np.vectorize(f)(s)
        return out
    return g


def _wrap2(f):
    """Vectorize a 2D IC callable T0(x, y); accepts scalar or callable."""
    if f is None:
        return lambda X, Y: np.zeros_like(np.asarray(X, dtype=float))
    if np.isscalar(f):
        c = float(f)
        return lambda X, Y: np.full_like(np.asarray(X, dtype=float), c)

    def g(X, Y):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        out = np.asarray(f(X, Y), dtype=float)
        if out.shape != X.shape:
            out = np.vectorize(f)(X, Y)
        return out
    return g


# ----------------------------------------------------------------------
# main solver
# ----------------------------------------------------------------------
class NeumannHeat2D:
    """
    Closed-form modal solver for the pure-Neumann unsteady heat problem.

    Parameters
    ----------
    a, b : float
        Domain width (x) and height (y).
    alpha : float
        Thermal diffusivity.
    p0, pa : callable/scalar/None
        dT/dx on the x=0 and x=a edges, as functions of y.
    r0, rb : callable/scalar/None
        dT/dy on the y=0 and y=b edges, as functions of x.
    T0 : callable/scalar/None
        Initial condition T0(x, y).
    n_modes_x, n_modes_y : int
        Number of cosine modes kept in x and y (m = 0..n_modes_x, etc.).
    n_quad : int
        Number of quadrature points per dimension for the projections.
    """

    def __init__(self, a, b, alpha,
                 p0=None, pa=None, r0=None, rb=None, T0=None,
                 n_modes_x=48, n_modes_y=48, n_quad=4000):
        self.a, self.b, self.alpha = float(a), float(b), float(alpha)
        self.M, self.N = int(n_modes_x), int(n_modes_y)

        p0, pa = _wrap(p0), _wrap(pa)
        r0, rb = _wrap(r0), _wrap(rb)
        T0f = _wrap2(T0)

        m = np.arange(self.M + 1)
        n = np.arange(self.N + 1)

        # eigenvalues lambda_mn and Neumann norm factors eps_m eps_n
        kx = m * np.pi / self.a
        ky = n * np.pi / self.b
        self.lam = kx[:, None] ** 2 + ky[None, :] ** 2         # (M+1, N+1)
        eps_m = np.where(m == 0, 1.0, 2.0)
        eps_n = np.where(n == 0, 1.0, 2.0)
        self.eps_mn = eps_m[:, None] * eps_n[None, :]

        # ---- boundary projections -> B_mn  (notes eq. (16)) --------------
        # 1D cosine transforms of the four edge functions.
        yq = np.linspace(0.0, self.b, n_quad)
        wy = _trap_weights(yq)
        xq = np.linspace(0.0, self.a, n_quad)
        wx = _trap_weights(xq)

        Cy = np.cos(n[:, None] * np.pi * yq[None, :] / self.b)   # (N+1, nq)
        Cx = np.cos(m[:, None] * np.pi * xq[None, :] / self.a)   # (M+1, nq)

        pa_hat = Cy @ (wy * pa(yq))     # (N+1,)  int_0^b cos(n pi y/b) pa dy
        p0_hat = Cy @ (wy * p0(yq))     # (N+1,)
        rb_hat = Cx @ (wx * rb(xq))     # (M+1,)
        r0_hat = Cx @ (wx * r0(xq))     # (M+1,)

        sign_m = (-1.0) ** m            # (M+1,)
        sign_n = (-1.0) ** n            # (N+1,)

        B = (np.outer(sign_m, pa_hat)          # (-1)^m * pa_hat[n]
             - p0_hat[None, :]                 # - p0_hat[n]
             + np.outer(rb_hat, sign_n)        # (-1)^n * rb_hat[m]
             - r0_hat[:, None])                # - r0_hat[m]
        self.B = B

        # net flux and forced drift rate (B[0,0] == Q_net identically)
        self.Q_net = B[0, 0]
        self.gamma = self.alpha * self.Q_net / (self.a * self.b)

        # ---- Poisson-field coefficients A_mn  (notes eq. (14)) -----------
        A = np.zeros_like(B)
        mask = np.ones_like(B, dtype=bool)
        mask[0, 0] = False                          # skip lambda_00 = 0
        A[mask] = (self.eps_mn[mask] * B[mask]
                   / (self.a * self.b * self.lam[mask]))
        A[0, 0] = 0.0                               # zero-mean gauge
        self.A = A

        # ---- IC projection -> That0_mn, then C_mn = That0_mn - A_mn ------
        # cosine coefficients of the initial condition on the same grid
        Xg, Yg = np.meshgrid(xq, yq)                # (nq_y, nq_x)
        T0grid = T0f(Xg, Yg)                        # (nq, nq)
        G = (wy[:, None] * T0grid) * wx[None, :]    # apply 2D weights
        I = Cx @ (Cy @ G).T                         # (M+1, N+1) raw integral
        That0 = self.eps_mn * I / (self.a * self.b)
        self.That0 = That0
        self.C = That0 - A                          # C_00 = mean(T0) since A_00=0

        # diagnostics
        self.Tbar0 = That0[0, 0]                    # initial mean temperature
        nonzero = self.lam[mask]
        self.lam_min = nonzero.min() if nonzero.size else 0.0
        self.tau_relax = (1.0 / (self.alpha * self.lam_min)
                          if self.lam_min > 0 else np.inf)

    # ------------------------------------------------------------------
    # field evaluation
    # ------------------------------------------------------------------
    def _cos_matrices(self, x, y):
        m = np.arange(self.M + 1)
        n = np.arange(self.N + 1)
        Cx = np.cos(m[:, None] * np.pi * np.asarray(x)[None, :] / self.a)  # (M+1,Nx)
        Cy = np.cos(n[:, None] * np.pi * np.asarray(y)[None, :] / self.b)  # (N+1,Ny)
        return Cx, Cy

    def _assemble(self, coeff, x, y):
        """Return field[j, i] = sum_mn coeff[m,n] cos(m pi x_i/a) cos(n pi y_j/b)."""
        Cx, Cy = self._cos_matrices(x, y)
        return (coeff @ Cy).T @ Cx                  # (Ny, Nx)

    def psi(self, x, y):
        """Steady Poisson field psi(x, y) (zero-mean gauge)."""
        return self._assemble(self.A, x, y)

    def transient(self, x, y, t):
        """Decaying transient theta(x, y, t)."""
        coeff = self.C * np.exp(-self.alpha * self.lam * t)
        return self._assemble(coeff, x, y)

    def field(self, x, y, t):
        """Full temperature field T(x, y, t) on the grid x (1D) by y (1D)."""
        coeff = self.A + self.C * np.exp(-self.alpha * self.lam * t)
        return self.gamma * t + self._assemble(coeff, x, y)

    def mean_temperature(self, t):
        """Spatial-mean temperature: gamma*t + Tbar0 (exact)."""
        return self.gamma * t + self.Tbar0

    # ------------------------------------------------------------------
    # alternative constructor from physical flux
    # ------------------------------------------------------------------
    @classmethod
    def from_flux(cls, a, b, alpha, k,
                  q_left=None, q_right=None, q_bottom=None, q_top=None,
                  T0=None, **kw):
        """
        Build from OUTWARD heat flux q_n (positive = heat leaving the slab),
        using dT/dn = -q_n/k. Edge functions:
            q_left(y), q_right(y), q_bottom(x), q_top(x).

        Mapping (outward-normal to coordinate-derivative):
            p0 = +q_left/k,  pa = -q_right/k,  r0 = +q_bottom/k,  rb = -q_top/k
        """
        qL, qR = _wrap(q_left), _wrap(q_right)
        qB, qT = _wrap(q_bottom), _wrap(q_top)
        p0 = lambda y: qL(y) / k
        pa = lambda y: -qR(y) / k
        r0 = lambda x: qB(x) / k
        rb = lambda x: -qT(x) / k
        return cls(a, b, alpha, p0=p0, pa=pa, r0=r0, rb=rb, T0=T0, **kw)

    # ------------------------------------------------------------------
    # diagnostics
    # ------------------------------------------------------------------
    def summary(self):
        lines = [
            "NeumannHeat2D solver",
            f"  domain            : [0, {self.a}] x [0, {self.b}]",
            f"  diffusivity alpha : {self.alpha:g}",
            f"  modes (M x N)     : {self.M} x {self.N}",
            f"  net flux Q_net    : {self.Q_net:.6g}",
            f"  drift rate gamma  : {self.gamma:.6g}   (dT_mean/dt)",
            f"  initial mean Tbar0: {self.Tbar0:.6g}",
            f"  slowest mode tau  : {self.tau_relax:.6g}   (1/(alpha lambda_min))",
        ]
        if abs(self.Q_net) < 1e-12 * max(1.0, self.a * self.b):
            lines.append("  -> balanced flux: a true steady state exists (gamma=0).")
        else:
            sense = "heating" if self.gamma > 0 else "cooling"
            lines.append(f"  -> net {sense}: no steady state; mean drifts linearly.")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # plotting
    # ------------------------------------------------------------------
    def _grid(self, nx=200, ny=200):
        x = np.linspace(0, self.a, nx)
        y = np.linspace(0, self.b, ny)
        return x, y

    def plot(self, t, nx=200, ny=200, levels=40, ax=None, cmap="inferno",
             show=False):
        """Filled-contour snapshot of T at time t."""
        x, y = self._grid(nx, ny)
        Z = self.field(x, y, t)
        if ax is None:
            fig, ax = plt.subplots(figsize=(6.2, 4.6 * self.b / self.a + 1.2))
        cf = ax.contourf(x, y, Z, levels=levels, cmap=cmap)
        ax.set_aspect("equal")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_title(f"T(x, y, t = {t:g})")
        plt.colorbar(cf, ax=ax, label="T")
        if show:
            plt.show()
        return ax

    def plot_decomposition(self, t, nx=200, ny=200, levels=40, show=False):
        """Four panels: drift+mean, psi, transient, and the total field."""
        x, y = self._grid(nx, ny)
        psi = self.psi(x, y)
        th = self.transient(x, y, t)
        drift = np.full_like(psi, self.gamma * t + self.Tbar0)
        total = drift + psi + th

        fig, axs = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
        drift_val = self.gamma * t + self.Tbar0
        panels = [
            (drift, f"drift + mean = gamma*t + Tbar0 = {drift_val:.3g}", "viridis"),
            (psi,   "psi(x, y)   (steady Poisson field)", "cividis"),
            (th,    f"theta(x, y, t={t:g})   (transient)", "magma"),
            (total, f"T(x, y, t={t:g})   (total)", "inferno"),
        ]
        for ax, (Z, title, cmap) in zip(axs.ravel(), panels):
            ax.set_aspect("equal")
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("x"); ax.set_ylabel("y")
            if np.ptp(Z) < 1e-9 * max(1.0, abs(drift_val)):
                # spatially uniform panel: contourf can't level a flat field
                im = ax.imshow(Z, origin="lower", extent=[0, self.a, 0, self.b],
                               cmap=cmap, aspect="equal")
                fig.colorbar(im, ax=ax, shrink=0.85)
            else:
                cf = ax.contourf(x, y, Z, levels=levels, cmap=cmap)
                fig.colorbar(cf, ax=ax, shrink=0.85)
        fig.suptitle("Solution decomposition:  T = gamma*t + psi + theta",
                     fontsize=12)
        if show:
            plt.show()
        return fig

    def plot_evolution(self, times, nx=200, ny=200, levels=40, cmap="inferno",
                       shared_scale=True, show=False):
        """Row of snapshots at the given times."""
        x, y = self._grid(nx, ny)
        fields = [self.field(x, y, t) for t in times]
        if shared_scale:
            vmin = min(f.min() for f in fields)
            vmax = max(f.max() for f in fields)
            lvls = np.linspace(vmin, vmax, levels)   # truly shared levels
        else:
            lvls = levels
        k = len(times)
        fig, axs = plt.subplots(1, k, figsize=(3.4 * k, 3.6),
                                constrained_layout=True, squeeze=False)
        for ax, t, Z in zip(axs[0], times, fields):
            cf = ax.contourf(x, y, Z, levels=lvls, cmap=cmap, extend="both")
            ax.set_aspect("equal")
            ax.set_title(f"t = {t:g}", fontsize=10)
            ax.set_xlabel("x")
        axs[0][0].set_ylabel("y")
        fig.colorbar(cf, ax=axs[0].tolist(), shrink=0.8, label="T")
        if show:
            plt.show()
        return fig


# ----------------------------------------------------------------------
# self-tests + demo
# ----------------------------------------------------------------------
def _run_verification():
    """
    Reproduce the 1D analytic check from the notes (Sec. 8):
    insulate left/top/bottom, drive right face with uniform gradient pa = c.
    Expected quasi-steady field:  psi(x) = (c/2a) x^2 - c a/6  (zero mean),
    drift rate gamma = alpha c / a.
    """
    print("=" * 64)
    print("VERIFICATION 1 - 1D uniform-flux analytic check")
    print("=" * 64)
    a, b, alpha, c = 2.0, 1.0, 0.7, 3.0

    # IC chosen = analytic (psi + mean) so the transient is identically zero:
    # this isolates and tests the Poisson-field (A_mn/B_mn) pipeline exactly.
    mean0 = 10.0
    psi_exact = lambda x: (c / (2 * a)) * x ** 2 - c * a / 6.0
    T0 = lambda X, Y: psi_exact(X) + mean0

    s = NeumannHeat2D(a, b, alpha, p0=0.0, pa=c, r0=0.0, rb=0.0, T0=T0,
                      n_modes_x=120, n_modes_y=4)
    print(s.summary())

    # gamma check
    gamma_exact = alpha * c / a
    print(f"\n  gamma  numeric = {s.gamma:.8f}")
    print(f"  gamma  exact   = {gamma_exact:.8f}")
    assert abs(s.gamma - gamma_exact) < 1e-8, "gamma mismatch"

    # psi field check. The cosine series converges fast (O(1/M^2)) in the
    # INTERIOR but only weakly (O(1/M)) right at the driven face x=a, where
    # psi has nonzero normal derivative -- this is Remark 1 in the notes, not
    # an error. So we test the interior tightly and the boundary loosely.
    xx = np.linspace(0, a, 400)
    psi_num = s.psi(xx, np.array([b / 2]))[0]
    psi_an = psi_exact(xx)
    err = np.abs(psi_num - psi_an)
    interior = (xx > 0.02 * a) & (xx < 0.98 * a)
    err_int = err[interior].max()
    err_bnd = err.max()
    print(f"  max |psi_num - psi_exact| interior = {err_int:.3e}")
    print(f"  max |psi_num - psi_exact| boundary = {err_bnd:.3e}"
          f"   (O(1/M) at x=a, expected)")
    assert err_int < 5e-3, "psi interior mismatch"

    # full-field check at several times (transient should be ~0), interior norm
    for t in [0.0, 0.5, 2.0, 10.0]:
        Tnum = s.field(xx, np.array([b / 2]), t)[0]
        Tan = gamma_exact * t + psi_an + mean0
        err_t = np.abs(Tnum - Tan)[interior].max()
        print(f"  t={t:5.2f}:  max |T_num - T_exact| interior = {err_t:.3e}")
        assert err_t < 5e-3, f"field mismatch at t={t}"
    print("  PASS\n")

    print("=" * 64)
    print("VERIFICATION 2 - constant IC reconstruction + energy balance")
    print("=" * 64)
    # constant IC (so there IS a real transient), balanced sinusoidal flux
    s2 = NeumannHeat2D(a=3.0, b=2.0, alpha=1.0,
                       p0=lambda y: np.sin(np.pi * y / 2.0),
                       pa=lambda y: np.sin(np.pi * y / 2.0),   # p0=pa -> no x net
                       r0=0.0, rb=0.0,
                       T0=5.0, n_modes_x=64, n_modes_y=64)
    xx = np.linspace(0, 3.0, 120)
    yy = np.linspace(0, 2.0, 120)
    T_at_0 = s2.field(xx, yy, 0.0)
    err_ic = np.max(np.abs(T_at_0 - 5.0))
    print(f"  balanced flux -> Q_net = {s2.Q_net:.3e}, gamma = {s2.gamma:.3e}")
    print(f"  max |T(.,.,0) - T0|  = {err_ic:.3e}   (IC reconstruction)")
    assert err_ic < 1e-2, "IC reconstruction failed"

    # energy balance: d/dt mean(T) should equal gamma (here ~0)
    dt = 1e-3
    dmean = (s2.mean_temperature(dt) - s2.mean_temperature(0)) / dt
    print(f"  d/dt mean(T) = {dmean:.3e}  vs  gamma = {s2.gamma:.3e}")
    print("  PASS\n")


def _demo():
    """A richer example with localized heating, sinusoidal side flux, bumpy IC."""
    print("=" * 64)
    print("DEMO - localized heating, net-positive flux (slab warms up)")
    print("=" * 64)
    a, b, alpha = 4.0, 2.0, 0.15

    # Right face: Gaussian heat influx centred at mid-height.
    # (coordinate derivative pa = dT/dx > 0 means heat flowing in from the right)
    pa = lambda y: 2.0 * np.exp(-((y - b / 2) ** 2) / (2 * 0.25 ** 2))
    # Left face insulated; bottom loses a little heat uniformly; top insulated.
    p0 = 0.0
    r0 = lambda x: -0.15 + 0.0 * x
    rb = 0.0
    # Wavy initial condition.
    T0 = lambda X, Y: 20.0 + 3.0 * np.cos(np.pi * X / a) * np.cos(2 * np.pi * Y / b)

    s = NeumannHeat2D(a, b, alpha, p0=p0, pa=pa, r0=r0, rb=rb, T0=T0,
                      n_modes_x=64, n_modes_y=64)
    print(s.summary())

    s.plot_decomposition(t=8.0)
    plt.savefig("neumann_demo_decomposition.png", dpi=130)
    print("\n  saved: neumann_demo_decomposition.png")

    s.plot_evolution(times=[0.0, 3.0, 10.0, 30.0])
    plt.savefig("neumann_demo_evolution.png", dpi=130)
    print("  saved: neumann_demo_evolution.png")


if __name__ == "__main__":
    _run_verification()
    _demo()
    print("\nAll checks complete.")
