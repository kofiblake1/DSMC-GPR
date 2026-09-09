r"""
plate_heating.py
================

Free-molecular (collisionless) aerodynamic heating of a 2-D flat plate at angle
of attack, solved analytically for the plate's internal temperature field and
rendered as an MP4.

Two boundary-condition models, both fed by the same Schaaf-Chambre surface flux:

  * NeumannHeat2D  -- wall temperature FROZEN in the flux law => constant flux
                      per face => pure-Neumann problem. The plate heats forever
                      (no steady state). Cleanest fixed-wall verification case.

  * RobinHeat2D    -- exact self-regulating condition. Re-emission scales with the
                      LOCAL surface temperature, giving a Robin (Newton-cooling) BC
                      per face:   k_s dT/dn + h_j T = sigma q_i,j ,
                      h_j = 2 sigma k_B N_i,j. Every mode decays; the plate relaxes
                      to a genuine steady state (each face -> its recovery temp
                      T_rec = q_i / (2 k_B N_i) = g_j / H_j).

Surface flux (per face, monatomic gas, translational energy only):
    s   = -S (u_hat . n_hat) = S sin(theta)          signed normal speed ratio
    N_i = n c_m /(2 sqrt pi) [ e^{-s^2} + sqrt(pi) s (1+erf s) ]
    q_i = rho c_m^3/(4 sqrt pi)[ (S^2+2) e^{-s^2} + sqrt(pi) s (S^2+5/2)(1+erf s) ]
    q_net = sigma ( q_i - 2 k_B T_w N_i )            (+ = into the plate)

Both models expose `from_freestream_plate(...)`: give it gas + freestream + angle
of attack + accommodation + plate size + solid properties, and it computes every
surface flux and builds the solver.

Run:  python plate_heating.py            # verifies, then writes plate_heating.mp4
"""

from __future__ import annotations
import numpy as np
from scipy.special import erf
from scipy.optimize import brentq

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# ----------------------------------------------------------------------
# physical constants
# ----------------------------------------------------------------------
K_B = 1.380649e-23         # J/K
AMU = 1.66053906660e-27    # kg


# ----------------------------------------------------------------------
# free-molecular surface fluxes (Schaaf-Chambre, monatomic)
# ----------------------------------------------------------------------
def free_molecular_fluxes(n_inf, T_inf, S, c_m, m_mol, s_n, T_w, sigma):
    """
    Per-face incident number flux N_i, incident energy flux q_i, and net heat
    flux into the wall q_net. Accepts scalar or array s_n. All SI.
    """
    s_n = np.asarray(s_n, dtype=float)
    rho = m_mol * n_inf
    E = np.exp(-s_n**2)
    P = 1.0 + erf(s_n)
    sp = np.sqrt(np.pi)
    N_i = n_inf * c_m / (2*sp) * (E + sp*s_n*P)
    q_i = rho * c_m**3 / (4*sp) * ((S**2 + 2)*E + sp*s_n*(S**2 + 2.5)*P)
    q_net = sigma * (q_i - 2*K_B*T_w*N_i)
    return N_i, q_i, q_net


def plate_face_speed_ratios(S, alpha_deg):
    """Signed normal speed ratios for the four faces of a plate at AoA.
    Returns dict keyed by edge: x0=leading, xa=trailing, y0=bottom(windward),
    yb=top(leeward)."""
    a = np.radians(alpha_deg)
    return {
        "x0_leading":  S*np.cos(a),
        "xa_trailing": -S*np.cos(a),
        "y0_bottom":   S*np.sin(a),
        "yb_top":      -S*np.sin(a),
    }


def freestream_kinematics(T_inf, m_mol, gamma=5/3, M=None, U=None):
    """Return (c_m, S, U). Provide either Mach M or speed U."""
    c_m = np.sqrt(2*K_B*T_inf/m_mol)
    if M is not None:
        S = M*np.sqrt(gamma/2)
        U = S*c_m
    elif U is not None:
        S = U/c_m
    else:
        raise ValueError("provide Mach M or speed U")
    return c_m, S, U


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _trap_weights(grid):
    w = np.empty_like(grid, float)
    w[1:-1] = 0.5*(grid[2:]-grid[:-2]); w[0] = 0.5*(grid[1]-grid[0])
    w[-1] = 0.5*(grid[-1]-grid[-2]); return w


def _wrap(f):
    if f is None:
        return lambda s: np.zeros_like(np.asarray(s, float))
    if np.isscalar(f):
        c = float(f); return lambda s: np.full_like(np.asarray(s, float), c)
    def g(s):
        s = np.asarray(s, float); o = np.asarray(f(s), float)
        return o if o.shape == s.shape else np.vectorize(f)(s)
    return g


def _wrap2(f):
    if f is None:
        return lambda X, Y: np.zeros_like(np.asarray(X, float))
    if np.isscalar(f):
        c = float(f); return lambda X, Y: np.full_like(np.asarray(X, float), c)
    def g(X, Y):
        X = np.asarray(X, float); Y = np.asarray(Y, float)
        o = np.asarray(f(X, Y), float)
        return o if o.shape == X.shape else np.vectorize(f)(X, Y)
    return g


# ======================================================================
# NEUMANN solver (frozen wall temperature)  -- cosine basis
# ======================================================================
class NeumannHeat2D:
    def __init__(self, a, b, alpha, p0=None, pa=None, r0=None, rb=None, T0=None,
                 n_modes_x=48, n_modes_y=48, n_quad=4000):
        self.a, self.b, self.alpha = float(a), float(b), float(alpha)
        self.M, self.N = int(n_modes_x), int(n_modes_y)
        p0, pa, r0, rb, T0f = _wrap(p0), _wrap(pa), _wrap(r0), _wrap(rb), _wrap2(T0)
        m = np.arange(self.M+1); n = np.arange(self.N+1)
        self.lam = (m[:, None]*np.pi/self.a)**2 + (n[None, :]*np.pi/self.b)**2
        eps_m = np.where(m == 0, 1.0, 2.0); eps_n = np.where(n == 0, 1.0, 2.0)
        self.eps_mn = eps_m[:, None]*eps_n[None, :]
        yq = np.linspace(0, self.b, n_quad); wy = _trap_weights(yq)
        xq = np.linspace(0, self.a, n_quad); wx = _trap_weights(xq)
        Cy = np.cos(n[:, None]*np.pi*yq[None, :]/self.b)
        Cx = np.cos(m[:, None]*np.pi*xq[None, :]/self.a)
        pa_h = Cy@(wy*pa(yq)); p0_h = Cy@(wy*p0(yq))
        rb_h = Cx@(wx*rb(xq)); r0_h = Cx@(wx*r0(xq))
        sgn_m = (-1.0)**m; sgn_n = (-1.0)**n
        B = (np.outer(sgn_m, pa_h) - p0_h[None, :]
             + np.outer(rb_h, sgn_n) - r0_h[:, None])
        self.B = B; self.Q_net = B[0, 0]
        self.gamma = self.alpha*self.Q_net/(self.a*self.b)
        A = np.zeros_like(B); mask = np.ones_like(B, bool); mask[0, 0] = False
        A[mask] = self.eps_mn[mask]*B[mask]/(self.a*self.b*self.lam[mask])
        self.A = A
        Xg, Yg = np.meshgrid(xq, yq); G = (wy[:, None]*T0f(Xg, Yg))*wx[None, :]
        That0 = self.eps_mn*(Cx@(Cy@G).T)/(self.a*self.b)
        self.C = That0 - A; self.Tbar0 = That0[0, 0]

    def _assemble(self, coeff, x, y):
        m = np.arange(self.M+1); n = np.arange(self.N+1)
        Cx = np.cos(m[:, None]*np.pi*np.asarray(x)[None, :]/self.a)
        Cy = np.cos(n[:, None]*np.pi*np.asarray(y)[None, :]/self.b)
        return (coeff@Cy).T@Cx

    def field(self, x, y, t):
        coeff = self.A + self.C*np.exp(-self.alpha*self.lam*t)
        return self.gamma*t + self._assemble(coeff, x, y)

    @classmethod
    def from_flux(cls, a, b, alpha, k, q_left=None, q_right=None,
                  q_bottom=None, q_top=None, T0=None, **kw):
        """Build from OUTWARD heat flux q_n (+ = leaving). dT/dn = -q_n/k."""
        qL, qR, qB, qT = _wrap(q_left), _wrap(q_right), _wrap(q_bottom), _wrap(q_top)
        return cls(a, b, alpha,
                   p0=lambda y: qL(y)/k, pa=lambda y: -qR(y)/k,
                   r0=lambda x: qB(x)/k, rb=lambda x: -qT(x)/k, T0=T0, **kw)

    @classmethod
    def from_freestream_plate(cls, n_inf, T_inf, m_mol, a, b, k_s, rho_cp,
                              alpha_deg, T_w, sigma=1.0, gamma_gas=5/3,
                              M=None, U=None, T0=None, **kw):
        """
        Frozen-wall free-molecular plate. Computes q_net on each face at fixed
        wall temperature T_w and routes it through `from_flux`. The net flux is
        INTO the plate, i.e. outward heat flux = -q_net.
        """
        c_m, S, U = freestream_kinematics(T_inf, m_mol, gamma_gas, M, U)
        sr = plate_face_speed_ratios(S, alpha_deg)
        qn = {}
        for key, s in sr.items():
            _, _, q_net = free_molecular_fluxes(n_inf, T_inf, S, c_m, m_mol, s, T_w, sigma)
            qn[key] = float(q_net)
        alpha_s = k_s/rho_cp
        self = cls.from_flux(a, b, alpha_s, k_s,
                             q_left=-qn["x0_leading"], q_right=-qn["xa_trailing"],
                             q_bottom=-qn["y0_bottom"], q_top=-qn["yb_top"],
                             T0=T0, **kw)
        self.meta = dict(c_m=c_m, S=S, U=U, q_net=qn, T_w=T_w, kind="Neumann")
        return self


# ======================================================================
# ROBIN solver (self-regulating wall)  -- transcendental Robin eigenbasis
# ======================================================================
def _robin_eigen_1d(L, H_lo, H_hi, n_modes, n_quad=3000):
    r"""
    Solve X'' + kappa^2 X = 0 with  X'(0) = H_lo X(0),  X'(L) = -H_hi X(L).
    Eigencondition (pole-free):
        F(k) = (H_lo H_hi - k^2) sin(kL) + k (H_lo + H_hi) cos(kL) = 0
    Eigenfunction: X_m(x) = cos(k x) + (H_lo/k) sin(k x).
    Returns (kappas, mus, Xq[m,i], xq, wq, normX[m], IX[m], Xlo=1, Xhi[m]).
    """
    def F(k):
        return (H_lo*H_hi - k*k)*np.sin(k*L) + k*(H_lo + H_hi)*np.cos(k*L)

    near_neumann = (H_lo + H_hi) < 1e-9
    kmax = (n_modes + 2)*np.pi/L
    ks = np.linspace(1e-7, kmax, int((n_modes + 2)*400))
    vals = F(ks)
    roots = []
    if near_neumann:
        roots.append(0.0)   # genuine constant mode when both ends insulated
    for i in range(len(ks)-1):
        if vals[i]*vals[i+1] < 0:
            roots.append(brentq(F, ks[i], ks[i+1], xtol=1e-13, rtol=1e-13))
        if len(roots) >= n_modes:
            break
    kappas = np.array(roots[:n_modes])

    xq = np.linspace(0, L, n_quad); wq = _trap_weights(xq)

    def Xeval(k, x):
        x = np.asarray(x, float)
        if k == 0.0:
            return np.ones_like(x)             # constant mode
        return np.cos(k*x) + (H_lo/k)*np.sin(k*x)

    Xq = np.array([Xeval(k, xq) for k in kappas])          # (n_modes, n_quad)
    normX = Xq**2 @ wq
    IX = Xq @ wq
    Xlo = np.array([1.0 for _ in kappas])                  # X_m(0) = 1
    Xhi = np.array([Xeval(k, L)[()] if k else 1.0 for k in kappas])
    return dict(kappas=kappas, mus=kappas**2, Xq=Xq, xq=xq, wq=wq,
                normX=normX, IX=IX, Xlo=Xlo, Xhi=Xhi, H_lo=H_lo, L=L, Xeval=Xeval)


class RobinHeat2D:
    r"""
    dT/dt = alpha laplacian(T),  with per-edge Robin data
        dT/dn + H_j T = g_j        (H_j >= 0 [1/m], g_j [K/m])
    on x0,xa (functions of the x-eigenbasis) and y0,yb (y-eigenbasis).

    T(x,y,t) = sum_{mn} [ c_ss + (c0 - c_ss) e^{-alpha lam_mn t} ] X_m(x) Y_n(y)
        c_ss = G_mn / (lam_mn |phi_mn|^2),   G_mn = boundary projection of g
        c0   = <T0, phi_mn> / |phi_mn|^2
    Every lam_mn > 0 (for H>0) so the series relaxes to the steady state
    T_s = sum c_ss phi_mn.
    """
    def __init__(self, a, b, alpha, H, g, T0=None,
                 n_modes_x=24, n_modes_y=24, n_quad=3000):
        self.a, self.b, self.alpha = float(a), float(b), float(alpha)
        Hx0, Hxa, Hy0, Hyb = H
        gx0, gxa, gy0, gyb = g
        self.H = dict(x0=Hx0, xa=Hxa, y0=Hy0, yb=Hyb)
        self.g = dict(x0=gx0, xa=gxa, y0=gy0, yb=gyb)

        self.X = _robin_eigen_1d(a, Hx0, Hxa, n_modes_x, n_quad)
        self.Y = _robin_eigen_1d(b, Hy0, Hyb, n_modes_y, n_quad)
        mux, muy = self.X["mus"], self.Y["mus"]
        self.lam = mux[:, None] + muy[None, :]                # (Mx, My)
        self.norm2 = np.outer(self.X["normX"], self.Y["normX"])

        # boundary drive projection G_mn
        IX, IY = self.X["IX"], self.Y["IX"]
        Xlo, Xhi = self.X["Xlo"], self.X["Xhi"]
        Ylo, Yhi = self.Y["Xlo"], self.Y["Xhi"]
        xterm = (gx0*Xlo + gxa*Xhi)[:, None] * IY[None, :]    # x-edges * int Y
        yterm = (gy0*Ylo + gyb*Yhi)[None, :] * IX[:, None]    # y-edges * int X
        self.G = xterm + yterm
        self.c_ss = self.G/(self.lam*self.norm2)

        # initial condition projection
        T0f = _wrap2(T0)
        xq, yq = self.X["xq"], self.Y["xq"]
        wx, wy = self.X["wq"], self.Y["wq"]
        Xg, Yg = np.meshgrid(xq, yq)                          # (ny, nx)
        T0grid = T0f(Xg, Yg)
        Xw = self.X["Xq"]*wx[None, :]                         # (Mx, nx)
        Yw = self.Y["Xq"]*wy[None, :]                         # (My, ny)
        proj = Xw @ (Yw @ T0grid).T                           # (Mx, My)
        self.c0 = proj/self.norm2

    # ---- field evaluation ----
    def _basis(self, x, y):
        Xp = np.array([self.X["Xeval"](k, x) for k in self.X["kappas"]])  # (Mx,nx)
        Yp = np.array([self.Y["Xeval"](k, y) for k in self.Y["kappas"]])  # (My,ny)
        return Xp, Yp

    def field(self, x, y, t):
        coeff = self.c_ss + (self.c0 - self.c_ss)*np.exp(-self.alpha*self.lam*t)
        Xp, Yp = self._basis(x, y)
        return (coeff@Yp).T@Xp

    def steady(self, x, y):
        Xp, Yp = self._basis(x, y)
        return (self.c_ss@Yp).T@Xp

    def slowest_time(self):
        return 1.0/(self.alpha*self.lam.min())

    def relaxation_time(self, frac=0.02):
        """Time for the transient's L2 amplitude to fall to `frac` of its
        initial value. This is the *visible* relaxation time -- unlike the
        slowest single mode, it is weighted by how much each mode is actually
        excited, so weakly-excited near-zero modes don't dominate it."""
        amp2 = (self.c0 - self.c_ss)**2 * self.norm2
        rate = self.alpha * self.lam
        tot = amp2.sum()
        if tot <= 0:
            return self.slowest_time()

        def resid(t):
            return np.sqrt((amp2*np.exp(-2*rate*t)).sum()/tot)

        hi = 1.0
        for _ in range(200):
            if resid(hi) <= frac:
                break
            hi *= 2
        lo = 0.0
        for _ in range(100):
            mid = 0.5*(lo+hi)
            if resid(mid) > frac:
                lo = mid
            else:
                hi = mid
        return hi

    @classmethod
    def from_freestream_plate(cls, n_inf, T_inf, m_mol, a, b, k_s, rho_cp,
                              alpha_deg, sigma=1.0, gamma_gas=5/3,
                              M=None, U=None, T0=None, **kw):
        """
        Exact self-regulating free-molecular plate. Per face:
            H = 2 sigma k_B N_i / k_s,   g = sigma q_i / k_s
        (both independent of wall temperature). Recovery temp per face = g/H.
        """
        c_m, S, U = freestream_kinematics(T_inf, m_mol, gamma_gas, M, U)
        sr = plate_face_speed_ratios(S, alpha_deg)
        H, g, Ni, qi, Trec = {}, {}, {}, {}, {}
        for key, s in sr.items():
            N_i, q_i, _ = free_molecular_fluxes(n_inf, T_inf, S, c_m, m_mol, s, 0.0, sigma)
            H[key] = 2*sigma*K_B*N_i/k_s
            g[key] = sigma*q_i/k_s
            Ni[key], qi[key] = float(N_i), float(q_i)
            Trec[key] = g[key]/H[key] if H[key] > 0 else np.nan
        alpha_s = k_s/rho_cp
        self = cls(a, b, alpha_s,
                   H=(H["x0_leading"], H["xa_trailing"], H["y0_bottom"], H["yb_top"]),
                   g=(g["x0_leading"], g["xa_trailing"], g["y0_bottom"], g["yb_top"]),
                   T0=T0, **kw)
        self.meta = dict(c_m=c_m, S=S, U=U, N_i=Ni, q_i=qi, T_rec=Trec, kind="Robin")
        return self


# ======================================================================
# verification of the Robin solver against explicit finite differences
# ======================================================================
def _verify_robin_fd():
    print("="*66)
    print("VERIFY  Robin analytic solution vs explicit finite differences")
    print("="*66)
    a, b, alpha = 1.0, 1.0, 1.0
    H = (2.0, 1.0, 3.0, 0.5)          # x0, xa, y0, yb
    g = (1.0, -0.5, 2.0, 0.3)
    T0 = 0.2
    sol = RobinHeat2D(a, b, alpha, H, g, T0=T0, n_modes_x=60, n_modes_y=60)

    # explicit FD with ghost nodes for Robin BC
    nx, ny = 81, 81
    x = np.linspace(0, a, nx); y = np.linspace(0, b, ny)
    dx, dy = x[1]-x[0], y[1]-y[0]
    dt = 0.2*min(dx, dy)**2/alpha
    T = np.full((ny, nx), T0, float)
    Hx0, Hxa, Hy0, Hyb = H; gx0, gxa, gy0, gyb = g

    def lap_step(T):
        Tn = T.copy()
        Te = np.zeros((ny+2, nx+2))
        Te[1:-1, 1:-1] = T
        # Robin ghosts:  dT/dn + H T = g  (outward normal)
        # x=0: -T_x + Hx0 T = gx0 -> T_ghostL = T[:,1] - 2 dx (Hx0 T[:,0] - gx0)
        Te[1:-1, 0]  = T[:, 1]  - 2*dx*(Hx0*T[:, 0]  - gx0)
        Te[1:-1, -1] = T[:, -2] - 2*dx*(Hxa*T[:, -1] - gxa)
        Te[0, 1:-1]  = T[1, :]  - 2*dy*(Hy0*T[0, :]  - gy0)
        Te[-1, 1:-1] = T[-2, :] - 2*dy*(Hyb*T[-1, :] - gyb)
        lap = ((Te[1:-1, 2:] - 2*T + Te[1:-1, :-2])/dx**2
               + (Te[2:, 1:-1] - 2*T + Te[:-2, 1:-1])/dy**2)
        return T + dt*alpha*lap

    # The Robin eigenbasis has homogeneous Robin BC, so (as with the Neumann
    # cosine basis) the series converges O(1/nm) at the edges but fast in the
    # interior. Verify the interior against FD; report the boundary separately.
    mi = (x > 0.06*a) & (x < 0.94*a)
    mj = (y > 0.06*b) & (y < 0.94*b)
    checks = [0.02, 0.1, 0.4]
    t = 0.0; nextj = 0
    tmax = checks[-1]
    results = []
    while t < tmax + dt/2:
        if nextj < len(checks) and t >= checks[nextj] - dt/2:
            Ta = sol.field(x, y, checks[nextj])
            rng = np.max(Ta) - np.min(Ta) + 1e-12
            err_full = np.max(np.abs(Ta - T))/rng
            err_int = np.max(np.abs(Ta - T)[np.ix_(mj, mi)])/rng
            results.append((checks[nextj], err_int, err_full))
            nextj += 1
        T = lap_step(T); t += dt
    for tc, ei, ef in results:
        print(f"  t={tc:.3f}:  interior |analytic-FD|/range = {ei:.3e}"
              f"   (boundary {ef:.3e}, O(1/nm)+FD)")
        assert ei < 1.5e-2, "Robin solver disagrees with FD in the interior"
    # steady BC residual check
    xf = np.linspace(0, a, 200); yf = np.linspace(0, b, 200)
    Ts = sol.steady(xf, yf)
    print(f"  steady-state field range: [{Ts.min():.4f}, {Ts.max():.4f}]")
    print("  PASS\n")


# ======================================================================
# animation
# ======================================================================
def animate_evolution(sol, outfile="plate_heating.mp4", n_frames=160,
                      t_end_factor=4.0, fps=30, nx=240, ny=120, dpi=140):
    """Render an MP4 of T(x,y,t) plus a through-thickness profile at mid-chord."""
    a, b = sol.a, sol.b
    x = np.linspace(0, a, nx); y = np.linspace(0, b, ny)
    tau = sol.relaxation_time(frac=0.02) if hasattr(sol, "relaxation_time") \
        else sol.slowest_time()
    t_end = t_end_factor*tau
    # denser sampling early (diffusive), quadratic time spacing
    frac = (np.arange(n_frames)/(n_frames-1))**2
    times = t_end*frac

    Ts = sol.steady(x, y)
    T_ic = sol.field(x, y, 0.0)
    vmin = min(T_ic.min(), Ts.min())
    vmax = max(T_ic.max(), Ts.max())
    pad = 0.03*(vmax-vmin); vmin -= pad; vmax += pad
    levels = np.linspace(vmin, vmax, 40)

    xmid = a/2
    Tprof_ss = sol.steady(np.array([xmid]), y)[:, 0]
    Tprof_ic = sol.field(np.array([xmid]), y, 0.0)[:, 0]

    fig, (axf, axp) = plt.subplots(
        1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [2.1, 1]})
    fig.subplots_adjust(left=0.06, right=0.97, wspace=0.28, bottom=0.14, top=0.86)

    cf = axf.contourf(x*1e3, y*1e3, T_ic, levels=levels, cmap="inferno", extend="both")
    cbar = fig.colorbar(cf, ax=axf, pad=0.02); cbar.set_label("T  [K]")
    axf.set_aspect("equal")
    axf.set_xlabel("chord  x  [mm]", labelpad=4)
    axf.set_ylabel("thickness  y  [mm]")

    # flow arrow (angle of attack) + windward/leeward labels inside the panel
    meta = getattr(sol, "meta", {})
    axf.annotate("", xy=(0.17*a*1e3, 0.62*b*1e3),
                 xytext=(0.01*a*1e3, 0.30*b*1e3),
                 arrowprops=dict(arrowstyle="-|>", lw=2.2, color="cyan"))
    axf.text(0.02*a*1e3, 0.30*b*1e3, "flow", color="cyan", fontsize=9,
             ha="left", va="top")
    _bb = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.65)
    axf.text(0.985, 0.05, "windward", transform=axf.transAxes, ha="right",
             va="bottom", fontsize=8, bbox=_bb)
    axf.text(0.985, 0.95, "leeward", transform=axf.transAxes, ha="right",
             va="top", fontsize=8, bbox=_bb)
    axf.text(0.02, 0.95, "leading\nedge", transform=axf.transAxes, ha="left",
             va="top", fontsize=8, bbox=_bb)

    ttl = axf.set_title("")

    axp.plot(Tprof_ic, y*1e3, "b:", lw=1.3, label="initial")
    axp.plot(Tprof_ss, y*1e3, "k--", lw=1.3, label="steady state")
    (line,) = axp.plot(Tprof_ic, y*1e3, "r-", lw=2.2, label="current")
    axp.set_xlim(vmin, vmax); axp.set_ylim(0, b*1e3)
    axp.set_xlabel("T at mid-chord  [K]"); axp.set_ylabel("thickness y  [mm]")
    axp.legend(loc="lower right", fontsize=8); axp.grid(alpha=0.3)

    kind = meta.get("kind", "")
    S = meta.get("S", None)
    supt = f"Free-molecular plate heating ({kind})"
    if S is not None:
        supt += f"   S={S:.2f}"
    fig.suptitle(supt, fontsize=12, y=0.98)

    state = {"cf": cf}

    def update(i):
        t = times[i]
        Z = sol.field(x, y, t)
        try:
            state["cf"].remove()                     # matplotlib >= 3.8
        except (AttributeError, ValueError):
            for c in state["cf"].collections:        # older matplotlib
                c.remove()
        state["cf"] = axf.contourf(x*1e3, y*1e3, Z, levels=levels,
                                   cmap="inferno", extend="both")
        prof = sol.field(np.array([xmid]), y, t)[:, 0]
        line.set_xdata(prof)
        ttl.set_text(f"t = {t:8.1f} s      mean T = {Z.mean():6.1f} K"
                     f"      t/tau = {t/tau:4.2f}")
        return [line, ttl]

    print(f"Rendering {n_frames} frames to {outfile}  (t_end={t_end:.2f}s, tau={tau:.2f}s) ...")
    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
    writer = FFMpegWriter(fps=fps, bitrate=2400,
                          metadata=dict(artist="plate_heating.py"))
    anim.save(outfile, writer=writer, dpi=dpi)
    plt.close(fig)
    print(f"  wrote {outfile}")


# ======================================================================
# main
# ======================================================================
def main():
    _verify_robin_fd()

    # ---- Mach 5 argon over a plate at angle of attack ----
    m_Ar = 39.948*AMU
    params = dict(
        n_inf=1.0e20, T_inf=350.0, m_mol=m_Ar, M=5.0, gamma_gas=5/3,
        alpha_deg=-15.0, sigma=1.0,
        a=0.06, b=0.02,                 # chord 60 mm, thickness 20 mm
        k_s=0.02, rho_cp=5.0e5,         # low-k ceramic -> Biot ~ 1.5 (finite-Biot regime)
        T0=300.0,                       # start cold and uniform
    )

    print("="*66)
    print("Mach 5 argon plate  --  Robin (self-regulating) model")
    print("="*66)
    robin = RobinHeat2D.from_freestream_plate(n_modes_x=32, n_modes_y=32, **params)
    robin.meta["alpha_deg"] = params["alpha_deg"]
    mt = robin.meta
    print(f"c_m={mt['c_m']:.1f} m/s  S={mt['S']:.3f}  U={mt['U']:.1f} m/s")
    print("per-face recovery temperature T_rec = q_i/(2 k_B N_i):")
    for kf, v in mt["T_rec"].items():
        print(f"   {kf:12s}: {v:8.1f} K   (q_i={mt['q_i'][kf]:.3e} W/m^2)")
    print(f"slowest relaxation time tau = {robin.slowest_time():.2f} s")
    ss = robin.steady(np.linspace(0, params['a'], 120),
                      np.linspace(0, params['b'], 120))
    print(f"steady-state T range: [{ss.min():.1f}, {ss.max():.1f}] K")

    animate_evolution(robin, outfile="plate_heating.mp4",
                      n_frames=160, t_end_factor=4.0, fps=30)

    # frozen-wall Neumann version is also available:
    neu = NeumannHeat2D.from_freestream_plate(T_w=300.0, n_modes_x=40,
                                              n_modes_y=40, **params)
    print(f"\n[Neumann/frozen-wall check]  drift rate gamma = {neu.gamma:.4f} K/s "
          f"(plate heats without bound)")


if __name__ == "__main__":
    main()
