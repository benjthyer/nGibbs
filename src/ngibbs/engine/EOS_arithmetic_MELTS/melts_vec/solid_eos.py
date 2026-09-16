"""
Vectorized closed-form solid-endmember equations of state for MELTS.

Translated line-for-line from `intEOSsolid()` in MAGMA's `sources/gibbs.c`
(read directly this session -- not reconstructed from the papers). Two EOS
families are implemented:

- ``eos_berman``: a direct polynomial in (P, T) -- no root-finding at all.
- ``eos_vinet``: finite-strain (Vinet 1986), which requires a small scalar
  Newton-Raphson solve for a compression variable ``x`` -- but only one
  scalar unknown per (endmember, P, T) triple, evaluated twice (once at the
  target P, once at the reference Pr), unlike HeFESTo's coupled BM3 +
  Mie-Gruneisen-Debye volume solve. Fully vectorized: every endmember/query
  pair is solved simultaneously via elementwise numpy Newton iteration, no
  Python-level loop over endmembers or batch rows.

`EOS_SAXENA` is declared in `includes/silmin.h` but does not appear in any
of the four `Solids` tables in `sol_struct_data.h` (`sol_struct_data.h`'s
own changelog notes "removed Saxena EOS treatment" for solids) -- it is
intentionally not implemented here. `CP_SAXENA` (a *heat-capacity* model,
unrelated to `EOS_SAXENA`) does appear and lives in `thermal.py`.

Both functions return a dict of the *additive* corrections gibbs.c adds
on top of the reference-state (Tr, Pr) H/S/Cp from `thermal.py`
(``g_add, h_add, s_add, cp_add, dcpdt_add``), plus the absolute volume and
its derivatives (``V, dVdT, dVdP, d2VdT2, d2VdTdP, d2VdP2``) -- these have
no separate "reference" term in the source (they are initialized to 0.0
and set entirely by the EOS), so they are returned as totals, not deltas.

Units: P in bars, T in K, V in J/bar (= 10 x cm^3/mol) -- see constants.py.
"""
from __future__ import annotations
import numpy as np

from .constants import Tr, Pr

_DBL_EPS = np.finfo(np.float64).eps


def eos_berman(P, T, v0, v1, v2, v3, v4):
    """EOS_BERMAN: closed-form polynomial volume EOS (Berman, 1988).

    V(P,T) = v0 * [1 + v1*(P-Pr) + v2*(P-Pr)^2 + v3*(T-Tr) + v4*(T-Tr)^2]

    Translated verbatim from gibbs.c lines 456-473.
    """
    P = np.asarray(P, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    dP = P - Pr
    dT = T - Tr

    g_add = v0*((v1/2.0 - v2)*(P**2 - Pr**2) + v2*(P**3 - Pr**3)/3.0
                + (1.0 - v1 + v2 + v3*dT + v4*dT**2)*dP)
    h_add = g_add - T*v0*(v3 + 2.0*dT*v4)*dP
    s_add = -v0*(v3 + 2.0*dT*v4)*dP
    cp_add = -T*v0*2.0*v4*dP
    dcpdt_add = -v0*2.0*v4*dP

    V       = v0*(1.0 + dP*v1 + dP**2*v2 + dT*v3 + dT**2*v4)
    dVdT    = v0*(v3 + 2.0*dT*v4)
    dVdP    = v0*(v1 + 2.0*dP*v2)
    d2VdT2  = v0*2.0*v4 * np.ones_like(V)
    d2VdTdP = np.zeros_like(V)
    d2VdP2  = v0*2.0*v2 * np.ones_like(V)

    return dict(g_add=g_add, h_add=h_add, s_add=s_add,
                cp_add=cp_add, dcpdt_add=dcpdt_add,
                V=V, dVdT=dVdT, dVdP=dVdP,
                d2VdT2=d2VdT2, d2VdTdP=d2VdTdP, d2VdP2=d2VdP2)


def _vinet_solve_x(P, T, K, Kp, alpha, eta, n_iter=60):
    """Batched scalar Newton-Raphson solve for the Vinet compression
    variable x(P,T), plus its analytic T/P derivatives at the converged x.

    Translated from gibbs.c lines 485-501 (the P-branch of the do/while
    solve; the Pr-branch is the same code with P replaced by Pr -- callers
    get that by passing P=Pr, T=T here a second time).

    All formulas work in "p_ei" = P/10000 units (bars -> the E-o-S's native
    pressure scale, matching K in the same units the parameter table uses)
    exactly as gibbs.c does; the 1e4 / 1e8 rescalings needed to turn these
    into dV/dP, d2V/dTdP, d2V/dP2 (P in bars) are applied by the caller,
    exactly where gibbs.c applies them.
    """
    P = np.asarray(P, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    p_ei = P / 10000.0
    dT = T - Tr

    x = np.ones(np.broadcast_shapes(P.shape, T.shape, np.shape(K)), dtype=np.float64)
    for _ in range(n_iter):
        expo = np.exp(eta*(1.0 - x))
        fn  = x**2*p_ei - 3.0*K*(1.0 - x)*expo - x**2*alpha*K*dT
        dfn = 2.0*x*p_ei + 3.0*K*(1.0 + eta*(1.0 - x))*expo - 2.0*x*alpha*K*dT
        dfn_safe = np.where(np.abs(dfn) < _DBL_EPS, _DBL_EPS, dfn)
        x = x - fn/dfn_safe

    expo = np.exp(eta*(1.0 - x))

    dxdt = (-(1.0/3.0)*x**3*alpha*K
            / (K*expo*(-2.0 + x - eta*x + eta*x**2)))

    denom_t = (2.0*p_ei*x + 3.0*K*expo + 3.0*K*eta*expo
               - 3.0*K*eta*x*expo - 2.0*alpha*K*x*dT)
    numer_t2 = (2.0*p_ei*dxdt - 6.0*K*eta*dxdt*expo - 3.0*K*eta**2*dxdt*expo
                + 3.0*K*eta**2*x*dxdt*expo - 4.0*alpha*K*x - 2.0*alpha*K*dxdt*dT)
    d2xdt2 = -dxdt*numer_t2/denom_t

    dxdp = -(1.0/3.0)*x**3/(K*expo*(2.0 - x + eta*x - eta*x**2))

    numer_tp = (2.0*x*dxdt + 2.0*p_ei*dxdp*dxdt - 2.0*alpha*K*dxdp*dxdt*dT
                - 3.0*K*eta**2*dxdt*dxdp*expo - 6.0*K*dxdt*eta*dxdp*expo
                - 2.0*alpha*K*x*dxdp + 3.0*K*eta**2*dxdt*dxdp*expo*x)
    d2xdtdp = -numer_tp/denom_t

    numer_p2 = ((-6.0 + 2.0*x - 4.0*eta*x + 2.0*eta*x**2 - eta**2*x**2
                 + eta**2*x**3)*expo)
    denom_p2 = x*expo*(2.0 - x + eta*x - eta*x**2)
    d2xdp2 = -dxdp**2*numer_p2/denom_p2

    return x, dxdt, d2xdt2, dxdp, d2xdtdp, d2xdp2


def eos_vinet(P, T, v0, alpha, K, Kp, n_iter=60):
    """EOS_VINET: finite-strain volume EOS (Vinet, 1986).

    Translated verbatim from gibbs.c lines 474-533. Requires solving for
    the compression variable x at both P and Pr (the reference-state
    volume at Pr is not simply v0 once a nonzero thermal-expansion term is
    present, since x0(T) still depends on T -- only P is fixed at Pr).
    """
    P = np.asarray(P, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    eta = 1.5*(Kp - 1.0)
    dT = T - Tr

    x,  dxdt,  d2xdt2,  dxdp,  d2xdtdp,  d2xdp2  = _vinet_solve_x(P,  T, K, Kp, alpha, eta, n_iter)
    x0, dx0dt, d2x0dt2, _dx0dp, _d2x0dtdp, _d2x0dp2 = _vinet_solve_x(
        Pr*np.ones_like(P), T, K, Kp, alpha, eta, n_iter)

    eta2 = eta*eta
    a  =  (9.0*v0*K/eta2)*(1.0 - eta*(1.0 - x))*np.exp(eta*(1.0 - x))
    a +=  v0*dT*K*alpha*(x**3 - 1.0) - 9.0*v0*K/eta2
    a -=  (9.0*v0*K/eta2)*(1.0 - eta*(1.0 - x0))*np.exp(eta*(1.0 - x0))
    a -=  v0*dT*K*alpha*(x0**3 - 1.0) - 9.0*v0*K/eta2

    g_add = -a*10000.0 + P*v0*x**3 - Pr*v0*x0**3
    h_add = g_add + 10000.0*T*alpha*K*v0*(x**3 - x0**3)
    s_add = 10000.0*alpha*K*v0*(x**3 - x0**3)
    cp_add = 10000.0*T*alpha*K*v0*3.0*(x**2*dxdt - x0**2*dx0dt)
    dcpdt_add = (10000.0*alpha*K*v0*3.0*(x**2*dxdt - x0**2*dx0dt)
                 + 10000.0*T*alpha*K*v0*3.0*(2.0*x*dxdt**2 + x**2*d2xdt2
                                             - 2.0*x0*dx0dt**2 - x0**2*d2x0dt2))

    V       = v0*x**3
    dVdT    = 3.0*v0*x**2*dxdt
    dVdP    = 3.0*v0*x**2*dxdp/10000.0
    d2VdT2  = 3.0*v0*(2.0*x*dxdt**2 + x**2*d2xdt2)
    d2VdTdP = 3.0*v0*(2.0*x*dxdt*dxdp + x**2*d2xdtdp)/10000.0
    d2VdP2  = 3.0*v0*(2.0*x*dxdp**2 + x**2*d2xdp2)/(10000.0*10000.0)

    return dict(g_add=g_add, h_add=h_add, s_add=s_add,
                cp_add=cp_add, dcpdt_add=dcpdt_add,
                V=V, dVdT=dVdT, dVdP=dVdP,
                d2VdT2=d2VdT2, d2VdTdP=d2VdTdP, d2VdP2=d2VdP2)
