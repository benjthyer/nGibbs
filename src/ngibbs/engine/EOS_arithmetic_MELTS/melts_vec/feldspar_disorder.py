"""
MELTS's sanidine Al-Si order-disorder correction, translated verbatim from
`sources/gibbs.c`'s dedicated named branch (``strcmp(name, "sanidine")``,
lines 2780-2823).

Why this needed its own module
-------------------------------
Like quartz/tridymite (`quartz_tridymite.py`), sanidine is one of the
"special terms for minerals that involve order-disorder functions" gibbs.c
layers ON TOP of the generic Berman-EOS pure-endmember result (gibbs.c
line ~2541's comment) -- but unlike quartz/tridymite, sanidine's own
generic Berman EOS integral (`compute.py`'s existing H0/S0/Cp0/V path) is
NOT replaced, just added to. This is an Einstein-polynomial-parameterized
Al-Si disordering enthalpy/entropy/volume correction (`dhdis`/`dsdis`/
`dvdis`, from a `d0..d5` coefficient set specific to sanidine), active
whenever T > Tr (298.15 K -- i.e. essentially always for a magmatic
calculation) and saturating at a disordering temperature `td` = min(1436 K,
T): full disordering is complete by 1436 K in this model, so above that the
correction becomes a T-independent additive constant plus a P-scaled
`dvdis` term (with additional heat-capacity terms only active in the
298.15-1436 K window, since Cp itself flattens to a P,T-independent
constant contribution above 1436 K).

Root cause identified for: the k-feldspar (sanidine-dominated) entropy/Cp
gap previously flagged (~11-13 J/(K.mol) absolute S offset, ~2% mean
relative, scaling with sanidine mole fraction) in the feldspar
solid-solution mixing benchmark -- that gap was hypothesized to come from
`feldspar.c`'s own excess-entropy Margules terms (verbatim-checked this
session against `sources/feldspar.c` and confirmed to be a faithful,
deliberate asymmetry versus its H-macro, not a bug: `wsanor`/`wsoran`/
`wsabanor` genuinely do not exist as calibrated parameters in the real
source -- see `compute.py`'s docstring). This module's `dsdis` term, missing
from every pure-sanidine-endmember evaluation feeding into that mixing
model, quantitatively matches: `x_or * dsdis` at typical magmatic T and a
sanidine mole fraction of 0.83-0.95 (the range the earlier benchmark's
near-pure k-feldspar rows fell in) lands squarely in the observed
11-13 J/(K.mol) gap.

Albite (added 2026-09-23): gibbs.c's albite branch (lines 2543-2561, taken
because HIGH_STRUCTURAL_STATE_FELDSPAR is not defined) calls `albite()`
(`sources/albite.c`), Salje's (1985) two-parameter Landau model of
displacive (q = s[0]) and Al-Si (q_od = s[1]) ordering relative to
monalbite. `albite_disorder_correction` translates it verbatim: the same
undamped 2-D Newton solve from s = (0.6, 0.9), the same convergence test and
200-iteration cap, the "skip" when both parameters are below
sqrt(DBL_EPSILON), and the source's own H/S/V/Cp/dCp/dT and V-derivative
assembly (H = G - T dG/dT with the source's DGDT, which carries an extra
P-dependent term beyond -S). This was the remaining feldspar Cp gap (~1-11%
for albite-rich compositions).

Units: P in bars, T in K, V in J/bar -- see constants.py.
"""
from __future__ import annotations
import numpy as np

from .constants import Tr, Pr

# Sanidine Al-Si disordering coefficients, gibbs.c lines 2789-2794.
_D0 = 282.98
_D1 = -4.83e3
_D2 = 36.21e5
_D3 = -15.733e-2
_D4 = 34.770e-6
_D5 = 41.063e4
_TD_MAX = 1436.0   # full-disorder saturation temperature (K)


def sanidine_disorder_correction(T, P, high_structural_state: bool = False):
    """Additive (dG, dH, dS, dV, dCp, ddCpdT, ddVdT, dd2VdT2) corrections
    for pure sanidine, to be added on top of the generic Berman-EOS result
    (`compute.py`'s H0/S0/Cp0/V/dVdT/d2VdT2 before this correction). dVdP,
    d2VdTdP, d2VdP2 are untouched -- gibbs.c's own branch never adjusts
    them for this correction.

    high_structural_state=True fixes td=1436.0 regardless of T (the
    ``HIGH_STRUCTURAL_STATE_FELDSPAR`` compile option); the default,
    False, matches this package's standard-MELTS target calibration
    (td = min(1436, T)).
    """
    T = np.asarray(T, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    shape = np.broadcast_shapes(T.shape, P.shape)
    T = np.broadcast_to(T, shape).astype(np.float64)
    P = np.broadcast_to(P, shape).astype(np.float64)

    td = np.full(shape, _TD_MAX) if high_structural_state else np.minimum(_TD_MAX, T)

    dhdis = (_D0*(td - Tr) + 2.0*_D1*(np.sqrt(td) - np.sqrt(Tr)) - _D2*(1.0/td - 1.0/Tr)
             + _D3*(td**2 - Tr**2)/2.0 + _D4*(td**3 - Tr**3)/3.0)
    dsdis = (_D0*(np.log(td) - np.log(Tr)) - 2.0*_D1*(1.0/np.sqrt(td) - 1.0/np.sqrt(Tr))
             - _D2*(1.0/td**2 - 1.0/Tr**2)/2.0 + _D3*(td - Tr) + _D4*(td**2 - Tr**2)/2.0)
    dvdis = dhdis/_D5 if _D5 != 0.0 else np.zeros(shape)

    above_tr = T > Tr
    in_window = above_tr & (T < _TD_MAX)

    dG = np.where(above_tr, dhdis - T*dsdis + dvdis*(P - Pr), 0.0)
    dH = np.where(above_tr, dhdis + dvdis*(P - Pr), 0.0)
    dS = np.where(above_tr, dsdis, 0.0)
    dV = np.where(above_tr, dvdis, 0.0)

    cp_poly = _D0 + _D1/np.sqrt(T) + _D2/T**2 + _D3*T + _D4*T**2
    dcpdt_poly = -_D1*0.5/T**1.5 - 2.0*_D2/T**3 + _D3 + 2.0*_D4*T

    dCp = np.where(in_window, cp_poly, 0.0)
    ddCpdT = np.where(in_window, dcpdt_poly, 0.0)
    if _D5 != 0.0:
        ddVdT = np.where(in_window, cp_poly/_D5, 0.0)
        dd2VdT2 = np.where(in_window, dcpdt_poly/_D5, 0.0)
        dH = dH + np.where(in_window, -(P - Pr)*T*cp_poly/_D5, 0.0)
        dS = dS + np.where(in_window, -(P - Pr)*cp_poly/_D5, 0.0)
        dCp = dCp + np.where(in_window, -(P - Pr)*T*dcpdt_poly/_D5, 0.0)
        d3cpdt3_poly = 1.5*0.5*_D1/T**2.5 + 6.0*_D2/T**4 + 2.0*_D4
        ddCpdT = ddCpdT + np.where(
            in_window,
            -(P - Pr)*dcpdt_poly/_D5 - (P - Pr)*T*d3cpdt3_poly/_D5,
            0.0,
        )
    else:
        ddVdT = np.zeros(shape)
        dd2VdT2 = np.zeros(shape)

    return dict(dG=dG, dH=dH, dS=dS, dV=dV, dCp=dCp, ddCpdT=ddCpdT,
                ddVdT=ddVdT, dd2VdT2=dd2VdT2)


# Salje (1985) albite ordering constants, sources/albite.c.
_A0, _B, _AOD0, _BOD, _COD = 5.479, 6854.0, 41.620, -9301.0, 43600.0
_AD0, _AD1, _AD2, _AD3 = -2.171, -3.043, -0.001569, 0.000002109
_TC, _TOD, _VSCALE = 1251.0, 824.1, 335282.925
_ALBITE_MAX_ITER = 200


def _albite_terms(s0, s1, t, p):
    """albite.c macros at (s, t, p); every quantity the albite() assembly uses."""
    f = 1.0 + (p - 1.0)/_VSCALE
    dd = _AD0 - _AD2*t*t - 2.0*_AD3*t**3          # (d0 - d2 t^2 - 2 d3 t^3)
    ee = _AD1 + 2.0*_AD2*t + 3.0*_AD3*t*t         # (d1 + 2 d2 t + 3 d3 t^2)
    S = -(0.5*_A0*s0*s0 + 0.5*_AOD0*s1*s1 + ee*s0*s1)
    H = (-0.5*_A0*_TC*s0*s0 + 0.25*_B*s0**4 - 0.5*_AOD0*_TOD*s1*s1 + 0.25*_BOD*s1**4
         + (_COD/6.0)*s1**6 + dd*s0*s1)
    V = H/_VSCALE
    G = H - t*S + (p - 1.0)*V
    g0 = (-_A0*_TC*s0 + _B*s0**3 + dd*s1)*f + t*(_A0*s0 + ee*s1)
    g1 = (-_AOD0*_TOD*s1 + _BOD*s1**3 + _COD*s1**5 + dd*s0)*f + t*(_AOD0*s1 + ee*s0)
    dgdt = -S + (-2.0*_AD2*t - 6.0*_AD3*t*t)*s0*s1*(p - 1.0)/_VSCALE
    g00 = (-_A0*_TC + 3.0*_B*s0*s0)*f + t*_A0
    g01 = dd*f + t*ee
    g11 = (-_AOD0*_TOD + 3.0*_BOD*s1*s1 + 5.0*_COD*s1**4)*f + t*_AOD0
    g0t = -2.0*(_AD2*t + 3.0*_AD3*t*t)*s1*(p - 1.0)/_VSCALE + _A0*s0 + ee*s1
    g1t = -2.0*(_AD2*t + 3.0*_AD3*t*t)*s0*(p - 1.0)/_VSCALE + _AOD0*s1 + ee*s0
    g0p = (-_A0*_TC*s0 + _B*s0**3 + dd*s1)/_VSCALE
    g1p = (-_AOD0*_TOD*s1 + _BOD*s1**3 + _COD*s1**5 + dd*s0)/_VSCALE
    gtt = -2.0*(_AD2 + 6.0*_AD3*t)*s0*s1*(p - 1.0)/_VSCALE + (2.0*_AD2 + 6.0*_AD3*t)*s0*s1
    gtp = -2.0*(_AD2*t + 3.0*_AD3*t*t)*s0*s1/_VSCALE
    g000 = 6.0*_B*s0*f
    g111 = (6.0*_BOD*s1 + 20.0*_COD*s1**3)*f
    g00t, g11t = np.full_like(t, _A0), np.full_like(t, _AOD0)
    g01t = (-2.0*_AD2*t - 6.0*_AD3*t*t)*f + _AD1 + 2.0*_AD2*t + 3.0*_AD3*t*t + t*(2.0*_AD2 + 6.0*_AD3*t)
    g00p = (-_A0*_TC + 3.0*_B*s0*s0)/_VSCALE
    g01p = dd/_VSCALE
    g11p = (-_AOD0*_TOD + 3.0*_BOD*s1*s1 + 5.0*_COD*s1**4)/_VSCALE
    g0tt = -2.0*(_AD2 + 6.0*_AD3*t)*s1*(p - 1.0)/_VSCALE + (2.0*_AD2 + 6.0*_AD3*t)*s1
    g1tt = -2.0*(_AD2 + 6.0*_AD3*t)*s0*(p - 1.0)/_VSCALE + (2.0*_AD2 + 6.0*_AD3*t)*s0
    g0tp = -2.0*(_AD2*t + 3.0*_AD3*t*t)*s1/_VSCALE
    g1tp = -2.0*(_AD2*t + 3.0*_AD3*t*t)*s0/_VSCALE
    gttt = -12.0*_AD3*s0*s1*(p - 1.0)/_VSCALE + 6.0*_AD3*s0*s1
    gttp = -2.0*(_AD2 + 6.0*_AD3*t)*s0*s1/_VSCALE
    z = np.zeros_like(t)
    return dict(
        G=G, H=H, S=S, V=V, dgdt=dgdt, gs=np.stack([g0, g1], -1),
        gss=np.stack([np.stack([g00, g01], -1), np.stack([g01, g11], -1)], -2),
        gst=np.stack([g0t, g1t], -1), gsp=np.stack([g0p, g1p], -1),
        gtt=gtt, gtp=gtp, gpp=z, gttt=gttt, gttp=gttp, gtpp=z, gppp=z,
        gsss=_sym3(g000, z, z, g111),
        gsst=np.stack([np.stack([g00t, g01t], -1), np.stack([g01t, g11t], -1)], -2),
        gssp=np.stack([np.stack([g00p, g01p], -1), np.stack([g01p, g11p], -1)], -2),
        gstt=np.stack([g0tt, g1tt], -1), gstp=np.stack([g0tp, g1tp], -1), gspp=np.stack([z, z], -1),
    )


def _sym3(a000, a001, a011, a111):
    """Symmetric (B, 2, 2, 2) third-derivative tensor from its 4 distinct entries."""
    t = np.empty(a000.shape + (2, 2, 2))
    t[..., 0, 0, 0] = a000
    t[..., 0, 0, 1] = t[..., 0, 1, 0] = t[..., 1, 0, 0] = a001
    t[..., 0, 1, 1] = t[..., 1, 0, 1] = t[..., 1, 1, 0] = a011
    t[..., 1, 1, 1] = a111
    return t


def ordering_response(d, s_t, s_p):
    """Second-order response of an equilibrium ordering state s(T, P) (dG/ds = 0).
    d: dict of G derivatives (gss (B,n,n), gsss (B,n,n,n), gsst, gssp (B,n,n),
    gst, gsp, gstt, gstp, gspp (B,n)); s_t, s_p: ds/dT, ds/dP (B,n).
    Returns d2s/dT2, d2s/dTdP, d2s/dP2 -- the source's order() recipe."""
    def rhs(a_st, gss_a, gss_b, s_a, s_b):
        return (a_st + np.einsum('bjk,bk->bj', gss_a, s_b) + np.einsum('bjk,bk->bj', gss_b, s_a)
                + np.einsum('bjkl,bk,bl->bj', d['gsss'], s_a, s_b))
    solve = lambda v: -np.linalg.solve(d['gss'], v[..., None])[..., 0]
    s_tt = solve(rhs(d['gstt'], d['gsst'], d['gsst'], s_t, s_t))
    s_tp = solve(rhs(d['gstp'], d['gsst'], d['gssp'], s_p, s_t))
    s_pp = solve(rhs(d['gspp'], d['gssp'], d['gssp'], s_p, s_p))
    return s_tt, s_tp, s_pp


def equilibrium_properties(d, s_t, s_p, s_tt, s_tp, s_pp, T):
    """Cp, dCp/dT, dV/dT, dV/dP, d2V/dT2, d2V/dTdP, d2V/dP2 of G(s(T, P), T, P)
    along the ordering equilibrium -- the assembly used verbatim by albite.c's
    albite() and by every MELTS cpmix/vmix with ordering parameters."""
    e = lambda a, b: np.einsum('bi,bi->b', a, b)
    q = lambda M, a, b: np.einsum('bij,bi,bj->b', M, a, b)
    c = lambda M, a, b, cc: np.einsum('bijk,bi,bj,bk->b', M, a, b, cc)
    gtt_tot = d['gtt'] + 2.0*e(d['gst'], s_t) + q(d['gss'], s_t, s_t)
    Cp = -T*gtt_tot
    gttt_tot = (d['gttt'] + 3.0*e(d['gstt'], s_t) + 3.0*e(d['gst'], s_tt) + 3.0*q(d['gss'], s_t, s_tt)
                + 3.0*q(d['gsst'], s_t, s_t) + c(d['gsss'], s_t, s_t, s_t))
    dCpdT = -T*gttt_tot - gtt_tot
    dVdT = d['gtp'] + e(d['gst'], s_p) + e(d['gsp'], s_t) + q(d['gss'], s_t, s_p)
    dVdP = d['gpp'] + 2.0*e(d['gsp'], s_p) + q(d['gss'], s_p, s_p)
    d2VdT2 = (d['gttp'] + e(d['gstt'], s_p) + 2.0*e(d['gst'], s_tp) + e(d['gsp'], s_tt) + 2.0*e(d['gstp'], s_t)
              + 2.0*q(d['gsst'], s_t, s_p) + q(d['gss'], s_tt, s_p) + 2.0*q(d['gss'], s_t, s_tp)
              + q(d['gssp'], s_t, s_t) + c(d['gsss'], s_t, s_t, s_p))
    d2VdTdP = (d['gtpp'] + 2.0*e(d['gstp'], s_p) + e(d['gst'], s_pp) + 2.0*e(d['gsp'], s_tp) + e(d['gspp'], s_t)
               + 2.0*q(d['gssp'], s_t, s_p) + q(d['gss'], s_t, s_pp) + 2.0*q(d['gss'], s_tp, s_p)
               + q(d['gsst'], s_p, s_p) + c(d['gsss'], s_t, s_p, s_p))
    d2VdP2 = (d['gppp'] + 3.0*e(d['gspp'], s_p) + 3.0*e(d['gsp'], s_pp) + 3.0*q(d['gss'], s_p, s_pp)
              + 3.0*q(d['gssp'], s_p, s_p) + c(d['gsss'], s_p, s_p, s_p))
    return dict(Cp=Cp, dCpdT=dCpdT, dVdT=dVdT, dVdP=dVdP, d2VdT2=d2VdT2, d2VdTdP=d2VdTdP, d2VdP2=d2VdP2)


def albite_disorder_correction(T, P):
    """Additive corrections for pure albite from sources/albite.c's albite()
    (see module docstring): dG, dH, dS, dV, dCp, ddCpdT, ddVdT, ddVdP,
    dd2VdT2, dd2VdTdP, dd2VdP2, plus the ordering state `s` (..., 2)."""
    T = np.asarray(T, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    shape = np.broadcast_shapes(T.shape, P.shape)
    t = np.broadcast_to(T, shape).astype(np.float64).ravel()
    p = np.broadcast_to(P, shape).astype(np.float64).ravel()
    n = t.size
    s = np.tile([0.6, 0.9], (n, 1))
    s_old = np.full((n, 2), 2.0)
    active = np.ones(n, dtype=bool)
    eps = np.finfo(float).eps
    for _ in range(_ALBITE_MAX_ITER):
        active &= (np.abs(s[:, 0] - s_old[:, 0]) > 10.0*eps) | (np.abs(s[:, 1] - s_old[:, 1]) > 10.0*eps)
        if not active.any():
            break
        d = _albite_terms(s[active, 0], s[active, 1], t[active], p[active])
        s_old[active] = s[active]
        s[active] = s[active] - np.linalg.solve(d['gss'], d['gs'][..., None])[..., 0]
    d = _albite_terms(s[:, 0], s[:, 1], t, p)
    s_t = -np.linalg.solve(d['gss'], d['gst'][..., None])[..., 0]
    s_p = -np.linalg.solve(d['gss'], d['gsp'][..., None])[..., 0]
    s_tt, s_tp, s_pp = ordering_response(d, s_t, s_p)
    eq = equilibrium_properties(d, s_t, s_p, s_tt, s_tp, s_pp, t)
    skip = (np.abs(s[:, 0]) < np.sqrt(eps)) & (np.abs(s[:, 1]) < np.sqrt(eps))
    keep = lambda a: np.where(skip, 0.0, a).reshape(shape)
    return dict(dG=keep(d['G']), dH=keep(d['G'] - t*d['dgdt']), dS=keep(-d['dgdt']), dV=keep(d['V']),
                dCp=keep(eq['Cp']), ddCpdT=keep(eq['dCpdT']), ddVdT=keep(eq['dVdT']), ddVdP=keep(eq['dVdP']),
                dd2VdT2=keep(eq['d2VdT2']), dd2VdTdP=keep(eq['d2VdTdP']), dd2VdP2=keep(eq['d2VdP2']),
                s=s.reshape(shape + (2,)))
