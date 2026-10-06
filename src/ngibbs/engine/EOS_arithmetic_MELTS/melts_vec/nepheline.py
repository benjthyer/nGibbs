"""
Nepheline solid-solution mixing model -- vectorized translation of MAGMA
`sources/nepheline.c` (Sack & Ghiorso 1995): na-, k-, vc- and ca-nepheline
on large (ls) and small (ss) cation sites, with one K-Na ordering parameter
s between them (NR = 3, NS = 1).

Composition (conNph): r0 = x_k-nepheline, r1 = x_vc-nepheline,
r2 = x_ca-nepheline, x_na-nepheline = 1 - r0 - r1 - r2. Site fractions:
    xkls = r0 + 3s/4,  xvcls = r1 + r2,  xnals = 1 - r0 - r1 - r2 - 3s/4,
    xkss = r0 - s/4,   xcass = r2/3,     xnass = 1 - r0 - r2/3 + s/4,
clipped to [DBL_EPSILON, 1 - DBL_EPSILON] (xcass to 1/3 - DBL_EPSILON).

order() is translated verbatim: start at the random-distribution guess
s = -4 r0 (r1 + 2 r2/3)/(4 - r1 - 2 r2); skip the solve for "irrelevant"
compositions (x_K or x_Na below sqrt(DBL_EPSILON), or the vacancy fraction
within sqrt(DBL_EPSILON) of 1/4); otherwise Newton on dG/ds with step
halving until every site fraction is in range, the clip, s = xkls - xkss,
and stop when |s_new - s_old| <= 10 DBL_EPSILON (at most 200 iterations).

Every macro used by MELTS's bulk recipe (G, H, S, V, DGDS0, D2GDS0S0,
D2GDS0DT, D2GDS0DP, D3GDS0S0S0, D3GDS0S0DT, D3GDS0S0DP; all D*GDT*/DP* with
no s are zero) is verbatim, with the DWKNALS ... AWVK terms dropped because
every one of those constants is 0.0 in the source. G includes the penalty
sqrt(DBL_EPSILON)/r1, so a nepheline without vc-nepheline has infinite G
and H, exactly as in MELTS.

Bulk recipe (read_write.c, cpmixNph, vmixNph): hmix = G + T S, smix = S,
vmix = V at the equilibrium s; cpmix = -T (2 G_sT s_T + G_ss s_T^2) with
s_T = -G_sT/G_ss; dV/dT and dV/dP of mixing from the same chain rule
(melts_vec.feldspar_disorder.equilibrium_properties).
"""
from __future__ import annotations
import numpy as np

from .constants import Rgas
from .solution_model import darken_activities
from .feldspar_disorder import ordering_response, equilibrium_properties

ENDMEMBERS = ["na-nepheline", "k-nepheline", "vc-nepheline", "ca-nepheline"]

_HEX, _SEX, _VEX = -31744.63, -20.92, -1.05
_HX, _SX, _VX = -13893.18, 12.55, 0.0
_H23, _H24, _S23, _S24 = 73520.0, 42560.0, 0.0, 0.0
_WHNAKLS, _WVNAKLS, _WHNAKSS, _WVNAKSS = 6861.76, 0.33, 51002.96, 0.54
_WVN, _WVK, _WCANA, _WCAK, _WPLAG = 14644.0, 8368.0, 0.0, 0.0, 0.0
_S4 = -15.8765
_EPS = np.finfo(float).eps
_PENALTY = np.sqrt(_EPS)
_MAX_ITER = 200


def _site_fractions(r0, r1, r2, s, clip):
    """(6, B) [xkls, xvcls, xnals, xkss, xnass, xcass]; clip=True applies
    order()'s DBL_EPSILON bounds (xcass <= 1/3 - DBL_EPSILON)."""
    x = np.stack([r0 + 3.0*s/4.0, r1 + r2, 1.0 - r0 - r1 - r2 - 3.0*s/4.0,
                  r0 - s/4.0, 1.0 - r0 - r2/3.0 + s/4.0, r2/3.0])
    if clip:
        hi = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0/3.0])[:, None] - _EPS
        x = np.minimum(np.maximum(x, _EPS), hi)
    return x


def _terms(r0, r1, r2, s, x, t, p, full=True):
    """nepheline.c macros at (r, s, t, p) with clipped site fractions x.
    full=False returns only DGDS0 and D2GDS0S0 (the Newton step)."""
    xkls, xvcls, xnals, xkss, xnass, xcass = x
    R = Rgas
    GEX = _HEX - t*_SEX + (p - 1.0)*_VEX
    GX = _HX - t*_SX + (p - 1.0)*_VX
    WNAKLS = _WHNAKLS + (p - 1.0)*_WVNAKLS
    WNAKSS = _WHNAKSS + (p - 1.0)*_WVNAKSS
    G23, G24 = _H23 - t*_S23, _H24 - t*_S24
    c24 = (18.0*_WVN - 18.0*_WVK + 36.0*_WCAK - 36.0*_WCANA - 18.0*WNAKLS + 4.0*WNAKSS
           - 9.0*GEX - 3.0*GX + 6.0*G23 - 36.0*G24)
    gs = (R*t*(0.75*np.log(xkls) - 0.75*np.log(xnals) - 3.0*0.25*np.log(xkss) + 3.0*0.25*np.log(xnass))
          + 0.25*(2.0*GEX + GX + 3.0*WNAKLS - WNAKSS) + (1.0/8.0)*(3.0*GX - 9.0*WNAKLS - WNAKSS)*s
          - 0.5*(GX + 3.0*WNAKLS - WNAKSS)*r0
          + 0.125*(GX - GEX - 2.0*G23 - 6.0*WNAKLS - 6.0*_WVN + 6.0*_WVK)*r1 + c24*r2/24.0)
    gss = (R*t*(0.75*0.75/xkls + 0.75*0.75/xnals + 3.0*0.25*0.25/xkss + 3.0*0.25*0.25/xnass)
           + (1.0/8.0)*(3.0*GX - 9.0*WNAKLS - WNAKSS))
    if not full:
        return gs, gss
    S = (_S4*r2 - R*(xkls*np.log(xkls) + xvcls*np.log(xvcls) + xnals*np.log(xnals)
                     - (1.0 - 3.0*xcass)*np.log(1.0 - 3.0*xcass) + 3.0*xkss*np.log(xkss)
                     + 3.0*xcass*np.log(xcass) + 3.0*xnass*np.log(xnass))
         + 0.25*(2.0*_SEX + _SX)*s + _SX*r0*(1.0 - r0) + (3.0/16.0)*_SX*s*s - 0.5*_SX*r0*s
         + 0.5*(2.0*_S23 + _SEX - _SX)*r0*r1 + 0.125*(_SX - _SEX - 2.0*_S23)*r1*s
         + (6.0*_S23 + 3.0*_SEX - 15.0*_SX)*r0*r2/18.0
         + (-9.0*_SEX - 3.0*_SX + 6.0*_S23 - 36.0*_S24)*r2*s/24.0)
    H = (0.25*(2.0*_HEX + _HX + 3.0*_WHNAKLS - _WHNAKSS)*s + (_HX + _WHNAKLS + _WHNAKSS)*r0*(1.0 - r0)
         + (1.0/16.0)*(3.0*_HX - 9.0*_WHNAKLS - _WHNAKSS)*s*s - 0.5*(_HX + 3.0*_WHNAKLS - _WHNAKSS)*r0*s
         + _WVN*r1*(1.0 - r1)
         + 0.5*(2.0*_H23 + _HEX - _HX - 2.0*_WHNAKLS - 2.0*_WVN + 2.0*_WVK)*r0*r1
         + 0.125*(_HX - _HEX - 2.0*_H23 - 6.0*_WHNAKLS - 6.0*_WVN + 6.0*_WVK)*r1*s
         + _WCANA*r2*(1.0 - r2)
         + (6.0*_H23 + 3.0*_HEX - 15.0*_HX - 18.0*_WHNAKLS - 4.0*_WHNAKSS + 36.0*_WCAK - 36.0*_WCANA
            + 18.0*_WVN - 18.0*_WVK)*r0*r2/18.0
         + (_WPLAG - _WCANA - _WVN)*r1*r2
         + (18.0*_WVN - 18.0*_WVK + 36.0*_WCAK - 36.0*_WCANA - 18.0*_WHNAKLS + 4.0*_WHNAKSS - 9.0*_HEX
            - 3.0*_HX + 6.0*_H23 - 36.0*_H24)*r2*s/24.0)
    V = (0.25*(2.0*_VEX + _VX + 3.0*_WVNAKLS - _WVNAKSS)*s + (_VX + _WVNAKLS + _WVNAKSS)*r0*(1.0 - r0)
         + (1.0/16.0)*(3.0*_VX - 9.0*_WVNAKLS - _WVNAKSS)*s*s - 0.5*(_VX + 3.0*_WVNAKLS - _WVNAKSS)*r0*s
         + 0.5*(_VEX - _VX - 2.0*_WVNAKLS)*r0*r1 + 0.125*(_VX - _VEX - 6.0*_WVNAKLS)*r1*s
         + (3.0*_VEX - 15.0*_VX - 18.0*_WVNAKLS - 4.0*_WVNAKSS)*r0*r2/18.0
         + (-18.0*_WVNAKLS + 4.0*_WVNAKSS - 9.0*_VEX - 3.0*_VX)*r2*s/24.0)
    with np.errstate(divide='ignore'):
        G = H - t*S + (p - 1.0)*V + _PENALTY/r1
    gst = (R*(0.75*np.log(xkls) - 0.75*np.log(xnals) - 3.0*0.25*np.log(xkss) + 3.0*0.25*np.log(xnass))
           - 0.25*(2.0*_SEX + _SX) - (1.0/8.0)*(3.0*_SX)*s + 0.5*_SX*r0 - 0.125*(_SX - _SEX - 2.0*_S23)*r1
           + (9.0*_SEX + 3.0*_SX - 6.0*_S23 + 36.0*_S24)*r2/24.0)
    gsp = (0.25*(2.0*_VEX + _VX + 3.0*_WVNAKLS - _WVNAKSS) + (1.0/8.0)*(3.0*_VX - 9.0*_WVNAKLS - _WVNAKSS)*s
           - 0.5*(_VX + 3.0*_WVNAKLS - _WVNAKSS)*r0 + 0.125*(_VX - _VEX - 6.0*_WVNAKLS)*r1
           + (-18.0*_WVNAKLS + 4.0*_WVNAKSS - 9.0*_VEX - 3.0*_VX)*r2/24.0)
    gsss = -R*t*(0.75**3/(xkls*xkls) - 0.75**3/(xnals*xnals) - 3.0*0.25**3/(xkss*xkss) + 3.0*0.25**3/(xnass*xnass))
    gsst = (R*(0.75*0.75/xkls + 0.75*0.75/xnals + 3.0*0.25*0.25/xkss + 3.0*0.25*0.25/xnass)
            - (1.0/8.0)*(3.0*_SX))
    gssp = np.full_like(t, (1.0/8.0)*(3.0*_VX - 9.0*_WVNAKLS - _WVNAKSS))
    # DGDR0..2 (activities); constant-zero D*WK*/AW* terms omitted
    dgdr0 = (R*t*(np.log(xkls) - np.log(xnals) + 3.0*np.log(xkss) - 3.0*np.log(xnass))
             + (GX + WNAKLS + WNAKSS)*(1.0 - 2.0*r0) - 0.5*(GX + 3.0*WNAKLS - WNAKSS)*s
             + 0.5*(2.0*G23 + GEX - GX - 2.0*WNAKLS - 2.0*_WVN + 2.0*_WVK)*r1
             + (6.0*G23 + 3.0*GEX - 15.0*GX - 18.0*WNAKLS - 4.0*WNAKSS + 36.0*_WCAK - 36.0*_WCANA
                + 18.0*_WVN - 18.0*_WVK)*r2/18.0)
    with np.errstate(divide='ignore'):
        dgdr1 = (R*t*(np.log(xvcls) - np.log(xnals)) + _WVN*(1.0 - 2.0*r1)
                 + 0.5*(2.0*G23 + GEX - GX - 2.0*WNAKLS - 2.0*_WVN + 2.0*_WVK)*r0
                 + 0.125*(GX - GEX - 2.0*G23 - 6.0*WNAKLS - 6.0*_WVN + 6.0*_WVK)*s
                 + (_WPLAG - _WCANA - _WVN)*r2 - _PENALTY/(r1*r1))
    dgdr2 = (-t*_S4 + R*t*(np.log(xvcls) - np.log(xnals) + np.log(1.0 - 3.0*xcass) + np.log(xcass)
                           - np.log(xnass) + 1.0)
             + _WCANA*(1.0 - 2.0*r2)
             + (6.0*G23 + 3.0*GEX - 15.0*GX - 18.0*WNAKLS - 4.0*WNAKSS + 36.0*_WCAK - 36.0*_WCANA
                + 18.0*_WVN - 18.0*_WVK)*r0/18.0
             + (_WPLAG - _WCANA - _WVN)*r1 + c24*s/24.0)
    z = np.zeros_like(t)
    return dict(G=G, H=H, S=S, V=V, gs=gs, gss=gss, gst=gst, gsp=gsp, gsss=gsss, gsst=gsst, gssp=gssp,
                dgdr=np.stack([dgdr0, dgdr1, dgdr2], -1), zero=z)


def order(r0, r1, r2, t, p):
    """Equilibrium ordering parameter s, clipped site fractions and the
    "irrelevant composition" mask for which order() does not solve (order(),
    verbatim)."""
    s0 = -4.0*r0*(r1 + 2.0*r2/3.0)/(4.0 - r1 - 2.0*r2)
    x = _site_fractions(r0, r1, r2, s0, True)
    s = x[0] - x[3]
    xk, xna, xvc = r0, 1.0 - r0 - r1/4.0 - r2/2.0, (r1 + r2)/4.0
    skip = (xk < np.sqrt(_EPS)) | (xna < np.sqrt(_EPS)) | (xvc > 0.25 - np.sqrt(_EPS))
    s_old = np.where(skip, s, 2.0)
    active = ~skip
    for _ in range(_MAX_ITER):
        active &= np.abs(s - s_old) > 10.0*_EPS
        if not active.any():
            break
        a = active
        gs, gss = _terms(r0[a], r1[a], r2[a], s[a], x[:, a], t[a], p[a], full=False)
        s_old[a] = s[a]
        corr = -gs/gss
        lam = np.ones_like(corr)
        trial = s[a] + corr
        for _h in range(64):   # step halving: lambda > DBL_EPSILON allows at most 52 halvings
            xs = _site_fractions(r0[a], r1[a], r2[a], trial, False)
            bad = ((xs < 0.0) | (xs > 1.0)).any(0) | (xs[5] > 1.0/3.0)
            bad &= lam > _EPS
            if not bad.any():
                break
            lam = np.where(bad, lam/2.0, lam)
            trial = np.where(bad, s_old[a] + lam*corr, trial)
        xc = _site_fractions(r0[a], r1[a], r2[a], trial, True)
        x[:, a] = xc
        s[a] = xc[0] - xc[3]
    return s, x, skip


def solution_thermo(X, T, P):
    """Mixing properties per mole of nepheline for (B, 4) mole fractions X in
    ENDMEMBERS order: G_mix, H_mix, S_mix, V_mix, Cp_mix, dCpdT_mix, dVdT_mix,
    dVdP_mix, mu, activities, s_eq."""
    X = np.asarray(X, dtype=np.float64)
    t = np.asarray(T, dtype=np.float64)
    p = np.asarray(P, dtype=np.float64)
    r0, r1, r2 = X[:, 1], X[:, 2], X[:, 3]
    s, x, skip = order(r0, r1, r2, t, p)
    d = _terms(r0, r1, r2, s, x, t, p)
    z = d['zero']
    # one ordering parameter: shape the scalar derivatives as (B, 1[, 1[, 1]])
    dd = dict(gss=d['gss'][:, None, None], gst=d['gst'][:, None], gsp=d['gsp'][:, None],
              gsss=d['gsss'][:, None, None, None], gsst=d['gsst'][:, None, None], gssp=d['gssp'][:, None, None],
              gstt=z[:, None], gstp=z[:, None], gspp=z[:, None],
              gtt=z, gtp=z, gpp=z, gttt=z, gttp=z, gtpp=z, gppp=z)
    # Where order() skips the solve (no K, no Na, or a vacancy-saturated
    # composition) s is fixed at its start value, so ds/dT = ds/dP = 0. MAGMA
    # reuses whatever LU factorization a previous call left behind for these
    # rows, i.e. its Cp/dV/dT/dV/dP there are undefined; this uses the fixed-s
    # values instead.
    keep = ~skip[:, None]
    s_t = np.where(keep, -dd['gst']/d['gss'][:, None], 0.0)
    s_p = np.where(keep, -dd['gsp']/d['gss'][:, None], 0.0)
    s_tt, s_tp, s_pp = (np.where(keep, v, 0.0) for v in ordering_response(dd, s_t, s_p))
    eq = equilibrium_properties(dd, s_t, s_p, s_tt, s_tp, s_pp, t)
    fr = np.stack([np.stack([-r0, -r1, -r2], -1), np.stack([1.0 - r0, -r1, -r2], -1),
                   np.stack([-r0, 1.0 - r1, -r2], -1), np.stack([-r0, -r1, 1.0 - r2], -1)], -2)
    with np.errstate(invalid='ignore', over='ignore'):
        mu, a = darken_activities(d['G'], d['dgdr'], fr, Rgas, t)
    return dict(G_mix=d['G'], H_mix=d['G'] + t*d['S'], S_mix=d['S'], V_mix=d['V'],
                Cp_mix=eq['Cp'], dCpdT_mix=eq['dCpdT'], dVdT_mix=eq['dVdT'], dVdP_mix=eq['dVdP'],
                mu=mu, activities=a, s_eq=s)
