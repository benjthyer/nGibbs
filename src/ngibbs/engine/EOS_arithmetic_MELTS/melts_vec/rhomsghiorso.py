"""
Rhombohedral-oxide (ilmenite-hematite-geikielite-pyrophanite-corundum)
solid-solution mixing model -- vectorized translation of
`sources/rhomsghiorso.c`'s `gmixMsg`/`hmixMsg`/`smixMsg`/`vmixMsg`/`actMsg`/
`order`, verified against the file directly (see `tests/verify_rhomsghiorso.c`).

Reference: Ghiorso, M.S., Evans, B.W. (2008), "Thermodynamics of
rhombohedral oxide solid solutions and a revision of the
Fe-Ti two-oxide geothermometer and oxygen-barometer", American Journal of
Science 308: 957-1039.

Endmembers (NA=5, matching `testMsg`'s NAMES/FORMULAS arrays):
    0 geikielite   MgTiO3
    1 hematite     Fe2O3
    2 ilmenite     FeTiO3
    3 pyrophanite  MnTiO3
    4 corundum     Al2O3

Independent composition variables (NR=4): r[0]=X_ilmenite, r[1]=X_geikielite,
r[2]=X_pyrophanite, r[3]=X_corundum (X_hematite = 1-r0-r1-r2-r3, the
implicit/dependent one -- confirmed via `conMsg`'s own m[]->r[] conversion,
rhomsghiorso.c lines 2877-2880). NOTE this NAMES-array order (geikielite,
hematite, ilmenite, pyrophanite, corundum) DIFFERS from the r0..r3 index
order (ilmenite, geikielite, pyrophanite, corundum) -- exactly analogous to
the same distinction in spinel.py, and tracked with equal care throughout
this module: every "ENDMEMBERS"-style weighted sum below uses r[1] to
weight the geikielite slot, (1-r0-r1-r2-r3) to weight hematite, r[0] for
ilmenite, r[2] for pyrophanite, r[3] for corundum (`ENDMEMBERS` macro,
rhomsghiorso.c line 1354).

Three internal ordering parameters (NS=3): s[0], s[1], s[2] -- each is an
INDEPENDENT 1-D Landau order-disorder parameter for the ilmenite (IL),
geikielite (GK), and pyrophanite (PY) components respectively (s0<->IL,
s1<->GK, s2<->PY). Hematite (HM) and corundum (CR) have NO internal
ordering variable: their pure-endmember H is identically 0.0 and their S
depends only on the short-range-order spline `fSRO(t,0)` (both HM_S and
CR_S use the exact same formula, `fSRO(t,0)*2*R*log(2)` -- confirmed by
direct macro comparison, rhomsghiorso.c lines 704-728).

fSRO short-range-order spline
------------------------------
`fSRO(t, order)` is a natural cubic spline over 8 fixed temperature knots
(873.15 K to 1573.15 K in 100 K steps), ported faithfully below including
its 2nd-derivative (`y2`) construction and piecewise evaluation + up to
3rd T-derivative (`order` selects which). In the CURRENT calibration all 9
SRO constants (`SROconst`, `SRO600`..`SRO1300`) are the SAME value
(0.0730205), which collapses the spline to a CONSTANT function:
`fSRO(t,0) = 0.0730205` for all t, and `fSRO(t,1) = fSRO(t,2) = fSRO(t,3)
= 0` for all t (a natural cubic spline through equal y-values has all
second derivatives, and hence the constructed piecewise polynomial's
non-zeroth derivatives, identically zero). The general spline machinery is
still implemented faithfully (not hardcoded to this degenerate case) in
case of a future recalibration with differing knot values.

CRITICAL subtlety -- reported H/S use G-T*dG/dT, NOT the raw H/S macros
-------------------------------------------------------------------------
Both at the pure-endmember level (`pureRhm`'s mask&FOURTH/FIFTH blocks,
rhomsghiorso.c lines 1020-1034) and at the bulk level (`hmixMsg`/`smixMsg`,
lines 3303-3348), the reported enthalpy/entropy are computed via the TRUE
analytic derivative dG/dT (`DGDT`/`DHM_GDT`/`DCR_GDT`), NOT the raw named
`H`/`S` macros:
    hmix = G - t*(DGDT)          (bulk, hmixMsg line 3312)
    smix = -(DGDT)                (bulk, smixMsg line 3340)
    pure hmix[hematite]  = HM_G - t*(DHM_GDT)   (NOT HM_H)
    pure smix[hematite]  = -(DHM_GDT)           (NOT HM_S)
    pure hmix[corundum]  = CR_G - t*(DCR_GDT)   (NOT CR_H)
    pure smix[corundum]  = -(DCR_GDT)           (NOT CR_S)
because DGDT/DHM_GDT/DCR_GDT include an extra `fSRO(t,1)*[...]` correction
term (arising from S's own T-dependence via fSRO(t,0)) that the raw named
S/H macros omit. Algebraically `G - t*DGDT = H + t*fSRO(t,1)*[...]`, which
is NOT identically equal to the raw H macro in general (only numerically
identical in the CURRENT calibration because fSRO(t,1)=0 always, per the
finding above). This module therefore implements `gibbs_total`/
`pure_endmember_ghsv`'s H/S outputs via the G-T*dG/dT / -dG/dT pattern,
not the raw H/S macros, to stay correct beyond the current calibration --
exactly the class of subtle formula-consistency issue this project has
caught before (orthopyroxene's H0/S0/V0 near-miss, clinopyroxene's missing
entropy term). For the three ordering endmembers (IL/GK/PY) this
recomputation is an EXACT algebraic identity with the raw H macro (their
DGDT = -(S) exactly, with no fSRO correction term, so G - t*DGDT = H
always) -- only HM/CR carry a genuine (currently zero) correction.
`G` itself (used for `gmix`) is always the plain `H_macro - t*S_macro`
combination -- there is no analogous subtlety there, since G's definition
IS that combination; the subtlety only affects H and S when reported
SEPARATELY as derived quantities.

Bulk Newton-solve clip bounds are RELATIVE to r[i], not absolute
-------------------------------------------------------------------
Unlike spinel's absolute `[-1,1]` clip, `order()`'s bulk Newton solve clips
each s[i] to `[-r[i]+DBL_EPSILON, r[i]-DBL_EPSILON]` (rhomsghiorso.c lines
2417-2418) -- the bound smoothly narrows to force s[i]->0 as r[i]->0,
rather than hitting a fixed clip wall. Empirically (see
`benchmark_rhomsghiorso_test.py`) this makes the plain (undamped) Newton
step used by the C source itself converge reliably across the tested
composition/T/P grid without needing spinel.py's additional line-search +
gating machinery; a numerically-estimated-Jacobian Newton solve (as in
`solution_model.py`) with this same relative clip is used below.

Site fractions (three tiers, rhomsghiorso.c lines 2338-2373)
-------------------------------------------------------------
- "ID" tier (`xfe2ID`, `xmg2ID`, `xmn2ID`, `xti4ID`, `xal3ID`, `xfe3ID`):
  depend on r ONLY (not s, not the ordering solve) -- the "disordered"/
  ideal-mixing-only site fractions used in the fSRO-weighted term of `S`
  and throughout `DGDR0-3`/`DGDT`.
- "a" tier (`xfe2a`, `xmg2a`, `xmn2a`, `xti4a`, `xal3a`, `xfe3a`) and "b"
  tier (mirror image, `r[i]-s[i]` instead of `r[i]+s[i]`): the two
  crystallographic sites' actual (ordered) site fractions, depending on
  both r and s. Both tiers are floored at DBL_EPSILON whenever the raw
  formula is <= 0 (rhomsghiorso.c lines 2375-2387), exactly ported below.

Pure-endmember reference (`pureOrder`/`pureRhm`, lines 739-1216)
-------------------------------------------------------------------
Unlike spinel.py (which needs a different (r,s) vertex per endmember),
`pureOrder` solves the SAME s0*,s1*,s2* triple regardless of which
endmember is being evaluated -- it takes only (t,p), not r, as input: three
DECOUPLED 1-D Newton solves (each endmember's Landau equation depends only
on its own s[i]), initial guess s[i]=0.98, iteration
`s[i] += -dgds[i]/d2gds2[i]`, clipped to [0.0, 1.0-10*DBL_EPSILON], max
1000 iterations, falling back to s=0 on non-convergence. `pureRhm` then
reads GK_G/H/S(s1*), HM_G/H/S (no s dependence), IL_G/H/S(s0*),
PY_G/H/S(s2*), CR_G/H/S (no s dependence) directly -- no per-vertex
dispatch logic is needed, unlike spinel.py's `_PURE_FUNCS` table.

V = DGDP directly, both at pure-endmember and bulk level -- there is no
separate `#define V` macro anywhere in the file (confirmed: `vmixMsg`'s and
`pureRhm`'s EIGHTH-mask blocks both assign the DGDP-style macro straight to
their vmix outputs, lines 3581/1089-1095).

pMELTS omits the corundum endmember -- confirmed, and handled "for free"
------------------------------------------------------------------------
`rhomsghiorso.c` itself is a single, generic NA=5/NR=4 implementation with
no `#ifdef`/compile-time branch for pMELTS anywhere in the file (checked
directly). But `MELTS_Parameters/sol_struct_data.json`'s two endmember
tables differ: `meltsSolids` (rhyolite-MELTS, this module's default target
per `params.DEFAULT_TABLE`) lists all 5 rhm-oxide rows (MgTiO3, Fe2O3,
FeTiO3, MnTiO3, Al2O3), while `pMeltsSolids` lists only 4 (MgTiO3, Fe2O3,
FeTiO3, MnTiO3) -- Al2O3/corundum is simply absent as a row. So the
"without corundum" pMELTS model is NOT a different formula, only a
different ACTIVE-component list at the calibration-table level: pMELTS
never allocates moles to the corundum endmember (r[3]=X_corundum is always
identically 0). This general NA=5/NR=4 implementation already reduces
correctly to that 4-endmember case at r[3]=0 -- every corundum-bearing term
in `H`/`DGDR0-3`/`DGDP` above vanishes cleanly there (e.g. `term4 =
whmcrn*r3*Xsum`, `term8 = (...)*r0*r3`, etc., and corundum itself carries
no internal ordering variable to begin with, so nothing else depends on
it being "active"). No separate pMELTS code path is needed here; a future
pMeltsSolids-table wiring in `solid_solutions.py` only needs to fix r[3]=0
and use just the first 4 of this module's 5-endmember outputs.
"""
from __future__ import annotations
import numpy as np

from .constants import Rgas as R

ENDMEMBERS = ["geikielite", "hematite", "ilmenite", "pyrophanite", "corundum"]

_DBL_EPS = 2.220446049250313e-16   # C's DBL_EPSILON

# =============================================================================
# Base parameters, verbatim from rhomsghiorso.c lines 84-161 (already in
# joules / joules-per-bar in the source -- NO kcal*4.184 conversion here,
# unlike spinel.c/clinopyroxene.c).
# =============================================================================
dvilm, dvgei, dvpyr = 0.010758, 0.010758, 0.010758
wvilm, wvgei, wvpyr = 0.035089, 0.035089, 0.035089
dwvhmilm, dwvhmgei, dwvhmpyr = 0.013701, 0.013701, 0.013701
wvhmilm2, wvhmgei2, wvhmpyr2 = 0.0, 0.0, 0.0
dwvcrnilm, dwvcrngei, dwvcrnpyr = 0.013701, 0.013701, 0.013701
wvhmilm, wvhmgei, wvhmpyr = -0.11764, -0.11764, -0.11764
wvilmgei, wvilmpyr, wvgeipyr = 0.0, 0.0, 0.0

dhilm, dhgei, dhpyr = 17477.0, 17477.0, 17477.0
whilm, whgei, whpyr = 3189.0, 3189.0, 3189.0
dwhhmilm, dwhhmgei, dwhhmpyr = -5626.63, -5626.63, -5626.63
whhmilm2, whhmgei2, whhmpyr2 = -833.14, -833.14, -833.14
dwhcrnilm, dwhcrngei, dwhcrnpyr = 0.0, 0.0, 0.0

whmcrn = 69000.0
whhmilm, whhmgei, whhmpyr = 22535.6, 22535.6, 22535.6
wcrnilm, wcrngei, wcrnpyr = 22535.6, 22535.6, 22535.6
whilmgei, whilmpyr, whgeipyr = 2600.0, 2200.0, 2600.0
whilmgeiT, whilmpyrT, whgeipyrT = 88099.9, 30244.0, 2600.0
whilmilmgei = whilmgeigei = whilmilmpyr = whilmpyrpyr = whgeigeipyr = whgeipyrpyr = 0.0

SROconst = 0.0730205
SRO600 = SRO700 = SRO800 = SRO900 = SRO1000 = SRO1100 = SRO1200 = SRO1300 = 0.0730205

# =============================================================================
# fSRO short-range-order natural cubic spline (rhomsghiorso.c lines 221-277),
# ported faithfully. Knots/second-derivatives computed once at import time
# (mirrors the source's `makeSpline` one-time cache -- there is no
# recalibration path in this Python translation).
# =============================================================================
_SPLINE_X = np.array([873.15, 973.15, 1073.15, 1173.15, 1273.15, 1373.15, 1473.15, 1573.15])
_SPLINE_Y = np.array([SRO600, SRO700, SRO800, SRO900, SRO1000, SRO1100, SRO1200, SRO1300])


def _build_spline_y2(x, y):
    n = len(x)
    y2 = np.zeros(n)
    u = np.zeros(n - 1)
    for i in range(1, n - 1):
        sig = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1])
        p = sig * y2[i - 1] + 2.0
        y2[i] = (sig - 1.0) / p
        u[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]) - (y[i] - y[i - 1]) / (x[i] - x[i - 1])
        u[i] = (6.0 * u[i] / (x[i + 1] - x[i - 1]) - sig * u[i - 1]) / p
    for i in range(n - 2, -1, -1):
        y2[i] = y2[i] * y2[i + 1] + u[i]
    return y2


_SPLINE_Y2 = _build_spline_y2(_SPLINE_X, _SPLINE_Y)


def fSRO(tk, order):
    """Vectorized natural-cubic-spline evaluation, rhomsghiorso.c lines
    254-276. `tk` may be an ndarray; `order` selects value(0)/1st(1)/
    2nd(2)/3rd(3) T-derivative; anything else returns 0.0 (matching the
    source's `else return 0.0` fallback)."""
    tk = np.asarray(tk, dtype=np.float64)
    x, y, y2 = _SPLINE_X, _SPLINE_Y, _SPLINE_Y2
    n = len(x)
    klo = np.clip(np.searchsorted(x, tk, side='right') - 1, 0, n - 2)
    khi = klo + 1
    xlo, xhi = x[klo], x[khi]
    ylo, yhi = y[klo], y[khi]
    y2lo, y2hi = y2[klo], y2[khi]
    h = xhi - xlo
    a = (xhi - tk) / h
    b = (tk - xlo) / h
    if order == 0:
        return a * ylo + b * yhi + ((a ** 3 - a) * y2lo + (b ** 3 - b) * y2hi) * (h * h) / 6.0
    elif order == 1:
        return (yhi - ylo) / h - ((3.0 * a * a - 1.0) * y2lo - (3.0 * b * b - 1.0) * y2hi) * h / 6.0
    elif order == 2:
        return a * y2lo + b * y2hi
    elif order == 3:
        return (y2hi - y2lo) / h
    else:
        return np.zeros_like(tk)


# =============================================================================
# Pure-endmember Landau/short-range-order macros (rhomsghiorso.c lines
# 623-728), verbatim. s0<->ilmenite, s1<->geikielite, s2<->pyrophanite;
# hematite/corundum have no internal ordering variable.
# =============================================================================
def _il_g_h_s(s0, t, p):
    H = (dhilm + (p - 1.0) * dvilm) * (1.0 - s0 * s0) + (whilm + (p - 1.0) * wvilm) * s0 * s0 * (1.0 - s0 * s0)
    S = -R * ((1.0 + s0) * np.log(1.0 + s0) + (1.0 - s0) * np.log(1.0 - s0) - 2.0 * np.log(2.0))
    G = H - t * S
    return G, H, S


def _il_dgds0(s0, t, p):
    return (R * t * (np.log(1.0 + s0) - np.log(1.0 - s0))
            - 2.0 * (dhilm + (p - 1.0) * dvilm) * s0
            + (whilm + (p - 1.0) * wvilm) * (2.0 * s0 - 4.0 * s0 ** 3))


def _il_dgdp(s0, p):
    return dvilm * (1.0 - s0 * s0) + wvilm * s0 * s0 * (1.0 - s0 * s0)


def _gk_g_h_s(s1, t, p):
    H = (dhgei + (p - 1.0) * dvgei) * (1.0 - s1 * s1) + (whgei + (p - 1.0) * wvgei) * s1 * s1 * (1.0 - s1 * s1)
    S = -R * ((1.0 + s1) * np.log(1.0 + s1) + (1.0 - s1) * np.log(1.0 - s1) - 2.0 * np.log(2.0))
    G = H - t * S
    return G, H, S


def _gk_dgds1(s1, t, p):
    return (R * t * (np.log(1.0 + s1) - np.log(1.0 - s1))
            - 2.0 * (dhgei + (p - 1.0) * dvgei) * s1
            + (whgei + (p - 1.0) * wvgei) * (2.0 * s1 - 4.0 * s1 ** 3))


def _gk_dgdp(s1, p):
    return dvgei * (1.0 - s1 * s1) + wvgei * s1 * s1 * (1.0 - s1 * s1)


def _py_g_h_s(s2, t, p):
    H = (dhpyr + (p - 1.0) * dvpyr) * (1.0 - s2 * s2) + (whpyr + (p - 1.0) * wvpyr) * s2 * s2 * (1.0 - s2 * s2)
    S = -R * ((1.0 + s2) * np.log(1.0 + s2) + (1.0 - s2) * np.log(1.0 - s2) - 2.0 * np.log(2.0))
    G = H - t * S
    return G, H, S


def _py_dgds2(s2, t, p):
    return (R * t * (np.log(1.0 + s2) - np.log(1.0 - s2))
            - 2.0 * (dhpyr + (p - 1.0) * dvpyr) * s2
            + (whpyr + (p - 1.0) * wvpyr) * (2.0 * s2 - 4.0 * s2 ** 3))


def _py_dgdp(s2, p):
    return dvpyr * (1.0 - s2 * s2) + wvpyr * s2 * s2 * (1.0 - s2 * s2)


def _hm_g_h_s_dgdt(t):
    """HM_G/HM_H/HM_S/DHM_GDT, rhomsghiorso.c lines 704-707: no internal
    ordering, S depends only on fSRO(t,0)."""
    S = fSRO(t, 0) * 2.0 * R * np.log(2.0)
    H = np.zeros_like(S)
    G = H - t * S
    DGDT = -fSRO(t, 0) * 2.0 * R * np.log(2.0) - fSRO(t, 1) * 2.0 * R * t * np.log(2.0)
    return G, H, S, DGDT


def _cr_g_h_s_dgdt(t):
    """CR_G/CR_H/CR_S/DCR_GDT, rhomsghiorso.c lines 717-720: identical
    formula to hematite's (confirmed by direct macro comparison)."""
    return _hm_g_h_s_dgdt(t)


# =============================================================================
# Pure-endmember ordering solve (`pureOrder`, rhomsghiorso.c lines 739-874):
# THREE decoupled 1-D Newton solves for s0*(IL), s1*(GK), s2*(PY), depending
# only on (t,p) -- NOT on r, unlike spinel.py's per-vertex approach.
# =============================================================================
def _newton_1d(dgds_func, s0_guess, T, P, n_iter=80, lo=0.0, hi=1.0 - 1e-9):
    T = np.asarray(T, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    s = np.full(T.shape, s0_guess, dtype=np.float64)
    eps = 1e-6
    for _ in range(n_iter):
        f0 = dgds_func(s, T, P)
        fp = dgds_func(np.clip(s + eps, lo, hi), T, P)
        fm = dgds_func(np.clip(s - eps, lo, hi), T, P)
        jac = (fp - fm) / (2.0 * eps)
        jac = np.where(jac == 0.0, 1.0, jac)
        s = np.clip(s - f0 / jac, lo, hi)
    return s


def pure_ordering_state(T, P, n_iter=80):
    """s0*, s1*, s2* -- the shared pure-endmember ordering triple used by
    ALL FIVE endmembers (ilmenite via s0, geikielite via s1, pyrophanite
    via s2; hematite/corundum ignore it)."""
    s0 = _newton_1d(lambda s, t, p: _il_dgds0(s, t, p), 0.98, T, P, n_iter=n_iter)
    s1 = _newton_1d(lambda s, t, p: _gk_dgds1(s, t, p), 0.98, T, P, n_iter=n_iter)
    s2 = _newton_1d(lambda s, t, p: _py_dgds2(s, t, p), 0.98, T, P, n_iter=n_iter)
    return s0, s1, s2


def pure_endmember_ghsv(T, P, n_iter=80):
    """G (raw, for gmix), H/S (G-T*dG/dT-corrected, for hmix/smix), V
    (=dG/dP, for vmix) for the 5 endmembers in ENDMEMBERS order -- see
    `pureRhm`'s FIRST/FOURTH/FIFTH/EIGHTH masks, rhomsghiorso.c lines
    1012-1095. Shape (..., 5)."""
    T = np.asarray(T, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    s0, s1, s2 = pure_ordering_state(T, P, n_iter=n_iter)

    gk_G, gk_H, gk_S = _gk_g_h_s(s1, T, P)
    hm_G, hm_H, hm_S, hm_DGDT = _hm_g_h_s_dgdt(T)
    il_G, il_H, il_S = _il_g_h_s(s0, T, P)
    py_G, py_H, py_S = _py_g_h_s(s2, T, P)
    cr_G, cr_H, cr_S, cr_DGDT = _cr_g_h_s_dgdt(T)

    # gmix reference: raw G macro (pureRhm mask&THIRD)
    G = np.stack([gk_G, hm_G, il_G, py_G, cr_G], axis=-1)
    # hmix reference: G+t*S for IL/GK/PY (exact identity, = raw H), G-t*DGDT
    # for HM/CR (pureRhm mask&FOURTH, lines 1020-1026)
    H = np.stack([gk_G + T * gk_S, hm_G - T * hm_DGDT, il_G + T * il_S,
                  py_G + T * py_S, cr_G - T * cr_DGDT], axis=-1)
    # smix reference: raw S macro for IL/GK/PY, -DGDT for HM/CR (pureRhm
    # mask&FIFTH, lines 1028-1034)
    S = np.stack([gk_S, -hm_DGDT, il_S, py_S, -cr_DGDT], axis=-1)
    # vmix reference: dG/dP directly (pureRhm mask&EIGHTH, lines 1089-1095)
    V = np.stack([_gk_dgdp(s1, P), np.zeros_like(T), _il_dgdp(s0, P),
                  _py_dgdp(s2, P), np.zeros_like(T)], axis=-1)
    return G, H, S, V


# =============================================================================
# Site fractions (rhomsghiorso.c lines 2338-2387).
# =============================================================================
def _floor_eps(x):
    return np.where(x > 0.0, x, _DBL_EPS)


def site_fractions_ID(r):
    """The r-only ("disordered") site fractions (lines 2338-2343)."""
    r = np.asarray(r, dtype=np.float64)
    r0, r1, r2, r3 = (r[..., i] for i in range(4))
    xfe2ID = _floor_eps(r0 / 2.0)
    xmg2ID = _floor_eps(r1 / 2.0)
    xmn2ID = _floor_eps(r2 / 2.0)
    xti4ID = _floor_eps((r0 + r1 + r2) / 2.0)
    xal3ID = _floor_eps(r3)
    xfe3ID = _floor_eps(1.0 - r0 - r1 - r2 - r3)
    return dict(xfe2ID=xfe2ID, xmg2ID=xmg2ID, xmn2ID=xmn2ID, xti4ID=xti4ID,
                xal3ID=xal3ID, xfe3ID=xfe3ID)


def site_fractions_ab(r, s):
    """The (r,s)-dependent "a"/"b" site tiers (lines 2361-2387)."""
    r = np.asarray(r, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    r0, r1, r2, r3 = (r[..., i] for i in range(4))
    s0, s1, s2 = s[..., 0], s[..., 1], s[..., 2]

    xfe2a = _floor_eps((r0 + s0) / 2.0)
    xmg2a = _floor_eps((r1 + s1) / 2.0)
    xmn2a = _floor_eps((r2 + s2) / 2.0)
    xti4a = _floor_eps((r0 - s0 + r1 - s1 + r2 - s2) / 2.0)
    xal3a = _floor_eps(r3)
    xfe3a = _floor_eps(1.0 - r0 - r1 - r2 - r3)

    xfe2b = _floor_eps((r0 - s0) / 2.0)
    xmg2b = _floor_eps((r1 - s1) / 2.0)
    xmn2b = _floor_eps((r2 - s2) / 2.0)
    xti4b = _floor_eps((r0 + s0 + r1 + s1 + r2 + s2) / 2.0)
    xal3b = _floor_eps(r3)
    xfe3b = _floor_eps(1.0 - r0 - r1 - r2 - r3)

    return dict(xfe2a=xfe2a, xmg2a=xmg2a, xmn2a=xmn2a, xti4a=xti4a, xal3a=xal3a, xfe3a=xfe3a,
                xfe2b=xfe2b, xmg2b=xmg2b, xmn2b=xmn2b, xti4b=xti4b, xal3b=xal3b, xfe3b=xfe3b)


# =============================================================================
# Bulk S/H/G (`S`/`H`/`G` macros, rhomsghiorso.c lines 1365-1395).
# =============================================================================
def _S_macro(t, x_id, x_ab):
    return (-R * (x_ab['xfe2a'] * np.log(x_ab['xfe2a']) + x_ab['xmg2a'] * np.log(x_ab['xmg2a'])
                  + x_ab['xmn2a'] * np.log(x_ab['xmn2a']) + x_ab['xti4a'] * np.log(x_ab['xti4a'])
                  + x_ab['xfe2b'] * np.log(x_ab['xfe2b']) + x_ab['xmg2b'] * np.log(x_ab['xmg2b'])
                  + x_ab['xmn2b'] * np.log(x_ab['xmn2b']) + x_ab['xti4b'] * np.log(x_ab['xti4b'])
                  - 2.0 * x_id['xfe2ID'] * np.log(x_id['xfe2ID']) - 2.0 * x_id['xmg2ID'] * np.log(x_id['xmg2ID'])
                  - 2.0 * x_id['xmn2ID'] * np.log(x_id['xmn2ID']) - 2.0 * x_id['xti4ID'] * np.log(x_id['xti4ID']))
            - (1.0 - fSRO(t, 0)) * 2.0 * R * (
                x_id['xfe3ID'] * np.log(x_id['xfe3ID']) + x_id['xal3ID'] * np.log(x_id['xal3ID'])
                + x_id['xfe2ID'] * np.log(x_id['xfe2ID']) + x_id['xmg2ID'] * np.log(x_id['xmg2ID'])
                + x_id['xmn2ID'] * np.log(x_id['xmn2ID']) + x_id['xti4ID'] * np.log(x_id['xti4ID']))
            + fSRO(t, 0) * 2.0 * R * np.log(2.0))


def _H_macro(r, s, p):
    r0, r1, r2, r3 = (r[..., i] for i in range(4))
    s0, s1, s2 = s[..., 0], s[..., 1], s[..., 2]
    Xsum = 1.0 - r0 - r1 - r2 - r3

    A_il = whhmilm + (p - 1.0) * wvhmilm
    B_il = dwhhmilm + (p - 1.0) * dwvhmilm
    A_gei = whhmgei + (p - 1.0) * wvhmgei
    B_gei = dwhhmgei + (p - 1.0) * dwvhmgei
    A_pyr = whhmpyr + (p - 1.0) * wvhmpyr
    B_pyr = dwhhmpyr + (p - 1.0) * dwvhmpyr

    term1 = (A_il + B_il * (1.0 - 2.0 * r0 - r1 - r2 - r3)) * r0 * Xsum
    term2 = (A_gei + B_gei * (1.0 - r0 - 2.0 * r1 - r2 - r3)) * r1 * Xsum
    term3 = (A_pyr + B_pyr * (1.0 - r0 - r1 - 2.0 * r2 - r3)) * r2 * Xsum
    term4 = whmcrn * r3 * Xsum
    term5 = (whilmgei + (p - 1.0) * wvilmgei + whilmgeiT) * r0 * r1 / 2.0 + (whilmgei - whilmgeiT) * s0 * s1 / 2.0
    term6 = (whilmpyr + (p - 1.0) * wvilmpyr + whilmpyrT) * r0 * r2 / 2.0 + (whilmpyr - whilmpyrT) * s0 * s2 / 2.0
    term7 = (whgeipyr + (p - 1.0) * wvgeipyr + whgeipyrT) * r1 * r2 / 2.0 + (whgeipyr - whgeipyrT) * s1 * s2 / 2.0
    term8 = (wcrnilm + (dwhcrnilm + (p - 1.0) * dwvcrnilm) * (r0 - r3)) * r0 * r3
    term9 = (wcrngei + (dwhcrngei + (p - 1.0) * dwvcrngei) * (r1 - r3)) * r1 * r3
    term10 = (wcrnpyr + (dwhcrnpyr + (p - 1.0) * dwvcrnpyr) * (r2 - r3)) * r2 * r3
    term11 = (dhilm + (p - 1.0) * dvilm + (B_il / 2.0 + (whhmilm2 + (p - 1.0) * wvhmilm2) / 4.0) * Xsum
              - (dwhcrnilm + (p - 1.0) * dwvcrnilm) * r3 / 2.0) * (r0 * r0 - s0 * s0)
    term12 = (dhgei + (p - 1.0) * dvgei + (B_gei / 2.0 + (whhmgei2 + (p - 1.0) * wvhmgei2) / 4.0) * Xsum
              - (dwhcrngei + (p - 1.0) * dwvcrngei) * r3 / 2.0) * (r1 * r1 - s1 * s1)
    term13 = (dhpyr + (p - 1.0) * dvpyr + (B_pyr / 2.0 + (whhmpyr2 + (p - 1.0) * wvhmpyr2) / 4.0) * Xsum
              - (dwhcrnpyr + (p - 1.0) * dwvcrnpyr) * r3 / 2.0) * (r2 * r2 - s2 * s2)
    term14 = (whilm + (p - 1.0) * wvilm) * s0 * s0 * (r0 * r0 - s0 * s0)
    term15 = (whgei + (p - 1.0) * wvgei) * s1 * s1 * (r1 * r1 - s1 * s1)
    term16 = (whpyr + (p - 1.0) * wvpyr) * s2 * s2 * (r2 * r2 - s2 * s2)
    term17 = whilmilmgei * (r0 * r0 - s0 * s0) * r1 / 4.0 + whilmgeigei * (r1 * r1 - s1 * s1) * r0 / 4.0
    term18 = whilmilmpyr * (r0 * r0 - s0 * s0) * r2 / 4.0 + whilmpyrpyr * (r2 * r2 - s2 * s2) * r0 / 4.0
    term19 = whgeigeipyr * (r1 * r1 - s1 * s1) * r2 / 4.0 + whgeipyrpyr * (r2 * r2 - s2 * s2) * r1 / 4.0

    return (term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9 + term10
            + term11 + term12 + term13 + term14 + term15 + term16 + term17 + term18 + term19)


def gibbs_total(r, s, T, P):
    """Bulk G (raw, for gmix), H/S (G-T*dG/dT-corrected, for hmix/smix), V
    (=DGDP, for vmix) -- rhomsghiorso.c's `G`/`H`/`S` macros plus the
    `hmixMsg`/`smixMsg`/`vmixMsg` reporting pattern (see module docstring)."""
    r = np.asarray(r, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    x_id = site_fractions_ID(r)
    x_ab = site_fractions_ab(r, s)

    S_macro = _S_macro(T, x_id, x_ab)
    H_macro = _H_macro(r, s, P)
    G = H_macro - T * S_macro

    DGDT_val = dgdt(r, s, T, x_id=x_id)
    H_reported = G - T * DGDT_val
    S_reported = -DGDT_val
    V = dgdp(r, s, P)
    return G, H_reported, S_reported, V


# =============================================================================
# DGDR0-3 / DGDS0-2 / DGDT / DGDP (rhomsghiorso.c lines 1397-1513).
# =============================================================================
def dgdr(r, s, T, P, x_id=None, x_ab=None):
    r = np.asarray(r, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    if x_id is None:
        x_id = site_fractions_ID(r)
    if x_ab is None:
        x_ab = site_fractions_ab(r, s)
    r0, r1, r2, r3 = (r[..., i] for i in range(4))
    s0, s1, s2 = s[..., 0], s[..., 1], s[..., 2]
    Xsum = 1.0 - r0 - r1 - r2 - r3
    xfe2ID, xmg2ID, xmn2ID, xti4ID = x_id['xfe2ID'], x_id['xmg2ID'], x_id['xmn2ID'], x_id['xti4ID']
    xal3ID, xfe3ID = x_id['xal3ID'], x_id['xfe3ID']
    xfe2a, xmg2a, xmn2a, xti4a = x_ab['xfe2a'], x_ab['xmg2a'], x_ab['xmn2a'], x_ab['xti4a']
    xfe2b, xmg2b, xmn2b, xti4b = x_ab['xfe2b'], x_ab['xmg2b'], x_ab['xmn2b'], x_ab['xti4b']

    A_il = whhmilm + (P - 1.0) * wvhmilm
    B_il = dwhhmilm + (P - 1.0) * dwvhmilm
    A_gei = whhmgei + (P - 1.0) * wvhmgei
    B_gei = dwhhmgei + (P - 1.0) * dwvhmgei
    A_pyr = whhmpyr + (P - 1.0) * wvhmpyr
    B_pyr = dwhhmpyr + (P - 1.0) * dwvhmpyr
    fs0 = fSRO(T, 0)

    d0 = (R * T * (0.5 * np.log(xfe2a) + 0.5 * np.log(xti4a) + 0.5 * np.log(xfe2b) + 0.5 * np.log(xti4b)
                   - np.log(xfe2ID) - np.log(xti4ID))
          + (1.0 - fs0) * 2.0 * R * T * (-np.log(xfe3ID) + 0.5 * np.log(xfe2ID) + 0.5 * np.log(xti4ID))
          + (A_il + B_il * (1.0 - 2.0 * r0 - r1 - r2 - r3)) * (1.0 - 2.0 * r0 - r1 - r2 - r3)
          - (A_gei + B_gei * (1.0 - r0 - 2.0 * r1 - r2 - r3)) * r1
          - (A_pyr + B_pyr * (1.0 - r0 - r1 - 2.0 * r2 - r3)) * r2
          - 2.0 * B_il * r0 * Xsum - B_gei * r1 * Xsum - B_pyr * r2 * Xsum
          - whmcrn * r3
          + (whilmgei + (P - 1.0) * wvilmgei + whilmgeiT) * r1 / 2.0
          + (whilmpyr + (P - 1.0) * wvilmpyr + whilmpyrT) * r2 / 2.0
          + (wcrnilm + (dwhcrnilm + (P - 1.0) * dwvcrnilm) * (r0 - r3)) * r3
          + (dwhcrnilm + (P - 1.0) * dwvcrnilm) * r0 * r3
          + 2.0 * (dhilm + (P - 1.0) * dvilm + (B_il / 2.0 + (whhmilm2 + (P - 1.0) * wvhmilm2) / 4.0) * Xsum
                   - (dwhcrnilm + (P - 1.0) * dwvcrnilm) * r3 / 2.0) * r0
          - (B_il / 2.0 + (whhmilm2 + (P - 1.0) * wvhmilm2) / 4.0) * (r0 * r0 - s0 * s0)
          - (B_gei / 2.0 + (whhmgei2 + (P - 1.0) * wvhmgei2) / 4.0) * (r1 * r1 - s1 * s1)
          - (B_pyr / 2.0 + (whhmpyr2 + (P - 1.0) * wvhmpyr2) / 4.0) * (r2 * r2 - s2 * s2)
          + 2.0 * (whilm + (P - 1.0) * wvilm) * s0 * s0 * r0
          + whilmilmgei * r0 * r1 / 2.0 + whilmgeigei * (r1 * r1 - s1 * s1) / 4.0
          + whilmilmpyr * r0 * r2 / 2.0 + whilmpyrpyr * (r2 * r2 - s2 * s2) / 4.0)

    d1 = (R * T * (0.5 * np.log(xmg2a) + 0.5 * np.log(xti4a) + 0.5 * np.log(xmg2b) + 0.5 * np.log(xti4b)
                   - np.log(xmg2ID) - np.log(xti4ID))
          + (1.0 - fs0) * 2.0 * R * T * (-np.log(xfe3ID) + 0.5 * np.log(xmg2ID) + 0.5 * np.log(xti4ID))
          - (A_il + B_il * (1.0 - 2.0 * r0 - r1 - r2 - r3)) * r0
          + (A_gei + B_gei * (1.0 - r0 - 2.0 * r1 - r2 - r3)) * (1.0 - r0 - 2.0 * r1 - r2 - r3)
          - (A_pyr + B_pyr * (1.0 - r0 - r1 - 2.0 * r2 - r3)) * r2
          - B_il * r0 * Xsum - 2.0 * B_gei * r1 * Xsum - B_pyr * r2 * Xsum
          - whmcrn * r3
          + (whilmgei + (P - 1.0) * wvilmgei + whilmgeiT) * r0 / 2.0
          + (whgeipyr + (P - 1.0) * wvgeipyr + whgeipyrT) * r2 / 2.0
          + (wcrngei + (dwhcrngei + (P - 1.0) * dwvcrngei) * (r1 - r3)) * r3
          + (dwhcrngei + (P - 1.0) * dwvcrngei) * r1 * r3
          - (B_il / 2.0 + (whhmilm2 + (P - 1.0) * wvhmilm2) / 4.0) * (r0 * r0 - s0 * s0)
          + 2.0 * (dhgei + (P - 1.0) * dvgei + (B_gei / 2.0 + (whhmgei2 + (P - 1.0) * wvhmgei2) / 4.0) * Xsum
                   - (dwhcrngei + (P - 1.0) * dwvcrngei) * r3 / 2.0) * r1
          - (B_gei / 2.0 + (whhmgei2 + (P - 1.0) * wvhmgei2) / 4.0) * (r1 * r1 - s1 * s1)
          - (B_pyr / 2.0 + (whhmpyr2 + (P - 1.0) * wvhmpyr2) / 4.0) * (r2 * r2 - s2 * s2)
          + 2.0 * (whgei + (P - 1.0) * wvgei) * s1 * s1 * r1
          + whilmilmgei * (r0 * r0 - s0 * s0) / 4.0 + whilmgeigei * r1 * r0 / 2.0
          + whgeigeipyr * r1 * r2 / 2.0 + whgeipyrpyr * (r2 * r2 - s2 * s2) / 4.0)

    d2 = (R * T * (0.5 * np.log(xmn2a) + 0.5 * np.log(xti4a) + 0.5 * np.log(xmn2b) + 0.5 * np.log(xti4b)
                   - np.log(xmn2ID) - np.log(xti4ID))
          + (1.0 - fs0) * 2.0 * R * T * (-np.log(xfe3ID) + 0.5 * np.log(xmn2ID) + 0.5 * np.log(xti4ID))
          - (A_il + B_il * (1.0 - 2.0 * r0 - r1 - r2 - r3)) * r0
          - (A_gei + B_gei * (1.0 - r0 - 2.0 * r1 - r2 - r3)) * r1
          + (A_pyr + B_pyr * (1.0 - r0 - r1 - 2.0 * r2 - r3)) * (1.0 - r0 - r1 - 2.0 * r2 - r3)
          - B_il * r0 * Xsum - B_gei * r1 * Xsum - 2.0 * B_pyr * r2 * Xsum
          - whmcrn * r3
          + (whilmpyr + (P - 1.0) * wvilmpyr + whilmpyrT) * r0 / 2.0
          + (whgeipyr + (P - 1.0) * wvgeipyr + whgeipyrT) * r1 / 2.0
          + (wcrnpyr + (dwhcrnpyr + (P - 1.0) * dwvcrnpyr) * (r2 - r3)) * r3
          + (dwhcrnpyr + (P - 1.0) * dwvcrnpyr) * r2 * r3
          - (B_il / 2.0 + (whhmilm2 + (P - 1.0) * wvhmilm2) / 4.0) * (r0 * r0 - s0 * s0)
          - (B_gei / 2.0 + (whhmgei2 + (P - 1.0) * wvhmgei2) / 4.0) * (r1 * r1 - s1 * s1)
          + 2.0 * (dhpyr + (P - 1.0) * dvpyr + (B_pyr / 2.0 + (whhmpyr2 + (P - 1.0) * wvhmpyr2) / 4.0) * Xsum
                   - (dwhcrnpyr + (P - 1.0) * dwvcrnpyr) * r3 / 2.0) * r2
          - (B_pyr / 2.0 + (whhmpyr2 + (P - 1.0) * wvhmpyr2) / 4.0) * (r2 * r2 - s2 * s2)
          + 2.0 * (whpyr + (P - 1.0) * wvpyr) * s2 * s2 * r2
          + whilmilmpyr * (r0 * r0 - s0 * s0) / 4.0 + whilmpyrpyr * r2 * r0 / 2.0
          + whgeigeipyr * (r1 * r1 - s1 * s1) / 4.0 + whgeipyrpyr * r2 * r1 / 2.0)

    d3 = ((1.0 - fs0) * 2.0 * R * T * (-np.log(xfe3ID) + np.log(xal3ID))
          - (A_il + B_il * (1.0 - 2.0 * r0 - r1 - r2 - r3)) * r0
          - (A_gei + B_gei * (1.0 - r0 - 2.0 * r1 - r2 - r3)) * r1
          - (A_pyr + B_pyr * (1.0 - r0 - r1 - 2.0 * r2 - r3)) * r2
          - B_il * r0 * Xsum - B_gei * r1 * Xsum - B_pyr * r2 * Xsum
          + whmcrn * (1.0 - r0 - r1 - r2 - 2.0 * r3)
          + (wcrnilm + (dwhcrnilm + (P - 1.0) * dwvcrnilm) * (r0 - r3)) * r0
          - (dwhcrnilm + (P - 1.0) * dwvcrnilm) * r0 * r3
          + (wcrngei + (dwhcrngei + (P - 1.0) * dwvcrngei) * (r1 - r3)) * r1
          - (dwhcrngei + (P - 1.0) * dwvcrngei) * r1 * r3
          + (wcrnpyr + (dwhcrnpyr + (P - 1.0) * dwvcrnpyr) * (r2 - r3)) * r2
          - (dwhcrnpyr + (P - 1.0) * dwvcrnpyr) * r2 * r3
          - (B_il / 2.0 + (whhmilm2 + (P - 1.0) * wvhmilm2) / 4.0
             + (dwhcrnilm + (P - 1.0) * dwvcrnilm) / 2.0) * (r0 * r0 - s0 * s0)
          - (B_gei / 2.0 + (whhmgei2 + (P - 1.0) * wvhmgei2) / 4.0
             + (dwhcrngei + (P - 1.0) * dwvcrngei) / 2.0) * (r1 * r1 - s1 * s1)
          - (B_pyr / 2.0 + (whhmpyr2 + (P - 1.0) * wvhmpyr2) / 4.0
             + (dwhcrnpyr + (P - 1.0) * dwvcrnpyr) / 2.0) * (r2 * r2 - s2 * s2))

    out = np.empty(r.shape)
    out[..., 0], out[..., 1], out[..., 2], out[..., 3] = d0, d1, d2, d3
    return out


def dgds(r, s, T, P, x_ab=None):
    r = np.asarray(r, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    if x_ab is None:
        x_ab = site_fractions_ab(r, s)
    r0, r1, r2, r3 = (r[..., i] for i in range(4))
    s0, s1, s2 = s[..., 0], s[..., 1], s[..., 2]
    Xsum = 1.0 - r0 - r1 - r2 - r3
    xfe2a, xti4a, xfe2b, xti4b = x_ab['xfe2a'], x_ab['xti4a'], x_ab['xfe2b'], x_ab['xti4b']
    xmg2a, xmg2b = x_ab['xmg2a'], x_ab['xmg2b']
    xmn2a, xmn2b = x_ab['xmn2a'], x_ab['xmn2b']

    B_il = dwhhmilm + (P - 1.0) * dwvhmilm
    B_gei = dwhhmgei + (P - 1.0) * dwvhmgei
    B_pyr = dwhhmpyr + (P - 1.0) * dwvhmpyr

    d0 = (0.5 * R * T * (np.log(xfe2a) - np.log(xti4a) - np.log(xfe2b) + np.log(xti4b))
          + (whilmgei - whilmgeiT) * s1 / 2.0 + (whilmpyr - whilmpyrT) * s2 / 2.0
          - 2.0 * (dhilm + (P - 1.0) * dvilm + (B_il / 2.0 + (whhmilm2 + (P - 1.0) * wvhmilm2) / 4.0) * Xsum
                   - (dwhcrnilm + (P - 1.0) * dwvcrnilm) * r3 / 2.0) * s0
          + 2.0 * (whilm + (P - 1.0) * wvilm) * s0 * (r0 * r0 - 2.0 * s0 * s0)
          - whilmilmgei * s0 * r1 / 2.0 - whilmilmpyr * s0 * r2 / 2.0)

    d1 = (0.5 * R * T * (np.log(xmg2a) - np.log(xti4a) - np.log(xmg2b) + np.log(xti4b))
          + (whilmgei - whilmgeiT) * s0 / 2.0 + (whgeipyr - whgeipyrT) * s2 / 2.0
          - 2.0 * (dhgei + (P - 1.0) * dvgei + (B_gei / 2.0 + (whhmgei2 + (P - 1.0) * wvhmgei2) / 4.0) * Xsum
                   - (dwhcrngei + (P - 1.0) * dwvcrngei) * r3 / 2.0) * s1
          + 2.0 * (whgei + (P - 1.0) * wvgei) * s1 * (r1 * r1 - 2.0 * s1 * s1)
          - whilmgeigei * s1 * r0 / 2.0 - whgeigeipyr * s1 * r2 / 2.0)

    d2 = (0.5 * R * T * (np.log(xmn2a) - np.log(xti4a) - np.log(xmn2b) + np.log(xti4b))
          + (whilmpyr - whilmpyrT) * s0 / 2.0 + (whgeipyr - whgeipyrT) * s1 / 2.0
          - 2.0 * (dhpyr + (P - 1.0) * dvpyr + (B_pyr / 2.0 + (whhmpyr2 + (P - 1.0) * wvhmpyr2) / 4.0) * Xsum
                   - (dwhcrnpyr + (P - 1.0) * dwvcrnpyr) * r3 / 2.0) * s2
          + 2.0 * (whpyr + (P - 1.0) * wvpyr) * s2 * (r2 * r2 - 2.0 * s2 * s2)
          - whilmpyrpyr * s2 * r0 / 2.0 - whgeipyrpyr * s2 * r1 / 2.0)

    out = np.empty(s.shape)
    out[..., 0], out[..., 1], out[..., 2] = d0, d1, d2
    return out


def dgdt(r, s, T, x_id=None):
    """DGDT macro, rhomsghiorso.c lines 1496-1497."""
    r = np.asarray(r, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    if x_id is None:
        x_id = site_fractions_ID(r)
    x_ab = site_fractions_ab(r, s)
    S_val = _S_macro(T, x_id, x_ab)
    xfe3ID, xal3ID = x_id['xfe3ID'], x_id['xal3ID']
    xfe2ID, xmg2ID, xmn2ID, xti4ID = x_id['xfe2ID'], x_id['xmg2ID'], x_id['xmn2ID'], x_id['xti4ID']
    fs1 = fSRO(T, 1)
    return (-S_val
            - fs1 * 2.0 * R * T * (xfe3ID * np.log(xfe3ID) + xal3ID * np.log(xal3ID) + xfe2ID * np.log(xfe2ID)
                                   + xmg2ID * np.log(xmg2ID) + xmn2ID * np.log(xmn2ID) + xti4ID * np.log(xti4ID))
            - fs1 * 2.0 * R * T * np.log(2.0))


def dgdp(r, s, P):
    """DGDP macro, rhomsghiorso.c lines 1499-1513."""
    r = np.asarray(r, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    r0, r1, r2, r3 = (r[..., i] for i in range(4))
    s0, s1, s2 = s[..., 0], s[..., 1], s[..., 2]
    Xsum = 1.0 - r0 - r1 - r2 - r3

    return ((wvhmilm + dwvhmilm * (1.0 - 2.0 * r0 - r1 - r2 - r3)) * r0 * Xsum
            + (wvhmgei + dwvhmgei * (1.0 - r0 - 2.0 * r1 - r2 - r3)) * r1 * Xsum
            + (wvhmpyr + dwvhmpyr * (1.0 - r0 - r1 - 2.0 * r2 - r3)) * r2 * Xsum
            + wvilmgei * r0 * r1 + wvilmpyr * r0 * r2 + wvgeipyr * r1 * r2
            + dwvcrnilm * (r0 - r3) * r0 * r3 + dwvcrngei * (r1 - r3) * r1 * r3
            + dwvcrnpyr * (r2 - r3) * r2 * r3
            + (dvilm + (dwvhmilm / 2.0 + wvhmilm2 / 4.0) * Xsum - dwvcrnilm * r3 / 2.0) * (r0 * r0 - s0 * s0)
            + (dvgei + (dwvhmgei / 2.0 + wvhmgei2 / 4.0) * Xsum - dwvcrngei * r3 / 2.0) * (r1 * r1 - s1 * s1)
            + (dvpyr + (dwvhmpyr / 2.0 + wvhmpyr2 / 4.0) * Xsum - dwvcrnpyr * r3 / 2.0) * (r2 * r2 - s2 * s2)
            + wvilm * s0 * s0 * (r0 * r0 - s0 * s0) + wvgei * s1 * s1 * (r1 * r1 - s1 * s1)
            + wvpyr * s2 * s2 * (r2 * r2 - s2 * s2))


# =============================================================================
# Endmember mole fractions <-> r (`ENDMEMBERS`/`conMsg` macros, lines 1354
# and 2877-2880).
# =============================================================================
def endmember_mole_fractions(r):
    r = np.asarray(r, dtype=np.float64)
    x_ilmenite = r[..., 0]
    x_geikielite = r[..., 1]
    x_pyrophanite = r[..., 2]
    x_corundum = r[..., 3]
    x_hematite = 1.0 - r[..., 0] - r[..., 1] - r[..., 2] - r[..., 3]
    return np.stack([x_geikielite, x_hematite, x_ilmenite, x_pyrophanite, x_corundum], axis=-1)


def x_to_r(X):
    """(B,5) mole fractions [geikielite, hematite, ilmenite, pyrophanite,
    corundum] -> (B,4) r = [X_ilmenite, X_geikielite, X_pyrophanite,
    X_corundum]."""
    X = np.asarray(X, dtype=np.float64)
    x_geikielite, x_hematite, x_ilmenite, x_pyrophanite, x_corundum = (X[..., i] for i in range(5))
    return np.stack([x_ilmenite, x_geikielite, x_pyrophanite, x_corundum], axis=-1)


# =============================================================================
# Bulk ordering solve (`order()`, rhomsghiorso.c lines 2312-2456): coupled
# 3-D Newton solve for s=(s0,s1,s2) at fixed (r,T,P). Initial guess
# s[i]=0.9*r[i] (line 2351); each step clipped RELATIVE to r[i] (lines
# 2417-2418), not to an absolute box -- see module docstring.
# =============================================================================
def _clip_rel(x, lo, hi):
    """Sequential MIN-then-MAX, exactly matching order()'s own two-step
    clip (rhomsghiorso.c lines 2417-2418: `s[i]=MIN(s[i],hi); s[i]=
    MAX(s[i],lo)`) rather than a symmetric np.clip. This matters when
    r[i] < DBL_EPSILON (a pure-endmember vertex where that composition
    axis is absent): there hi=r[i]-eps < lo=-r[i]+eps, and the SEQUENTIAL
    min-then-max always collapses the result to `lo` (=-r[i]+eps)
    regardless of the Newton step -- a plain np.clip(x, lo, hi) with
    lo>hi instead collapses to `hi` and, worse, makes any two clipped
    values equal (a 0/0 in the Jacobian's finite difference)."""
    return np.maximum(np.minimum(x, hi), lo)


def solve_ordering(r, T, P, n_iter=100, jac_eps=1e-6):
    r = np.asarray(r, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    s = 0.9 * r[..., :3]
    lo = -r[..., :3] + _DBL_EPS
    hi = r[..., :3] - _DBL_EPS
    B, NS = s.shape

    def _f(ss):
        return dgds(r, ss, T, P)

    for _ in range(n_iter):
        f0 = _f(s)
        J = np.empty((B, NS, NS), dtype=np.float64)
        for j in range(NS):
            s_plus = s.copy()
            s_plus[:, j] = _clip_rel(s[:, j] + jac_eps, lo[:, j], hi[:, j])
            s_minus = s.copy()
            s_minus[:, j] = _clip_rel(s[:, j] - jac_eps, lo[:, j], hi[:, j])
            denom = s_plus[:, j] - s_minus[:, j]
            denom = np.where(denom == 0.0, 1.0, denom)
            J[:, :, j] = (_f(s_plus) - _f(s_minus)) / denom[:, None]
        J = J + 1e-10 * np.eye(NS)[None, :, :]
        try:
            step = np.linalg.solve(J, f0[:, :, None])[:, :, 0]
        except np.linalg.LinAlgError:
            step = np.zeros_like(f0)
        s = _clip_rel(s - step, lo, hi)
    return s


# =============================================================================
# Darken activities (`actMsg`): standard (non-normalized) Darken formula,
# `mu_i = g - mu0_i + sum_j FR_ij dG/dr_j`, exactly as `solution_model.
# darken_activities` expects -- see FR0-3 macros, rhomsghiorso.c lines
# 1335-1338 (component order NAMES = [geikielite, hematite, ilmenite,
# pyrophanite, corundum]; r-column order = [Xil, Xgk, Xpy, Xcn]).
# =============================================================================
def _fr_matrix(r):
    """(B,5,4) Darken FR matrix, component order = ENDMEMBERS."""
    r = np.asarray(r, dtype=np.float64)
    B = r.shape[0]
    r0, r1, r2, r3 = (r[:, i] for i in range(4))
    fr = np.zeros((B, 5, 4), dtype=np.float64)
    # FR0(i): r0(=Xil) column -- component 2 (ilmenite) is 1-r0, else -r0
    fr[:, 2, 0] = 1.0 - r0
    for i in (0, 1, 3, 4):
        fr[:, i, 0] = -r0
    # FR1(i): r1(=Xgk) column -- component 0 (geikielite) is 1-r1, else -r1
    fr[:, 0, 1] = 1.0 - r1
    for i in (1, 2, 3, 4):
        fr[:, i, 1] = -r1
    # FR2(i): r2(=Xpy) column -- component 3 (pyrophanite) is 1-r2, else -r2
    fr[:, 3, 2] = 1.0 - r2
    for i in (0, 1, 2, 4):
        fr[:, i, 2] = -r2
    # FR3(i): r3(=Xcn) column -- component 4 (corundum) is 1-r3, else -r3
    fr[:, 4, 3] = 1.0 - r3
    for i in (0, 1, 2, 3):
        fr[:, i, 3] = -r3
    return fr


def solution_thermo(r, T, P, n_iter=100, dT=0.02, dP=0.02):
    """Full rhombohedral-oxide solid-solution mixing thermodynamics.

    Returns gmix, H_mix, S_mix, V_mix (analytic, at converged s*),
    Cp_mix/dVdT_mix/dVdP_mix (central-differenced, s* re-solved at each
    stencil point), mu, activities (5,) and s_eq (3,).
    """
    r = np.asarray(r, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)

    def _g_h_s_v(rr, TT, PP, n_it=n_iter):
        s = solve_ordering(rr, TT, PP, n_iter=n_it)
        G, H, S, V = gibbs_total(rr, s, TT, PP)
        Gp, Hp, Sp, Vp = pure_endmember_ghsv(TT, PP)
        x = endmember_mole_fractions(rr)
        gmix = G - np.sum(x * Gp, axis=-1)
        hmix = H - np.sum(x * Hp, axis=-1)
        smix = S - np.sum(x * Sp, axis=-1)
        vmix = V - np.sum(x * Vp, axis=-1)
        return gmix, hmix, smix, vmix, s, G

    gmix, hmix, smix, vmix, s_eq, G_total = _g_h_s_v(r, T, P)

    _, h_pT, _, v_pT, _, _ = _g_h_s_v(r, T + dT, P)
    _, h_mT, _, v_mT, _, _ = _g_h_s_v(r, T - dT, P)
    _, _, _, v_pP, _, _ = _g_h_s_v(r, T, P + dP)
    _, _, _, v_mP, _, _ = _g_h_s_v(r, T, P - dP)

    Cp_mix = (h_pT - h_mT) / (2.0 * dT)
    dVdT_mix = (v_pT - v_mT) / (2.0 * dT)
    dVdP_mix = (v_pP - v_mP) / (2.0 * dP)

    dgdr_val = dgdr(r, s_eq, T, P)
    fr = _fr_matrix(r)
    mu_bulk = G_total[:, None] + np.einsum('bij,bj->bi', fr, dgdr_val)
    Gp, _, _, _ = pure_endmember_ghsv(T, P)
    mu = mu_bulk - Gp
    a = np.exp(mu / (R * T[:, None]))

    dCpdT_mix = np.zeros_like(Cp_mix)

    return dict(gmix=gmix, H_mix=hmix, S_mix=smix, V_mix=vmix,
                Cp_mix=Cp_mix, dCpdT_mix=dCpdT_mix, dVdT_mix=dVdT_mix, dVdP_mix=dVdP_mix,
                mu=mu, activities=a, s_eq=s_eq, endmembers=ENDMEMBERS)
