"""
Spinel solid-solution mixing model -- vectorized translation of
`sources/spinel.c` (order, pureOrder, gmixSpn, hmixSpn, smixSpn, vmixSpn,
cpmixSpn, actSpn), verified against the compiled MAGMA library (see
`tests/benchmark_spinel_test.py`).

Reference: Sack, R.O., Ghiorso, M.S. (1991), "An internally consistent
model for the thermodynamic properties of Fe-Mg-titanomagnetite-aluminate
spinels", Contributions to Mineralogy and Petrology 106: 474-505; and the
companion "Chromian spinels and petrogenetic indicators" paper.

Endmembers (NA=5, matching `testSpn`'s NAMES/FORMULAS arrays and `conSpn`'s
own `m[0..4]` order):
    0 chromite    FeCr2O4
    1 hercynite   FeAl2O4
    2 magnetite   Fe3O4
    3 spinel      MgAl2O4
    4 ulvospinel  Fe2TiO4

Independent composition variables (NR=4): r[0]=X_spinel, r[1]=X_chromite,
r[2]=X_ulvospinel, r[3]=X_magnetite (X_hercynite = 1-r0-r1-r2-r3, the
implicit/dependent one -- see `conSpn`'s `THIRD` block, spinel.c lines
2825-2829). Three internal ordering parameters (NS=3): s[0]=S1 (tet/oct
Mg-Al exchange), s[1]=S2 (tet/oct Al-Fe2+ exchange), s[2]=S4 (tet/oct
Fe2+-Fe3+ exchange). A FOURTH nominal ordering direction, "S3", exists in
the source's own Taylor-coefficient naming (`gs3`, `gx2s3`, `gs1s3`, ...)
but is NOT an independent variable -- the source's own comment says so
explicitly ("Independent variables are x2, x3, x4, x5, s1, s2, s4 (s3 =
X3)"), and every S3-tagged coefficient is used multiplied by `r[1]`
(=X3=X_chromite), never by a genuine fourth Newton unknown. This module
reproduces that identity by ALIASING the "S3" token to the very same
variable slot as "X3" in the generic polynomial engine below (see
`_TOKEN_INDEX`) -- confirmed, term-by-term, to reproduce the source's own
DGDR1 macro (which literally sums a "dH/dX3" block and a "dH/dS3" block,
i.e. the chain rule applied to X3 appearing as both an explicit variable
and, via the S3 alias, inside coefficients tagged S3) exactly, including
every doubled cross term (`gx3s3`, `gs3s3`) arising from X3 and S3 being
the same variable. See `_pure_vertex_note` below for how this same alias
also reproduces the SP_H pure-endmember macro's `(1.0+s[0])/2.0`
substitution as a special case, not a separate mechanism.

Correction to this project's own prior-session notes: `#define
CALIBRATE_SPINEL` is defined UNCONDITIONALLY near the top of spinel.c, so
the `#ifndef CALIBRATE_SPINEL` branch of the Taylor-coefficient block is
actually DEAD code and the `#else` branch is what's compiled -- the
reverse of what an earlier pass assumed. This turns out not to matter for
translation: both branches define the exact same coefficient FORMULAS (the
only difference is `static const double` vs. mutable `static double` plus
four `resetValueOfW45andW45p`/`resetValueOfH55`/`resetValueOfS55`/
`resetValueOfW55` calibration-only setter functions that are never called
during an ordinary forward property evaluation -- they exist only for an
external least-squares calibration driver this project does not use).

Structural relief compared to clinopyroxene.py/orthopyroxene.py: spinel.c
has NO PURE-vs-MIX two-reference-frame split. `pureOrder()`/`pureSpn()`
(the pure-endmember reference used by `gmixSpn()`) evaluate the exact same
global `g0..gs4s4` Taylor coefficients as the main bulk `order()`/`gmixSpn`
calculation -- confirmed by direct inspection of the `HC_G`/`SP_G`/`CR_G`/
`UV_G`/`MT_G` macros (spinel.c lines 725-828), which reference those same
module-level `gs1`/`gs2`/.../`gs4s4` names with no local shadow/override
declared anywhere (unlike `purePyx()`'s local `static const int clino =
TRUE;` in clinopyroxene.c/orthopyroxene.c). `gmixSpn()` is therefore
expected to be ~0 at spinel's own pure-endmember vertices, like
clinopyroxene's `gmixCpx()` and UNLIKE orthopyroxene's `gmixOpx()`; this is
verified numerically (not just architecturally) in
`benchmark_spinel_test.py`.

Also unlike clinopyroxene.py, there is no separate excess-entropy or
excess-volume Taylor-coefficient family here:
  - The ONLY entropy is (a) the ideal (configurational) mixing entropy
    over the 10 site fractions (the tetrahedral/octahedral 2-site spinel
    structure, `DECLARE_SITE_FRACTIONS`/`GET_SITE_FRACTIONS`/
    `SET_SITE_FRACTIONS`, the same macro pattern already used for olivine
    and both pyroxenes) and (b) one extra linear term `ss4*s[2]` (S55 is
    0 in this build, so `ss4` too is 0 for the compiled coefficient set,
    but is carried through symbolically rather than hardcoded to 0, in
    case that ever changes upstream).
  - The ONLY volume/pressure term is `r[2]*r[3]*((WV1)*r[3]+(WV2)*r[2])*
    (p-1.0)` -- just two W-parameters (WV1=-0.1250, WV2=0.1018), not a
    full V-Taylor-coefficient family. This also means V is IDENTICALLY
    ZERO at every one of the 5 pure-endmember vertices (since at most one
    of r[2],r[3] is nonzero there) -- confirmed directly: `DSP_GDP`,
    `DHC_GDP`, `DCR_GDP`, `DUV_GDP`, `DMT_GDP` are all `0.0` in the
    source.

Because of this simpler structure, only the ENTHALPY macro (`H`, spinel.c
lines 1580-1594) needs the generic-polynomial-dict treatment; entropy and
volume are each a handful of directly-transcribed closed-form terms.

Ordering solve and derived properties (translated verbatim, verified to
~1e-15 relative against the MAGMA library itself, incl. infeasible
emulator compositions -- see tests/benchmark_spinel_test.py):
  - `solve_ordering` is spinel.c order(): Newton on s = (s0, s1, s2) with
    the analytic D2GDS2, started at the random tet/oct distribution, each
    step shortened so no (unclipped) site fraction leaves [0, 1], stopped
    when every |ds| <= 10*DBL_EPSILON or after MAX_ITER = 200 steps, with
    the state at the start of the last step returned. Site fractions are
    clipped at DBL_EPSILON as in the source.
  - G, H, S, V and the Darken activities are read off at that state; Cp
    and dCp/dT come from the implicit ordering derivatives ds/dT =
    -D2GDS2^-1 D2GDSDT and d2s/dT2 (cpmixSpn FIRST|SECOND); V depends on
    r only, so dV/dT and dV/dP of mixing are 0.
  - hmix = G + T*S as in hmixSpn, so it carries the WV1/WV2 (p-1) term.
  - An earlier translation used a finite-difference Jacobian, a line
    search, composition "gates" and finite-difference Cp. That solver
    failed on emulator spinel with negative total Al (dG up to 6 kJ, Cp
    off by 1e3x) and froze the MgAl2O4 vertex (gmix 1.7 kJ instead of
    0); its hmix also lacked the (p-1) volume term (~9 J on BishopTuff
    spinel at ~1.5 kbar).

Pure-endmember reference (`pure_order`/`pure_endmember_props`, spinel.c
pureOrder()/pureSpn()): three decoupled 1-D Newton solves, verbatim
(start 0.5 / 0.9 / 0.1, clipped to (-1+eps | eps, 1-eps)):
  - spinel    (SP): free s0 at r=[1,0,0,0], with s1 = (1.0+s0)/2.0 built
    into the SP_* macros (the S2 token of the H polynomial written as
    (1+s0)/2), s2 = 0;
  - hercynite (HC): free s1 at r=[0,0,0,0], s0 = s2 = 0;
  - magnetite (MT): free s2 at r=[0,0,0,1], s0 = s1 = 0;
  - chromite (CR) and ulvospinel (UV): no ordering.
"""
from __future__ import annotations
import re
import warnings
import numpy as np

from .constants import Rgas

ENDMEMBERS = ["chromite", "hercynite", "magnetite", "spinel", "ulvospinel"]

_DBL_EPS = 2.220446049250313e-16   # C's DBL_EPSILON, matched exactly so
                                     # near-boundary site-fraction clipping
                                     # agrees bit-for-bit with spinel.c's
                                     # order()

# =============================================================================
# Base W/H parameters, verbatim from spinel.c lines 180-224 (kcal -> J via
# the same *1000.0*4.184 conversion used throughout MAGMA; WV1/WV2 are
# already in J/bar in the source).
# =============================================================================
H11 = -8.7 * 1000.0 * 4.184
W11 = 4.5 * 1000.0 * 4.184
W14 = 20.8 * 1000.0 * 4.184
W1P4 = 12.4 * 1000.0 * 4.184
W15 = 10.0 * 1000.0 * 4.184
W1P5 = 14.4 * 1000.0 * 4.184
W15P = 11.7 * 1000.0 * 4.184
W1P5P = 7.0 * 1000.0 * 4.184
W22 = 3.6 * 1000.0 * 4.184
H24 = 6.55 * 1000.0 * 4.184
W24U = 12.6 * 1000.0 * 4.184
W2P4U = 10.9 * 1000.0 * 4.184
H25 = 8.05 * 1000.0 * 4.184
W25PU = 15.3 * 1000.0 * 4.184
W2P5U = 14.4 * 1000.0 * 4.184
W45 = 6.0 * 1000.0 * 4.184
W45P = 2.0 * 1000.0 * 4.184
W4U5PU = 1.8 * 1000.0 * 4.184
H55 = 6.25 * 1000.0 * 4.184
S55 = 0.0
W55 = 0.0 * 1000.0 * 4.184
HEX = -3.6 * 1000.0 * 4.184
HX = 2.4 * 1000.0 * 4.184
WOCT = 2.0 * 1000.0 * 4.184
WTET = 2.0 * 1000.0 * 4.184
H33 = -20.0 * 1000.0 * 4.184
H23 = 0.0 * 1000.0 * 4.184
W33 = 7.0 * 1000.0 * 4.184
W13 = 10.0 * 1000.0 * 4.184
W1P3 = 15.2 * 1000.0 * 4.184
W13P = 11.3 * 1000.0 * 4.184
W1P3P = 5.9 * 1000.0 * 4.184
W2P3U = 10.0 * 1000.0 * 4.184
W23PU = 9.7 * 1000.0 * 4.184
W34 = 12.5 * 1000.0 * 4.184
W3P4 = 10.0 * 1000.0 * 4.184
W3PU4U = 10.4 * 1000.0 * 4.184
W35 = 0.0 * 1000.0 * 4.184
W3P5 = 10.0 * 1000.0 * 4.184
W35P = 8.0 * 1000.0 * 4.184
W3P5P = 0.0 * 1000.0 * 4.184

WV1 = -0.1250
WV2 = 0.1018

# =============================================================================
# Taylor-expansion coefficients (spinel.c lines 226-278), verbatim.
# g0 = gx2 = gx3 = gx4 = gx5 = 0.0 always in this build; carried through
# symbolically (not hardcoded away) for clarity/future-proofing, exactly
# as the source itself does.
# =============================================================================
g0 = 0.0
gx2 = 0.0
gx3 = 0.0
gx4 = 0.0
gx5 = 0.0
gs1 = 0.5 * (-(WOCT) + (W24U) - (W14) + 0.5 * (HX) + 0.5 * (HEX) + (H24))
gs2 = (W11) + (H11)
gs3 = (W13P) - (W13) + (H33)
hs4 = (W15P) - (W15) + (H55)
ss4 = (S55)
gx2x2 = -0.25 * ((WTET) + (WOCT) + (HX))
gx2x3 = 0.5 * ((W2P3U) - (W22) - (W1P3) + (W11) + 2.0 * (H23))
gx2x4 = 0.5 * ((WTET) - (W24U) + (W14) + 0.5 * (HX) - 0.5 * (HEX) + (H24))
gx2x5 = 0.5 * (-(W22) + (W11) + (W2P5U) - (W1P5) + 2.0 * (H25))
gx2s1 = 0.5 * ((WOCT) - (WTET))
gx2s2 = 0.5 * ((WTET) - (W22) + (W11) - (WOCT) + 2.0 * (W2P4U) - 2.0 * (W1P4)
               - (HEX) + 2.0 * (H24))
gx2s3 = 0.5 * ((WTET) - (WOCT) + 2.0 * (W3PU4U) - 2.0 * (W3P4) - (W2P3U)
               - (W23PU) + (W22) + (W1P3) + (W13P) - (W11) - (HEX)
               + 2.0 * (H24) - 2.0 * (H23))
gx2s4 = 0.5 * ((WTET) - (WOCT) - (W11) + (W22) + 2.0 * (W4U5PU) - 2.0 * (W45P)
               - (W2P5U) - (W25PU) + (W1P5) + (W15P) - (HEX) - 2.0 * (H25)
               + 2.0 * (H24))
gx3x3 = -(W13)
gx3x4 = (W34) - (W14) - (W13)
gx3x5 = (W35) - (W15) - (W13)
gx3s1 = 0.5 * ((W2P3U) - (W22) - (W1P3) + (W11))
gx3s2 = (W1P3) - (W13) - (W11)
gx3s3 = (W33) - (W13P) + (W13)
gx3s4 = (W35P) - (W35) - (W15P) + (W15)
gx4x4 = -(W14)
gx4x5 = (W45) - (W15) - (W14)
gx4s1 = 0.5 * ((WTET) - (W24U) + (W14) - 0.5 * (HX) + 0.5 * (HEX) - (H24))
gx4s2 = -(W11) + (W1P4) - (W14)
gx4s3 = (W3P4) - (W34) - (W13P) + (W13)
gx4s4 = (W45P) - (W45) - (W15P) + (W15)
gx5x5 = -(W15)
gx5s1 = 0.5 * (-(W22) + (W11) + (W2P5U) - (W1P5))
gx5s2 = -(W11) + (W1P5) - (W15)
gx5s3 = (W3P5) - (W35) - (W13P) + (W13)
gx5s4 = (W55) - (W15P) + (W15)
gs1s1 = 0.25 * (-(WTET) - (WOCT) + (HX))
gs1s2 = 0.5 * ((WTET) - (W22) + (W11) + (WOCT) - (HX))
gs1s3 = 0.5 * ((WTET) + (WOCT) - (W2P3U) - (W23PU) + (W22) + (W1P3) + (W13P)
               - (W11) - (HX))
gs1s4 = 0.5 * ((WTET) + (WOCT) + (W22) - (W11) - (W2P5U) - (W25PU) + (W1P5)
               + (W15P) - (HX))
gs2s2 = -(W11)
gs2s3 = (W1P3P) - (W1P3) - (W13P) + (W13)
gs2s4 = (W1P5P) - (W1P5) - (W15P) + (W15)
gs3s3 = -(W33)
gs3s4 = (W3P5P) - (W3P5) - (W35P) + (W35)
gs4s4 = -(W55)

# =============================================================================
# Generic polynomial engine over 7 "variables": X2,X3,X4,X5 (<-> r[0..3]),
# S1,S2,S4 (<-> s[0..2]) -- with "S3" ALIASED to the same slot as X3 (see
# module docstring). Coefficient names are all lowercase (`gx2x3`, `gs1s4`,
# ...); stripping the leading `g` and uppercasing the rest reproduces the
# token spelling directly.
# =============================================================================
_TOKEN_RE = re.compile(r'[XS]\d')
_TOKEN_INDEX = {'X2': 0, 'X3': 1, 'X4': 2, 'X5': 3, 'S1': 4, 'S2': 5, 'S3': 1, 'S4': 6}


def _var(idx, r, s):
    return r[..., idx] if idx < 4 else s[..., idx - 4]


def _parsed(coef: dict) -> list:
    out = []
    for key, val in coef.items():
        if val == 0.0:
            continue
        idxs = tuple(_TOKEN_INDEX[t] for t in _TOKEN_RE.findall(key.upper()))
        out.append((val, idxs))
    return out


def _poly_value(parsed, r, s):
    out = 0.0
    for val, idxs in parsed:
        m = val
        for i in idxs:
            m = m * _var(i, r, s)
        out = out + m
    return out


def _poly_d(parsed, r, s, wrt):
    out = 0.0
    for val, idxs in parsed:
        n = idxs.count(wrt)
        if n == 0:
            continue
        rest = list(idxs)
        rest.remove(wrt)
        m = val * n
        for i in rest:
            m = m * _var(i, r, s)
        out = out + m
    return out


_LOCAL = dict(locals())
G_COEF = {k[1:].upper(): v for k, v in _LOCAL.items()
          if k.startswith('g') and re.match(r'^g[xs]', k)}
del _LOCAL
_G_PARSED = _parsed(G_COEF)


# =============================================================================
# Site fractions (spinel.c order(), the current-iterate formulas and their
# clipping: x <= 0 -> DBL_EPSILON, x >= 1 -> 1 - DBL_EPSILON, verbatim).
# =============================================================================
def site_fractions(r, s):
    r = np.asarray(r, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    r0, r1, r2, r3 = (r[..., i] for i in range(4))
    s0, s1, s2 = s[..., 0], s[..., 1], s[..., 2]

    def clip(x):
        x = np.where(x <= 0.0, _DBL_EPS, x)
        return np.where(x >= 1.0, 1.0 - _DBL_EPS, x)

    xmg2tet = clip((r0 + s0) / 2.0)
    xfe2tet = clip(r2 - 0.5 * r0 - 0.5 * s0 + s1 + r1 + s2)
    xal3tet = clip(1.0 - r1 - r2 - r3 - s1)
    xfe3tet = clip(r3 - s2)
    xmg2oct = clip((r0 - s0) / 4.0)
    xfe2oct = clip((2.0 - r0 + s0 - 2.0 * s1 - 2.0 * r1 - 2.0 * s2) / 4.0)
    xal3oct = clip((1.0 - r1 - r2 - r3 + s1) / 2.0)
    xfe3oct = clip((r3 + s2) / 2.0)
    xcr3oct = clip(r1)
    xti4oct = clip(r2 / 2.0)

    return dict(xmg2tet=xmg2tet, xfe2tet=xfe2tet, xal3tet=xal3tet, xfe3tet=xfe3tet,
                xmg2oct=xmg2oct, xfe2oct=xfe2oct, xal3oct=xal3oct, xfe3oct=xfe3oct,
                xcr3oct=xcr3oct, xti4oct=xti4oct)


def _entropy_ideal(x):
    """The ideal-mixing part of the `S` macro (spinel.c lines 1572-1577),
    verbatim: tetrahedral-site terms weight 1, octahedral-site terms
    weight 2 (two formula units of octahedral cations per tetrahedral)."""
    return -Rgas * (
        x['xmg2tet'] * np.log(x['xmg2tet']) + x['xfe2tet'] * np.log(x['xfe2tet'])
        + x['xal3tet'] * np.log(x['xal3tet']) + x['xfe3tet'] * np.log(x['xfe3tet'])
        + 2.0 * x['xmg2oct'] * np.log(x['xmg2oct']) + 2.0 * x['xfe2oct'] * np.log(x['xfe2oct'])
        + 2.0 * x['xal3oct'] * np.log(x['xal3oct']) + 2.0 * x['xfe3oct'] * np.log(x['xfe3oct'])
        + 2.0 * x['xcr3oct'] * np.log(x['xcr3oct']) + 2.0 * x['xti4oct'] * np.log(x['xti4oct'])
    )


# =============================================================================
# Total G, H, S, V (the "H"/"S"/"G" macros, spinel.c lines 1579-1595 -- NOT
# gmix, see module docstring).
# =============================================================================
def gibbs_total(r, s, T, P):
    r = np.asarray(r, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    x = site_fractions(r, s)
    s2 = s[..., 2]

    H = g0 + _poly_value(_G_PARSED, r, s) + hs4 * s2
    S = _entropy_ideal(x) + ss4 * s2
    V = r[..., 2] * r[..., 3] * (WV1 * r[..., 3] + WV2 * r[..., 2])
    G = H - T * S + V * (P - 1.0)
    return G, H, S, V


# =============================================================================
# DGDR0-3 / DGDS0-2 (spinel.c lines 1599-1624): generic-polynomial-derivative
# leading term (with the X3/S3 alias automatically summing both "dH/dX3" and
# "dH/dS3" contributions, see module docstring) plus the verbatim ideal-
# mixing (R*t*log(...)) term, plus the small WV1/WV2 pressure term on r2/r3.
# =============================================================================
def dgdr(r, s, T, P, x=None):
    r = np.asarray(r, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    if x is None:
        x = site_fractions(r, s)
    r2, r3 = r[..., 2], r[..., 3]

    d0 = (_poly_d(_G_PARSED, r, s, 0)
          + 0.5 * Rgas * T * (np.log(x['xmg2tet'] / x['xfe2tet'])
                               + np.log(x['xmg2oct'] / x['xfe2oct'])))
    d1 = (_poly_d(_G_PARSED, r, s, 1)
          + Rgas * T * (np.log(x['xfe2tet'] / x['xal3tet'])
                         + 2.0 * np.log(x['xcr3oct']) - np.log(x['xfe2oct'])
                         - np.log(x['xal3oct'])))
    d2 = (_poly_d(_G_PARSED, r, s, 2)
          + Rgas * T * (np.log(x['xfe2tet'] / x['xal3tet'])
                         + np.log(x['xti4oct'] / x['xal3oct']))
          + r3 * (WV1 * r3 + WV2 * r2) * (P - 1.0) + r2 * r3 * WV2 * (P - 1.0))
    d3 = (_poly_d(_G_PARSED, r, s, 3)
          + Rgas * T * (np.log(x['xfe3tet'] / x['xal3tet'])
                         + np.log(x['xfe3oct'] / x['xal3oct']))
          + r2 * (WV1 * r3 + WV2 * r2) * (P - 1.0) + r2 * r3 * WV1 * (P - 1.0))

    out = np.empty(r.shape)
    out[..., 0], out[..., 1], out[..., 2], out[..., 3] = d0, d1, d2, d3
    return out


def dgds(r, s, T, P, x=None):
    r = np.asarray(r, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    if x is None:
        x = site_fractions(r, s)

    d0 = (_poly_d(_G_PARSED, r, s, 4)
          + 0.5 * Rgas * T * (np.log(x['xmg2tet'] / x['xfe2tet'])
                               + np.log(x['xfe2oct'] / x['xmg2oct'])))
    d1 = (_poly_d(_G_PARSED, r, s, 5)
          + Rgas * T * (np.log(x['xfe2tet'] / x['xal3tet'])
                         + np.log(x['xal3oct'] / x['xfe2oct'])))
    d2 = (hs4 - T * ss4 + _poly_d(_G_PARSED, r, s, 6)
          + Rgas * T * (np.log(x['xfe2tet'] / x['xfe3tet'])
                         + np.log(x['xfe3oct'] / x['xfe2oct'])))

    out = np.empty(s.shape)
    out[..., 0], out[..., 1], out[..., 2] = d0, d1, d2
    return out


# =============================================================================
# Endmember mole fractions <-> r (`ENDMEMBERS`/`conSpn` macros, spinel.c
# lines 1563 and 2825-2829).
# =============================================================================
def endmember_mole_fractions(r):
    r = np.asarray(r, dtype=np.float64)
    x_chromite = r[..., 1]
    x_hercynite = 1.0 - r[..., 0] - r[..., 1] - r[..., 2] - r[..., 3]
    x_magnetite = r[..., 3]
    x_spinel = r[..., 0]
    x_ulvospinel = r[..., 2]
    return np.stack([x_chromite, x_hercynite, x_magnetite, x_spinel, x_ulvospinel], axis=-1)


def x_to_r(X):
    """(B,5) mole fractions [chromite, hercynite, magnetite, spinel,
    ulvospinel] -> (B,4) r = [X_spinel, X_chromite, X_ulvospinel, X_magnetite]."""
    X = np.asarray(X, dtype=np.float64)
    x_chromite, x_hercynite, x_magnetite, x_spinel, x_ulvospinel = (X[..., i] for i in range(5))
    return np.stack([x_spinel, x_chromite, x_ulvospinel, x_magnetite], axis=-1)




# =============================================================================
# Second and third s-derivatives (D2GDS*, D2GDS*DT, D3GDS*, D3GDS*DT macros,
# spinel.c lines ~1690-1891). Every one of them is W + R*t*(site-fraction
# terms), so the Hessian is returned together with its T-derivative
# (d3gds2dt = the R*(...) part). D2GDT2 = D3GDT3 = D3GDSDT2 = 0 and every
# P-derivative of the s-gradient is 0 in this model.
# =============================================================================
def d2gds2(x, T):
    """(B,3,3) D2GDS2 and (B,3,3) D3GDS2DT from the clipped site fractions."""
    R = Rgas
    fe2t, fe2o = x['xfe2tet'], x['xfe2oct']
    M = np.empty(fe2t.shape + (3, 3))
    M[..., 0, 0] = 0.25*R*(1.0/x['xmg2tet'] + 1.0/fe2t + 0.5/fe2o + 0.5/x['xmg2oct'])
    M[..., 0, 1] = 0.5*R*(-1.0/fe2t - 0.5/fe2o)
    M[..., 0, 2] = 0.5*R*(-1.0/fe2t - 0.5/fe2o)
    M[..., 1, 1] = R*(1.0/fe2t + 1.0/x['xal3tet'] + 0.5/x['xal3oct'] + 0.5/fe2o)
    M[..., 1, 2] = R*(1.0/fe2t + 0.5/fe2o)
    M[..., 2, 2] = R*(1.0/fe2t + 1.0/x['xfe3tet'] + 0.5/x['xfe3oct'] + 0.5/fe2o)
    M[..., 1, 0], M[..., 2, 0], M[..., 2, 1] = M[..., 0, 1], M[..., 0, 2], M[..., 1, 2]
    W = np.array([[2.0*gs1s1, gs1s2, gs1s4], [gs1s2, 2.0*gs2s2, gs2s4], [gs1s4, gs2s4, 2.0*gs4s4]])
    return W + np.asarray(T, dtype=np.float64)[..., None, None]*M, M


def d2gdsdt(x):
    """(B,3) D2GDS0DT, D2GDS1DT, D2GDS2DT."""
    R = Rgas
    return np.stack([
        0.5*R*(np.log(x['xmg2tet']/x['xfe2tet']) + np.log(x['xfe2oct']/x['xmg2oct'])),
        R*(np.log(x['xfe2tet']/x['xal3tet']) + np.log(x['xal3oct']/x['xfe2oct'])),
        R*(np.log(x['xfe2tet']/x['xfe3tet']) + np.log(x['xfe3oct']/x['xfe2oct'])) - ss4], axis=-1)


def d3gds3(x, T):
    """(B,3,3,3) fully symmetric D3GDS3 tensor (fillD3GDS3)."""
    Rt = Rgas*np.asarray(T, dtype=np.float64)
    q = {k: 1.0/np.square(v) for k, v in x.items()}
    fe2t, fe2o = q['xfe2tet'], q['xfe2oct']
    e = {(0, 0, 0): -0.125*Rt*(q['xmg2tet'] - fe2t + 0.25*fe2o - 0.25*q['xmg2oct']),
         (0, 0, 1): -0.25*Rt*(fe2t - 0.25*fe2o),
         (0, 0, 2): -0.25*Rt*(fe2t - 0.25*fe2o),
         (0, 1, 1): -0.5*Rt*(-fe2t + 0.25*fe2o),
         (0, 1, 2): -0.5*Rt*(-fe2t + 0.25*fe2o),
         (0, 2, 2): -0.5*Rt*(-fe2t + 0.25*fe2o),
         (1, 1, 1): -Rt*(fe2t - q['xal3tet'] + 0.25*q['xal3oct'] - 0.25*fe2o),
         (1, 1, 2): -Rt*(fe2t - 0.25*fe2o),
         (1, 2, 2): -Rt*(fe2t - 0.25*fe2o),
         (2, 2, 2): -Rt*(fe2t - q['xfe3tet'] + 0.25*q['xfe3oct'] - 0.25*fe2o)}
    out = np.empty(fe2t.shape + (3, 3, 3))
    for (i, j, k), v in e.items():
        for a, b, c in {(i, j, k), (i, k, j), (j, i, k), (j, k, i), (k, i, j), (k, j, i)}:
            out[..., a, b, c] = v
    return out


# =============================================================================
# Bulk ordering solve: spinel.c `order()` (lines 2152-2379), verbatim.
# Start from the random tet/oct distribution; each iteration evaluates DGDS
# and the analytic D2GDS2 at the clipped site fractions of the current s,
# takes the Newton step deltaS = (-D2GDS2)^-1 DGDS, and shortens it (lambda)
# so no site fraction leaves [0, 1], testing the eight unclipped site
# fractions in the source's order. Iteration stops when every |ds| <=
# 10*DBL_EPSILON or after MAX_ITER = 200 steps; the returned state is sOld
# (the start of the last step) with its clipped site fractions, and the
# Hessian is the one factored in that last step -- exactly what order()
# leaves behind for gmix/hmix/cpmix and the ds/dT solves.
#
# Nothing here assumes a physically feasible composition: emulator output
# with, e.g., negative total Al (x_hercynite + x_spinel < 0) is handled the
# way MAGMA handles it (clipped site fractions, truncated steps).
# =============================================================================
MAX_ITER = 200


def _initial_guess(r):
    """order()'s starting point: cations spread over tet/oct sites in
    proportion to site availability (spinel.c lines 2189-2214)."""
    r = np.asarray(r, dtype=np.float64)
    r0, r1, r2, r3 = r[..., 0], r[..., 1], r[..., 2], r[..., 3]
    totAl = 2.0 * (1.0 - r1 - r2 - r3)
    totCr = 2.0 * r1
    totFe3 = 2.0 * r3
    totMg = r0
    totTi = r2
    ratio = 2.0 - totCr - totTi
    xmg2oct = totMg * ratio / (1.0 + ratio)
    xal3oct = totAl * ratio / (1.0 + ratio)
    xfe3oct = totFe3 * ratio / (1.0 + ratio)
    xmg2tet = totMg - xmg2oct
    xal3tet = totAl - xal3oct
    xfe3tet = totFe3 - xfe3oct
    return np.stack([xmg2tet - xmg2oct, xal3oct/2.0 - xal3tet/2.0, xfe3oct/2.0 - xfe3tet/2.0], axis=-1)


def _feasible_step(x, dS):
    """order()'s lambda: the largest step fraction (<= 1) keeping every
    unclipped site fraction in [0, 1], applied site by site in source
    order (`if (f + lambda*df < 0) ... else if (f + lambda*df > 1) ...`)."""
    d0, d1, d2 = dS[:, 0], dS[:, 1], dS[:, 2]
    lam = np.ones(d0.shape)
    for f, df in ((x['xmg2tet'], d0/2.0), (x['xfe2tet'], -d0/2.0 + d1 + d2), (x['xal3tet'], -d1),
                  (x['xfe3tet'], -d2), (x['xmg2oct'], -d0/4.0), (x['xfe2oct'], d0/4.0 - d1/2.0 - d2/2.0),
                  (x['xal3oct'], d1/2.0), (x['xfe3oct'], d2/2.0)):
        nz = df != 0.0
        with np.errstate(divide='ignore', invalid='ignore'):
            lo = nz & (f + lam*df < 0.0)
            hi = nz & ~lo & (f + lam*df > 1.0)
            lam = np.where(lo, -f/df, np.where(hi, (1.0 - f)/df, lam))
    return lam


def solve_ordering(r, T, P):
    """spinel.c order() for (B,4) r at (B,) T: returns (s (B,3), clipped
    site fractions at s, D2GDS2 (B,3,3) at s, converged (B,) bool).
    `converged` is False where MAX_ITER was reached; MAGMA silently uses
    that last state, and so does this function (the caller warns)."""
    r = np.asarray(r, dtype=np.float64)
    T = np.broadcast_to(np.asarray(T, dtype=np.float64), r.shape[:-1])
    P = np.broadcast_to(np.asarray(P, dtype=np.float64), r.shape[:-1])
    B = r.shape[0]
    sNew = _initial_guess(r)
    sOld = np.full((B, 3), 2.0)
    active = np.ones(B, dtype=bool)
    for _ in range(MAX_ITER):
        active &= np.any(np.abs(sNew - sOld) > 10.0*_DBL_EPS, axis=-1)
        if not active.any():
            break
        a = np.flatnonzero(active)
        s = sNew[a]
        ra, Ta = r[a], T[a]
        x = site_fractions(ra, s)
        g = dgds(ra, s, Ta, P[a], x)
        H, _ = d2gds2(x, Ta)
        sOld[a] = s
        try:
            dS = np.linalg.solve(-H, g[..., None])[..., 0]
        except np.linalg.LinAlgError as err:
            raise np.linalg.LinAlgError(f"spinel order(): singular D2GDS2 in rows {a.tolist()}: {err}") from err
        lam = _feasible_step(x, dS)
        sNew[a] = s + np.minimum(lam, 1.0)[:, None]*dS
    converged = ~np.any(np.abs(sNew - sOld) > 10.0*_DBL_EPS, axis=-1)
    x = site_fractions(r, sOld)
    H, _ = d2gds2(x, T)
    return sOld, x, H, converged


# =============================================================================
# Pure-endmember reference: spinel.c pureOrder()/pureSpn() (lines 725-1330),
# verbatim. Three decoupled 1-D Newton solves -- s0 of spinel (with
# s1 = (1+s0)/2 built into the SP_* macros), s1 of hercynite, s2 of
# magnetite -- started at (0.5, 0.9, 0.1), clipped to (-1+eps | eps,
# 1-eps), iterated until all three steps are <= 10*DBL_EPSILON, returning
# sOld. Chromite and ulvospinel have no ordering. Endmember order here and
# in the returned arrays is ENDMEMBERS = [chromite, hercynite, magnetite,
# spinel, ulvospinel] (MAGMA's ends[0..4]).
# =============================================================================
def _pure_derivs(s, T):
    """Per pure ordering parameter (columns: SP s0, HC s1, MT s2): dG/ds,
    d2G/ds2, d2G/dsdT, d3G/ds3, d3G/ds2dT."""
    R = Rgas
    t = T
    a, b, c = s[..., 0], s[..., 1], s[..., 2]
    dg = np.stack([
        gs1 + gs2/2.0 + gx2s1 + gx2s2/2.0 + 2.0*gs1s1*a + gs1s2*(0.5 + a) + gs2s2*(1.0 + a)/2.0
        + R*t*(0.5*np.log(1.0 + a) + 0.5*np.log(3.0 + a) - np.log(1.0 - a)),
        gs2 + 2.0*gs2s2*b + R*t*(np.log(b) + np.log(1.0 + b) - 2.0*np.log(1.0 - b)),
        hs4 - t*ss4 + gx5s4 + 2.0*gs4s4*c + R*t*(np.log(c) - 2.0*np.log(1.0 - c) + np.log(1.0 + c))], axis=-1)
    d2 = np.stack([
        2.0*gs1s1 + gs1s2 + gs2s2/2.0 + R*t*(0.5/(1.0 + a) + 0.5/(3.0 + a) + 1.0/(1.0 - a)),
        2.0*gs2s2 + R*t*(1.0/b + 1.0/(1.0 + b) + 2.0/(1.0 - b)),
        2.0*gs4s4 + R*t*(1.0/c + 2.0/(1.0 - c) + 1.0/(1.0 + c))], axis=-1)
    dt = np.stack([
        R*(0.5*np.log(1.0 + a) + 0.5*np.log(3.0 + a) - np.log(1.0 - a)),
        R*(np.log(b) + np.log(1.0 + b) - 2.0*np.log(1.0 - b)),
        R*(np.log(c) - 2.0*np.log(1.0 - c) + np.log(1.0 + c)) - ss4], axis=-1)
    d3 = np.stack([
        -R*t*(0.5/np.square(1.0 + a) + 0.5/np.square(3.0 + a) - 1.0/np.square(1.0 - a)),
        -R*t*(1.0/np.square(b) + 1.0/np.square(1.0 + b) - 2.0/np.square(1.0 - b)),
        -R*t*(1.0/np.square(c) - 2.0/np.square(1.0 - c) + 1.0/np.square(1.0 + c))], axis=-1)
    d3t = np.stack([
        R*(0.5/(1.0 + a) + 0.5/(3.0 + a) + 1.0/(1.0 - a)),
        R*(1.0/b + 1.0/(1.0 + b) + 2.0/(1.0 - b)),
        R*(1.0/c + 2.0/(1.0 - c) + 1.0/(1.0 + c))], axis=-1)
    return dg, d2, dt, d3, d3t


def pure_order(T):
    """pureOrder(): (B,3) [s0 of spinel, s1 of hercynite, s2 of magnetite]."""
    T = np.asarray(T, dtype=np.float64)
    sNew = np.broadcast_to(np.array([0.5, 0.9, 0.1]), T.shape + (3,)).copy()
    sOld = np.full(T.shape + (3,), 2.0)
    lo = np.array([-1.0 + _DBL_EPS, _DBL_EPS, _DBL_EPS])
    for _ in range(10*MAX_ITER):
        active = np.any(np.abs(sNew - sOld) > 10.0*_DBL_EPS, axis=-1)
        if not active.any():
            return sOld
        s = sNew[active]
        dg, d2, _, _, _ = _pure_derivs(s, T[active])
        sOld[active] = s
        sNew[active] = np.maximum(np.minimum(s - dg/d2, 1.0 - _DBL_EPS), lo)
    raise RuntimeError(f"spinel pureOrder() did not converge at T = {T[active][:5].tolist()} ...")


def pure_endmember_props(T):
    """pureSpn(): G, H, S, Cp, dCpdT (each (B,5), ENDMEMBERS order) of the
    model's own endmember references (V and its derivatives are 0)."""
    T = np.asarray(T, dtype=np.float64)
    s = pure_order(T)
    a, b, c = s[..., 0], s[..., 1], s[..., 2]
    R = Rgas
    S = np.stack([
        np.zeros_like(T),
        -R*(b*np.log(b) + 2.0*(1.0 - b)*np.log(1.0 - b) + (1.0 + b)*np.log(1.0 + b) - 2.0*np.log(2.0)),
        -R*(c*np.log(c) + 2.0*(1.0 - c)*np.log(1.0 - c) + (1.0 + c)*np.log(1.0 + c) - 2.0*np.log(2.0)) + ss4*c,
        -R*(0.5*(1.0 + a)*np.log(1.0 + a) + (1.0 - a)*np.log(1.0 - a) + 0.5*(3.0 + a)*np.log(3.0 + a)
            - 5.0*np.log(2.0)),
        np.full_like(T, R*2.0*np.log(2.0))], axis=-1)
    Hm = np.stack([
        np.full_like(T, g0 + gx3 + gs3 + gx3x3 + gx3s3 + gs3s3),
        g0 + gs2*b + gs2s2*b*b,
        g0 + gx5 + hs4*c + gx5x5 + gx5s4*c + gs4s4*c*c,
        g0 + gx2 + gs1*a + gs2*(1.0 + a)/2.0 + gx2x2 + gx2s1*a + gx2s2*(1.0 + a)/2.0 + gs1s1*a*a
        + gs1s2*a*(1.0 + a)/2.0 + gs2s2*np.square(1.0 + a)/4.0,
        np.full_like(T, g0 + gx4 + gx4x4)], axis=-1)
    G = Hm - T[..., None]*S
    # SIXTH / SEVENTH masks: ds/dT = -d2gdsdt/d2gds2, d2s/dT2 from pureOrder FOURTH
    _, d2, dt, d3, d3t = _pure_derivs(s, T)
    dsdt = -dt/d2
    d2sdt2 = -(2.0*d3t*dsdt + d3*dsdt*dsdt)/d2
    t = T[..., None]
    temp = 2.0*dt*dsdt + d2*np.square(dsdt)
    cp3 = -t*temp
    dcp3 = -t*(3.0*dt*d2sdt2 + 3.0*d2*dsdt*d2sdt2 + 3.0*d3t*dsdt*dsdt + d3*dsdt*dsdt*dsdt) - temp
    z = np.zeros_like(T)
    # columns of the pure-order arrays are (SP, HC, MT); endmembers are (CR, HC, MT, SP, UV)
    Cp = np.stack([z, cp3[..., 1], cp3[..., 2], cp3[..., 0], z], axis=-1)
    dCpdT = np.stack([z, dcp3[..., 1], dcp3[..., 2], dcp3[..., 0], z], axis=-1)
    return dict(G=G, H=G + T[..., None]*S, S=S, Cp=Cp, dCpdT=dCpdT)


# =============================================================================
# Darken activities (`actSpn`, spinel.c lines 2995-3070): mu_i = G - mu0_i
# + sum_j FR_ij dG/dr_j with G, dG/dr at order()'s state and mu0 = pureSpn G.
# =============================================================================
def _fr_matrix(r):
    """(B,5,4) Darken FR matrix, spinel.c FR2(i)..FR5(i) macros, component
    order = ENDMEMBERS = [chromite, hercynite, magnetite, spinel,
    ulvospinel]; column j is r_j's FR: 1 - r_j for its own endmember
    (spinel, chromite, ulvospinel, magnetite), -r_j otherwise."""
    r = np.asarray(r, dtype=np.float64)
    fr = np.broadcast_to(-r[:, None, :], (r.shape[0], 5, 4)).copy()
    for j, i in enumerate((3, 0, 4, 2)):
        fr[:, i, j] = 1.0 - r[:, j]
    return fr


def solution_thermo(r, T, P):
    """Spinel mixing properties per mole (gmixSpn, hmixSpn, smixSpn,
    vmixSpn, cpmixSpn FIRST|SECOND, actSpn SECOND), all at order()'s state.

    Cp_mix = -T (2 gst.dsdt + dsdt.gss.dsdt) with dsdt = -gss^-1 gst (the
    D2GDT2 = 0 ordering contribution), dCpdT_mix from d2s/dT2 (order()
    EIGHTH mask), each minus the ENDMEMBERS-weighted pureSpn values. V_mix
    = DGDP depends on r only, so dVdT_mix = dVdP_mix = 0 exactly.
    """
    r = np.asarray(r, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    s, x, Hs, converged = solve_ordering(r, T, P)
    if not converged.all():
        bad = np.flatnonzero(~converged)
        warnings.warn(f"spinel order() hit MAX_ITER={MAX_ITER} on {bad.size} row(s) (first: {bad[:5].tolist()}); "
                      "using the last iterate, as MAGMA does.", RuntimeWarning)
    Gt, Hpoly, St, Vt = gibbs_total(r, s, T, P)
    pure = pure_endmember_props(T)
    xe = endmember_mole_fractions(r)
    gmix = Gt - np.sum(xe*pure['G'], axis=-1)
    smix = St - np.sum(xe*pure['S'], axis=-1)
    hmix = (Gt + T*St) - np.sum(xe*pure['H'], axis=-1)
    vmix = Vt

    gst = d2gdsdt(x)
    dsdt = np.linalg.solve(-Hs, gst[..., None])[..., 0]
    _, d3t = d2gds2(x, T)
    d3 = d3gds3(x, T)
    temp = 2.0*np.sum(gst*dsdt, -1) + np.einsum('bi,bij,bj->b', dsdt, Hs, dsdt)
    Cp_mix = -T*temp - np.sum(xe*pure['Cp'], axis=-1)
    rhs = 2.0*np.einsum('bjk,bk->bj', d3t, dsdt) + np.einsum('bjkl,bk,bl->bj', d3, dsdt, dsdt)
    d2sdt2 = np.linalg.solve(-Hs, rhs[..., None])[..., 0]
    dt = (3.0*np.sum(gst*d2sdt2, -1) + 3.0*np.einsum('bij,bi,bj->b', Hs, dsdt, d2sdt2)
          + 3.0*np.einsum('bij,bi,bj->b', d3t, dsdt, dsdt) + np.einsum('bijk,bi,bj,bk->b', d3, dsdt, dsdt, dsdt))
    dCpdT_mix = -T*dt - temp - np.sum(xe*pure['dCpdT'], axis=-1)

    fr = _fr_matrix(r)
    mu = Gt[:, None] - pure['G'] + np.einsum('bij,bj->bi', fr, dgdr(r, s, T, P, x))
    with np.errstate(over='ignore'):
        a = np.exp(mu / (Rgas * T[:, None]))
    z = np.zeros_like(Cp_mix)
    return dict(gmix=gmix, H_mix=hmix, S_mix=smix, V_mix=vmix,
                Cp_mix=Cp_mix, dCpdT_mix=dCpdT_mix, dVdT_mix=z, dVdP_mix=z,
                mu=mu, activities=a, s_eq=s, endmembers=ENDMEMBERS)
