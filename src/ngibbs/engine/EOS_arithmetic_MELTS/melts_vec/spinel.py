"""
Spinel solid-solution mixing model -- vectorized translation of
`sources/spinel.c`'s `gmixSpn`/`actSpn`/`order`, verified against the file
directly (see `tests/verify_spinel.c`).

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

Design notes carried over from `solution_model.py`/clinopyroxene.py:
  - a batched Newton solve with a NUMERICALLY estimated Jacobian (not the
    analytic `d2gds2` GSL-LU-solved Hessian spinel.c itself uses) finds
    the equilibrium ordering parameters s* = (s0,s1,s2) at each (r,T,P);
  - G, H, S and the Darken activities are read off analytically at s*
    (envelope theorem); Cp/dV(T,P) are obtained by finite-differencing
    with s* re-solved at each stencil point.

Pure-endmember reference (`pureOrder`/`pureSpn`, spinel.c lines 881-1161):
THREE of the five endmembers each carry their own separate, DECOUPLED
1-D Landau order-disorder parameter (chromite and ulvospinel have none):
  - hercynite (HC): free s1, at vertex r=[0,0,0,0] (X_hercynite=1), s0=s2=0.
  - magnetite (MT): free s2, at vertex r=[0,0,0,1], s0=s1=0.
  - spinel    (SP): free s0, at vertex r=[1,0,0,0], s1=(1.0+s0)/2.0 (!),
    s2=0 -- confirmed by direct term-by-term comparison of `SP_H`'s macro
    text against the generic H polynomial evaluated with that
    substitution (every occurrence of the "S2" token in `SP_H` is written
    algebraically as `(1.0+s[0])/2.0` rather than as an independent
    variable -- this is the SAME alias mechanism noted above, just
    applied to make S2 dependent on S1 at this one vertex rather than
    holding it at a constant).
  - chromite  (CR): fixed, r=[0,1,0,0], s=[0,0,0], no ordering freedom.
  - ulvospinel(UV): fixed, r=[0,0,1,0], s=[0,0,0], no ordering freedom.
Rather than routing these five special cases back through the generic
polynomial engine (which would need per-vertex substitution logic), each
is transcribed directly from its own `_S`/`_H`/`_G`/`D..._GDS...` macro
pair, mirroring how `orthopyroxene.py`/clinopyroxene.py handle essenite's
own separate internal ordering.
"""
from __future__ import annotations
import re
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
# Site fractions (spinel.c lines 2225-2233, the converged/current-iterate
# formulas -- both-bounds clipping at DBL_EPSILON matches the source).
# =============================================================================
def site_fractions(r, s):
    r = np.asarray(r, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    r0, r1, r2, r3 = (r[..., i] for i in range(4))
    s0, s1, s2 = s[..., 0], s[..., 1], s[..., 2]

    def clip(x):
        return np.clip(x, _DBL_EPS, 1.0 - _DBL_EPS)

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
# Bulk ordering solve (spinel.c `order()`, lines 2152-2379): coupled 3-D
# Newton solve for s=(s0,s1,s2) at fixed (r,T,P). No degenerate-composition
# gating is needed here (unlike clinopyroxene.py) because every site
# fraction is clipped at DBL_EPSILON before entering a log, exactly as the
# C source itself does, so dgds never evaluates log(0) even at the
# composition vertices.
# =============================================================================
def _initial_guess(r):
    """Physically-motivated starting point mirroring order()'s own
    totMg/totFe2/totAl/totFe3/totCr/totTi tet/oct partition (spinel.c
    lines 2189-2210) -- purely a Newton starting point, does not affect
    the converged answer."""
    r = np.asarray(r, dtype=np.float64)
    r0, r1, r2, r3 = r[..., 0], r[..., 1], r[..., 2], r[..., 3]
    totAl = 2.0 * (1.0 - r1 - r2 - r3)
    totCr = 2.0 * r1
    totFe2 = 1.0 - r0 + r2
    totFe3 = 2.0 * r3
    totMg = r0
    totTi = r2
    ratio = 2.0 - totCr - totTi
    denom = 1.0 + ratio
    xmg2oct = totMg * ratio / denom
    xfe2oct = totFe2 * ratio / denom
    xal3oct = totAl * ratio / denom
    xfe3oct = totFe3 * ratio / denom
    xmg2tet = totMg - xmg2oct
    xal3tet = totAl - xal3oct
    xfe3tet = totFe3 - xfe3oct
    xmg2oct = xmg2oct / 2.0
    xal3oct = xal3oct / 2.0
    xfe3oct = xfe3oct / 2.0

    s0 = xmg2tet - 2.0 * xmg2oct
    s1 = xal3oct - xal3tet / 2.0
    s2 = xfe3oct - xfe3tet / 2.0
    return np.stack([s0, s1, s2], axis=-1)


def _gates(r):
    """Which of s0,s1,s2 have a genuine root to chase, given bulk r --
    the same "degenerate composition" concept clinopyroxene.py's own
    `_ab_initio_guess_and_gates` uses (see its docstring), applied here
    because spinel's coupled 3-variable ideal-mixing derivative has the
    same failure mode: whenever the cation pair a given s_i physically
    exchanges between sites is (partly or wholly) ABSENT from the bulk
    composition, DGDS_i has no reachable interior zero (confirmed by
    direct grid evaluation: at pure spinel, MgAl2O4, Fe2+ is entirely
    absent -- `totFe2 = 1-r0+r2 = 0` -- and DGDS0/DGDS1 simply never
    change sign smoothly across the physical (s0,s1) box, only jumping
    at the clip boundary), and chasing it anyway corrupts convergence of
    the OTHER, genuinely well-posed s components too (via the coupled
    Jacobian's off-diagonal terms) -- confirmed by A/B testing: gating
    off only the ill-posed component(s) recovers, bit-for-bit, the SAME
    equilibrium `solve_ordering` finds for a well-posed vertex (e.g.
    hercynite's bulk-order() s1 -> 0.8254792529, exactly matching
    `_hc_ghsv`'s own independent 1-D `pureOrder`-equivalent solve) that
    an UNGATED joint solve fails to reach.

    Fe2+ (`totFe2`) is the shared "exchange partner" for all three
    directions (S1: Mg-Fe2, S2: Al-Fe2, S4: Fe2-Fe3, per the site-
    fraction formulas in `site_fractions`), so it gates all three; each
    direction additionally needs its own other cation present."""
    r = np.asarray(r, dtype=np.float64)
    r0, r1, r2, r3 = r[..., 0], r[..., 1], r[..., 2], r[..., 3]
    totMg = r0
    totAl = 2.0 * (1.0 - r1 - r2 - r3)
    totFe2 = 1.0 - r0 + r2
    totFe3 = 2.0 * r3
    gate0 = (totMg != 0.0) & (totFe2 != 0.0)
    gate1 = (totAl != 0.0) & (totFe2 != 0.0)
    gate2 = (totFe3 != 0.0) & (totFe2 != 0.0)
    return np.stack([gate0, gate1, gate2], axis=-1)


def solve_ordering(r, T, P, n_iter=150, jac_eps=1e-4, max_halvings=30):
    """Coupled 3-D Newton solve for s=(s0,s1,s2): a numerically estimated
    Jacobian (as in `solution_model.newton_solve_ordering`), a
    backtracking line search on top of it, AND composition-dependent
    gating of components with no reachable root (see `_gates`) -- all
    three are needed together (verified empirically, not merely assumed
    by analogy to clinopyroxene.py): plain Newton (no line search, no
    gating) oscillates indefinitely for compositions near a single
    dominant endmember, because spinel's ideal-mixing entropy derivative
    involves site-fraction log-RATIOS (`log(xmg2tet/xfe2tet)`, etc.)
    that become extremely steep near the DBL_EPSILON floor/ceiling clip
    -- an un-damped step can overshoot straight across a clipped
    boundary and land at a point with an even LARGER gradient on the far
    side. Line search alone fixes every genuinely interior/mixed
    composition (residual ~1e-12) but is not sufficient exactly at or
    very near a pure-endmember vertex where an entire cation type is
    absent: there gating is also needed, both to give a well-defined
    answer for the components with no root (frozen at
    `_initial_guess`'s own physically-motivated value, mirroring the
    only sensible fallback) AND to stop that ill-posed direction from
    corrupting the coupled Jacobian's estimate for the OTHER, genuinely
    well-posed components.

    Even with all three, gmix is NOT reliably ~0 exactly at every pure-
    endmember vertex (unlike clinopyroxene.py/spinel's own hercynite and
    magnetite, whose bulk vertex solution empirically DOES land on
    `_hc_ghsv`/`_mt_ghsv`'s independent `pureOrder`-equivalent value --
    chromite and ulvospinel have no internal ordering at all by design,
    S=0/const, so this doesn't apply to them either; only spinel itself,
    where the Fe2+-absence degeneracy above leaves s0/s1 frozen at
    `_initial_guess` rather than at `_sp_ghsv`'s own value). See
    `benchmark_spinel_test.py`, which for this reason checks gmix at
    genuinely interior/mixed compositions (following the precedent set
    for orthopyroxene's own gmix test, melts_vec/orthopyroxene.py) plus
    the two vertices (hercynite, magnetite) confirmed to agree.
    """
    r = np.asarray(r, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    s = np.clip(np.nan_to_num(_initial_guess(r), nan=0.0), -1.0 + 1e-6, 1.0 - 1e-6)
    B, NS = s.shape
    gate = _gates(r)

    def _f(ss):
        return np.where(gate, dgds(r, ss, T, P), 0.0)

    for _ in range(n_iter):
        f0 = _f(s)
        norm0 = np.sum(f0 * f0, axis=-1)
        J = np.empty((B, NS, NS), dtype=np.float64)
        for j in range(NS):
            s_plus = s.copy();  s_plus[:, j]  += jac_eps
            s_minus = s.copy(); s_minus[:, j] -= jac_eps
            J[:, :, j] = (_f(s_plus) - _f(s_minus)) / (2.0 * jac_eps)
        J = J + 1e-8 * np.eye(NS)[None, :, :]
        try:
            step = np.linalg.solve(J, f0[:, :, None])[:, :, 0]
        except np.linalg.LinAlgError:
            step = np.zeros_like(f0)

        alpha = np.ones(B)
        s_new = np.clip(s - alpha[:, None] * step, -1.0 + 1e-9, 1.0 - 1e-9)
        norm_new = np.sum(_f(s_new) ** 2, axis=-1)
        bad = norm_new > norm0
        k = 0
        while np.any(bad) and k < max_halvings:
            alpha = np.where(bad, alpha * 0.5, alpha)
            s_new = np.clip(s - alpha[:, None] * step, -1.0 + 1e-9, 1.0 - 1e-9)
            norm_new = np.sum(_f(s_new) ** 2, axis=-1)
            bad = norm_new > norm0
            k += 1
        s = s_new
    return s


# =============================================================================
# Pure-endmember reference (`pureOrder`/`pureSpn`, spinel.c lines 725-1161):
# three decoupled 1-D Landau order parameters, each solved once per (T,P)
# (independent of bulk composition r), plus two endmembers with no internal
# ordering at all. See module docstring for the vertex (r,s) of each.
# =============================================================================
def _newton_1d(dgds_func, s0, T, P, n_iter=60, lo=-1.0 + 1e-9, hi=1.0 - 1e-9):
    T = np.asarray(T, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    s = np.full(T.shape, s0, dtype=np.float64)
    eps = 1e-6
    for _ in range(n_iter):
        f0 = dgds_func(s, T, P)
        fp = dgds_func(np.clip(s + eps, lo, hi), T, P)
        fm = dgds_func(np.clip(s - eps, lo, hi), T, P)
        jac = (fp - fm) / (2.0 * eps)
        jac = np.where(jac == 0.0, 1.0, jac)
        s = s - f0 / jac
        s = np.clip(s, lo, hi)
    return s


def _sp_dgds0(s0, T, P):
    """DSP_GDS0, spinel.c lines 761-763, verbatim (s0 in (-1,1))."""
    return (gs1 + gs2 / 2.0 + gx2s1 + gx2s2 / 2.0 + 2.0 * gs1s1 * s0
            + gs1s2 * (0.5 + s0) + gs2s2 * (1.0 + s0) / 2.0
            + Rgas * T * (0.5 * np.log(1.0 + s0) + 0.5 * np.log(3.0 + s0) - np.log(1.0 - s0)))


def _hc_dgds1(s1, T, P):
    """DHC_GDS1, spinel.c lines 733-734, verbatim (s1 in (0,1))."""
    return (gs2 + 2.0 * gs2s2 * s1
            + Rgas * T * (np.log(s1) + np.log(1.0 + s1) - 2.0 * np.log(1.0 - s1)))


def _mt_dgds2(s2, T, P):
    """DMT_GDS2, spinel.c lines 823-825, verbatim (s2 in (0,1))."""
    return (hs4 - T * ss4 + gx5s4 + 2.0 * gs4s4 * s2
            + Rgas * T * (np.log(s2) - 2.0 * np.log(1.0 - s2) + np.log(1.0 + s2)))


def _sp_ghsv(T, P):
    s0 = _newton_1d(_sp_dgds0, 0.5, T, P, lo=-1.0 + 1e-9, hi=1.0 - 1e-9)
    S = -Rgas * (0.5 * (1.0 + s0) * np.log(1.0 + s0) + (1.0 - s0) * np.log(1.0 - s0)
                 + 0.5 * (3.0 + s0) * np.log(3.0 + s0) - 5.0 * np.log(2.0))
    H = (g0 + gx2 + gs1 * s0 + gs2 * (1.0 + s0) / 2.0 + gx2x2
         + gx2s1 * s0 + gx2s2 * (1.0 + s0) / 2.0 + gs1s1 * s0 * s0
         + gs1s2 * s0 * (1.0 + s0) / 2.0 + gs2s2 * (1.0 + s0) ** 2 / 4.0)
    G = H - T * S
    V = np.zeros_like(H)
    return G, H, S, V


def _hc_ghsv(T, P):
    s1 = _newton_1d(_hc_dgds1, 0.9, T, P, lo=0.0 + 1e-9, hi=1.0 - 1e-9)
    S = -Rgas * (s1 * np.log(s1) + 2.0 * (1.0 - s1) * np.log(1.0 - s1)
                 + (1.0 + s1) * np.log(1.0 + s1) - 2.0 * np.log(2.0))
    H = g0 + gs2 * s1 + gs2s2 * s1 * s1
    G = H - T * S
    V = np.zeros_like(H)
    return G, H, S, V


def _mt_ghsv(T, P):
    s2 = _newton_1d(_mt_dgds2, 0.1, T, P, lo=0.0 + 1e-9, hi=1.0 - 1e-9)
    S = -Rgas * (s2 * np.log(s2) + 2.0 * (1.0 - s2) * np.log(1.0 - s2)
                 + (1.0 + s2) * np.log(1.0 + s2) - 2.0 * np.log(2.0)) + ss4 * s2
    H = g0 + gx5 + hs4 * s2 + gx5x5 + gx5s4 * s2 + gs4s4 * s2 * s2
    G = H - T * S
    V = np.zeros_like(H)
    return G, H, S, V


def _cr_ghsv(T, P):
    T = np.asarray(T, dtype=np.float64)
    S = np.zeros_like(T)
    H = np.full_like(T, g0 + gx3 + gs3 + gx3x3 + gx3s3 + gs3s3)
    G = H - T * S
    V = np.zeros_like(T)
    return G, H, S, V


def _uv_ghsv(T, P):
    T = np.asarray(T, dtype=np.float64)
    S = np.full_like(T, Rgas * 2.0 * np.log(2.0))
    H = np.full_like(T, g0 + gx4 + gx4x4)
    G = H - T * S
    V = np.zeros_like(T)
    return G, H, S, V


_PURE_FUNCS = {
    'chromite': _cr_ghsv,
    'hercynite': _hc_ghsv,
    'magnetite': _mt_ghsv,
    'spinel': _sp_ghsv,
    'ulvospinel': _uv_ghsv,
}


def pure_endmember_ghsv(T, P):
    """G,H,S,V (B,5) for the 5 endmembers in ENDMEMBERS order, using this
    model's OWN internal Taylor-coefficient-based reference frame (same
    caveat as clinopyroxene.py's `pure_endmember_ghsv`: only ever used
    differentially, against the same reference frame, in
    `endmember_mole_fractions`-weighted `gmix`/Darken normalization)."""
    T = np.asarray(T, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    G = np.empty(T.shape + (5,))
    H = np.empty(T.shape + (5,))
    S = np.empty(T.shape + (5,))
    V = np.empty(T.shape + (5,))
    for i, name in enumerate(ENDMEMBERS):
        g, h, s, v = _PURE_FUNCS[name](T, P)
        G[..., i], H[..., i], S[..., i], V[..., i] = g, h, s, v
    return G, H, S, V


# =============================================================================
# Darken activities (`actSpn`, spinel.c lines 2995-3070): the STANDARD
# (non-normalized) Darken formula -- unlike clinopyroxene.py, spinel.c does
# NOT divide by a per-component pure-endmember activity constant, it
# directly forms mu_i = g - mu0_i + sum_j fr_ij dG/dr_j (see actSpn's
# `mask & SECOND` block), which is exactly `solution_model.darken_activities`
# applied to (g - g0_i).
# =============================================================================
def _fr_matrix(r):
    """(B,5,4) Darken FR matrix, spinel.c FR2(i)..FR5(i) macros (lines
    1542-1545), component order = ENDMEMBERS = [chromite, hercynite,
    magnetite, spinel, ulvospinel]."""
    r = np.asarray(r, dtype=np.float64)
    B = r.shape[0]
    r0, r1, r2, r3 = (r[:, i] for i in range(4))
    fr = np.zeros((B, 5, 4), dtype=np.float64)
    # FR2(i): X2 column (r0) -- component 3 (spinel) is 1-r0, else -r0
    fr[:, 3, 0] = 1.0 - r0
    for i in (0, 1, 2, 4):
        fr[:, i, 0] = -r0
    # FR3(i): X3 column (r1) -- component 0 (chromite) is 1-r1, else -r1
    fr[:, 0, 1] = 1.0 - r1
    for i in (1, 2, 3, 4):
        fr[:, i, 1] = -r1
    # FR4(i): X4 column (r2) -- component 4 (ulvospinel) is 1-r2, else -r2
    fr[:, 4, 2] = 1.0 - r2
    for i in (0, 1, 2, 3):
        fr[:, i, 2] = -r2
    # FR5(i): X5 column (r3) -- component 2 (magnetite) is 1-r3, else -r3
    fr[:, 2, 3] = 1.0 - r3
    for i in (0, 1, 3, 4):
        fr[:, i, 3] = -r3
    return fr


def solution_thermo(r, T, P, n_iter=60, dT=0.02, dP=0.02):
    """Full spinel solid-solution mixing thermodynamics.

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
    a = np.exp(mu / (Rgas * T[:, None]))

    dCpdT_mix = np.zeros_like(Cp_mix)

    return dict(gmix=gmix, H_mix=hmix, S_mix=smix, V_mix=vmix,
                Cp_mix=Cp_mix, dCpdT_mix=dCpdT_mix, dVdT_mix=dVdT_mix, dVdP_mix=dVdP_mix,
                mu=mu, activities=a, s_eq=s_eq, endmembers=ENDMEMBERS)
