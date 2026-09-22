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

Out of scope, deliberately not addressed here: albite's own order-disorder
branch (gibbs.c lines 2543-2561) is structurally different -- when
`HIGH_STRUCTURAL_STATE_FELDSPAR` is *not* defined (the branch this
package's standard-MELTS target calibration takes), it calls a whole
separate `albite()` function (`sources/albite.c`) that solves an internal
order parameter via Newton-Raphson (the same class of iterative machinery
already implemented for the six solid-solution mixing models, but not yet
translated for this single-endmember correction) rather than a closed-form
polynomial. Left as a follow-up: plagioclase's (albite/anorthite-dominated)
S/Cp mismatch was already much smaller (<0.4%/~1%) than k-feldspar's in the
existing benchmark, so this is a smaller, lower-priority gap than
sanidine's.

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
