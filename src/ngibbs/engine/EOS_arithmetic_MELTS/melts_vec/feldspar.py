"""
Feldspar (albite-anorthite-sanidine ternary) solid-solution mixing model --
vectorized translation of `sources/feldspar.c`'s `gmixFld`/`actFld`,
verified against the file directly (asymmetric/subregular Margules model,
Elkins & Grove 1990).

No internal ordering parameters (NS = 0) -- G is an explicit closed-form
function of (xab, xan, T, P), so unlike olivine.py this needs neither a
Newton solve nor finite differences: dG/dr, dG/dT = -S, dG/dP = V are all
exact by construction (S and V are themselves closed-form and
T,P-independent, since every W here is a plain constant, not a
P-dependent GX/GEX-style combination like olivine's -- confirmed by
inspection of feldspar.c: `G = H - t*S + (p-1.0)*V` with H, S, V all
pure functions of composition only).

Composition convention: r0 = x_albite, r1 = x_anorthite,
x_sanidine = 1 - r0 - r1 (matching FR0(i)/FR1(i) in feldspar.c, i.e.
component order [albite, anorthite, sanidine] = indices [0, 1, 2]).
"""
from __future__ import annotations
import numpy as np

from .constants import Rgas
from .solution_model import darken_activities

ENDMEMBERS = ["albite", "anorthite", "sanidine"]

# Margules parameters, verbatim from feldspar.c (J for H-type, J/K for
# S-type, J/bar for V-type).
_WHABOR = 18810.0
_WSABOR = 10.3
_WVABOR = 0.4602
_WHORAB = 27320.0
_WSORAB = 10.3
_WVORAB = 0.3264
_WHABAN = 7924.0
_WHANAB = 0.0
_WHORAN = 40317.0
_WHANOR = 38974.0
_WVANOR = -0.1037
_WHABANOR = 12545.0
_WVABANOR = -1.095


def _site_fractions(r0, r1):
    xab = r0
    xan = r1
    xor = 1.0 - r0 - r1
    return xab, xan, xor


def gibbs_mixing(r0, r1, T, P):
    """G_mix(xab, xan, T, P) -- feldspar.c's `gmixFld` macro, verbatim.

    r0, r1, T, P broadcast against each other in the usual numpy way.
    """
    r0 = np.asarray(r0, dtype=np.float64)
    r1 = np.asarray(r1, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    xab, xan, xor = _site_fractions(r0, r1)

    with np.errstate(divide='ignore', invalid='ignore'):
        xab_safe = np.where(xab > 0.0, xab, 1.0)
        xan_safe = np.where(xan > 0.0, xan, 1.0)
        xor_safe = np.where(xor > 0.0, xor, 1.0)
        S_config = -Rgas*(
            np.where(xab > 0.0, xab*np.log(xab_safe), 0.0)
            + np.where(xan > 0.0, xan*np.log(xan_safe), 0.0)
            + np.where(xor > 0.0, xor*np.log(xor_safe), 0.0)
        )
    S = S_config + _WSABOR*xab*xor*(xor + xan/2.0) + _WSORAB*xab*xor*(xab + xan/2.0)

    H = (_WHABAN*xab*xan*(xan + xor/2.0) + _WHANAB*xab*xan*(xab + xor/2.0)
         + _WHABOR*xab*xor*(xor + xan/2.0) + _WHORAB*xab*xor*(xab + xan/2.0)
         + _WHANOR*xan*xor*(xor + xab/2.0) + _WHORAN*xan*xor*(xan + xab/2.0)
         + _WHABANOR*xab*xan*xor)

    V = (_WVABOR*xab*xor*(xor + xan/2.0) + _WVORAB*xab*xor*(xab + xan/2.0)
         + _WVANOR*xan*xor*(xor + xab/2.0) + _WVABANOR*xab*xan*xor)

    G = H - T*S + (P - 1.0)*V
    return G, H, S, V


# FR matrix (component i, r-index j), verbatim from feldspar.c's
# FR0(i)/FR1(i) macros: FR0(i) = (i==0) ? 1-xab : -xab;
# FR1(i) = (i==1) ? 1-xan : -xan.
def _fr_matrix(xab, xan):
    """Returns (B, 3, 2) batched FR matrix."""
    B = xab.shape[0]
    fr = np.empty((B, 3, 2), dtype=np.float64)
    fr[:, 0, 0] = 1.0 - xab
    fr[:, 1, 0] = -xab
    fr[:, 2, 0] = -xab
    fr[:, 0, 1] = -xan
    fr[:, 1, 1] = 1.0 - xan
    fr[:, 2, 1] = -xan
    return fr


def solution_thermo(r0, r1, T, P, dr=1e-6):
    """Full feldspar solid-solution mixing thermodynamics: gmix, its
    exact analytic T,P derivatives (S = -dG/dT, V = dG/dP; both already
    closed-form, see module docstring), and Darken activities (via a
    small central-difference dG/dr -- safe here since, unlike olivine,
    there is no auxiliary equilibrium variable to worry about, so
    ordinary finite differences of the explicit G(r0,r1) polynomial are
    exact up to O(dr^2) truncation, negligible at dr=1e-6 given G is
    O(1e4-1e5) in magnitude).

    Returns dict with gmix, H, S, V (mixing-only, i.e. to be ADDED to
    Sum_i x_i * G_i,pure(T,P) from the pure endmember EOS -- see
    compute.py's compute_feldspar()), Cp_mix (=0 exactly), dVdT/dVdP
    (=0 exactly, see docstring), and activities/mu for [albite,
    anorthite, sanidine].
    """
    r0 = np.asarray(r0, dtype=np.float64)
    r1 = np.asarray(r1, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)

    G, H, S, V = gibbs_mixing(r0, r1, T, P)

    Gp0, _, _, _ = gibbs_mixing(r0 + dr, r1, T, P)
    Gm0, _, _, _ = gibbs_mixing(r0 - dr, r1, T, P)
    Gp1, _, _, _ = gibbs_mixing(r0, r1 + dr, T, P)
    Gm1, _, _, _ = gibbs_mixing(r0, r1 - dr, T, P)
    dGdr0 = (Gp0 - Gm0)/(2.0*dr)
    dGdr1 = (Gp1 - Gm1)/(2.0*dr)
    dGdr = np.stack([dGdr0, dGdr1], axis=-1)   # (B, 2)

    xab, xan, xor = _site_fractions(r0, r1)
    fr = _fr_matrix(xab, xan)
    mu, a = darken_activities(G, dGdr, fr, Rgas, T)

    Cp_mix = np.zeros_like(G)
    dVdT = np.zeros_like(G)
    dVdP = np.zeros_like(G)

    return dict(
        gmix=G, H_mix=H, S_mix=S, V_mix=V, Cp_mix=Cp_mix,
        dVdT_mix=dVdT, dVdP_mix=dVdP,
        mu=mu, activities=a, endmembers=ENDMEMBERS,
        x=np.stack([xab, xan, xor], axis=-1),
    )
