"""
Bulk solid-solution properties: combine the pure-endmember EOS
(`compute.compute()`, Berman/Vinet) with each phase's mixing model
(`feldspar.py`, `olivine.py`, ...) into total (composition-weighted +
mixing-correction) V, K, alpha, Cp, G, H, S for the solution as a whole --
the solid-solution analogue of `compute.compute()` itself, and (with
`liquid_eos.compute_liquid_bulk`) the two pieces `API.py`'s
`MELTSAPI.get_property_melts_vectorized_from_assemblage()` combines per
phase in an assemblage.

General combination rule (confirmed by inspection of feldspar.c/olivine.c:
the mixing-model "G/H/S/V" macros contain no linear-in-composition
reference term -- they vanish at every pure-endmember limit, i.e. they
are genuinely *excess plus ideal-configurational* corrections only):

    solution_prop = sum_i x_i * pure_endmember_prop_i(T, P)
                    + mixing_correction(x, T, P)

applied identically to V, dV/dT, dV/dP, H, S, Cp, dCp/dT; G is then
recomputed as G = H - T*S for consistency (rather than separately summed,
since the mixing model's own gmix already legitimately equals
H_mix - T*S_mix + (P-1)*V_mix and G_pure_i = H_pure_i - T*S_pure_i by the
same identity -- summing G directly gives the same answer, done here via
H,S for symmetry with how K/alpha are derived from V).

clinopyroxene.c's own "G/H/S/V" macros are NOT built this way (they DO
carry a linear-in-composition term, plus a full cubic Taylor expansion --
see clinopyroxene.py's module docstring) -- but clinopyroxene.py's
gibbs_mixing-equivalent (`solution_thermo`) already performs the model's
own ENDMEMBERS mole-fraction-weighted pure-G subtraction internally
before returning `gmix`/`H_mix`/`S_mix`/`V_mix`, so those quantities
still vanish exactly at every pure-endmember composition and the same
combination rule above applies unmodified here too.
"""
from __future__ import annotations
import numpy as np

from .compute import compute
from . import feldspar as _feldspar
from . import olivine as _olivine
from . import clinopyroxene as _clinopyroxene


def _weighted_pure_sum(T, P, params, names, X):
    """Sum_i x_i * pure_endmember_prop_i(T,P) for each of the standard
    compute() output quantities, given (B, N) mole fractions X over the
    N endmembers `names` (in that order)."""
    pure = compute(T, P, params, names=names)
    X = np.asarray(X, dtype=np.float64)
    out = {}
    for key in ("V", "dVdT", "dVdP", "d2VdT2", "d2VdTdP", "d2VdP2",
                "Cp", "dCpdT", "G", "H", "S"):
        out[key] = np.sum(X*pure[key], axis=1)
    return out


def compute_feldspar_solution(T, P, X, solid_params) -> dict:
    """Bulk (albite, anorthite, sanidine) feldspar solid-solution
    properties.

    Parameters
    ----------
    T, P : (B,) arrays.
    X : (B, 3) mole fractions [x_albite, x_anorthite, x_sanidine]
        (should sum to 1 per row; not enforced).
    solid_params : MELTSSolidParams (e.g. from params.load_solids()).

    Returns dict of (B,) arrays: V, dVdT, dVdP, K, alpha, Cp, dCpdT, G,
    H, S, plus `activities` (B,3) and `mu` (B,3) for [albite, anorthite,
    sanidine].
    """
    T = np.asarray(T, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)

    pure_sum = _weighted_pure_sum(T, P, solid_params, _feldspar.ENDMEMBERS, X)
    mix = _feldspar.solution_thermo(X[:, 0], X[:, 1], T, P)

    V = pure_sum["V"] + mix["V_mix"]
    dVdT = pure_sum["dVdT"] + mix["dVdT_mix"]
    dVdP = pure_sum["dVdP"] + mix["dVdP_mix"]
    H = pure_sum["H"] + mix["H_mix"]
    S = pure_sum["S"] + mix["S_mix"]
    Cp = pure_sum["Cp"] + mix["Cp_mix"]
    dCpdT = pure_sum["dCpdT"]   # feldspar mixing model contributes 0 to Cp
    G = H - T*S

    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        K = np.where(dVdP != 0.0, -V/dVdP, np.inf)
        alpha = dVdT/V

    return dict(V=V, dVdT=dVdT, dVdP=dVdP, K=K, alpha=alpha,
                Cp=Cp, dCpdT=dCpdT, G=G, H=H, S=S,
                activities=mix["activities"], mu=mix["mu"],
                endmembers=_feldspar.ENDMEMBERS)


def compute_olivine_solution(T, P, X, solid_params, n_iter=60) -> dict:
    """Bulk (tephroite, fayalite, co-olivine, ni-olivine, monticellite,
    forsterite) olivine solid-solution properties.

    Parameters
    ----------
    T, P : (B,) arrays.
    X : (B, 6) mole fractions, in olivine.ENDMEMBERS order (tephroite,
        fayalite, co-olivine, ni-olivine, monticellite, forsterite;
        should sum to 1 per row).
    solid_params : MELTSSolidParams.

    Returns dict of (B,) arrays: V, dVdT, dVdP, K, alpha, Cp, dCpdT, G,
    H, S, plus `activities`/`mu` (B,6) and `s_eq` (B,4) the converged
    ordering parameters.
    """
    T = np.asarray(T, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)

    r = _olivine.x_to_r(X)   # (B,5)

    pure_sum = _weighted_pure_sum(T, P, solid_params, _olivine.ENDMEMBERS, X)
    mix = _olivine.solution_thermo(r, T, P, n_iter=n_iter)

    V = pure_sum["V"] + mix["V_mix"]
    dVdT = pure_sum["dVdT"] + mix["dVdT_mix"]
    dVdP = pure_sum["dVdP"] + mix["dVdP_mix"]
    H = pure_sum["H"] + mix["H_mix"]
    S = pure_sum["S"] + mix["S_mix"]
    Cp = pure_sum["Cp"] + mix["Cp_mix"]
    dCpdT = pure_sum["dCpdT"] + mix["dCpdT_mix"]
    G = H - T*S

    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        K = np.where(dVdP != 0.0, -V/dVdP, np.inf)
        alpha = dVdT/V

    return dict(V=V, dVdT=dVdT, dVdP=dVdP, K=K, alpha=alpha,
                Cp=Cp, dCpdT=dCpdT, G=G, H=H, S=S,
                activities=mix["activities"], mu=mix["mu"], s_eq=mix["s_eq"],
                endmembers=_olivine.ENDMEMBERS)


def compute_clinopyroxene_solution(T, P, X, solid_params, n_iter=60) -> dict:
    """Bulk (diopside, clinoenstatite, hedenbergite, alumino-buffonite,
    buffonite, essenite, jadeite) clinopyroxene solid-solution properties.

    See clinopyroxene.py's module docstring for the scope of this
    translation (the "clino=TRUE" structural-state simplification that
    applies to this build of MAGMA's clinopyroxene.c, essenite's own
    separate internal ordering, and the non-standard per-component
    pure-G-normalized Darken activity formula this phase uses).

    Parameters
    ----------
    T, P : (B,) arrays.
    X : (B, 7) mole fractions, in clinopyroxene.ENDMEMBERS order
        (diopside, clinoenstatite, hedenbergite, alumino-buffonite,
        buffonite, essenite, jadeite; should sum to 1 per row).
    solid_params : MELTSSolidParams.

    Returns dict of (B,) arrays: V, dVdT, dVdP, K, alpha, Cp, dCpdT, G,
    H, S, plus `activities`/`mu` (B,7) and `s_eq` (B,2) the converged
    main-ordering parameters.
    """
    T = np.asarray(T, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)

    r = _clinopyroxene.x_to_r(X)   # (B,6)

    pure_sum = _weighted_pure_sum(T, P, solid_params, _clinopyroxene.ENDMEMBERS, X)
    mix = _clinopyroxene.solution_thermo(r, T, P, n_iter=n_iter)

    V = pure_sum["V"] + mix["V_mix"]
    dVdT = pure_sum["dVdT"] + mix["dVdT_mix"]
    dVdP = pure_sum["dVdP"] + mix["dVdP_mix"]
    H = pure_sum["H"] + mix["H_mix"]
    S = pure_sum["S"] + mix["S_mix"]
    Cp = pure_sum["Cp"] + mix["Cp_mix"]
    dCpdT = pure_sum["dCpdT"] + mix["dCpdT_mix"]
    G = H - T*S

    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        K = np.where(dVdP != 0.0, -V/dVdP, np.inf)
        alpha = dVdT/V

    return dict(V=V, dVdT=dVdT, dVdP=dVdP, K=K, alpha=alpha,
                Cp=Cp, dCpdT=dCpdT, G=G, H=H, S=S,
                activities=mix["activities"], mu=mix["mu"], s_eq=mix["s_eq"],
                endmembers=_clinopyroxene.ENDMEMBERS)
