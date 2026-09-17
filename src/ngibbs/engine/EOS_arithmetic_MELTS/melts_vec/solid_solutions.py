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
from . import orthopyroxene as _orthopyroxene
from . import spinel as _spinel
from . import rhomsghiorso as _rhomsghiorso


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


def compute_orthopyroxene_solution(T, P, X, solid_params, n_iter=60) -> dict:
    """Bulk (diopside, clinoenstatite, hedenbergite, alumino-buffonite,
    buffonite, essenite, jadeite) orthopyroxene solid-solution properties.

    See orthopyroxene.py's module docstring for the scope of this
    translation (the "clino=FALSE" MIX-branch Taylor coefficients used
    for the bulk composition, vs. the "clino=TRUE" PURE-branch reused
    verbatim from clinopyroxene.py for the pure-endmember reference and
    essenite's own internal ordering -- both C-harness-verified this
    session, including the nonzero H0/S0/V0 constant terms that have no
    analogue in clinopyroxene.py).

    `_weighted_pure_sum` below looks up the standard-state (Berman/Vinet)
    pure-endmember EOS by name via `orthopyroxene.ENDMEMBERS` (the same
    7 names as clinopyroxene.ENDMEMBERS -- literally the same list
    object, since orthopyroxene.py imports it directly rather than
    redefining it). `sol_struct_data.json`'s `meltsSolids` table lists
    these 7 names TWICE (once in a clinopyroxene-context block, once in
    an orthopyroxene-context block) with byte-identical h/s/v/Cp/EOS
    values in both occurrences (confirmed by direct inspection) -- so
    which of the two duplicate rows `params.select()`'s "first match"
    label lookup resolves to is immaterial here.

    Parameters
    ----------
    T, P : (B,) arrays.
    X : (B, 7) mole fractions, in orthopyroxene.ENDMEMBERS order
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

    r = _orthopyroxene.x_to_r(X)   # (B,6)

    pure_sum = _weighted_pure_sum(T, P, solid_params, _orthopyroxene.ENDMEMBERS, X)
    mix = _orthopyroxene.solution_thermo(r, T, P, n_iter=n_iter)

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
                endmembers=_orthopyroxene.ENDMEMBERS)


def compute_spinel_solution(T, P, X, solid_params, n_iter=60) -> dict:
    """Bulk (chromite, hercynite, magnetite, spinel, ulvospinel) spinel
    solid-solution properties.

    See spinel.py's module docstring for the scope of this translation:
    unlike clinopyroxene.py/orthopyroxene.py, spinel.c has NO separate
    PURE-vs-MIX Taylor-coefficient reference frame (pureOrder() and the
    bulk order() share the exact same coefficients), and the internal
    ordering solve (3 site-occupancy parameters s0/s1/s2) needed a
    combined backtracking-line-search + physical-gating Newton solver
    (`spinel.solve_ordering`) beyond the plain numerically-Jacobian'd
    solve every other phase uses, to correctly handle both interior
    compositions AND the genuine physical degeneracies at 3 of the 5
    pure-endmember vertices (an entire cation type absent) -- both
    verified bit-for-bit against a standalone C harness
    (`tests/verify_spinel.c`) this session.

    Parameters
    ----------
    T, P : (B,) arrays.
    X : (B, 5) mole fractions, in spinel.ENDMEMBERS order (chromite,
        hercynite, magnetite, spinel, ulvospinel; should sum to 1 per
        row).
    solid_params : MELTSSolidParams.

    Returns dict of (B,) arrays: V, dVdT, dVdP, K, alpha, Cp, dCpdT, G,
    H, S, plus `activities`/`mu` (B,5) and `s_eq` (B,3) the converged
    main-ordering parameters.
    """
    T = np.asarray(T, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)

    r = _spinel.x_to_r(X)   # (B,4)

    pure_sum = _weighted_pure_sum(T, P, solid_params, _spinel.ENDMEMBERS, X)
    mix = _spinel.solution_thermo(r, T, P, n_iter=n_iter)

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
                endmembers=_spinel.ENDMEMBERS)


def compute_rhm_oxide_solution(T, P, X, solid_params, n_iter=100) -> dict:
    """Bulk (geikielite, hematite, ilmenite, pyrophanite, corundum)
    rhombohedral-oxide solid-solution properties.

    See rhomsghiorso.py's module docstring for the scope of this
    translation: three DECOUPLED per-component 1-D Landau ordering
    parameters (s0<->ilmenite, s1<->geikielite, s2<->pyrophanite;
    hematite/corundum have none), a short-range-order cubic spline
    (`fSRO`, currently calibrated to a constant), and the G-T*dG/dT (not
    raw H/S macro) reporting convention `hmixMsg`/`smixMsg` use for
    hmix/smix (both bulk and pure) -- all verified bit-for-bit against a
    standalone C harness (`tests/verify_rhomsghiorso.c`) this session.
    pMELTS omits the corundum endmember entirely (`sol_struct_data.json`'s
    `pMeltsSolids` table has only 4 rhm-oxide rows, no Al2O3) -- this
    general 5-endmember model reduces correctly to that case simply by
    never allocating moles to `X[:, 4]` (corundum's own r[3]=X_corundum
    then stays 0 and every corundum term vanishes cleanly); no separate
    code path exists or is needed for that calibration.

    Parameters
    ----------
    T, P : (B,) arrays.
    X : (B, 5) mole fractions, in rhomsghiorso.ENDMEMBERS order
        (geikielite, hematite, ilmenite, pyrophanite, corundum; should
        sum to 1 per row -- for a pMELTS-calibrated composition with no
        corundum, pass 0 in the last column).
    solid_params : MELTSSolidParams.

    Returns dict of (B,) arrays: V, dVdT, dVdP, K, alpha, Cp, dCpdT, G,
    H, S, plus `activities`/`mu` (B,5) and `s_eq` (B,3) the converged
    ilmenite/geikielite/pyrophanite ordering parameters.
    """
    T = np.asarray(T, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)

    r = _rhomsghiorso.x_to_r(X)   # (B,4)

    pure_sum = _weighted_pure_sum(T, P, solid_params, _rhomsghiorso.ENDMEMBERS, X)
    mix = _rhomsghiorso.solution_thermo(r, T, P, n_iter=n_iter)

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
                endmembers=_rhomsghiorso.ENDMEMBERS)
