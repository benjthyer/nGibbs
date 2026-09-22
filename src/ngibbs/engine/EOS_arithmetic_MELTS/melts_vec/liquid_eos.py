"""
Vectorized MELTS liquid EOS -- per-component reference-state thermodynamics
(H, S, Cp via a solid-Berman-integration-to-Tfusion + constant-Cp-above-Tfusion
scheme) and the Kress (Lange-type) volume EOS, translated verbatim from
`sources/gibbs.c`'s generic liquid branch (the `else` at line 1281, "all
other components, MELTS, pMELTS, XMELTS", confirmed active for the default
`MODE__MELTS` / rhyolite-MELTS calculation mode -- the alternate Ghiorso
volume EOS at line 1358 is gated to `MODE_xMELTS` only and is not
implemented here).

Scope
-----
This module gives EXACT per-component liquid properties (V and its P,T
derivatives; G, H, S, Cp at the component's own composition, i.e. as a
pure liquid). Bulk multicomponent liquid mixing in `compute_liquid_bulk()`
below is:

    V_bulk  = sum_i x_i * V_i(T,P)          (exact: MAGMA's own W(i,j)
                                              volume-of-mixing parameters
                                              are all zero -- confirmed
                                              directly, see below)
    Cp_bulk = sum_i x_i * Cp_i(T,P)         (exact -- cpmixLiq_v34 returns
                                              0 unconditionally)
    H_bulk  = sum_i x_i * H_i(T,P) + Hex
    S_bulk  = sum_i x_i * S_i(T,P) - R*sum_i x_i*log(x_i) + Sex

where Hex/Sex is the NON-IDEAL (regular-solution) correction from
`liquid_nonideal.py` -- see that module's docstring for the full
derivation. This closes the gap this docstring used to flag as
unimplemented: MELTS's real liquid model has a non-ideal excess Gibbs
energy of mixing parameterized by the W(i,j) enthalpy table in
`param_struct_data.h` (confirmed nonzero for enthalpy; entropy and volume
interaction terms are confirmed identically zero for every one of the 171
component pairs in this calibration, so Vex=0 and the only excess terms
are Hex, from the regular-solution enthalpy sum, and a small additional
Sex from H2O's own extra ideal-mixing term -- both closed-form, no Newton
solve needed). This is still not a full liquid ACTIVITY/speciation model
(no chemical potentials/activities are exposed here) -- flagged clearly
rather than silently claimed.
"""
from __future__ import annotations
import numpy as np

from .constants import Rgas, Pr, Trl
from .thermal import berman_ref_state
from .liquid_nonideal import load_wij_matrix, liquid_nonideal_correction


def kress_component(T, P, v_liq, dvdt, dvdp, d2vdtp, d2vdp2,
                     t_fusion, s_fusion, cp_liquid,
                     ref_h, ref_s, ref_k0, ref_k1, ref_k2, ref_k3,
                     ref_cp_t=0.0, ref_cp_h=0.0, ref_l1=0.0, ref_l2=0.0):
    """Per-component liquid G, H, S, Cp, V (and P,T derivatives of V),
    verbatim from gibbs.c lines 1281-1354 (t > tglass branch only -- all
    19 `meltsLiquid` entries have Tglass = 0, so T > Tglass always holds
    for any physical T > 0 and the glass-transition sub-branch, lines
    1292-1326, is dead code for this table and is not translated).

    All arguments broadcast against each other in the usual numpy way.

    ref_cp_t/ref_cp_h/ref_l1/ref_l2 are the solid reference phase's
    optional order-disorder ("lambda transition") Berman correction --
    zero for most components, but NOT for Fe2O3, KAlSiO4 and Ca3(PO4)2
    (see gibbs.c line 1279: `gibbs(liquid->tfus, pr, "dummy", phase, ...,
    fusion)` computes the fusion state via the *general* solid gibbs()
    dispatch, which includes this correction whenever the endmember's own
    Berman.Tt != 0 -- it must be passed through here or those three
    components' fusion enthalpy/entropy (and everything built on top of
    it) comes out wrong).
    """
    T = np.asarray(T, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)

    # 1. "fusion" state: integrate the SOLID (crystalline) reference
    #    phase's Berman Cp from Tr up to Tfusion at P = Pr, then add the
    #    enthalpy/entropy of fusion -- gibbs.c lines 1282-1286.
    fusion_h, fusion_s, _, _ = berman_ref_state(
        t_fusion, ref_h, ref_s, ref_k0, ref_k1, ref_k2, ref_k3,
        cp_t=ref_cp_t, cp_h=ref_cp_h, l1=ref_l1, l2=ref_l2)
    fusion_h = fusion_h + t_fusion*s_fusion
    fusion_s = fusion_s + s_fusion

    # 2. Liquid reference state (P = Pr) via constant liquid Cp from
    #    Tfusion up to T -- gibbs.c lines 1287-1291.
    hl = fusion_h + cp_liquid*(T - t_fusion)
    sl = fusion_s + cp_liquid*np.log(T/t_fusion)
    cpl = np.broadcast_to(cp_liquid, np.broadcast_shapes(T.shape, np.shape(cp_liquid))).astype(np.float64)
    dcpldt = np.zeros_like(cpl)

    # 3. Kress volume EOS + its pressure contribution to G, H, S --
    #    gibbs.c lines 1334-1353, exactly parallel in structure to the
    #    solid Berman EOS's g_add/h_add/s_add pattern in solid_eos.py.
    dT = T - Trl
    dP = P - Pr

    press_integral = (
        (v_liq + dvdt*dT)*dP
        + 0.5*(dvdp + dT*d2vdtp)*(P**2 - Pr**2)
        - (dvdp + dT*d2vdtp)*Pr*dP
        + d2vdp2*((P**3 - Pr**3)/6.0 - Pr*(P**2 - Pr**2)/2.0 + Pr**2*dP/2.0)
    )
    s_press_corr = -(dvdt*dP + 0.5*d2vdtp*dP*dP)

    G = hl - T*sl + press_integral
    H = hl + press_integral + T*s_press_corr
    S = sl + s_press_corr
    Cp = cpl          # Kress model has no d^2V/dT^2 term -> no pressure
    dCpdT = dcpldt     # correction to Cp (matches gibbs.c, which doesn't
                        # touch cpl/dcpldt after the Kress block).

    V = v_liq + dvdt*dT + (dvdp + d2vdtp*dT)*dP + d2vdp2*(0.5*P**2 - Pr*dP)
    dVdT = dvdt + d2vdtp*dP
    dVdP = dvdp + d2vdtp*dT + d2vdp2*dP
    d2VdT2 = np.zeros_like(V)
    d2VdTdP = np.broadcast_to(d2vdtp, V.shape).astype(np.float64)
    d2VdP2 = np.broadcast_to(d2vdp2, V.shape).astype(np.float64)

    return dict(G=G, H=H, S=S, Cp=Cp, dCpdT=dCpdT,
                V=V, dVdT=dVdT, dVdP=dVdP,
                d2VdT2=d2VdT2, d2VdTdP=d2VdTdP, d2VdP2=d2VdP2)


def compute_liquid_components(T, P, params, names=None) -> dict:
    """Per-component (B, N) liquid properties -- the liquid analogue of
    `melts_vec.compute.compute()` for solid endmembers.

    Parameters
    ----------
    T, P : (B,) arrays.
    params : MELTSLiquidParams (see params.py's load_liquid()).
    names : optional list of component labels to restrict to.
    """
    if names is not None:
        params = params.select(names)

    T = np.asarray(T, dtype=np.float64)[:, None]
    P = np.asarray(P, dtype=np.float64)[:, None]

    out = kress_component(
        T, P,
        params.v_liq[None, :], params.dvdt[None, :], params.dvdp[None, :],
        params.d2vdtp[None, :], params.d2vdp2[None, :],
        params.t_fusion[None, :], params.s_fusion[None, :], params.cp_liquid[None, :],
        params.ref_h[None, :], params.ref_s[None, :],
        params.ref_k0[None, :], params.ref_k1[None, :],
        params.ref_k2[None, :], params.ref_k3[None, :],
        ref_cp_t=params.ref_cp_t[None, :], ref_cp_h=params.ref_cp_h[None, :],
        ref_l1=params.ref_l1[None, :], ref_l2=params.ref_l2[None, :],
    )
    out["labels"] = params.labels
    return out


def compute_liquid_bulk(T, P, X, params, names=None, apply_nonideal: bool = True,
                         wij_matrix=None, h2o_index=None) -> dict:
    """Bulk (bulk-composition-weighted) liquid properties: ideal mixing for
    V/Cp (exact, see module docstring) plus, by default, the non-ideal
    (regular-solution) excess G/H/S correction from `liquid_nonideal.py`.

    Parameters
    ----------
    T, P : (B,) arrays.
    X : (B, N) array of mole fractions (rows should sum to 1; not enforced
        here -- caller's responsibility, matching how compute() for solid
        solutions expects normalized composition input).
    params, names : as in compute_liquid_components.
    apply_nonideal : if False, reproduces the OLD ideal-only behavior (no
        Hex/Sex correction) -- kept for benchmark/regression comparisons
        against the pre-fix baseline; always True for production use.
    wij_matrix : optional pre-built (N, N) W(i,j) enthalpy matrix (from
        `liquid_nonideal.load_wij_matrix`); built fresh from `params.labels`
        if not supplied (cheap, but callers evaluating this in a tight loop
        may want to build it once and pass it in).
    h2o_index : optional explicit index of H2O in `params.labels`; looked
        up via `params.label_index["H2O"]` if not supplied.

    Returns a dict of (B,) arrays: V, dVdT, dVdP, K, alpha, Cp, dCpdT,
    G, H, S.
    """
    if names is not None:
        params = params.select(names)
    X = np.asarray(X, dtype=np.float64)

    comp = compute_liquid_components(T, P, params)
    # comp arrays are (B, N); weight by mole fraction and sum over N.
    V = np.sum(X*comp["V"], axis=1)
    dVdT = np.sum(X*comp["dVdT"], axis=1)
    dVdP = np.sum(X*comp["dVdP"], axis=1)
    d2VdT2 = np.sum(X*comp["d2VdT2"], axis=1)
    d2VdTdP = np.sum(X*comp["d2VdTdP"], axis=1)
    d2VdP2 = np.sum(X*comp["d2VdP2"], axis=1)
    H = np.sum(X*comp["H"], axis=1)
    Cp = np.sum(X*comp["Cp"], axis=1)
    dCpdT = np.sum(X*comp["dCpdT"], axis=1)

    with np.errstate(divide='ignore', invalid='ignore'):
        Xsafe = np.where(X > 0.0, X, 1.0)
        S_config = -Rgas*np.sum(np.where(X > 0.0, X*np.log(Xsafe), 0.0), axis=1)
    S = np.sum(X*comp["S"], axis=1) + S_config

    T1 = np.asarray(T, dtype=np.float64)

    if apply_nonideal:
        if wij_matrix is None:
            wij_matrix = load_wij_matrix(params.labels)
        if h2o_index is None:
            h2o_index = params.label_index["H2O"]
        corr = liquid_nonideal_correction(X, T1, wij_matrix, h2o_index)
        H = H + corr["dH"]
        S = S + corr["dS"]

    G = H - T1*S

    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        K = np.where(dVdP != 0.0, -V/dVdP, np.inf)
        alpha = dVdT/V

    return dict(V=V, dVdT=dVdT, dVdP=dVdP, d2VdT2=d2VdT2, d2VdTdP=d2VdTdP, d2VdP2=d2VdP2,
                K=K, alpha=alpha, Cp=Cp, dCpdT=dCpdT, G=G, H=H, S=S)
