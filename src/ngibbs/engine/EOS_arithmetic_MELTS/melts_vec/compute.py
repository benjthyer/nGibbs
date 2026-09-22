"""
Top-level batched compute() for MELTS solid endmembers: (T, P) -> V, K,
alpha, Cp, G, H, S -- the solid-phase analogue of
`EOS_arithmetic/hefesto_vec/compute.py`.

Scope of this module (deliberately): pure per-endmember standard-state
thermodynamics via the closed-form Berman/Vinet EOS and Berman/Saxena Cp
models read out of MAGMA's `gibbs.c`. It does *not* yet cover:

- the liquid volume/solution model (Kress + Ghiorso-Kress partial molar
  volumes -- next piece of work, per the plan doc),
- solid-solution mixing (feldspar.c-style W(i,j) excess-G models) --
  endmember properties feed into these, but this module stops at the
  endmember level, matching how `gibbs.c`'s own `gibbs()` function is
  itself only one layer of the full calculation,
- mineral-specific order-disorder corrections applied *after* the generic
  branch for a handful of named phases (e.g. albite Al-Si ordering) --
  gibbs.c lines ~2540 onward; not translated here.

One exception to the "generic branch only" scope above: quartz and
tridymite are NOT ordinary Berman-EOS endmembers in real MELTS -- they
have dedicated, hardcoded alpha<->beta polymorphic phase-transition
branches in gibbs.c (quartz: lines 1702-1919; tridymite: lines 1920-1993)
that swap to different reference H/S/V/EOS constants above a transition
temperature, plus (quartz only) a pressure-dependent transition
temperature and a Landau-type volumetric correction below it. Applying the
generic Berman EOS uniformly to these two, as every other named phase
correctly does, was the root cause of a previously-flagged quartz/
tridymite volume+Cp mismatch against raw MELTS output. `compute()` below
runs the generic dispatch for every endmember as before, then overrides
just the quartz/tridymite columns with `quartz_tridymite.compute_quartz`/
`compute_tridymite`, which are validated to match a standalone C harness
built from the verbatim gibbs.c source to full printed precision (see that
module's docstring).

A second special case, of a different shape: sanidine carries an
additional Al-Si order-disorder correction (gibbs.c lines 2780-2823,
`feldspar_disorder.sanidine_disorder_correction`) layered ON TOP of its
own ordinary generic Berman-EOS result (unlike quartz/tridymite, which
*replace* the generic result for their branch). This is the root cause of
a previously-flagged k-feldspar entropy/Cp mismatch in the feldspar
solid-solution mixing benchmark -- see that module's docstring for the
quantitative match. Albite's own order-disorder branch (gibbs.c lines
2543-2561, an iterative Newton-Raphson order-parameter solve in a
separate `albite()` function) is a structurally bigger lift and is NOT
translated here -- left as a follow-up, lower priority since the earlier
benchmark's plagioclase (albite/anorthite-dominated) mismatch was already
much smaller than k-feldspar's.

So: this is the generic-case solid EOS plus the two bespoke special cases
that were validated and wired in, verified against the same formulas
gibbs.c uses for every endmember -- covering, per `MELTS_Parameters/
README.md`'s survey, the large majority of the `meltsSolids` table plus
quartz/tridymite/sanidine; other named special-case branches (e.g. albite
ordering) remain out of scope.
"""
from __future__ import annotations
import numpy as np

from .constants import CP_BERMAN, CP_SAXENA, EOS_BERMAN, EOS_VINET, Pr, Tr
from .feldspar_disorder import sanidine_disorder_correction
from .params import MELTSSolidParams
from .quartz_tridymite import compute_quartz, compute_tridymite
from .solid_eos import eos_berman, eos_vinet
from .thermal import berman_ref_state, saxena_ref_state


def compute(T, P, params: MELTSSolidParams, names=None, is_pmelts: bool = False) -> dict:
    """Compute standard-state solid thermodynamics for a batch of (T, P)
    against some or all endmembers in `params`.

    Parameters
    ----------
    T, P : (B,) arrays -- temperature (K), pressure (bars).
    params : MELTSSolidParams (N endmembers).
    names : optional list of endmember labels to restrict to (else all N
        in `params`, in table order).
    is_pmelts : passed through to quartz's QUARTZ_ADJUSTMENT term (gibbs.c
        line 194) -- False (the default) matches standard/rhyolite-MELTS,
        as does the rest of this package's calibration.

    Returns
    -------
    dict of (B, N) arrays: V, dVdT, dVdP, d2VdT2, d2VdTdP, d2VdP2,
    K (isothermal bulk modulus, bars), alpha (volumetric thermal
    expansion, 1/K), Cp, dCpdT (J/mol/K, J/mol/K^2), G, H, S
    (J/mol, J/mol, J/mol/K), plus `labels` (the N endmember names, in the
    order the columns are in).
    """
    if names is not None:
        params = params.select(names)

    T = np.asarray(T, dtype=np.float64)[:, None]   # (B,1)
    P = np.asarray(P, dtype=np.float64)[:, None]   # (B,1)

    h  = params.h[None, :]          # (1,N)
    s  = params.s[None, :]
    v0 = params.v0[None, :]
    cp_type  = params.cp_type[None, :]
    eos_type = params.eos_type[None, :]
    cpc = params.cp_coeffs         # (N,8)
    eoc = params.eos_coeffs        # (N,4)

    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        # -- 1. Reference-state (Pr) H, S, Cp, dCp/dT ------------------------
        H_berman, S_berman, Cp_berman, dCpdT_berman = berman_ref_state(
            T, h, s,
            cpc[None, :, 0], cpc[None, :, 1], cpc[None, :, 2], cpc[None, :, 3],
            cp_t=cpc[None, :, 4], cp_h=cpc[None, :, 5],
            l1=cpc[None, :, 6], l2=cpc[None, :, 7],
        )
        H_saxena, S_saxena, Cp_saxena, dCpdT_saxena = saxena_ref_state(
            T, h, s,
            cpc[None, :, 0], cpc[None, :, 1], cpc[None, :, 2], cpc[None, :, 3],
            cpc[None, :, 4], cpc[None, :, 5], cpc[None, :, 6],
        )
        is_berman_cp = (cp_type == CP_BERMAN)
        H0     = np.where(is_berman_cp, H_berman, H_saxena)
        S0     = np.where(is_berman_cp, S_berman, S_saxena)
        Cp0    = np.where(is_berman_cp, Cp_berman, Cp_saxena)
        dCpdT0 = np.where(is_berman_cp, dCpdT_berman, dCpdT_saxena)
        G0 = H0 - T*S0   # gs = hs - t*ss, computed before the EOS term (gibbs.c line 2537)

        # -- 2. EOS pressure correction + absolute V(P,T) --------------------
        berm = eos_berman(
            P, T, v0,
            eoc[None, :, 0], eoc[None, :, 1], eoc[None, :, 2], eoc[None, :, 3],
        )
        # Guard Vinet's K against the zero padding used for Berman rows
        # (K appears in denominators inside the Newton solve) -- results
        # for those rows are discarded by the np.where below regardless.
        K_safe     = np.where(eos_type == EOS_VINET, eoc[None, :, 1], 1.0)
        Kp_safe    = np.where(eos_type == EOS_VINET, eoc[None, :, 2], 4.0)
        alpha_safe = np.where(eos_type == EOS_VINET, eoc[None, :, 0], 0.0)
        vin = eos_vinet(P, T, v0, alpha_safe, K_safe, Kp_safe)

        is_berman_eos = (eos_type == EOS_BERMAN)
        out = {}
        for key in ("g_add", "h_add", "s_add", "cp_add", "dcpdt_add",
                    "V", "dVdT", "dVdP", "d2VdT2", "d2VdTdP", "d2VdP2"):
            out[key] = np.where(is_berman_eos, berm[key], vin[key])

        G = G0 + out["g_add"]
        H = H0 + out["h_add"]
        S = S0 + out["s_add"]
        Cp = Cp0 + out["cp_add"]
        dCpdT = dCpdT0 + out["dcpdt_add"]
        V = out["V"]
        dVdT, dVdP = out["dVdT"], out["dVdP"]
        d2VdT2, d2VdTdP, d2VdP2 = out["d2VdT2"], out["d2VdTdP"], out["d2VdP2"]

        # -- 3. quartz/tridymite alpha<->beta special case (overrides the
        #    generic Berman-EOS result above for just those two columns;
        #    see this module's and quartz_tridymite.py's docstrings). -----
        labels_arr = np.asarray(params.labels)
        is_quartz = (labels_arr == "quartz")[None, :]
        is_tridymite = (labels_arr == "tridymite")[None, :]
        if is_quartz.any():
            qz = compute_quartz(
                T, P, h, s, v0,
                eoc[None, :, 0], eoc[None, :, 1], eoc[None, :, 2], eoc[None, :, 3],
                cpc[None, :, 0], cpc[None, :, 1], cpc[None, :, 2], cpc[None, :, 3],
                cpc[None, :, 4], cpc[None, :, 6], cpc[None, :, 7],
                is_pmelts=is_pmelts,
            )
            G, H, S = np.where(is_quartz, qz["G"], G), np.where(is_quartz, qz["H"], H), np.where(is_quartz, qz["S"], S)
            Cp, dCpdT = np.where(is_quartz, qz["Cp"], Cp), np.where(is_quartz, qz["dCpdT"], dCpdT)
            V = np.where(is_quartz, qz["V"], V)
            dVdT, dVdP = np.where(is_quartz, qz["dVdT"], dVdT), np.where(is_quartz, qz["dVdP"], dVdP)
            d2VdT2 = np.where(is_quartz, qz["d2VdT2"], d2VdT2)
            d2VdTdP = np.where(is_quartz, qz["d2VdTdP"], d2VdTdP)
            d2VdP2 = np.where(is_quartz, qz["d2VdP2"], d2VdP2)
        if is_tridymite.any():
            tdy = compute_tridymite(
                T, P, h, s, v0,
                eoc[None, :, 0], eoc[None, :, 1], eoc[None, :, 2], eoc[None, :, 3],
                cpc[None, :, 0], cpc[None, :, 1], cpc[None, :, 2], cpc[None, :, 3],
                cpc[None, :, 4], cpc[None, :, 6], cpc[None, :, 7],
            )
            G, H, S = np.where(is_tridymite, tdy["G"], G), np.where(is_tridymite, tdy["H"], H), np.where(is_tridymite, tdy["S"], S)
            Cp, dCpdT = np.where(is_tridymite, tdy["Cp"], Cp), np.where(is_tridymite, tdy["dCpdT"], dCpdT)
            V = np.where(is_tridymite, tdy["V"], V)
            dVdT, dVdP = np.where(is_tridymite, tdy["dVdT"], dVdT), np.where(is_tridymite, tdy["dVdP"], dVdP)
            d2VdT2 = np.where(is_tridymite, tdy["d2VdT2"], d2VdT2)
            d2VdTdP = np.where(is_tridymite, tdy["d2VdTdP"], d2VdTdP)
            d2VdP2 = np.where(is_tridymite, tdy["d2VdP2"], d2VdP2)

        # -- 4. sanidine Al-Si order-disorder correction (additive on top
        #    of its own generic Berman-EOS result -- see this module's and
        #    feldspar_disorder.py's docstrings). ---------------------------
        is_sanidine = (labels_arr == "sanidine")[None, :]
        if is_sanidine.any():
            dis = sanidine_disorder_correction(T, P)
            G = G + np.where(is_sanidine, dis["dG"], 0.0)
            H = H + np.where(is_sanidine, dis["dH"], 0.0)
            S = S + np.where(is_sanidine, dis["dS"], 0.0)
            Cp = Cp + np.where(is_sanidine, dis["dCp"], 0.0)
            dCpdT = dCpdT + np.where(is_sanidine, dis["ddCpdT"], 0.0)
            V = V + np.where(is_sanidine, dis["dV"], 0.0)
            dVdT = dVdT + np.where(is_sanidine, dis["ddVdT"], 0.0)
            d2VdT2 = d2VdT2 + np.where(is_sanidine, dis["dd2VdT2"], 0.0)
            # dVdP, d2VdTdP, d2VdP2 untouched -- gibbs.c's sanidine branch
            # never adjusts them.

        out["dVdT"], out["dVdP"] = dVdT, dVdP
        out["d2VdT2"], out["d2VdTdP"], out["d2VdP2"] = d2VdT2, d2VdTdP, d2VdP2

        # Isothermal bulk modulus K_T = -V (dP/dV)_T = V / (-dV/dP); and
        # volumetric thermal expansion alpha = (1/V)(dV/dT)_P. Bars for K
        # (same pressure unit as P throughout this module).
        K_mod = np.where(out["dVdP"] != 0.0, -V/out["dVdP"], np.inf)
        alpha_exp = out["dVdT"]/V

    return dict(
        labels=params.labels,
        V=V, dVdT=out["dVdT"], dVdP=out["dVdP"],
        d2VdT2=out["d2VdT2"], d2VdTdP=out["d2VdTdP"], d2VdP2=out["d2VdP2"],
        K=K_mod, alpha=alpha_exp,
        Cp=Cp, dCpdT=dCpdT,
        G=G, H=H, S=S,
    )
