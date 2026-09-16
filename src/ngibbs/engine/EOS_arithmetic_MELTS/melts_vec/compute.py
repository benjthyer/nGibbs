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

So: this is the generic-case solid EOS, verified against the same formulas
gibbs.c uses for every endmember that doesn't hit one of those special
named branches -- which, per `MELTS_Parameters/README.md`'s survey, is the
large majority of the `meltsSolids` table.
"""
from __future__ import annotations
import numpy as np

from .constants import CP_BERMAN, CP_SAXENA, EOS_BERMAN, EOS_VINET, Pr, Tr
from .params import MELTSSolidParams
from .solid_eos import eos_berman, eos_vinet
from .thermal import berman_ref_state, saxena_ref_state


def compute(T, P, params: MELTSSolidParams, names=None) -> dict:
    """Compute standard-state solid thermodynamics for a batch of (T, P)
    against some or all endmembers in `params`.

    Parameters
    ----------
    T, P : (B,) arrays -- temperature (K), pressure (bars).
    params : MELTSSolidParams (N endmembers).
    names : optional list of endmember labels to restrict to (else all N
        in `params`, in table order).

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
