"""
Reference-state (P = Pr) heat capacity, enthalpy and entropy for MELTS solid
endmembers -- vectorized translation of the CP_BERMAN / CP_SAXENA branches in
`sources/gibbs.c` (the `else` "General equations from Berman (1988) or
Saxena (1993)" block, lines ~2474-2535 of the MAGMA source read this
session).

These are pure closed-form polynomial evaluations -- no root-finding.  The
pressure-dependent correction on top of these (which also perturbs G, H, S,
Cp once V(P,T) enters the picture) lives in `solid_eos.py`; call that
*after* this module, exactly as `gibbs()` does (compute hs/ss/cps at Pr
first, then let `intEOSsolid()` add the P-V terms in place).

All inputs/outputs are plain numpy arrays and broadcast against each other
in the usual numpy way -- e.g. T can be (B,) and the per-endmember
coefficients (B,) or scalar, or T can be (B,1) against coefficients (N,)
to get a (B,N) result; the caller (`compute.py`) decides the batch layout.
"""
from __future__ import annotations
import numpy as np

from .constants import Tr


def berman_cp(T, k0, k1, k2, k3):
    """Cp(T) and dCp/dT at P = Pr, Berman (1988) polynomial form.

    Cp = k0 + k1/sqrt(T) + k2/T^2 + k3/T^3
    """
    T = np.asarray(T, dtype=np.float64)
    cp    = k0 + k1/np.sqrt(T) + k2/T**2 + k3/T**3
    dcpdt = -0.5*k1*T**-1.5 - 2.0*k2/T**3 - 3.0*k3/T**4
    return cp, dcpdt


def berman_lambda_terms(T, cp_t, cp_h, l1, l2):
    """Optional order-disorder / first-order-transition correction.

    Only applied where cp_t != 0 (most endmembers have cp_t == 0, i.e. no
    transition, and get zero contribution here -- matching gibbs.c's
    ``if (cp_t != 0.0) { ... }`` guard exactly).  Two sub-branches by
    T vs cp_t, translated verbatim from gibbs.c lines 2498-2517.

    Returns (dH, dS, dCp, dCpdT) additive corrections -- add these to the
    plain Berman Cp/H/S/dCpdT before the EOS pressure term.
    """
    T  = np.asarray(T, dtype=np.float64)
    k0 = np.zeros_like(T)  # broadcast helper, unused directly

    has_lambda = cp_t != 0.0
    hot        = T > cp_t   # only meaningful where has_lambda

    tr2, tr3, tr4 = Tr**2, Tr**3, Tr**4
    cp_t_safe = np.where(has_lambda, cp_t, 1.0)   # avoid div-by-zero; masked out below

    dH_hot = (cp_h
              + 0.5*l1*l1*(cp_t**2 - tr2)
              + (2.0/3.0)*l1*l2*(cp_t**3 - tr3)
              + 0.25*l2*l2*(cp_t**4 - tr4))
    dS_hot = (cp_h/cp_t_safe
              + l1*l1*(cp_t - Tr)
              + l1*l2*(cp_t**2 - tr2)
              + (1.0/3.0)*l2*l2*(cp_t**3 - tr3))

    dH_cold = (0.5*l1*l1*(T**2 - tr2)
               + (2.0/3.0)*l1*l2*(T**3 - tr3)
               + 0.25*l2*l2*(T**4 - tr4))
    dS_cold = (l1*l1*(T - Tr)
               + l1*l2*(T**2 - tr2)
               + (1.0/3.0)*l2*l2*(T**3 - tr3))
    dCp_cold    = T*(l1 + l2*T)**2
    dCpdT_cold  = (l1 + l2*T)**2 + 2.0*T*(l1 + l2*T)*l2

    dH  = np.where(has_lambda, np.where(hot, dH_hot, dH_cold), 0.0)
    dS  = np.where(has_lambda, np.where(hot, dS_hot, dS_cold), 0.0)
    # the "hot" branch in gibbs.c never touches cps/dcpsdt -- only the
    # "cold" (T <= cp_t) branch does.
    dCp    = np.where(has_lambda & ~hot, dCp_cold, 0.0)
    dCpdT  = np.where(has_lambda & ~hot, dCpdT_cold, 0.0)
    return dH, dS, dCp, dCpdT


def berman_ref_state(T, h_ref, s_ref, k0, k1, k2, k3,
                      cp_t=0.0, cp_h=0.0, l1=0.0, l2=0.0):
    """Full CP_BERMAN reference-state (P = Pr) H, S, Cp, dCp/dT.

    Translated from gibbs.c lines 2475-2518 (``hs = phase->h; ss = phase->s;``
    through the end of the CP_BERMAN branch, including the optional lambda
    transition).
    """
    T = np.asarray(T, dtype=np.float64)
    cp, dcpdt = berman_cp(T, k0, k1, k2, k3)

    H = (h_ref + k0*(T - Tr) + 2.0*k1*(np.sqrt(T) - np.sqrt(Tr))
         - k2*(1.0/T - 1.0/Tr) - 0.5*k3*(1.0/T**2 - 1.0/Tr**2))
    S = (s_ref + k0*np.log(T/Tr)
         - 2.0*k1*(1.0/np.sqrt(T) - 1.0/np.sqrt(Tr))
         - 0.5*k2*(1.0/T**2 - 1.0/Tr**2)
         - (1.0/3.0)*k3*(1.0/T**3 - 1.0/Tr**3))

    dH, dS, dCp, dCpdT = berman_lambda_terms(T, cp_t, cp_h, l1, l2)
    return H + dH, S + dS, cp + dCp, dcpdt + dCpdT


def saxena_ref_state(T, h_ref, s_ref, a, b, c, d, e, g, h):
    """CP_SAXENA reference-state (P = Pr) H, S, Cp, dCp/dT.

    Translated from gibbs.c lines 2520-2534 (Saxena 1993 form). Not used by
    any endmember in the default `meltsSolids` (rhyolite-MELTS) table, but
    present in `xMeltsSolids` / `meltsFluidSolids` -- implemented here for
    completeness with those modes.
    """
    T = np.asarray(T, dtype=np.float64)
    cp     = a + b*T + c/T**2 + d*T**2 + e/T**3 + g/np.sqrt(T) + h/T
    dcpdt  = b - 2.0*c/T**3 + 2.0*d*T - 3.0*e/T**4 - 0.5*g/T**1.5 - h/T**2

    H = (h_ref + a*(T - Tr) + (b/2.0)*(T**2 - Tr**2) - c*(1.0/T - 1.0/Tr)
         + (d/3.0)*(T**3 - Tr**3) - (e/2.0)*(1.0/T**2 - 1.0/Tr**2)
         + 2.0*g*(np.sqrt(T) - np.sqrt(Tr)) + h*np.log(T/Tr))
    S = (s_ref + a*np.log(T/Tr) + b*(T - Tr) - (c/2.0)*(1.0/T**2 - 1.0/Tr**2)
         + (d/2.0)*(T**2 - Tr**2) - (e/3.0)*(1.0/T**3 - 1.0/Tr**3)
         - 2.0*g*(1.0/np.sqrt(T) - 1.0/np.sqrt(Tr)) - h*(1.0/T - 1.0/Tr))
    return H, S, cp, dcpdt
