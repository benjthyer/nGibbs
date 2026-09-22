"""
MELTS's quartz/tridymite alpha<->beta phase-transition special case,
translated verbatim from `sources/gibbs.c`'s two dedicated named branches
(``strcmp(name, "quartz")``, lines 1702-1919; ``strcmp(name, "tridymite")``,
lines 1920-1993) -- NOT the generic EOS_BERMAN path every other solid
endmember goes through (`compute.py`/`solid_eos.eos_berman`).

Why this needed its own module
-------------------------------
Unlike every other Berman-EOS solid this package translates, quartz and
tridymite are alpha<->beta polymorphic transitions where real MELTS swaps
to an ENTIRELY DIFFERENT, hardcoded set of reference thermodynamic
constants (H, S, and the full Berman volumetric EOS v0/v1/v2/v3/v4) once T
crosses a transition temperature -- not just a smooth Cp/H/S lambda
correction layered on ONE fixed set of Berman EOS parameters (which is what
`sol_struct_data.json`'s single calibration row per phase, and this
package's existing `thermal.berman_ref_state` lambda-transition term,
already handle correctly). Quartz's transition temperature is ALSO
pressure-dependent (`cp_t = Tt + 0.0237*(P-1)`, Berman 1988's dT/dP for the
alpha-beta quartz boundary) and quartz alone carries an additional
"lambda volume" pressure/temperature-derivative correction below the
transition, entirely absent from the generic path.

This is the root cause identified for the raw-MELTS-output quartz/tridymite
volume+Cp discrepancy (~1-3% rho/V, up to ~14% max Cp for quartz)
previously flagged as unresolved: the generic Berman EOS path this package
used before this module existed applied ONE (alpha, Tr=298K-calibrated)
set of v0/v1-v4 uniformly at every temperature, including deep into the
beta-quartz/beta-tridymite stability field (~700-1050 C, where essentially
all of that benchmark's sampled rows fall) where real MELTS has already
swapped to a different, hardcoded reference volume and compressibility.

Beta-form constants (hs, ss, v0, v1, v2, v3, v4 above the transition) are
NOT extractable from `sol_struct_data.h`/`.json` at all -- they are literal
numeric constants hardcoded directly in this gibbs.c branch, not read from
any calibration table. They are hardcoded here as module-level constants,
extracted verbatim from the same source lines.

Verification: no C harness ships in this package's `tests/` for this
module yet (unlike every other translated phase) -- pending a follow-up
pass to fold the standalone driver this translation was checked against
(constructed this session, sed-extracted quartz/tridymite blocks + a
minimal `intEOSsolid` EOS_BERMAN-branch driver) into `tests/` alongside the
existing C harnesses. Every (T, P) point checked against that driver
matched to floating-point noise across both the alpha and beta branches,
including a P-dependent transition-temperature crossover for quartz.

Units: P in bars, T in K, V in J/bar -- same convention as the rest of
melts_vec (see constants.py).
"""
from __future__ import annotations
import numpy as np

from .constants import Tr, Pr
from .solid_eos import eos_berman

# --------------------------------------------------------------------------- #
# Beta-form constants, hardcoded verbatim in gibbs.c's quartz/tridymite
# branches (NOT present in sol_struct_data.h -- there is no calibration-
# table row for the high-T polymorph at all).
# --------------------------------------------------------------------------- #
_QUARTZ_BETA_H = -908627.0       # J/mol
_QUARTZ_BETA_S = 44.207          # J/(mol K)
_QUARTZ_BETA_V0 = 2.370          # J/bar
_QUARTZ_BETA_V1 = -1.238e-6
_QUARTZ_BETA_V2 = 7.087e-13
_QUARTZ_BETA_V3 = 0.0
_QUARTZ_BETA_V4 = 0.0
_QUARTZ_DTDP = 0.0237            # Berman (1988) dTt/dP for the alpha-beta
                                  # quartz transition (K/bar)

_TRIDYMITE_BETA_H = -907045.0    # J/mol
_TRIDYMITE_BETA_S = 45.524       # J/(mol K)
_TRIDYMITE_BETA_V0 = 2.737       # J/bar
_TRIDYMITE_BETA_V1 = -0.740e-6
_TRIDYMITE_BETA_V2 = 3.735e-12
_TRIDYMITE_BETA_V3 = 4.829e-6
_TRIDYMITE_BETA_V4 = 0.0

# QUARTZ_ADJUSTMENT: an empirical G/H offset applied to quartz only,
# gibbs.c line 194 -- active (-1291.0 J/mol) whenever RHYOLITE_ADJUSTMENTS
# is compiled in AND calculationMode != MODE_pMELTS; 0.0 for pMELTS. This
# package targets standard MELTS (rhyolite-MELTS) calibration throughout
# (per the rest of this package's own docstrings), so the default here
# matches that -- pass is_pmelts=True for a pMELTS-calibrated calculation.
_QUARTZ_ADJUSTMENT_MELTS = -1291.0
_WETTING_ANGLE_CORR = 0.0        # gibbs.c line 201 -- always 0 in this build


def _berman_cp_hs(T, k0, k1, k2, k3, h_ref, s_ref):
    """The plain (non-lambda) Berman Cp integral from Tr to T added to a
    fixed (h_ref, s_ref) pair -- gibbs.c's repeated
    ``hs = hs + k0*(t-tr) + 2*k1*(sqrt(t)-sqrt(tr)) - k2*(1/t-1/tr) -
    0.5*k3*(1/t^2-1/tr^2)`` / matching ss pattern, used identically in both
    the alpha and beta branches of both quartz and tridymite (and NOT the
    same as `thermal.berman_ref_state`, which additionally folds in the
    lambda-transition term this module handles separately below)."""
    hs = h_ref + k0*(T - Tr) + 2.0*k1*(np.sqrt(T) - np.sqrt(Tr)) \
        - k2*(1.0/T - 1.0/Tr) - 0.5*k3*(1.0/T**2 - 1.0/Tr**2)
    ss = s_ref + k0*np.log(T/Tr) - 2.0*k1*(1.0/np.sqrt(T) - 1.0/np.sqrt(Tr)) \
        - 0.5*k2*(1.0/T**2 - 1.0/Tr**2) - (1.0/3.0)*k3*(1.0/T**3 - 1.0/Tr**3)
    return hs, ss


def compute_quartz(T, P, h, s, v0, v1, v2, v3, v4, k0, k1, k2, k3, Tt, l1, l2,
                    is_pmelts=False):
    """Quartz G/H/S/Cp/V(+derivatives), verbatim from gibbs.c's quartz
    branch (lines 1702-1919). All inputs broadcast; h/s/v0/v1-v4/k0-k3/Tt/
    l1/l2 are the ALPHA-quartz calibration values (`sol_struct_data.json`'s
    quartz row).
    """
    T = np.asarray(T, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    h, s, v0, v1, v2, v3, v4 = np.broadcast_arrays(h, s, v0, v1, v2, v3, v4)
    k0, k1, k2, k3, Tt, l1, l2 = np.broadcast_arrays(k0, k1, k2, k3, Tt, l1, l2)
    shape = np.broadcast_shapes(T.shape, P.shape, h.shape)
    T = np.broadcast_to(T, shape).astype(np.float64)
    P = np.broadcast_to(P, shape).astype(np.float64)
    h, s, v0, v1, v2, v3, v4 = (np.broadcast_to(a, shape).astype(np.float64)
                                 for a in (h, s, v0, v1, v2, v3, v4))
    k0, k1, k2, k3, Tt, l1, l2 = (np.broadcast_to(a, shape).astype(np.float64)
                                   for a in (k0, k1, k2, k3, Tt, l1, l2))

    cp_t = Tt + _QUARTZ_DTDP*(P - 1.0)
    is_beta = T > cp_t

    cps = k0 + k1/np.sqrt(T) + k2/T**2 + k3/T**3
    dcpsdt = -0.5*k1/T**1.5 - 2.0*k2/T**3 - 3.0*k3/T**4

    # -- beta branch: hardcoded reference constants + swapped EOS params --
    hs_beta, ss_beta = _berman_cp_hs(T, k0, k1, k2, k3, _QUARTZ_BETA_H, _QUARTZ_BETA_S)
    v0_beta = np.full(shape, _QUARTZ_BETA_V0)
    v1_beta = np.full(shape, _QUARTZ_BETA_V1)
    v2_beta = np.full(shape, _QUARTZ_BETA_V2)
    v3_beta = np.full(shape, _QUARTZ_BETA_V3)
    v4_beta = np.full(shape, _QUARTZ_BETA_V4)

    # -- alpha branch: calibration-table reference + Landau lambda term ----
    hs_alpha, ss_alpha = _berman_cp_hs(T, k0, k1, k2, k3, h, s)
    delt = Tt - cp_t   # = -DTDP*(P-1); 0 at P=1

    x1 = l1*l1*delt + 2.0*l1*l2*delt**2 + l2*l2*delt**3
    x2 = l1*l1 + 4.0*l1*l2*delt + 3.0*l2*l2*delt**2
    x3 = 2.0*l1*l2 + 3.0*l2*l2*delt
    x4 = l2*l2

    DdeltDp = -_QUARTZ_DTDP * np.ones(shape)
    Dx1Dp = l1*l1*DdeltDp + 4.0*l1*l2*delt*DdeltDp + 3.0*l2*l2*delt**2*DdeltDp
    D2x1Dp2 = 4.0*l1*l2*DdeltDp**2 + 6.0*l2*l2*delt*DdeltDp**2
    D3x1Dp3 = 6.0*l2*l2*DdeltDp**3
    Dx2Dp = 4.0*l1*l2*DdeltDp + 6.0*l2*l2*delt*DdeltDp
    D2x2Dp2 = 6.0*l2*l2*DdeltDp**2
    Dx3Dp = 3.0*l2*l2*DdeltDp
    Dx4Dp = np.zeros(shape)
    D2x3Dp2 = np.zeros(shape)
    D3x2Dp3 = np.zeros(shape)
    D2x4Dp2 = np.zeros(shape)
    D3x3Dp3 = np.zeros(shape)
    D3x4Dp3 = np.zeros(shape)

    ref = 373.0 - delt   # gibbs.c's literal "373.0" (not tr) for quartz
    hs_alpha = hs_alpha + x1*(T - ref) + x2*(T**2 - ref**2)/2.0 \
        + x3*(T**3 - ref**3)/3.0 + x4*(T**4 - ref**4)/4.0
    ss_alpha = ss_alpha + x1*(np.log(T) - np.log(ref)) + x2*(T - ref) \
        + x3*(T**2 - ref**2)/2.0 + x4*(T**3 - ref**3)/3.0

    lambdaV = (Dx1Dp*(T - ref) + Dx2Dp*(T**2 - ref**2)/2.0
               + Dx3Dp*(T**3 - ref**3)/3.0 + Dx4Dp*(T**4 - ref**4)/4.0
               - T*(Dx1Dp*(np.log(T) - np.log(ref)) + Dx2Dp*(T - ref)
                    + Dx3Dp*(T**2 - ref**2)/2.0 + Dx4Dp*(T**3 - ref**3)/3.0)
               + x1*DdeltDp + x2*ref*DdeltDp + x3*ref**2*DdeltDp + x4*ref**3*DdeltDp
               - T*(x1*DdeltDp/ref + x2*DdeltDp + x3*ref*DdeltDp + x4*ref**2*DdeltDp))

    lambdadVdt = (Dx1Dp + Dx2Dp*T + Dx3Dp*T**2 + Dx4Dp*T**3
                  - (Dx1Dp*(np.log(T) - np.log(ref)) + Dx2Dp*(T - ref)
                     + Dx3Dp*(T**2 - ref**2)/2.0 + Dx4Dp*(T**3 - ref**3)/3.0)
                  - T*(Dx1Dp/T + Dx2Dp + Dx3Dp*T + Dx4Dp*T**2)
                  - (x1*DdeltDp/ref + x2*DdeltDp + x3*ref*DdeltDp + x4*ref**2*DdeltDp))

    lambdadVdp = (D2x1Dp2*(T - ref) + Dx1Dp*DdeltDp
                  + D2x2Dp2*(T**2 - ref**2)/2.0 + Dx2Dp*ref*DdeltDp
                  + D2x3Dp2*(T**3 - ref**3)/3.0 + Dx3Dp*ref**2*DdeltDp
                  + D2x4Dp2*(T**4 - ref**4)/4.0 + Dx4Dp*ref**3*DdeltDp
                  - T*(D2x1Dp2*(np.log(T) - np.log(ref)) + Dx1Dp*DdeltDp/ref
                       + D2x2Dp2*(T - ref) + Dx2Dp*DdeltDp
                       + D2x3Dp2*(T**2 - ref**2)/2.0 + Dx3Dp*ref*DdeltDp
                       + D2x4Dp2*(T**3 - ref**3)/3.0 + Dx4Dp*ref**2*DdeltDp)
                  + Dx1Dp*DdeltDp
                  + Dx2Dp*ref*DdeltDp - x2*DdeltDp**2
                  + Dx3Dp*ref**2*DdeltDp - x3*2.0*ref*DdeltDp**2
                  + Dx4Dp*ref**3*DdeltDp - x4*3.0*ref**2*DdeltDp**2
                  - T*(Dx1Dp*DdeltDp/ref + x1*DdeltDp**2/ref**2
                       + Dx2Dp*DdeltDp
                       + Dx3Dp*ref*DdeltDp - x3*DdeltDp**2
                       + Dx4Dp*ref**2*DdeltDp - x4*2.0*ref*DdeltDp**2))

    lambdad2Vdt2 = (Dx2Dp + 2.0*Dx3Dp*T + 3.0*Dx4Dp*T**2
                    - 2.0*(Dx1Dp/T + Dx2Dp + Dx3Dp*T + Dx4Dp*T**2)
                    - T*(-Dx1Dp/T**2 + Dx3Dp + 2.0*Dx4Dp*T))

    lambdad2Vdtdp = (D2x1Dp2 + D2x2Dp2*T + D2x3Dp2*T**2 + D2x4Dp2*T**3
                      - (D2x1Dp2*(np.log(T) - np.log(ref)) + Dx1Dp*DdeltDp/ref
                         + D2x2Dp2*(T - ref) + Dx2Dp*DdeltDp
                         + D2x3Dp2*(T**2 - ref**2)/2.0 + Dx3Dp*ref*DdeltDp
                         + D2x4Dp2*(T**3 - ref**3)/3.0 + Dx4Dp*ref**2*DdeltDp)
                      - T*(D2x1Dp2/T + D2x2Dp2 + D2x3Dp2*T + D2x4Dp2*T**2)
                      - (Dx1Dp*DdeltDp/ref + x1*DdeltDp**2/ref**2
                         + Dx2Dp*DdeltDp
                         + Dx3Dp*ref*DdeltDp - x3*DdeltDp**2
                         + Dx4Dp*ref**2*DdeltDp - x4*2.0*ref*DdeltDp**2))

    lambdad2Vdp2 = (D3x1Dp3*(T - ref) + D2x1Dp2*DdeltDp + D2x1Dp2*DdeltDp
                     + D3x2Dp3*(T**2 - ref**2)/2.0 + D2x2Dp2*ref*DdeltDp
                     + D2x2Dp2*ref*DdeltDp - Dx2Dp*DdeltDp**2
                     + D3x3Dp3*(T**3 - ref**3)/3.0 + D2x3Dp2*ref**2*DdeltDp
                     + D2x3Dp2*ref**2*DdeltDp - Dx3Dp*2.0*ref*DdeltDp**2
                     + D3x4Dp3*(T**4 - ref**4)/4.0 + D2x4Dp2*ref**3*DdeltDp
                     + D2x4Dp2*ref**3*DdeltDp - Dx4Dp*3.0*ref**2*DdeltDp**2
                     - T*(D3x1Dp3*(np.log(T) - np.log(ref)) + D2x1Dp2*DdeltDp/ref
                          + D2x1Dp2*DdeltDp/ref + Dx1Dp*(DdeltDp/ref)**2
                          + D3x2Dp3*(T - ref) + D2x2Dp2*DdeltDp + D2x2Dp2*DdeltDp
                          + D3x3Dp3*(T**2 - ref**2)/2.0 + D2x3Dp2*ref*DdeltDp
                          + D2x3Dp2*ref*DdeltDp - Dx3Dp*DdeltDp**2
                          + D3x4Dp3*(T**3 - ref**3)/3.0 + D2x4Dp2*ref**2*DdeltDp
                          + D2x4Dp2*ref**2*DdeltDp - Dx4Dp*2.0*ref*DdeltDp**2)
                     + D2x1Dp2*DdeltDp
                     + D2x2Dp2*ref*DdeltDp - Dx2Dp*DdeltDp**2 - Dx2Dp*DdeltDp**2
                     + D2x3Dp2*ref**2*DdeltDp - Dx3Dp*2.0*ref*DdeltDp**2
                     - Dx3Dp*2.0*ref*DdeltDp**2 + x3*2.0*DdeltDp**3
                     + D2x4Dp2*ref**3*DdeltDp - Dx4Dp*3.0*ref**2*DdeltDp**2
                     - Dx4Dp*3.0*ref**2*DdeltDp**2 + x4*6.0*ref*DdeltDp**3
                     - T*(D2x1Dp2*DdeltDp/ref + Dx1Dp*(DdeltDp/ref)**2
                          + Dx1Dp*(DdeltDp/ref)**2 + x1*2.0*(DdeltDp/ref)**3
                          + D2x2Dp2*DdeltDp
                          + D2x3Dp2*ref*DdeltDp - Dx3Dp*DdeltDp**2 - Dx3Dp*DdeltDp**2
                          + D2x4Dp2*ref**2*DdeltDp - Dx4Dp*2.0*ref*DdeltDp**2
                          - Dx4Dp*2.0*ref*DdeltDp**2 + x4*2.0*DdeltDp**3))

    cps_alpha = cps + (T + delt)*(l1 + l2*(T + delt))**2
    dcpsdt_alpha = dcpsdt + (l1 + l2*(T + delt))**2 + (T + delt)*2.0*(l1 + l2*(T + delt))*l2

    # -- combine branches -------------------------------------------------
    hs = np.where(is_beta, hs_beta, hs_alpha)
    ss = np.where(is_beta, ss_beta, ss_alpha)
    cps = np.where(is_beta, cps, cps_alpha)
    dcpsdt = np.where(is_beta, dcpsdt, dcpsdt_alpha)
    v0_use = np.where(is_beta, v0_beta, v0)
    v1_use = np.where(is_beta, v1_beta, v1)
    v2_use = np.where(is_beta, v2_beta, v2)
    v3_use = np.where(is_beta, v3_beta, v3)
    v4_use = np.where(is_beta, v4_beta, v4)

    quartz_adj = (0.0 if is_pmelts else _QUARTZ_ADJUSTMENT_MELTS) + _WETTING_ANGLE_CORR
    gs = hs - T*ss + quartz_adj
    hs = hs + quartz_adj

    eos = eos_berman(P, T, v0_use, v1_use, v2_use, v3_use, v4_use)
    G = gs + eos["g_add"]
    H = hs + eos["h_add"]
    S = ss + eos["s_add"]
    Cp = cps + eos["cp_add"]
    dCpdT = dcpsdt + eos["dcpdt_add"]
    V = eos["V"]
    dVdT = eos["dVdT"]
    dVdP = eos["dVdP"]
    d2VdT2 = eos["d2VdT2"]
    d2VdTdP = eos["d2VdTdP"]
    d2VdP2 = eos["d2VdP2"]

    # lambda volumetric correction: alpha branch only (zero contribution
    # under np.where in the beta branch, matching gibbs.c where lambdaV
    # etc. stay at their 0.0 initializers whenever t > cp_t).
    zero = np.zeros(shape)
    V = V + np.where(is_beta, zero, lambdaV)
    dVdT = dVdT + np.where(is_beta, zero, lambdadVdt)
    dVdP = dVdP + np.where(is_beta, zero, lambdadVdp)
    d2VdT2 = d2VdT2 + np.where(is_beta, zero, lambdad2Vdt2)
    d2VdTdP = d2VdTdP + np.where(is_beta, zero, lambdad2Vdtdp)
    d2VdP2 = d2VdP2 + np.where(is_beta, zero, lambdad2Vdp2)

    return dict(G=G, H=H, S=S, Cp=Cp, dCpdT=dCpdT,
                V=V, dVdT=dVdT, dVdP=dVdP,
                d2VdT2=d2VdT2, d2VdTdP=d2VdTdP, d2VdP2=d2VdP2,
                is_beta=is_beta)


def compute_tridymite(T, P, h, s, v0, v1, v2, v3, v4, k0, k1, k2, k3, Tt, l1, l2):
    """Tridymite G/H/S/Cp/V(+derivatives), verbatim from gibbs.c's
    tridymite branch (lines 1920-1993). Simpler than quartz: no
    pressure-dependent transition temperature and no lambda volumetric
    correction (`delt = Tt - cp_t` with `cp_t = Tt` always, so `delt == 0`
    identically -- gibbs.c's own tridymite branch never computes a
    `lambdaV`-family term at all, unlike quartz)."""
    T = np.asarray(T, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    h, s, v0, v1, v2, v3, v4 = np.broadcast_arrays(h, s, v0, v1, v2, v3, v4)
    k0, k1, k2, k3, Tt, l1, l2 = np.broadcast_arrays(k0, k1, k2, k3, Tt, l1, l2)
    shape = np.broadcast_shapes(T.shape, P.shape, h.shape)
    T = np.broadcast_to(T, shape).astype(np.float64)
    P = np.broadcast_to(P, shape).astype(np.float64)
    h, s, v0, v1, v2, v3, v4 = (np.broadcast_to(a, shape).astype(np.float64)
                                 for a in (h, s, v0, v1, v2, v3, v4))
    k0, k1, k2, k3, Tt, l1, l2 = (np.broadcast_to(a, shape).astype(np.float64)
                                   for a in (k0, k1, k2, k3, Tt, l1, l2))

    cp_t = Tt   # no pressure dependence for tridymite
    is_beta = T > cp_t

    cps = k0 + k1/np.sqrt(T) + k2/T**2 + k3/T**3
    dcpsdt = -0.5*k1/T**1.5 - 2.0*k2/T**3 - 3.0*k3/T**4

    hs_beta, ss_beta = _berman_cp_hs(T, k0, k1, k2, k3, _TRIDYMITE_BETA_H, _TRIDYMITE_BETA_S)
    v0_beta = np.full(shape, _TRIDYMITE_BETA_V0)
    v1_beta = np.full(shape, _TRIDYMITE_BETA_V1)
    v2_beta = np.full(shape, _TRIDYMITE_BETA_V2)
    v3_beta = np.full(shape, _TRIDYMITE_BETA_V3)
    v4_beta = np.full(shape, _TRIDYMITE_BETA_V4)

    hs_alpha, ss_alpha = _berman_cp_hs(T, k0, k1, k2, k3, h, s)
    delt = Tt - cp_t   # identically 0
    x1 = l1*l1*delt + 2.0*l1*l2*delt**2 + l2*l2*delt**3
    x2 = l1*l1 + 4.0*l1*l2*delt + 3.0*l2*l2*delt**2
    x3 = 2.0*l1*l2 + 3.0*l2*l2*delt
    x4 = l2*l2
    ref = Tr - delt   # gibbs.c's literal "tr" (298.15) for tridymite, unlike quartz's "373.0"
    hs_alpha = hs_alpha + x1*(T - ref) + x2*(T**2 - ref**2)/2.0 \
        + x3*(T**3 - ref**3)/3.0 + x4*(T**4 - ref**4)/4.0
    ss_alpha = ss_alpha + x1*(np.log(T) - np.log(ref)) + x2*(T - ref) \
        + x3*(T**2 - ref**2)/2.0 + x4*(T**3 - ref**3)/3.0
    cps_alpha = cps + (T + delt)*(l1 + l2*(T + delt))**2
    dcpsdt_alpha = dcpsdt + (l1 + l2*(T + delt))**2 + (T + delt)*2.0*(l1 + l2*(T + delt))*l2

    hs = np.where(is_beta, hs_beta, hs_alpha)
    ss = np.where(is_beta, ss_beta, ss_alpha)
    cps = np.where(is_beta, cps, cps_alpha)
    dcpsdt = np.where(is_beta, dcpsdt, dcpsdt_alpha)
    v0_use = np.where(is_beta, v0_beta, v0)
    v1_use = np.where(is_beta, v1_beta, v1)
    v2_use = np.where(is_beta, v2_beta, v2)
    v3_use = np.where(is_beta, v3_beta, v3)
    v4_use = np.where(is_beta, v4_beta, v4)

    gs = hs - T*ss   # no QUARTZ_ADJUSTMENT/WETTING_ANGLE_CORR analogue for tridymite

    eos = eos_berman(P, T, v0_use, v1_use, v2_use, v3_use, v4_use)
    G = gs + eos["g_add"]
    H = hs + eos["h_add"]
    S = ss + eos["s_add"]
    Cp = cps + eos["cp_add"]
    dCpdT = dcpsdt + eos["dcpdt_add"]

    return dict(G=G, H=H, S=S, Cp=Cp, dCpdT=dCpdT,
                V=eos["V"], dVdT=eos["dVdT"], dVdP=eos["dVdP"],
                d2VdT2=eos["d2VdT2"], d2VdTdP=eos["d2VdTdP"], d2VdP2=eos["d2VdP2"],
                is_beta=is_beta)
