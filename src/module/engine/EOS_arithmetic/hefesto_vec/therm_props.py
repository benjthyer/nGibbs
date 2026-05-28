"""
Full thermodynamic property calculation at converged volume.

Implements the core of therm.f: given converged V, T, and precomputed
Uth, Uto, Cv, Cvo (from the volume solver), compute K, Ks, Gsh, alp, Cp.

Also computes per-species molar entropy S_landau [J/mol/K] for the Landau
correction term; the full entropy (including vibrational + electronic terms)
is assembled in compute.py.

Key ordering (matching Fortran therm.f exactly):
  1. Cold BM terms (Kc, cold Gsh) using pre-Landau V
  2. stishtran correction to Gsh for stishovite (applied to cold Gsh only)
  3. Thermal corrections to K and Gsh
  4. Landau corrections (modifies V, K, alp, Cp, Cv, Ks, and adds S_landau)
     - Gsh thermal correction uses post-Landau Vi (not pre-Landau V)
"""

from __future__ import annotations
import numpy as np

from .constants import Rgas
from .pressure  import cold_bulk_modulus_vec, cold_pressure_vec


# ---------------------------------------------------------------------------
# Stishovite acoustic-mode softening (stishtran.f, Hellwig et al. 2003)
# Called for species whose name starts with "st" before the ittyp correction.
# ---------------------------------------------------------------------------

def stishtran_vec(Pi: np.ndarray, Ti: np.ndarray,
                  Gsh: np.ndarray, is_st: np.ndarray) -> np.ndarray:
    """Vectorised stishtran: shear softening in stishovite.

    Parameters
    ----------
    Pi   : (B, S)  pressure (GPa)
    Ti   : (B, S)  temperature (K)
    Gsh  : (B, S)  cold shear modulus (GPa) — modified in place for is_st cells
    is_st: (1, S)  boolean mask, True for stishovite species

    Returns
    -------
    (B, S) corrected Gsh
    """
    if not np.any(is_st):
        return Gsh

    Gbare = Gsh
    Pcs = 51.6 + 11.1 * (Ti - 300.0) / 1000.0          # (B, S)
    Pc  = Pcs + 50.7
    Cso = 128.0 * Pc / Pcs

    # ordered phase (Pi < Pcs)
    safe_Pc_m_Pi_ord = np.where(np.abs(Pc - Pi) > 1.0e-10, Pc - Pi, 1.0e-10)
    Cs_ord = Cso - Cso * (Pc - Pcs) / safe_Pc_m_Pi_ord

    # disordered phase (Pi >= Pcs)
    denom_dis = Pc - Pi + 3.0 * (Pi - Pcs)              # = Pc + 2*Pi - 3*Pcs
    safe_denom_dis = np.where(np.abs(denom_dis) > 1.0e-10, denom_dis, 1.0e-10)
    Cs_dis = Cso - Cso * (Pc - Pcs) / safe_denom_dis   # Delta = 0

    Cs = np.where(Pi < Pcs, Cs_ord, Cs_dis)

    # VRH-like average of bare Gsh and softened (C11-C12)/2
    safe_Gbare = np.where(Gbare > 1.0e-10, Gbare, 1.0e-10)
    safe_Cs    = np.where(np.abs(Cs)   > 1.0e-10, Cs,    1.0e-10)
    Gsh_new = (0.5 * (13.0 / 15.0 * Gbare + 2.0 / 15.0 * Cs)
               + 0.5 / (13.0 / 15.0 / safe_Gbare + 2.0 / 15.0 / safe_Cs))

    return np.where(is_st, Gsh_new, Gsh)


# ---------------------------------------------------------------------------
# Main thermodynamic property kernel
# ---------------------------------------------------------------------------

def compute_therm_props(
    V   : np.ndarray,
    Ti  : np.ndarray,
    To  : np.ndarray,
    Pi  : np.ndarray,
    gamma  : np.ndarray,
    q      : np.ndarray,
    qp     : np.ndarray,
    etas   : np.ndarray,
    detasdv: np.ndarray,
    Uth : np.ndarray,
    Uto : np.ndarray,
    Cv  : np.ndarray,
    Cvo : np.ndarray,
    apar: np.ndarray,
    iltyp_arr: np.ndarray,
    *,
    ivtyp: int = 1,
    ittyp: int = 1,
    is_stishovite: np.ndarray | None = None,  # (S,) bool — species name starts "st"
) -> dict:
    """Compute K, Ks, Gsh, alp, Cp, rho, S_landau for all (B, S) pairs."""
    S = apar.shape[0]

    def p(col): return apar[None, :, col]

    fn   = p(0); wm  = p(2)
    Vo  = p(5)
    Ko   = p(6); Kop  = p(7); Kopp = p(8)
    wd1  = p(9)
    be   = p(27); ge   = p(28)
    izp  = p(33)
    Go   = p(34); Gop  = p(35); Got = p(36)
    Tc_p = p(37); Smax = p(38); Vmax = p(39)
    a5   = p(42)
    ibv  = p(31).astype(int)

    f = 0.5 * ((V / Vo) ** (-2.0 / 3.0) - 1.0)

    Kc = cold_bulk_modulus_vec(V, Vo, Ko, Kop, Kopp, a5, ibv=ibv)
    pc = cold_pressure_vec(V, Vo, Ko, Kop, Kopp, a5, ibv=ibv)

    ezp = np.where(np.abs(izp) == 1,
                   np.sign(izp) * (9.0 / 8.0) * fn * Rgas * wd1, 0.0)
    pzp = np.where(np.abs(izp) == 1,
                   np.sign(izp) * 1.0e-3 * (9.0 / 8.0) * fn * Rgas * wd1 * gamma / V,
                   0.0)

    ph  = 1.0e-3 * (gamma / V) * (Uth - Uto)
    Kth = ((gamma + 1.0 - q) * (ph + pzp)
           - 1.0e-3 * (gamma**2 / V) * (Ti * Cv - To * Cvo))

    beta  = be * (V / Vo) ** ge
    Cvel  = beta * Ti
    Cvelo = beta * To
    pel   = 1.0e-3 * 0.5 * ge * beta * (Ti**2 - To**2) / V
    Kel   = pel * (1.0 - ge)

    gamma_eff = np.where(Ti > 0.0,
                         (gamma * Cv + ge * Cvel) / (Cv + Cvel),
                         gamma)
    Cv_tot  = Cv  + Cvel
    Cvo_tot = Cvo + Cvelo

    K   = Kc + Kth + Kel
    alp = 1.0e-3 * gamma_eff * Cv_tot / (V * K)
    agT = alp * gamma_eff * Ti
    Cp  = Cv_tot * (1.0 + gamma_eff**2 * Cv_tot * Ti / (1.0e3 * V * K))
    Ks  = K * (1.0 + agT)

    # ── Cold shear modulus (pre-Landau, pre-stishtran, matching Fortran) ──────
    if ivtyp == 1:
        b0 = Go
        b1 = 3.0 * Ko * Gop - 5.0 * Go
        b2 = 6.0 * Ko * Gop - 24.0 * Ko - 14.0 * Go + 4.5 * Ko * Kop
        Gsh = (1.0 + 2.0 * f) ** 2.5 * (b0 + b1 * f + b2 * f**2)
    elif ivtyp == 2:
        b0 = Go; b1 = 3.0 * Ko * Gop - 5.0 * Go
        Gsh = (1.0 + 2.0 * f) ** 2.5 * (b0 + b1 * f)
    else:
        b0 = Go; b1 = 3.0 * Ko * Gop - 5.0 * Go
        Gsh = (1.0 + 2.0 * f) ** 2.5 * (b0 + b1 * f)

    # ── Stishovite shear softening (stishtran.f) ──────────────────────────────
    # Applied to cold Gsh before thermal correction, matching Fortran ordering.
    if is_stishovite is not None and np.any(is_stishovite):
        is_st = is_stishovite[None, :]   # (1, S)
        Gsh = stishtran_vec(Pi, Ti, Gsh, is_st)

    # ── Thermal shear correction ──────────────────────────────────────────────
    # Fortran applies this AFTER Landau V update (using post-Landau Vi).
    # We apply it here with V_base, then _apply_landau will correct for the
    # V_base → V_new change via the etas_therm_num argument.
    if ittyp == 1:
        etas_therm_num = 1.0e-3 * etas * (Uth - Uto + ezp)   # V-independent
        Gsh = Gsh - etas_therm_num / V
    elif ittyp == 2:
        etas_therm_num = None   # uses Vo not Vi, unaffected by Landau V shift
        Gsh = Gsh - 1.0e-3 * Got / Vo * (Uth - Uto)
    elif ittyp == 3:
        etas_therm_num = 1.0e-3 * Got * (Uth - Uto)          # V-independent
        Gsh = Gsh - etas_therm_num / V
    elif ittyp == 4:
        gamma0 = apar[None, :, 25]; q0 = apar[None, :, 26]
        safe_g0 = np.where(gamma0 == 0.0, 1.0, gamma0)
        safe_q0 = np.where(q0 == 0.0, 1.0, q0)
        etas_therm_num = 1.0e-3 * etas * (gamma / safe_g0) * (q / safe_q0) * (Uth - Uto)
        Gsh = Gsh - etas_therm_num / V
    else:
        etas_therm_num = None

    rho = wm / V

    # ── Landau corrections (iltyp=1 -> landauqr) ─────────────────────────────
    # iltyp_arr carries the GLOBAL iltyp for all species.
    # Applicability gated inside _apply_landau by Tco>0.
    iltyp_b = iltyp_arr[None, :]
    has_landau = (iltyp_b == 1) & (Tc_p > 0.0)

    if np.any(has_landau):
        K, Ks, Gsh, alp, Cp, Cv_tot, rho, V_corrected, S_landau = _apply_landau(
            K, Ks, Gsh, alp, Cp, Cv_tot, rho, V, Pi,
            Tc_p, Smax, Vmax, Ti, gamma_eff, has_landau,
            etas_therm_num=etas_therm_num,
        )
    else:
        V_corrected = V
        S_landau    = np.zeros_like(V)

    return dict(
        K=K, Ks=Ks, Gsh=Gsh, alp=alp, Cp=Cp, Cv=Cv_tot,
        Kc=Kc, Kth=Kth, Kel=Kel, ph=ph, pzp=pzp, ezp=ezp,
        rho=rho, V_corrected=V_corrected,
        f=f, gamma_eff=gamma_eff,
        S_landau=S_landau,
    )


# ---------------------------------------------------------------------------
# Landau correction kernel
# ---------------------------------------------------------------------------

def _apply_landau(K, Ks, Gsh, alp, Cp, Cv, rho, V, Pi,
                  Tc0_arr, Smax, Vmax, Ti, gamma, has_landau,
                  etas_therm_num=None):
    """Apply Landau corrections matching landauqr.f / hillert.f (iltyp=1).

    Branch 1 - Hillert magnetic (landauqr.f calls hillert.f):
      Magnetite: |Tco - 845.5| <= 1 K
      Iron:      |smax - 9.46028| <= 1e-5 AND |Tco - 1043.01| <= 1e-5
      p=0.40; only cplan is nonzero (alplan=betlan=vlan=0).

    Branch 2 - Standard landauqr:
      Q^4 = (Tc-Ti)/Tco if ordered, else Q=0
      cplan=Smax*Ti/(2*Tco*Q^2) if ordered, else 0
      alplan, betlan likewise 0 when disordered
      vlan = -Vmax*(Q^2-1) always (= +Vmax when disordered, Q=0)

    Entropy (Ftotsub.f line 83/110):
      Standard:  slan = -smax*(Q^2 - 1)  [= smax*(1-Q^2), = smax when disordered]
      Hillert:   slan = -smax*(gfunc + t*dgdt)  (hillert.f line 31)

    Final update matching therm.f:
      Vi  = V + vlan
      K   = Vi/V / (1/K + betlan)
      alp = V/Vi * (alp + alplan)
      Cp  = Cp + cplan
      Cv  = Cp - 1e3*Ti*Vi*alp^2*K
      Ks  = K*(1 + alp*gamma*Ti)

    Gsh thermal correction fix (Fortran applies ittyp term with post-Landau Vi):
      etas_therm_num = 1e-3 * etas * (Uth - Uto + ezp)  for ittyp=1
      Gsh was computed as: Gsh_cold - etas_therm_num/V_base
      Fortran computes:    Gsh_cold - etas_therm_num/V_new
      Fix (where vlan != 0): Gsh += etas_therm_num * (1/V - 1/V_new)
    """
    Tc0 = Tc0_arr

    safe_Smax = np.where(Smax > 0.0, Smax, 1.0)
    Tc  = Tc0 + Vmax / (1.0e-3 * safe_Smax) * Pi
    Tc  = np.minimum(Tc, 10.0 * Tc0)
    safe_Tc0 = np.where(Tc0 > 0.0, Tc0, 1.0)
    safe_Tc  = np.where(Tc  > 0.0, Tc,  1.0)

    # --- Hillert branch -------------------------------------------------------
    is_mag     = np.abs(Tc0 - 845.5) <= 1.0
    is_fe      = ((np.abs(Smax - 9.46028) <= 1.0e-5)
                  & (np.abs(Tc0 - 1043.01) <= 1.0e-5))
    is_hillert = is_mag | is_fe

    p_h   = 0.40
    denom = 518.0/1125.0 + 11692.0/15975.0 * (1.0/p_h - 1.0)
    afe   = 79.0 / (140.0 * p_h) / denom
    bfe   = 474.0 / 497.0 * (1.0/p_h - 1.0) / denom
    cfe   = 1.0 / denom

    t      = Ti / safe_Tc
    t_ord  = t <= 1.0
    safe_t = np.where(t > 0.0, t, 1.0)

    # ordered (t<=1)
    dgdt_o  = -bfe * (t**2/2.0 + t**8/15.0 + t**14/40.0)
    d2g_o   = -bfe * (t + 8.0*t**7/15.0 + 14.0*t**13/40.0)
    gfunc_o = -bfe * (t**3/6.0 + t**9/135.0 + t**15/600.0)
    # disordered (t>1)
    dgdt_d  = (cfe*(safe_t**(-6)/2.0 + safe_t**(-16)/21.0 + safe_t**(-26)/60.0)
               - afe/safe_t**2)
    d2g_d   = (-cfe*(3.0*safe_t**(-7) + 16.0*safe_t**(-17)/21.0 + 26.0*safe_t**(-27)/60.0)
               + 2.0*afe/safe_t**3)
    gfunc_d = (-cfe*(safe_t**(-5)/10.0 + safe_t**(-15)/315.0 + safe_t**(-25)/1500.0)
               - 1.0 + afe/safe_t)

    dgdt_h  = np.where(t_ord, dgdt_o, dgdt_d)
    d2g_h   = np.where(t_ord, d2g_o,  d2g_d)
    gfunc_h = np.where(t_ord, gfunc_o, gfunc_d)
    cplan_h = -Smax * (2.0*t*dgdt_h + t**2*d2g_h)
    # Hillert entropy: slan = -smax*(gfunc + t*dgdt)  (hillert.f line 31)
    slan_h  = -Smax * (gfunc_h + t * dgdt_h)

    # --- Standard landauqr branch --------------------------------------------
    ordered = Ti <= Tc
    Q4  = np.where(ordered, np.maximum((Tc - Ti) / safe_Tc0, 0.0), 0.0)
    Q   = Q4 ** 0.25
    Q2  = Q**2
    safe_Q2 = np.where(Q2 > 1.0e-30, Q2, 1.0e-30)  # only used inside ordered mask

    vlan_std   = -Vmax * (Q2 - 1.0)   # = +Vmax when Q=0 (disordered)
    cplan_std  = np.where(ordered, Smax * Ti / (2.0 * safe_Tc0 * safe_Q2), 0.0)
    alplan_std = np.where(ordered, Vmax / V / (2.0 * safe_Tc0 * safe_Q2), 0.0)
    betlan_std = np.where(ordered,
                          Vmax / V * Vmax / (1.0e-3 * safe_Smax)
                          / (2.0 * safe_Tc0 * safe_Q2),
                          0.0)
    # Standard Landau entropy: slan = -smax*(Q^2 - 1)  (landauqr.f line 83)
    # When disordered Q2=0 → slan=smax; when ordered slan=smax*(1-Q^2)
    slan_std = -Smax * (Q2 - 1.0)

    # --- Merge: hillert zeroes out volume/K/alp corrections ------------------
    cplan  = np.where(is_hillert, cplan_h,   cplan_std)
    alplan = np.where(is_hillert, 0.0,       alplan_std)
    betlan = np.where(is_hillert, 0.0,       betlan_std)
    vlan   = np.where(is_hillert, 0.0,       vlan_std)
    slan   = np.where(is_hillert, slan_h,    slan_std)

    # --- Apply (matching therm.f) --------------------------------------------
    volnl = V
    V_new = V + vlan
    V_new = np.where(V_new > 0.0, V_new, V)

    safe_K = np.where(K > 0.0, K, 1.0e-10)
    K_new  = V_new / volnl / (1.0 / safe_K + betlan)

    alp_new = volnl / V_new * (alp + alplan)
    Cp_new  = Cp + cplan                  # Fortran: Cp = Cp + cplan
    Cv_new  = Cp_new - 1.0e3 * Ti * V_new * alp_new**2 * K_new
    Ks_new  = K_new * (1.0 + alp_new * gamma * Ti)

    active_corr = has_landau & (Smax > 0.0)
    K_out   = np.where(active_corr, K_new,   K)
    Ks_out  = np.where(active_corr, Ks_new,  Ks)
    alp_out = np.where(active_corr, alp_new, alp)
    Cp_out  = np.where(active_corr, Cp_new,  Cp)
    Cv_out  = np.where(active_corr, Cv_new,  Cv)
    V_out   = np.where(active_corr, V_new,   V)
    rho_out = rho * V / V_out

    # --- Gsh fix: Fortran uses post-Landau Vi in the thermal correction -------
    # For species with vlan != 0 (standard Landau, e.g. qtz, neph), the thermal
    # shear correction was computed as -etas_therm_num/V_base. The Fortran uses
    # -etas_therm_num/V_new. Correct by adding the difference.
    if etas_therm_num is not None:
        has_vlan = active_corr & (np.abs(vlan) > 1.0e-30)
        safe_V_out = np.where(V_out > 0.0, V_out, V)
        gsh_fix = etas_therm_num * (1.0 / V - 1.0 / safe_V_out)
        Gsh_out = np.where(has_vlan, Gsh + gsh_fix, Gsh)
    else:
        Gsh_out = Gsh

    # Landau entropy [J/mol/K]: only applied where Landau correction is active
    S_landau = np.where(active_corr, slan, 0.0)

    return K_out, Ks_out, Gsh_out, alp_out, Cp_out, Cv_out, rho_out, V_out, S_landau
