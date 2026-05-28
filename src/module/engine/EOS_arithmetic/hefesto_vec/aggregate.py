"""
Two-level Voigt-Reuss-Hill phase averaging -> bulk density, Vp, Vs, S.

Matches physub.f logic exactly:

Level 1 - Intra-phase Reuss (within each solid-solution phase):
    volph   = sum_s n_s * V_s
    K_T_ph  = volph / sum_s(n_s*V_s/K_s)        (isothermal Reuss)
    G_ph    = volph / sum_s(n_s*V_s/Gsh_s)      (shear Reuss)
    alp_ph  = sum_s(n_s*V_s*alp_s) / volph      (volume-weighted mean)
    Cp_spec = sum_s(n_s*Cp_s) / sum_s(n_s*wm_s) (specific heat J/K/g)
    Cv_spec = Cp_spec - Ti*volph*alp_ph**2*K_T_ph / wmph * 1000
    gam_ph  = volph*alp_ph*K_T_ph / (Cv_spec*wmph) * 1000
    Ks_ph   = K_T_ph * (1 + alp_ph*gam_ph*Ti)   (adiabatic)

Level 2 - Inter-phase Voigt + Reuss:
    Kv  = sum_ph(volph*Ks_ph) / volagg
    Kr  = volagg / sum_ph(volph/Ks_ph)
    Gv  = sum_ph(volph*G_ph) / volagg
    Gr  = volagg / sum_ph(volph/G_ph)

Aggregate density:
    rho = sum_s(n_s*wm_s) / sum_s(n_s*V_s)

Aggregate entropy (physub.f):
    S_agg = sum_s(n_s * S_s) / wmagg  [J/g/K]

Hill = average of velocities (NOT velocity of averaged moduli):
    Vsv = sqrt(Gv/rho), Vsr = sqrt(Gr/rho), Vsh = (Vsv+Vsr)/2
    Vpv = sqrt((Kv+4/3*Gv)/rho), Vpr = sqrt((Kr+4/3*Gr)/rho), Vph=(Vpv+Vpr)/2

Unit notes:
    V   : cm^3/mol
    K,G : GPa
    wm  : g/mol
    alp : K^-1
    Cp  : J/mol/K
    Ti  : K
    rho : g/cm^3
    Vel : km/s
    1 GPa*cm^3 = 1000 J  (the *1000 in gamma formula converts GPa*cm^3 -> J)
    sqrt(GPa / (g/cm^3)) = km/s
"""

from __future__ import annotations
import numpy as np


def vrh_average(
    X,             # (B, S) mole amounts / fractions
    V,             # (B, S) converged molar volumes (cm^3/mol)
    K,             # (B, S) isothermal bulk modulus (GPa)
    Ks,            # (B, S) adiabatic bulk modulus (GPa)
    Gsh,           # (B, S) shear modulus (GPa)
    alp,           # (B, S) thermal expansivity (K^-1)
    Cp,            # (B, S) heat capacity (J/mol/K)
    S,             # (B, S) molar entropy (J/mol/K)
    wm,            # (S,)   molar mass (g/mol)
    Ti,            # (B,)   temperature (K)
    phase_members, # list of lists: phase_members[iph] = [0-based ispec, ...]
    active,        # (B, S) bool mask
):
    """Two-level Voigt-Reuss-Hill matching physub.f.

    Returns dict with:
        rho       : (B,) g/cm^3
        Kh, Gh    : (B,) GPa  Hill moduli = (Kv+Kr)/2, (Gv+Gr)/2
        Kv, Kr, Gv, Gr : (B,) GPa
        Vp, Vs, Vb : (B,) km/s  Hill = mean of Voigt & Reuss velocities
        Vpv, Vpr, Vsv, Vsr : (B,) km/s  component velocities
        S         : (B,) J/g/K  aggregate specific entropy
        Ks_phases, G_phases, vol_phases : lists of (B,) per-phase arrays
    """
    B, Sp = X.shape
    wm2d = wm[None, :]   # (1, S)

    # Aggregate density (species level)
    X_a    = np.where(active, X, 0.0)
    wmagg  = (X_a * wm2d).sum(axis=1)
    volagg = (X_a * V).sum(axis=1)
    safe_volagg = np.where(volagg > 1.0e-30, volagg, 1.0e-30)
    rho = wmagg / safe_volagg

    # Aggregate entropy: S_agg = sum(n_s * S_s) / wmagg  [J/g/K]
    # (physub.f: entagg = entagg + n(ispec)*sspeca(ispec); output: entagg/wmagg)
    entagg = (X_a * S).sum(axis=1)
    safe_wmagg = np.where(wmagg > 1.0e-30, wmagg, 1.0e-30)
    S_agg = entagg / safe_wmagg

    # Inter-phase accumulators
    baggv     = np.zeros(B)
    baggr     = np.zeros(B)
    gaggv     = np.zeros(B)
    gaggr     = np.zeros(B)
    volagg_ph = np.zeros(B)

    Ks_phases  = []
    G_phases   = []
    vol_phases = []

    for members in phase_members:
        if not members:
            continue
        idx = np.array(members, dtype=np.int64)

        act_ph = active[:, idx]
        X_ph   = np.where(act_ph, X[:, idx], 0.0)
        V_ph   = V[:, idx]
        K_ph   = K[:, idx]
        G_ph   = Gsh[:, idx]
        alp_ph = alp[:, idx]
        Cp_ph  = Cp[:, idx]
        wm_ph  = wm[idx]

        nV = X_ph * V_ph

        volph = nV.sum(axis=1)
        phase_active = volph > 1.0e-30
        safe_volph = np.where(phase_active, volph, 1.0e-30)

        # Intra-phase Reuss K_T: K_T_ph = volph / sum(n*V/K_T)
        safe_K = np.where(act_ph & (K_ph > 1.0e-10), K_ph, 1.0e30)
        buktph = (nV / safe_K).sum(axis=1)
        KT_phase = np.where(
            phase_active & (buktph > 1.0e-30),
            safe_volph / buktph, 0.0
        )

        # Intra-phase Reuss G: G_ph = volph / sum(n*V/G)
        safe_G = np.where(act_ph & (G_ph > 1.0e-10), G_ph, 1.0e30)
        gshph  = (nV / safe_G).sum(axis=1)
        G_phase = np.where(
            phase_active & (gshph > 1.0e-30),
            safe_volph / gshph, 0.0
        )

        # Phase alpha: alp_ph = sum(n*V*alp) / volph
        alpph     = (nV * alp_ph).sum(axis=1)
        alp_phase = alpph / safe_volph

        # Phase mass: wmph = sum(n*wm)
        wmph      = (X_ph * wm_ph[None, :]).sum(axis=1)
        safe_wmph = np.where(wmph > 1.0e-30, wmph, 1.0e-30)

        # Specific Cp (J/K/g): cpphtot = sum(n*Cap) / wmph
        cpph    = (X_ph * Cp_ph).sum(axis=1)
        cpphtot = cpph / safe_wmph

        # Specific Cv from physub.f:
        # cvphtot = cpphtot - Ti*volph*alp_ph^2*KT_ph/wmph * 1000
        # (1 GPa*cm^3 = 1000 J)
        correction = (Ti * safe_volph * alp_phase**2 * KT_phase
                      / safe_wmph * 1000.0)
        cvphtot = cpphtot - correction
        safe_cv = np.where(cvphtot > 1.0e-10, cvphtot, 1.0e-10)

        # Phase Gruneisen: gamphtot = volph*alp*K_T/(cvphtot*wmph)*1000
        gamphtot = (safe_volph * alp_phase * KT_phase
                    / (safe_cv * safe_wmph) * 1000.0)

        # Phase adiabatic Ks = K_T * (1 + alp*gamma*Ti)
        Ks_phase = KT_phase * (1.0 + alp_phase * gamphtot * Ti)
        Ks_phase = np.where(phase_active, Ks_phase, 0.0)
        G_phase  = np.where(phase_active, G_phase,  0.0)

        # Accumulate inter-phase Voigt/Reuss sums
        safe_Ks = np.where(phase_active & (Ks_phase > 1.0e-10), Ks_phase, 1.0e30)
        safe_Gp = np.where(phase_active & (G_phase  > 1.0e-10), G_phase,  1.0e30)

        baggv     += np.where(phase_active, volph * Ks_phase, 0.0)
        baggr     += np.where(phase_active, volph / safe_Ks,  0.0)
        gaggv     += np.where(phase_active, volph * G_phase,  0.0)
        gaggr     += np.where(phase_active, volph / safe_Gp,  0.0)
        volagg_ph += np.where(phase_active, volph, 0.0)

        Ks_phases.append(Ks_phase)
        G_phases.append(G_phase)
        vol_phases.append(volph)

    # Normalize
    safe_va = np.where(volagg_ph > 1.0e-30, volagg_ph, 1.0e-30)

    Kv = baggv / safe_va
    Kr = np.where(baggr > 1.0e-30, safe_va / baggr, 0.0)
    Gv = gaggv / safe_va
    Gr = np.where(gaggr > 1.0e-30, safe_va / gaggr, 0.0)
    Kh = 0.5 * (Kv + Kr)
    Gh = 0.5 * (Gv + Gr)

    # Wave velocities: sqrt(GPa/(g/cm^3)) = km/s
    safe_rho = np.where(rho > 1.0e-10, rho, 1.0e-10)

    Vsv = np.sqrt(np.maximum(Gv / safe_rho, 0.0))
    Vsr = np.sqrt(np.maximum(Gr / safe_rho, 0.0))
    Vpv = np.sqrt(np.maximum((Kv + 4.0 / 3.0 * Gv) / safe_rho, 0.0))
    Vpr = np.sqrt(np.maximum((Kr + 4.0 / 3.0 * Gr) / safe_rho, 0.0))
    Vbv = np.sqrt(np.maximum(Kv / safe_rho, 0.0))
    Vbr = np.sqrt(np.maximum(Kr / safe_rho, 0.0))

    # Hill = average of Voigt and Reuss VELOCITIES (physub.f lines 540-542)
    Vsh = 0.5 * (Vsv + Vsr)
    Vph = 0.5 * (Vpv + Vpr)
    Vbh = 0.5 * (Vbv + Vbr)

    return dict(
        rho=rho,
        Kh=Kh, Gh=Gh,
        Kv=Kv, Kr=Kr, Gv=Gv, Gr=Gr,
        Vp=Vph, Vs=Vsh, Vb=Vbh,
        Vpv=Vpv, Vpr=Vpr, Vsv=Vsv, Vsr=Vsr,
        Ks_phases=Ks_phases, G_phases=G_phases, vol_phases=vol_phases,
        S=S_agg,
    )
