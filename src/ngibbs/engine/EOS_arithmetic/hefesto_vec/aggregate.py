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

from .constants import Rgas


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
    site_data=None, # list indexed by species: [(element, stoich), ...] per site
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
    # (physub.f: entagg += n(ispec)*sspeca(ispec); output: entagg/wmagg)
    # sspeca = sspeco + smixi (func.f:92), where smixi = -R*ln(x_s_in_phase)
    # is the ideal mixing entropy contribution from solid-solution phases.
    entagg = (X_a * S).sum(axis=1)

    # Multi-site ideal mixing entropy matching cp.f / func.f:
    #   smixi_s = R * sum_kst[ stoich_s_kst * ln(n_kst_total / n_element_on_kst) ]
    # where components sharing the same element on the same site are grouped together.
    # This correctly handles:
    #   - Stoichiometric weighting (fo: Mg_2 → factor 2; py: Mg_3 → factor 3)
    #   - Multi-site phases (pv: A-site Mg/Fe/Al + B-site Si/Al)
    #   - Repeated sites (pe: Mg_2Mg_2O_4 → two identical sites, each contributing)
    entagg_mix = np.zeros(B, dtype=np.float64)

    use_sites = site_data is not None
    for ph_idx, members in enumerate(phase_members):
        if len(members) < 2:
            continue
        idx = np.array(members, dtype=np.int64)
        X_ph = np.where(active[:, idx], X[:, idx], 0.0)   # (B, m)

        if use_sites:
            # site_data[s] = List[List[(el,q)]] — each entry is one crystallographic
            # site with one or more (element, stoich) pairs.  Parenthesised formula
            # groups (e.g. (Na_2Mg_1) in namj, (Si_1Al_1) in cats) share a site.
            member_sites = [site_data[s] for s in members]
            n_sites = max((len(sp) for sp in member_sites), default=0)
            m = len(members)

            # Two layers of iastate-equivalent ordering (matching readin.f logic):
            #
            # Layer 1 — same-formula ordering (all sites):
            #   Species with identical site_data are different quantum states of the
            #   same compound (wu/wuls = FeO HS/LS, hepv/hlpv = Fe2O3-pv HS/LS).
            #   The Fortran sets iastate=TRUE for them at every site.
            #
            # Layer 2 — iron co-occupancy ordering (per site):
            #   readin.f: if two DIFFERENT-formula species both have Fe on site k →
            #   iastate=TRUE (they're treated as distinct iron species on that site).
            #   This isolates Fe²⁺ species (e.g. fepv, which has Fe+Si) from Fe³⁺
            #   species (hepv, hlpv, fapv, all Fe+non-Si) at the perovskite A-site,
            #   and separates wu from mag in the mw phase, etc.
            #
            # Layer 2 is applied per-site inside the kst loop below.
            formula_keys = [str(sp_sites) for sp_sites in member_sites]
            # global_excl[i] = frozenset of j excluded at every site (same formula)
            global_excl = [
                frozenset(j for j in range(m)
                          if formula_keys[j] == formula_keys[i] and j != i)
                for i in range(m)
            ]

            for kst in range(n_sites):
                # spec_comps[i] = [(el,q), ...] for species i at site kst.
                spec_comps = []
                for sp_sites in member_sites:
                    if kst < len(sp_sites):
                        spec_comps.append(sp_sites[kst])
                    else:
                        spec_comps.append([])

                # Layer 2: iron co-occupancy — build per-site extra exclusions.
                # If species i and j (different formulas) both have Fe at this site,
                # add j to i's exclusion set (and vice versa).
                has_fe = [any(el == 'Fe' for el, _ in comps) for comps in spec_comps]
                site_excl = [set(global_excl[i]) for i in range(m)]
                for i in range(m):
                    for j in range(m):
                        if i == j or j in global_excl[i]:
                            continue
                        if has_fe[i] and has_fe[j]:
                            site_excl[i].add(j)

                # n_kst_total = sum_s( sum_{c in s} q_c * X_s ) — total atoms on site
                nkp = np.zeros(B, dtype=np.float64)
                for i, comps in enumerate(spec_comps):
                    q_total = sum(q for _, q in comps)
                    nkp += q_total * X_ph[:, i]
                safe_nkp = np.where(nkp > 1.0e-30, nkp, 1.0e-30)

                # For each species i compute its own nikp excluding ordered partners.
                for i, comps_i in enumerate(spec_comps):
                    if not comps_i:
                        continue

                    nikp_i: dict = {}
                    for j, comps_j in enumerate(spec_comps):
                        if j in site_excl[i]:
                            continue
                        for el, q in comps_j:
                            if el not in nikp_i:
                                nikp_i[el] = np.zeros(B, dtype=np.float64)
                            nikp_i[el] += q * X_ph[:, j]

                    if len(nikp_i) < 2:
                        continue

                    contrib = np.zeros(B, dtype=np.float64)
                    for el, q in comps_i:
                        if el not in nikp_i:
                            continue
                        safe_nikp = np.where(
                            nikp_i[el] > 1.0e-30, nikp_i[el], 1.0e-30
                        )
                        contrib += q * np.log(safe_nkp / safe_nikp)
                    entagg_mix += X_ph[:, i] * Rgas * contrib

        else:
            # Fallback: simple single-site ideal mixing (no stoich weighting)
            n_ph = X_ph.sum(axis=1, keepdims=True)
            safe_n_ph = np.where(n_ph > 1.0e-30, n_ph, 1.0e-30)
            x_site = X_ph / safe_n_ph
            log_x = np.where(x_site > 1.0e-30, np.log(x_site), 0.0)
            entagg_mix += (X_ph * (-Rgas * log_x)).sum(axis=1)

    safe_wmagg = np.where(wmagg > 1.0e-30, wmagg, 1.0e-30)
    S_agg = (entagg + entagg_mix) / safe_wmagg

    # Species-level aggregate derivative properties (physub.f:373-375, 559-563).
    # These are the ISOMORPHIC (fixed-n) values, i.e. fort.59's alpiso/cpiso.
    # The metamorphic (phase-change) corrections live in metamorphic.py.
    alpagg = (X_a * V * alp).sum(axis=1) / safe_volagg      # 1/K
    cpagg  = (X_a * Cp).sum(axis=1) / safe_wmagg            # J/g/K

    # Inter-phase accumulators
    baggv     = np.zeros(B)
    baggr     = np.zeros(B)
    btaggr    = np.zeros(B)   # isothermal Reuss accumulator (physub.f:479)
    gaggv     = np.zeros(B)
    gaggr     = np.zeros(B)
    volagg_ph = np.zeros(B)

    Ks_phases  = []
    KT_phases  = []
    G_phases   = []
    vol_phases = []
    # Per-phase intermediates kept so that apply_fast_metamorphic() can redo the
    # K_T -> Cv -> gamma -> Ks -> velocity chain with a softened compliance
    # WITHOUT re-running the EOS.  See apply_fast_metamorphic() below.
    phase_cache = []

    for iph, members in enumerate(phase_members):
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

        safe_KT = np.where(phase_active & (KT_phase > 1.0e-10), KT_phase, 1.0e30)

        baggv     += np.where(phase_active, volph * Ks_phase, 0.0)
        baggr     += np.where(phase_active, volph / safe_Ks,  0.0)
        btaggr    += np.where(phase_active, volph / safe_KT,  0.0)
        gaggv     += np.where(phase_active, volph * G_phase,  0.0)
        gaggr     += np.where(phase_active, volph / safe_Gp,  0.0)
        volagg_ph += np.where(phase_active, volph, 0.0)

        Ks_phases.append(Ks_phase)
        KT_phases.append(np.where(phase_active, KT_phase, 0.0))
        G_phases.append(G_phase)
        vol_phases.append(volph)

        phase_cache.append(dict(
            iph=iph, volph=volph, phase_active=phase_active,
            buktph=buktph,              # cm^3/GPa  intra-phase Reuss compliance
            alp_phase=alp_phase, cpphtot=cpphtot, wmph=wmph,
            G_phase=G_phase,
        ))

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

    # Isothermal Reuss aggregate K_T  (physub.f:527  btaggr = volagg/btaggr).
    # This is the modulus that the metamorphic softening bmet is applied to.
    KTr = np.where(btaggr > 1.0e-30, safe_va / btaggr, 0.0)

    return dict(
        rho=rho,
        Kh=Kh, Gh=Gh,
        Kv=Kv, Kr=Kr, Gv=Gv, Gr=Gr,
        Vp=Vph, Vs=Vsh, Vb=Vbh,
        Vpv=Vpv, Vpr=Vpr, Vsv=Vsv, Vsr=Vsr,
        Ks_phases=Ks_phases, KT_phases=KT_phases,
        G_phases=G_phases, vol_phases=vol_phases,
        S=S_agg,
        # ── isomorphic aggregate derivative properties (fort.59 columns) ─────
        KTr=KTr,          # GPa    btaggr / btiso
        alpagg=alpagg,    # 1/K    alpiso
        cpagg=cpagg,      # J/g/K  cpiso
        volagg=volagg,    # cm^3   sum n_i V_i
        wmagg=wmagg,      # g      sum n_i wm_i
        entagg=entagg + entagg_mix,   # J/K  extensive aggregate entropy
        smix_total=entagg_mix,        # J/K  ideal-mixing part only
        phase_cache=phase_cache,      # for apply_fast_metamorphic()
    )


# ---------------------------------------------------------------------------
# Fast ("seismic-frequency") metamorphic softening of the velocity path
# ---------------------------------------------------------------------------

def _fast_term(seq, iph, B):
    """Pull phase ``iph`` out of a per-phase fast-term list, or zeros."""
    if seq is None:
        return np.zeros(B)
    val = seq[iph]
    if val is None:
        return np.zeros(B)
    return np.asarray(val, dtype=np.float64)


def apply_fast_metamorphic(agg, Ti, *,
                           bmet_fast_phases,
                           alpmet_fast_phases=None,
                           cpmet_fast_phases=None):
    """Apply the order-disorder relaxation to the velocity path, physub.f:414-431.

    HeFESTo's reported ``VP``/``VB`` are **not** isomorphic.  Before forming the
    per-phase adiabatic modulus it applies the per-phase analogues of the three
    aggregate metamorphic accumulators, each restricted to the *order-disorder*
    species of that phase (species with identical stoichiometry -- cation
    ordering, Fe spin-state pairs).  The argument is that intra-phase ordering
    relaxes fast enough to be seismic frequency, while cross-phase reactions do
    not::

        bukph  = sum_s(n_s V_s / K_s)  +  bmet_ph        (compliance, softens K_T)
        K_T_ph = volph / bukph
        alp_ph = alp_ph   + alpmet_ph                    (feeds gamma)
        Cp_ph  = Cp_ph    + cpmet_ph                     (feeds Cv, hence gamma)
        ... -> Cv_ph -> gamma_ph -> Ks_ph                (unchanged formulae)

    ``bmet_ph`` alone is not enough and in fact overshoots: it removes the whole
    +0.43% high-pressure Vb bias and carries on to -0.17%.  ``alpmet_ph`` raises
    gamma and pushes Ks back up.  With all three, the residual Vp/Vb bias is flat
    in pressure and equal to the isomorphic rho/Vs baseline.

    ``G_ph`` is untouched: **the shear modulus carries no metamorphic term**, so
    ``Vs`` must come back bit-identical.  The benchmark asserts this.

    Deliberately NOT softened
    -------------------------
    ``KTr`` (the isothermal Reuss aggregate, fort.59's ``btiso``) is left on the
    unsoftened ``K_T_ph``.  It is the modulus that ``metamorphic.combine_totals``
    then softens with the *full*, all-species ``bmet``; folding the fast part in
    here as well would double-count the order-disorder species.  Empirically this
    is right: ``KTtot``/``KStot`` agree with fort.56 to 0.017% as it stands.

    Out of scope (notated, not implemented)
    ---------------------------------------
    Anelastic attenuation (``qr19.f`` / ``vred.f``) is a separate correction that
    affects only the ``VSQ``/``VPQ`` columns of fort.56, not ``VP``/``VS``.

    Parameters
    ----------
    agg : dict
        Return value of :func:`vrh_average`, or a :func:`compute` result.  Must
        still carry ``phase_cache``.
    Ti : (B,) array
        Temperature, K.
    bmet_fast_phases : sequence of (B,) arrays
        Per-phase order-disorder compliance, cm^3/GPa.  Indexed by phase in the
        same order as ``phase_members``; entries for phases with no members are
        present but zero.
    alpmet_fast_phases, cpmet_fast_phases : sequence of (B,) arrays, optional
        Per-phase order-disorder expansivity (1/K) and *specific* heat capacity
        (J/g/K).  Omitting them reproduces the bmet-only variant, which is
        measurably worse -- they default to None only so the function stays
        usable for bisecting a regression.

    Returns
    -------
    dict
        ``Kv, Kr, Kh, Vp, Vs, Vb, Vpv, Vpr, Vbv, Vbr, Ks_phases, KT_phases``
        recomputed with the correction.  Merge into the ``compute()`` result to
        overwrite the isomorphic velocities.
    """
    cache = agg.get('phase_cache')
    if not cache:
        raise ValueError(
            'agg has no phase_cache — apply_fast_metamorphic() needs the '
            'per-phase intermediates recorded by vrh_average()'
        )

    Ti = np.asarray(Ti, dtype=np.float64)
    rho = agg['rho']
    B = rho.shape[0]

    baggv = np.zeros(B)
    baggr = np.zeros(B)
    gaggv = np.zeros(B)
    gaggr = np.zeros(B)
    volagg_ph = np.zeros(B)

    Ks_phases, KT_phases = [], []

    for c in cache:
        volph = c['volph']
        phase_active = c['phase_active']
        safe_volph = np.where(phase_active, volph, 1.0e-30)
        alp_phase = c['alp_phase']
        safe_wmph = np.where(c['wmph'] > 1.0e-30, c['wmph'], 1.0e-30)
        G_phase = c['G_phase']

        iph = c['iph']
        bmet = _fast_term(bmet_fast_phases, iph, B)
        alpmet = _fast_term(alpmet_fast_phases, iph, B)
        cpmet = _fast_term(cpmet_fast_phases, iph, B)

        # ── soften the Reuss compliance ──────────────────────────────────────
        buktph = c['buktph'] + bmet

        # A pathological relaxation could drive the compliance to zero or
        # negative (an unstable, negative-K_T assemblage).  The Fortran has no
        # guard; clamp to the unsoftened value rather than emit a NaN.
        bad = ~(buktph > 1.0e-30)
        if bad.any():
            buktph = np.where(bad, c['buktph'], buktph)

        KT_phase = np.where(phase_active & (buktph > 1.0e-30),
                            safe_volph / np.where(buktph > 1.0e-30, buktph, 1.0),
                            0.0)

        # ── and shift alpha / Cp, which feed gamma and hence Ks ──────────────
        alp_phase = alp_phase + alpmet

        correction = (Ti * safe_volph * alp_phase**2 * KT_phase
                      / safe_wmph * 1000.0)
        cvphtot = c['cpphtot'] + cpmet - correction
        safe_cv = np.where(cvphtot > 1.0e-10, cvphtot, 1.0e-10)

        gamphtot = (safe_volph * alp_phase * KT_phase
                    / (safe_cv * safe_wmph) * 1000.0)

        Ks_phase = KT_phase * (1.0 + alp_phase * gamphtot * Ti)
        Ks_phase = np.where(phase_active, Ks_phase, 0.0)

        safe_Ks = np.where(phase_active & (Ks_phase > 1.0e-10), Ks_phase, 1.0e30)
        safe_Gp = np.where(phase_active & (G_phase  > 1.0e-10), G_phase,  1.0e30)

        baggv += np.where(phase_active, volph * Ks_phase, 0.0)
        baggr += np.where(phase_active, volph / safe_Ks,  0.0)
        gaggv += np.where(phase_active, volph * G_phase,  0.0)
        gaggr += np.where(phase_active, volph / safe_Gp,  0.0)
        volagg_ph += np.where(phase_active, volph, 0.0)

        Ks_phases.append(Ks_phase)
        KT_phases.append(np.where(phase_active, KT_phase, 0.0))

    safe_va = np.where(volagg_ph > 1.0e-30, volagg_ph, 1.0e-30)

    Kv = baggv / safe_va
    Kr = np.where(baggr > 1.0e-30, safe_va / baggr, 0.0)
    Gv = gaggv / safe_va
    Gr = np.where(gaggr > 1.0e-30, safe_va / gaggr, 0.0)

    safe_rho = np.where(rho > 1.0e-10, rho, 1.0e-10)

    Vsv = np.sqrt(np.maximum(Gv / safe_rho, 0.0))
    Vsr = np.sqrt(np.maximum(Gr / safe_rho, 0.0))
    Vpv = np.sqrt(np.maximum((Kv + 4.0 / 3.0 * Gv) / safe_rho, 0.0))
    Vpr = np.sqrt(np.maximum((Kr + 4.0 / 3.0 * Gr) / safe_rho, 0.0))
    Vbv = np.sqrt(np.maximum(Kv / safe_rho, 0.0))
    Vbr = np.sqrt(np.maximum(Kr / safe_rho, 0.0))

    return dict(
        Kv=Kv, Kr=Kr, Kh=0.5 * (Kv + Kr),
        Vp=0.5 * (Vpv + Vpr), Vs=0.5 * (Vsv + Vsr), Vb=0.5 * (Vbv + Vbr),
        Vpv=Vpv, Vpr=Vpr, Vbv=Vbv, Vbr=Vbr,
        Ks_phases=Ks_phases, KT_phases=KT_phases,
    )


# ---------------------------------------------------------------------------
# Torch-native VRH averaging (GPU-compatible)
# ---------------------------------------------------------------------------

def vrh_average_torch(
    X,              # (B, S) torch tensor
    V,              # (B, S)
    K,              # (B, S)
    Ks,             # (B, S)
    Gsh,            # (B, S)
    alp,            # (B, S)
    Cp,             # (B, S)
    S,              # (B, S) per-species molar entropy (J/mol/K)
    wm,             # (S,)   molar mass (g/mol) — torch tensor
    Ti,             # (B,)   temperature (K) — torch tensor
    phase_members,  # list of lists (Python ints, not tensors)
    active,         # (B, S) bool torch tensor
    site_data=None,
) -> dict:
    """Torch equivalent of vrh_average."""
    import torch

    device = X.device
    B, Sp  = X.shape
    wm2d   = wm.unsqueeze(0)                     # (1, S)

    X_a     = torch.where(active, X, torch.zeros_like(X))
    wmagg   = (X_a * wm2d).sum(dim=1)            # (B,)
    volagg  = (X_a * V).sum(dim=1)               # (B,)
    sv      = volagg.clamp(min=1e-30)
    rho     = wmagg / sv

    entagg  = (X_a * S).sum(dim=1)               # (B,)
    entagg_mix = torch.zeros(B, dtype=torch.float64, device=device)

    use_sites = site_data is not None
    for ph_idx, members in enumerate(phase_members):
        if len(members) < 2:
            continue
        idx  = torch.tensor(members, dtype=torch.long, device=device)
        X_ph = torch.where(active[:, idx], X[:, idx], torch.zeros(B, len(members), dtype=X.dtype, device=device))

        if use_sites:
            member_sites = [site_data[s] for s in members]
            n_sites = max((len(sp) for sp in member_sites), default=0)
            m       = len(members)
            formula_keys = [str(sp) for sp in member_sites]
            global_excl  = [
                frozenset(j for j in range(m)
                          if formula_keys[j] == formula_keys[i] and j != i)
                for i in range(m)
            ]
            for kst in range(n_sites):
                spec_comps = []
                for sp_sites in member_sites:
                    spec_comps.append(sp_sites[kst] if kst < len(sp_sites) else [])
                has_fe    = [any(el == 'Fe' for el, _ in comps) for comps in spec_comps]
                site_excl = [set(global_excl[i]) for i in range(m)]
                for i in range(m):
                    for j in range(m):
                        if i == j or j in global_excl[i]:
                            continue
                        if has_fe[i] and has_fe[j]:
                            site_excl[i].add(j)
                nkp = torch.zeros(B, dtype=torch.float64, device=device)
                for i, comps in enumerate(spec_comps):
                    q_tot = sum(q for _, q in comps)
                    nkp  += q_tot * X_ph[:, i]
                snkp = nkp.clamp(min=1e-30)
                for i, comps_i in enumerate(spec_comps):
                    if not comps_i:
                        continue
                    nikp_i: dict = {}
                    for j, comps_j in enumerate(spec_comps):
                        if j in site_excl[i]:
                            continue
                        for el, q in comps_j:
                            if el not in nikp_i:
                                nikp_i[el] = torch.zeros(B, dtype=torch.float64, device=device)
                            nikp_i[el] = nikp_i[el] + q * X_ph[:, j]
                    if len(nikp_i) < 2:
                        continue
                    contrib = torch.zeros(B, dtype=torch.float64, device=device)
                    for el, q in comps_i:
                        if el not in nikp_i:
                            continue
                        sn = nikp_i[el].clamp(min=1e-30)
                        contrib = contrib + q * torch.log(snkp / sn)
                    entagg_mix = entagg_mix + X_ph[:, i] * Rgas * contrib
        else:
            n_ph   = X_ph.sum(dim=1, keepdim=True).clamp(min=1e-30)
            x_site = X_ph / n_ph
            log_x  = torch.where(x_site > 1e-30, torch.log(x_site), torch.zeros_like(x_site))
            entagg_mix = entagg_mix + (X_ph * (-Rgas * log_x)).sum(dim=1)

    swmagg = wmagg.clamp(min=1e-30)
    S_agg  = (entagg + entagg_mix) / swmagg

    # Species-level isomorphic aggregate derivative properties (fort.59), the
    # inputs metamorphic.add_metamorphic() needs.  Kept in step with the numpy
    # path so the GPU route can feed the metamorphic pass too.
    salp = wmagg.clamp(min=1e-30)
    alpagg = (X_a * V * alp).sum(dim=1) / sv
    cpagg  = (X_a * Cp).sum(dim=1) / salp

    baggv     = torch.zeros(B, dtype=torch.float64, device=device)
    baggr     = torch.zeros(B, dtype=torch.float64, device=device)
    btaggr    = torch.zeros(B, dtype=torch.float64, device=device)
    gaggv     = torch.zeros(B, dtype=torch.float64, device=device)
    gaggr     = torch.zeros(B, dtype=torch.float64, device=device)
    volagg_ph = torch.zeros(B, dtype=torch.float64, device=device)

    Ks_phases  = [];  G_phases = [];  vol_phases = []
    phase_cache = []

    for iph, members in enumerate(phase_members):
        if not members:
            continue
        idx   = torch.tensor(members, dtype=torch.long, device=device)
        act_ph = active[:, idx]
        X_ph   = torch.where(act_ph, X[:, idx], torch.zeros(B, len(members), dtype=X.dtype, device=device))
        V_ph   = V[:, idx];    K_ph  = K[:, idx]
        G_ph   = Gsh[:, idx];  alp_ph = alp[:, idx]
        Cp_ph  = Cp[:, idx];   wm_ph  = wm[idx]

        nV      = X_ph * V_ph
        volph   = nV.sum(dim=1)
        ph_act  = volph > 1e-30
        svolph  = volph.clamp(min=1e-30)

        sK_p    = torch.where(act_ph & (K_ph > 1e-10), K_ph,
                              torch.full_like(K_ph, 1e30))
        buktph  = (nV / sK_p).sum(dim=1)
        KT_ph   = torch.where(ph_act & (buktph > 1e-30), svolph / buktph,
                              torch.zeros_like(volph))

        sG_p    = torch.where(act_ph & (G_ph > 1e-10), G_ph,
                              torch.full_like(G_ph, 1e30))
        gshph   = (nV / sG_p).sum(dim=1)
        G_phase = torch.where(ph_act & (gshph > 1e-30), svolph / gshph,
                              torch.zeros_like(volph))

        alp_phase = (nV * alp_ph).sum(dim=1) / svolph
        wmph      = (X_ph * wm_ph.unsqueeze(0)).sum(dim=1)
        swmph     = wmph.clamp(min=1e-30)
        cpph      = (X_ph * Cp_ph).sum(dim=1)
        cpphtot   = cpph / swmph
        correction = Ti * svolph * alp_phase**2 * KT_ph / swmph * 1000.0
        cvphtot    = cpphtot - correction
        scv        = cvphtot.clamp(min=1e-10)
        gamphtot   = svolph * alp_phase * KT_ph / (scv * swmph) * 1000.0
        Ks_phase   = KT_ph * (1.0 + alp_phase * gamphtot * Ti)
        Ks_phase   = torch.where(ph_act, Ks_phase, torch.zeros_like(Ks_phase))
        G_phase    = torch.where(ph_act, G_phase,  torch.zeros_like(G_phase))

        sKs_p  = torch.where(ph_act & (Ks_phase > 1e-10), Ks_phase, torch.full_like(Ks_phase, 1e30))
        sGp    = torch.where(ph_act & (G_phase  > 1e-10), G_phase,  torch.full_like(G_phase,  1e30))
        sKT_p  = torch.where(ph_act & (KT_ph    > 1e-10), KT_ph,    torch.full_like(KT_ph,    1e30))

        baggv     = baggv + torch.where(ph_act, volph * Ks_phase, torch.zeros_like(volph))
        baggr     = baggr + torch.where(ph_act, volph / sKs_p,    torch.zeros_like(volph))
        btaggr    = btaggr + torch.where(ph_act, volph / sKT_p,   torch.zeros_like(volph))
        gaggv     = gaggv + torch.where(ph_act, volph * G_phase,  torch.zeros_like(volph))
        gaggr     = gaggr + torch.where(ph_act, volph / sGp,      torch.zeros_like(volph))
        volagg_ph = volagg_ph + torch.where(ph_act, volph,        torch.zeros_like(volph))

        Ks_phases.append(Ks_phase);  G_phases.append(G_phase);  vol_phases.append(volph)

        # Mirrors the numpy path's phase_cache.  apply_fast_metamorphic() is
        # numpy-only (as is the whole metamorphic module), so these come back to
        # the host here rather than staying on device -- they are (B,) vectors,
        # a negligible transfer next to the (B, nspec) EOS arrays.
        phase_cache.append(dict(
            iph=iph,
            volph=volph.detach().cpu().numpy(),
            phase_active=ph_act.detach().cpu().numpy(),
            buktph=buktph.detach().cpu().numpy(),
            alp_phase=alp_phase.detach().cpu().numpy(),
            cpphtot=cpphtot.detach().cpu().numpy(),
            wmph=wmph.detach().cpu().numpy(),
            G_phase=G_phase.detach().cpu().numpy(),
        ))

    sva  = volagg_ph.clamp(min=1e-30)
    Kv   = baggv / sva
    Kr   = torch.where(baggr > 1e-30, sva / baggr, torch.zeros_like(baggr))
    Gv   = gaggv / sva
    Gr   = torch.where(gaggr > 1e-30, sva / gaggr, torch.zeros_like(gaggr))
    Kh   = 0.5 * (Kv + Kr)
    Gh   = 0.5 * (Gv + Gr)

    srho = rho.clamp(min=1e-10)
    Vsv  = torch.sqrt(torch.clamp(Gv / srho, min=0.0))
    Vsr  = torch.sqrt(torch.clamp(Gr / srho, min=0.0))
    Vpv  = torch.sqrt(torch.clamp((Kv + 4.0/3.0*Gv) / srho, min=0.0))
    Vpr  = torch.sqrt(torch.clamp((Kr + 4.0/3.0*Gr) / srho, min=0.0))
    Vbv  = torch.sqrt(torch.clamp(Kv / srho, min=0.0))
    Vbr  = torch.sqrt(torch.clamp(Kr / srho, min=0.0))
    Vsh  = 0.5 * (Vsv + Vsr)
    Vph  = 0.5 * (Vpv + Vpr)
    Vbh  = 0.5 * (Vbv + Vbr)

    return dict(
        rho=rho,
        Kh=Kh, Gh=Gh,
        Kv=Kv, Kr=Kr, Gv=Gv, Gr=Gr,
        Vp=Vph, Vs=Vsh, Vb=Vbh,
        Vpv=Vpv, Vpr=Vpr, Vsv=Vsv, Vsr=Vsr,
        Ks_phases=Ks_phases, G_phases=G_phases, vol_phases=vol_phases,
        S=S_agg,
        # ── isomorphic aggregate derivative properties (fort.59 columns) ─────
        KTr=torch.where(btaggr > 1e-30, sva / btaggr, torch.zeros_like(btaggr)),
        alpagg=alpagg, cpagg=cpagg, volagg=volagg, wmagg=wmagg,
        phase_cache=phase_cache,
    )
