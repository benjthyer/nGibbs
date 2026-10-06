"""
MELTS-version-aware liquid: the rhyolite-MELTS 1.0.2 / 1.1.0 / 1.2.0 liquid
component standard states (including gibbs.c's hard-coded special cases) and
the 1.1/1.2 CO2-bearing liquid mixing model with its CaCO3 speciation.

Why this module exists
----------------------
`liq_struct_data.json` carries all-zero placeholder rows for liquid H2O and CO2
because MAGMA's `gibbs()` never reads them: SiO2, H2O, CO2 (and CaCO3 in the
CO2-aware modes) are special-cased in code *before* the generic Kress branch
that `liquid_eos.kress_component` translates. Without these branches dissolved
volatiles contributed no V/H/S/Cp at all, and SiO2 lacked its glass-transition
heat capacity -- the systematic BishopTuff/MORB liquid errors diagnosed in
`MELTS_EOS_Liquid_Volatiles_Findings.md`.

Modes (`mode` argument everywhere; MAGMA calculationMode in brackets):
- 'MELTS102'  rhyolite-MELTS 1.0.2 [MODE__MELTS]: 19-component `meltsLiquid`,
              liquid_v34 mixing (regular-solution W(i,j) + H2O extra ideal term)
              with `param_struct_data_v34.h`'s `meltsModelParameters` table.
              That is the table liquid_v34.c's WH() macro reads in MODE__MELTS;
              it differs from `param_struct_data.h`'s `originalModelParameters`
              (the source of `liq_wij_data.json`) in the eleven nonzero CO2 pairs.
- 'MELTS110'  rhyolite-MELTS 1.1.0 [MODE__MELTSandCO2]: `meltsFluidLiquid` (19
              basis components + CaCO3 species), liquid_CO2.c mixing with the
              `meltsAndCO2ModelParameters` W table.
- 'MELTS120'  rhyolite-MELTS 1.2.0 [MODE__MELTSandCO2_H2O]: as 1.1.0 but the
              `meltsAndCO2_H2OModelParameters` W table (recalibrated H2O
              interactions) and the Ochs & Lange liquid-H2O volume.

Special cases (gibbs.c, liquid section), all translated verbatim:
- SiO2 (all three modes): own h0/s0 and heat capacity with a glass transition
  at Tg = 1480 K (Berman-like Cp below, constant 81.373 above), then the usual
  Kress pressure terms.
- H2O, 1.0.2/1.1.0: whaar(1 bar) + Robie correction + the phiP pressure
  polynomial (which also gives V).
- H2O, 1.2.0: whaar(1 bar) + Robie correction with the 1.2 a/b shifts and the
  Ochs & Lange (2.775 J/bar at 1673.15 K) volume.
- CO2, 1.1.0/1.2.0: Duan pure CO2 at 1 bar + fixed h/s offsets + linear V.
- CaCO3, 1.1.0/1.2.0: CaSiO3 + CO2 - SiO2 (each via the above) + hCorr/vCorr.

The 1.1/1.2 mixing model (liquid_CO2.c / liquid_CO2_H2O.c, byte-identical apart
from the parameter table) is a quadratic regular solution over 20 species
(19 basis + CaCO3) plus configurational entropy with H2O's extra ideal term,
with one internal speciation parameter s = X(CaCO3) set by dG/ds = 0
(CaSiO3 + CO2 = CaCO3 + SiO2). MAGMA builds that quadratic as a 210-term Taylor
expansion fitted by SVD to the species-vertex values and binary W's; the
species model here is the same quadratic evaluated directly (the fit is square
and full rank -- 20 barycentric species coordinates in 19 variables -- so the
two are identical up to MAGMA's own SVD round-off). Verified against the MAGMA
C library to ~1e-12 relative (see melts_vec/tests/benchmark_liquid_modes_test.py).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .constants import Rgas, Pr, Trl
from .liquid_params import load_liquid, MELTSLiquidParams
from .liquid_eos import kress_component
from .liquid_nonideal import liquid_nonideal_correction
from .water import whaar
from .fluid_duan import pure_properties as _duan_pure

MODES = ('MELTS102', 'MELTS110', 'MELTS120')
_TABLE = {'MELTS102': 'meltsLiquid', 'MELTS110': 'meltsFluidLiquid', 'MELTS120': 'meltsFluidLiquid'}
# Solid tables per version: meltsFluidSolids = meltsSolids minus 'water', plus the
# Duan fluid endmembers and the carbon phases (calcite ... graphite, diamond).
SOLID_TABLE = {'MELTS102': 'meltsSolids', 'MELTS110': 'meltsFluidSolids', 'MELTS120': 'meltsFluidSolids'}
# Pure carbon-bearing solids that exist only in the 1.1/1.2 solid table
CARBON_PHASES = ('calcite', 'aragonite', 'magnesite', 'siderite', 'dolomite', 'spurrite', 'tilleyite',
                 'diamond', 'graphite')
_W_JSON = Path(__file__).resolve().parent.parent / 'MELTS_Parameters' / 'liq_mode_wij_data.json'

_R = Rgas            # gibbs.c r = 8.3143
_TRL = Trl           # gibbs.c trl = 1673.0


def _check_mode(mode):
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")


def load_liquid_mode(mode: str, json_path=None) -> MELTSLiquidParams:
    """Liquid parameter table for a MELTS version: 19 components (1.0.2) or
    19 basis components + the CaCO3 species (1.1.0/1.2.0, `meltsFluidLiquid`).
    json_path overrides the packaged liq_struct_data.json."""
    _check_mode(mode)
    return load_liquid(json_path, table=_TABLE[mode])


# --------------------------------------------------------------------------- #
# Special-case component standard states (gibbs.c, liquid section)
# --------------------------------------------------------------------------- #
def _sio2_reference(T):
    """SiO2 special case, P = Pr part: (hl, sl, cpl, dcpldt)."""
    h0, s0, al, bl, cl, dl, tg, cp = -901554.0, 48.475, 127.200, -10.777e-3, 4.3127e5, -1463.8, 1480.0, 81.373
    tr = 298.15
    def integ(t):
        h = h0 + al*(t - tr) + bl*(t*t - tr*tr)/2.0 - cl*(1.0/t - 1.0/tr) + 2.0*dl*(np.sqrt(t) - np.sqrt(tr))
        s = s0 + al*np.log(t/tr) + bl*(t - tr) - (cl/2.0)*(1.0/t**2 - 1.0/tr**2) - 2.0*dl*(1.0/np.sqrt(t) - 1.0/np.sqrt(tr))
        return h, s
    hg, sg = integ(np.float64(tg))
    hb, sb = integ(T)
    above = T >= tg
    hl = np.where(above, hg + (T - tg)*cp, hb)
    sl = np.where(above, sg + cp*np.log(T/tg), sb)
    cpl = np.where(above, cp, al + bl*T + cl/(T*T) + dl/np.sqrt(T))
    dcpldt = np.where(above, 0.0, bl - 2.0*cl/T**3 - 0.5*dl/T**1.5)
    return hl, sl, cpl, dcpldt


def _robie(t):
    r = _R
    g = r*t*(2.9147*np.log(t) - 9.6863e-4*t + 6.8593e-8*t*t + 77.8899/np.sqrt(t) - 28954.8/t - 2263.27/t**2 - 15.8997)
    dg = r*(2.9147*np.log(t) - 9.6863e-4*t + 6.8593e-8*t*t + 77.8899/np.sqrt(t) - 28954.8/t - 2263.27/t**2 - 15.8997) \
        + r*t*(2.9147/t - 9.6863e-4 + 2.0*6.8593e-8*t - 0.5*77.8899/t**1.5 + 28954.8/t**2 + 2.0*2263.27/t**3)
    d2g = 2.0*r*(2.9147/t - 9.6863e-4 + 2.0*6.8593e-8*t - 0.5*77.8899/t**1.5 + 28954.8/t**2 + 2.0*2263.27/t**3) \
        + r*t*(-2.9147/t**2 + 2.0*6.8593e-8 + 1.5*0.5*77.8899/t**2.5 - 2.0*28954.8/t**3 - 6.0*2263.27/t**4)
    d3g = 3.0*r*(-2.9147/t**2 + 2.0*6.8593e-8 + 1.5*0.5*77.8899/t**2.5 - 2.0*28954.8/t**3 - 6.0*2263.27/t**4) \
        + r*t*(2.0*2.9147/t**3 - 2.5*1.5*0.5*77.8899/t**3.5 + 6.0*28954.8/t**4 + 24.0*2263.27/t**5)
    return g, dg, d2g, d3g


def _h2o_liquid(T, P, ref_h, ref_s, mode):
    """Liquid H2O special case (per mode). Returns dict G, H, S, Cp, dCpdT, V, dVdT, dVdP,
    d2VdT2, d2VdTdP, d2VdP2."""
    r, t, p = _R, T, P
    w = whaar(np.ones_like(t), t, melts_mode=True)
    gR, dgR, d2gR, d3gR = _robie(t)
    a = -33676.0 + ref_h/r
    b = 18.3527 - ref_s/r
    if mode in ('MELTS102', 'MELTS110'):
        phiP = (0.110/t + 4.432e-5 + 1.405e-7*t - 2.394e-11*t*t)*p \
            + (7.337e-8/t - 1.170e-8 - 9.502e-13*t)*p*p + (1.876e-10/t + 4.586e-13)*p**3 - 1.191e-14*p**4/t
        dphiPdt = (-0.110/t**2 + 1.405e-7 - 2.0*2.394e-11*t)*p + (-7.337e-8/t**2 - 9.502e-13)*p*p \
            - 1.876e-10*p**3/t**2 + 1.191e-14*p**4/t**2
        d2phiPdt2 = (2.0*0.110/t**3 - 2.0*2.394e-11)*p + 2.0*7.337e-8*p*p/t**3 + 2.0*1.876e-10*p**3/t**3 \
            - 2.0*1.191e-14*p**4/t**3
        d3phiPdt3 = -6.0*0.110*p/t**4 - 6.0*7.337e-8*p*p/t**4 - 6.0*1.876e-10*p**3/t**4 + 6.0*1.191e-14*p**4/t**4
        gl = r*t*(a/t + b + phiP) + w['g'] - gR
        dgdt = r*(a/t + b + phiP) + r*t*(-a/t**2 + dphiPdt) - dgR
        sl = w['s'] - dgdt
        hl = gl + t*sl
        d2gdt2 = 2.0*r*(-a/t**2 + dphiPdt) + r*t*(2.0*a/t**3 + d2phiPdt2) - d2gR
        d3gdt3 = 3.0*r*(2.0*a/t**3 + d2phiPdt2) + r*t*(-6.0*a/t**4 + d3phiPdt3) - d3gR
        cpl = w['cp'] - t*d2gdt2
        dcpldt = w['dcpdt'] - d2gdt2 - t*d3gdt3
        vl = r*(0.110 + 4.432e-5*t + 1.405e-7*t*t - 2.394e-11*t**3) \
            + 2.0*r*(7.337e-8 - 1.170e-8*t - 9.502e-13*t*t)*p + 3.0*r*(1.876e-10 + 4.586e-13*t)*p*p \
            - 4.0*r*1.191e-14*p**3
        dvldt = r*(4.432e-5 + 2.0*1.405e-7*t - 3.0*2.394e-11*t*t) - 2.0*r*(1.170e-8 + 2.0*9.502e-13*t)*p \
            + 3.0*r*4.586e-13*p*p
        dvldp = 2.0*r*(7.337e-8 - 1.170e-8*t - 9.502e-13*t*t) + 6.0*r*(1.876e-10 + 4.586e-13*t)*p \
            - 12.0*r*1.191e-14*p*p
        d2vldt2 = r*(2.0*1.405e-7 - 6.0*2.394e-11*t) - 4.0*r*9.502e-13*p
        d2vldp2 = 6.0*r*(1.876e-10 + 4.586e-13*t) - 24.0*r*1.191e-14*p
        d2vldtdp = -2.0*r*(1.170e-8 + 2.0*9.502e-13*t) + 6.0*r*4.586e-13*p
    else:  # MELTS120 (and xMELTS): Ochs & Lange volume
        vOL, dvdtOL, dvdpOL = 2.775, 1.086e-3, -0.382e-4
        a = a + 2783.6851512128/r
        b = b - 2.3838467967178/r
        pv = vOL*(p - 1.0) + dvdtOL*(t - 1673.15)*(p - 1.0) + 0.5*dvdpOL*(p - 1.0)*(p - 1.0)
        gl = r*t*(a/t + b) + w['g'] - gR + pv
        dgdt = r*(a/t + b) + r*t*(-a/t**2) - dgR + dvdtOL*(p - 1.0)
        sl = w['s'] - dgdt
        hl = r*t*(a/t + b) + w['g'] - gR + t*(w['s'] - dgdt) + pv
        d2gdt2 = 2.0*r*(-a/t**2) + r*t*(2.0*a/t**3) - d2gR
        d3gdt3 = 3.0*r*(2.0*a/t**3) + r*t*(-6.0*a/t**4) - d3gR
        cpl = w['cp'] - t*d2gdt2
        dcpldt = w['dcpdt'] - d2gdt2 - t*d3gdt3
        vl = vOL + dvdtOL*(t - 1673.15) + dvdpOL*(p - 1.0)
        dvldt = np.full_like(t, dvdtOL); dvldp = np.full_like(t, dvdpOL)
        d2vldt2 = np.zeros_like(t); d2vldp2 = np.zeros_like(t); d2vldtdp = np.zeros_like(t)
    return dict(G=gl, H=hl, S=sl, Cp=cpl, dCpdT=dcpldt, V=vl, dVdT=dvldt, dVdP=dvldp,
                d2VdT2=d2vldt2, d2VdTdP=d2vldtdp, d2VdP2=d2vldp2)


def _co2_liquid(T, P):
    """Liquid CO2 special case (1.1.0/1.2.0): Duan pure CO2 at 1 bar + fixed offsets."""
    hCO2, sCO2, vCO2 = -630.93193811701, -109.39331414050, 4.0157994267547
    dvdt, dvdp = 1.213189e-3, -0.4267387e-4
    d = _duan_pure(T, np.ones_like(T), 'CO2')
    t, p, pr, trl = T, P, Pr, _TRL
    pint = vCO2*(p - pr) + dvdt*(t - trl)*(p - pr) + dvdp*(p*p/2.0 - pr*pr/2.0 - pr*(p - pr))
    return dict(G=d['g'] + hCO2 - t*sCO2 + pint,
                S=d['s'] + sCO2 - dvdt*(p - pr),
                H=d['h'] + hCO2 + pint - t*dvdt*(p - pr),
                Cp=d['cp'], dCpdT=d['dcpdt'],
                V=vCO2 + dvdt*(t - trl) + dvdp*(p - pr), dVdT=np.full_like(t, dvdt), dVdP=np.full_like(t, dvdp),
                d2VdT2=np.zeros_like(t), d2VdTdP=np.zeros_like(t), d2VdP2=np.zeros_like(t))


_KEYS = ('G', 'H', 'S', 'Cp', 'dCpdT', 'V', 'dVdT', 'dVdP', 'd2VdT2', 'd2VdTdP', 'd2VdP2')


def liquid_species_properties(T, P, params: MELTSLiquidParams, mode: str) -> dict:
    """Per-component/species (B, N) liquid standard-state properties exactly as
    MAGMA's gibbs() returns them for `mode` (generic Kress branch plus the SiO2,
    H2O, CO2 and CaCO3 special cases). Keys: G, H, S, Cp, dCpdT, V, dVdT, dVdP,
    d2VdT2, d2VdTdP, d2VdP2, labels."""
    _check_mode(mode)
    T = np.atleast_1d(np.asarray(T, dtype=np.float64))
    P = np.atleast_1d(np.asarray(P, dtype=np.float64))
    T, P = np.broadcast_arrays(T, P)
    Tc, Pc = T[:, None], P[:, None]
    gen = kress_component(
        Tc, Pc, params.v_liq[None, :], params.dvdt[None, :], params.dvdp[None, :],
        params.d2vdtp[None, :], params.d2vdp2[None, :], params.t_fusion[None, :], params.s_fusion[None, :],
        params.cp_liquid[None, :], params.ref_h[None, :], params.ref_s[None, :],
        params.ref_k0[None, :], params.ref_k1[None, :], params.ref_k2[None, :], params.ref_k3[None, :],
        ref_cp_t=params.ref_cp_t[None, :], ref_cp_h=params.ref_cp_h[None, :],
        ref_l1=params.ref_l1[None, :], ref_l2=params.ref_l2[None, :])
    out = {k: np.array(np.broadcast_to(gen[k], (T.size, params.nspec)), dtype=np.float64) for k in _KEYS}
    ix = params.label_index

    # SiO2: special reference (P = Pr) state; Kress pressure terms are the generic ones.
    i = ix['SiO2']
    hl_s, sl_s, cpl_s, dcpl_s = _sio2_reference(T)
    ref0 = kress_component(T, np.full_like(T, Pr), params.v_liq[i], params.dvdt[i], params.dvdp[i],
                           params.d2vdtp[i], params.d2vdp2[i], params.t_fusion[i], params.s_fusion[i],
                           params.cp_liquid[i], params.ref_h[i], params.ref_s[i], params.ref_k0[i],
                           params.ref_k1[i], params.ref_k2[i], params.ref_k3[i],
                           ref_cp_t=params.ref_cp_t[i], ref_cp_h=params.ref_cp_h[i],
                           ref_l1=params.ref_l1[i], ref_l2=params.ref_l2[i])
    dh, ds = hl_s - ref0['H'], sl_s - ref0['S']
    out['H'][:, i] += dh
    out['S'][:, i] += ds
    out['G'][:, i] += dh - T*ds
    out['Cp'][:, i] = cpl_s
    out['dCpdT'][:, i] = dcpl_s

    # H2O
    i = ix['H2O']
    h = _h2o_liquid(T, P, params.ref_h[i], params.ref_s[i], mode)
    for k in _KEYS:
        out[k][:, i] = h[k]

    if mode in ('MELTS110', 'MELTS120'):
        i = ix['CO2']
        c = _co2_liquid(T, P)
        for k in _KEYS:
            out[k][:, i] = c[k]
        # CaCO3 = CaSiO3 + CO2 - SiO2 (+ hCorr, vCorr). getCaCO3properties' own d2v
        # bookkeeping is garbled in the source (d2vdp2 assigned from d2vdt2); only the
        # quantities MELTS actually consumes (G, H, S, Cp, dCpdT, V, dVdT, dVdP) are formed.
        i = ix['CaCO3']
        a, b, s = ix['CaSiO3'], ix['CO2'], ix['SiO2']
        for k in _KEYS:
            out[k][:, i] = out[k][:, a] + out[k][:, b] - out[k][:, s]
        hCorr, vCorr = -17574.497522747, -1.9034060173857
        out['H'][:, i] += hCorr
        out['V'][:, i] += vCorr
        out['G'][:, i] += hCorr + vCorr*(P - Pr)
    out['labels'] = list(params.labels)
    return out


# --------------------------------------------------------------------------- #
# Bulk liquid
# --------------------------------------------------------------------------- #
_W_CACHE: dict = {}


def _mode_w(mode):
    """(species labels, W (3, NE, NE) [H, S, V], species adjustments (NE, 3) or None)
    for a MELTS version, from `MELTS_Parameters/liq_mode_wij_data.json`."""
    if mode not in _W_CACHE:
        rec = json.loads(_W_JSON.read_text())[mode]
        ne = len(rec['species'])
        W = np.zeros((3, ne, ne))
        n = 0
        for i in range(ne):
            for l in range(i + 1, ne):
                W[:, i, l] = W[:, l, i] = rec['W'][n]
                n += 1
        adj = np.array(rec['species_adjust'], dtype=np.float64) if 'species_adjust' in rec else None
        _W_CACHE[mode] = (rec['species'], W, adj)
    return _W_CACHE[mode]


def _solve_caco3(x_basis, g_mu, Wg, T):
    """Speciation parameter s = X(CaCO3): root of dG/ds on (0, min(X_CaSiO3, X_CO2)).
    x_basis: (B, 20) species vector at s = 0; g_mu: (B, 20) species G(i); Wg: (B, 20, 20)."""
    d = np.zeros(20); d[0], d[10], d[14], d[19] = 1.0, -1.0, -1.0, 1.0
    smax = np.minimum(x_basis[:, 10], x_basis[:, 14])
    ok = smax > 0.0
    s = np.zeros(x_basis.shape[0])
    if not ok.any():
        return s, d
    xb, gm, W, t, sm = x_basis[ok], g_mu[ok], Wg[ok], T[ok], smax[ok]
    RT = _R*t
    lin = gm @ d                                   # sum d_i G(i)
    dWd = np.einsum('i,bij,j->b', d, W, d)
    Wd_xb = np.einsum('bij,j->bi', W, d)           # W d  (B, 20)
    act = [0, 10, 14, 19]

    def f_and_fp(sv):
        x = xb + sv[:, None]*d
        fx = lin + np.einsum('bi,bi->b', Wd_xb, x) + RT*sum(d[k]*np.log(x[:, k]) for k in act)
        fpx = dWd + RT*sum(d[k]**2/x[:, k] for k in act)
        return fx, fpx
    lo = np.zeros_like(sm); hi = sm.copy()
    sv = 0.5*sm
    for _ in range(200):   # safeguarded Newton on a bracket (dG/ds -> -inf at 0, +inf at smax)
        fx, fpx = f_and_fp(sv)
        lo = np.where(fx < 0.0, sv, lo)
        hi = np.where(fx > 0.0, sv, hi)
        step = sv - fx/fpx
        bad = ~((step > lo) & (step < hi))
        new = np.where(bad, 0.5*(lo + hi), step)
        if np.all(np.abs(new - sv) <= 4.0*np.finfo(float).eps*np.maximum(sv, 1e-300)):
            sv = new
            break
        sv = new
    s[ok] = sv
    return s, d


def _bulk_co2(T, P, X, params, mode):
    """liquid_CO2(.c)/liquid_CO2_H2O(.c) bulk properties per mole of basis components."""
    species, W3, adj = _mode_w(mode)
    if list(params.labels) != species:
        raise ValueError(f"liquid params labels {params.labels} != model species {species}")
    sp = liquid_species_properties(T, P, params, mode)
    B = T.size
    if X.shape[1] != 19:
        raise ValueError(f"{mode} liquid composition must have the 19 basis components, got {X.shape[1]}")
    xb = np.concatenate([X, np.zeros((B, 1))], axis=1)
    # G(i) etc. include the per-species parameter adjustments (all zero in the shipped tables)
    Gi = sp['G'] + adj[None, :, 0] - T[:, None]*adj[None, :, 1] + (P[:, None] - 1.0)*adj[None, :, 2]
    Si = sp['S'] + adj[None, :, 1]
    Vi = sp['V'] + adj[None, :, 2]
    Wg = W3[0][None] - T[:, None, None]*W3[1][None] + (P[:, None, None] - 1.0)*W3[2][None]
    s, d = _solve_caco3(xb, Gi, Wg, T)
    x = xb + s[:, None]*d
    ih = 18
    with np.errstate(divide='ignore', invalid='ignore'):
        xlx = np.where(x > 0.0, x*np.log(np.where(x > 0.0, x, 1.0)), 0.0)
        xh = x[:, ih]
        h2o = np.where((xh > 0.0) & (xh < 1.0),
                       xh*np.log(np.where(xh > 0, xh, 1.0)) + (1.0 - xh)*np.log(np.where(xh < 1, 1.0 - xh, 1.0)), 0.0)
    config = xlx.sum(1) + h2o
    quad = lambda M: 0.5*np.einsum('bi,bij,bj->b', x, M, x)
    WS = np.broadcast_to(W3[1], (B, 20, 20)); WV = np.broadcast_to(W3[2], (B, 20, 20))
    G = np.sum(x*Gi, 1) + quad(Wg) + _R*T*config
    S = np.sum(x*Si, 1) + quad(WS) - _R*config
    H = G + T*S
    V = np.sum(x*Vi, 1) + quad(WV)
    # ordering (speciation) contributions to second derivatives
    has = s > 0.0
    act = [0, 10, 14, 19]
    with np.errstate(divide='ignore', invalid='ignore'):
        Gss = np.einsum('i,bij,j->b', d, Wg, d) + _R*T*sum(np.where(has, d[k]**2/x[:, k], 0.0) for k in act)
        Gst = -(Si @ d) - np.einsum('i,bij,bj->b', d, WS, x) \
            + _R*sum(np.where(has, d[k]*np.log(np.where(x[:, k] > 0, x[:, k], 1.0)), 0.0) for k in act)
        Gsp = Vi @ d + np.einsum('i,bij,bj->b', d, WV, x)
        corr_cp = np.where(has, T*Gst*Gst/Gss, 0.0)
        corr_vt = np.where(has, -Gst*Gsp/Gss, 0.0)
        corr_vp = np.where(has, -Gsp*Gsp/Gss, 0.0)
    Cp = np.sum(x*sp['Cp'], 1) + corr_cp
    dVdT = np.sum(x*sp['dVdT'], 1) + corr_vt
    dVdP = np.sum(x*sp['dVdP'], 1) + corr_vp
    # dCp/dT along the equilibrium path, ds/dT = -Gst/Gss; Cp = sum x cp_i + T Gst^2/Gss.
    # Third derivatives: Gstt = -(d.cp)/T (W and RT terms are linear in T),
    # Gsst = -d.WS.d + R sum d_k^2/x_k, Gsss = -RT sum d_k^3/x_k^2.
    dCpdT = np.sum(x*sp['dCpdT'], 1)
    if has.any():
        with np.errstate(divide='ignore', invalid='ignore'):
            dcp = sp['Cp'] @ d
            dsdT = np.where(has, -Gst/Gss, 0.0)
            Gstt = -dcp/T
            Gsst = -np.einsum('i,bij,j->b', d, WS, d) + _R*sum(np.where(has, d[k]**2/x[:, k], 0.0) for k in act)
            Gsss = -_R*T*sum(np.where(has, d[k]**3/x[:, k]**2, 0.0) for k in act)
            dGst = Gstt + Gsst*dsdT
            dGss = Gsst + Gsss*dsdT
            ord_part = Gst*Gst/Gss + T*(2.0*Gst*dGst/Gss - Gst*Gst*dGss/Gss**2)
            dCpdT = dCpdT + np.where(has, dcp*dsdT + ord_part, 0.0)
    return dict(G=G, H=H, S=S, V=V, Cp=Cp, dVdT=dVdT, dVdP=dVdP, dCpdT=dCpdT, s_CaCO3=s)


def compute_liquid_bulk_mode(T, P, X, mode: str, params: MELTSLiquidParams | None = None) -> dict:
    """Bulk liquid properties per mole of liquid (basis) components for a MELTS
    version, exactly as alphaMELTS reports them (mixing + standard states).

    X : (B, 19) moles or mole fractions of the 19 basis liquid components in
        `LIQUID_COMPONENT_LABELS` order (SiO2 ... H2O); rows are normalized
        here and must have a positive, finite sum. For 1.1/1.2 the CaCO3
        species is internal (speciation), not an input column.
    Returns dict of (B,): G, H, S, V (J/bar), Cp, dVdT, dVdP, dCpdT, K, alpha,
    and for 1.1/1.2 's_CaCO3' (CaCO3 species mole fraction).
    """
    _check_mode(mode)
    if params is None:
        params = load_liquid_mode(mode)
    T = np.atleast_1d(np.asarray(T, dtype=np.float64)); P = np.atleast_1d(np.asarray(P, dtype=np.float64))
    T, P = np.broadcast_arrays(T, P)
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    if np.any(X < 0.0):
        raise ValueError(f"negative liquid component amounts in rows {np.nonzero((X < 0.0).any(1))[0][:10]}")
    n = X.sum(1)
    bad = ~(np.isfinite(n) & (n > 0.0))
    if bad.any():
        raise ValueError(f"liquid composition rows must have a positive finite sum; bad rows: {np.nonzero(bad)[0][:10]}")
    X = X/n[:, None]
    if mode == 'MELTS102':
        sp = liquid_species_properties(T, P, params, mode)
        out = {k: np.sum(X*sp[k], 1) for k in ('H', 'Cp', 'dCpdT', 'V', 'dVdT', 'dVdP')}
        with np.errstate(divide='ignore', invalid='ignore'):
            S_config = -_R*np.sum(np.where(X > 0.0, X*np.log(np.where(X > 0.0, X, 1.0)), 0.0), 1)
        out['S'] = np.sum(X*sp['S'], 1) + S_config
        species, W3, _ = _mode_w(mode)
        if list(params.labels) != species:
            raise ValueError(f"liquid params labels {params.labels} != model components {species}")
        if np.any(W3[1:]):
            raise ValueError("MELTS102 W table has nonzero WS/WV; liquid_nonideal_correction assumes WH only")
        corr = liquid_nonideal_correction(X, T, W3[0], params.label_index['H2O'])
        out['H'] = out['H'] + corr['dH']
        out['S'] = out['S'] + corr['dS']
        out['G'] = out['H'] - T*out['S']
    else:
        out = _bulk_co2(T, P, X, params, mode)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        out['K'] = np.where(out['dVdP'] != 0.0, -out['V']/out['dVdP'], np.inf)
        out['alpha'] = out['dVdT']/out['V']
    return out
