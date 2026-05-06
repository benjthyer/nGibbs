"""Python translation scaffold for Fortran therm.f (condensed phases)."""
from __future__ import annotations

from math import isfinite
from typing import TYPE_CHECKING

from .ctherm import Ctherm
from .etherm import Etherm
from .ftherm import Ftherm
from .gamset import gamset
from .landau import landau
from .landauqr import landauqr
from .thetacal import thetacal
from .ztherm import Ztherm
from .therm_common import ThermResult, apar_value

if TYPE_CHECKING:
    from .state import HeFESToState


def _thermal_modes_from_apar(state: HeFESToState, ispec: int) -> tuple[float, ...]:
    """Extract mode parameters with conservative defaults."""
    apar = state.apar

    fn = max(apar_value(apar, ispec, 1, 1.0), 1.0e-12)
    zu = max(apar_value(apar, ispec, 2, 1.0), 1.0e-12)

    # Legacy files do not expose a stable global map for all mode slots in this
    # code path yet, so we use robust defaults if values are missing.
    wd1o = max(apar_value(apar, ispec, 16, 800.0), 1.0)
    wd2o = 0.0
    wd3o = 0.0
    ws1o = 0.0
    ws2o = 0.0
    ws3o = 0.0
    we1o = 0.0
    qe1 = 0.0
    we2o = 0.0
    qe2 = 0.0
    we3o = 0.0
    qe3 = 0.0
    we4o = 0.0
    qe4 = 0.0
    wouo = 0.0
    wolo = 0.0

    gammo = 1.2
    qo = 1.0
    if abs(gammo) < 1.0e-12:
        gammo = 1.2
    if abs(qo) < 1.0e-12:
        qo = 1.0
    got = apar_value(apar, ispec, 43, 0.0)

    return (
        fn,
        zu,
        wd1o,
        wd2o,
        wd3o,
        ws1o,
        ws2o,
        ws3o,
        we1o,
        qe1,
        we2o,
        qe2,
        we3o,
        qe3,
        we4o,
        qe4,
        wouo,
        wolo,
        gammo,
        qo,
        got,
    )


def therm(ispec: int, Vi: float, volnl: float, state: HeFESToState) -> ThermResult:
    """Compute condensed-phase thermodynamic properties."""
    apar = state.apar
    rgas = state.Rgas
    ti = max(state.Ti, 1.0e-12)

    to = max(apar_value(apar, ispec, 4, 298.15), 1.0e-12)
    fo = apar_value(apar, ispec, 5, 0.0)
    vo = max(apar_value(apar, ispec, 6, max(Vi, 1.0e-6)), 1.0e-6)
    ko = max(apar_value(apar, ispec, 7, 160.0), 1.0e-6)
    kop = apar_value(apar, ispec, 8, 4.0)
    kopp = apar_value(apar, ispec, 9, 0.0)
    be = apar_value(apar, ispec, 35, 0.0)
    ge = apar_value(apar, ispec, 36, 0.0)
    q2a2 = apar_value(apar, ispec, 37, 0.0)

    (
        fn,
        zu,
        wd1o,
        wd2o,
        wd3o,
        ws1o,
        ws2o,
        ws3o,
        we1o,
        qe1,
        we2o,
        qe2,
        we3o,
        qe3,
        we4o,
        qe4,
        wouo,
        wolo,
        gammo,
        qo,
        got,
    ) = _thermal_modes_from_apar(state, ispec)

    gam = gamset(
        wd1o,
        wd2o,
        wd3o,
        ws1o,
        ws2o,
        ws3o,
        we1o,
        we2o,
        we3o,
        we4o,
        wouo,
        wolo,
        gammo,
        qo,
        got,
        Vi,
        vo,
    )

    wd1 = gam.wd1
    wd2 = gam.wd2
    wd3 = gam.wd3
    ws1 = gam.ws1
    ws2 = gam.ws2
    ws3 = gam.ws3
    we1 = gam.we1
    we2 = gam.we2
    we3 = gam.we3
    we4 = gam.we4
    wou = gam.wou
    wol = gam.wol
    gamma = gam.gamma if isfinite(gam.gamma) else 1.2
    qq = gam.q if isfinite(gam.q) else 1.0
    etas = gam.etas

    # Protect thermal kernels from extreme underflow (x=theta/T -> 0) that
    # can hit log(1-exp(-x)) domain limits in floating-point arithmetic.
    min_mode = 1.0e-6
    wd1 = max(abs(wd1), min_mode)
    wd2 = max(abs(wd2), 0.0)
    wd3 = max(abs(wd3), 0.0)
    ws1 = max(abs(ws1), 0.0)
    ws2 = max(abs(ws2), 0.0)
    ws3 = max(abs(ws3), 0.0)
    we1 = max(abs(we1), 0.0)
    we2 = max(abs(we2), 0.0)
    we3 = max(abs(we3), 0.0)
    we4 = max(abs(we4), 0.0)
    wou = max(abs(wou), 0.0)
    wol = max(abs(wol), 0.0)

    uth = Etherm(ti, fn, zu, wd1, wd2, wd3, ws1, ws2, ws3, wou, wol, we1, we2, we3, we4, qe1, qe2, qe3, qe4)
    uto = Etherm(to, fn, zu, wd1, wd2, wd3, ws1, ws2, ws3, wou, wol, we1, we2, we3, we4, qe1, qe2, qe3, qe4)
    cv = Ctherm(ti, fn, zu, wd1, wd2, wd3, ws1, ws2, ws3, wou, wol, we1, we2, we3, we4, qe1, qe2, qe3, qe4)
    cvo = Ctherm(to, fn, zu, wd1, wd2, wd3, ws1, ws2, ws3, wou, wol, we1, we2, we3, we4, qe1, qe2, qe3, qe4)
    fth = Ftherm(ti, fn, zu, wd1, wd2, wd3, ws1, ws2, ws3, wou, wol, we1, we2, we3, we4, qe1, qe2, qe3, qe4)
    ftho = Ftherm(to, fn, zu, wd1, wd2, wd3, ws1, ws2, ws3, wou, wol, we1, we2, we3, we4, qe1, qe2, qe3, qe4)

    cvn = cv / max(3.0 * fn * rgas, 1.0e-12)
    tcal = ti * thetacal(cvn)
    zeta = Ztherm(ti, fn, zu, wd1, wd2, wd3, ws1, ws2, ws3, wou, wol, we1, we2, we3, we4, qe1, qe2, qe3, qe4, gamma) / max(cvn, 1.0e-12)

    # Birch-Murnaghan core.
    Vi = max(Vi, vo, 1.0e-6)
    f = 0.5 * ((Vi / vo) ** (-2.0 / 3.0) - 1.0)
    a3 = 3.0 * (kop - 4.0)
    a4 = 9.0 * (kopp + kop * (kop - 7.0) + 143.0 / 9.0)
    if kopp == 0.0:
        a4 = 0.0
    a5 = apar_value(apar, ispec, 43, 0.0)

    fbm = 4500.0 * ko * vo * f * f * (1.0 + (a3 / 3.0) * f + (a4 / 12.0) * f * f + (a5 / 60.0) * f * f * f)
    kc = ko * (1.0 + 2.0 * f) ** 2.5 * (
        1.0
        + (7.0 + a3) * f
        + (4.5 * a3 + 0.5 * a4) * f * f
        + ((11.0 / 6.0) * a4 + (1.0 / 6.0) * a5) * f * f * f
        + (13.0 / 24.0) * a5 * f * f * f * f
    )

    pzp = 0.0
    ph = 0.001 * (gamma / max(Vi, 1.0e-12)) * (uth - uto)
    kth = (gamma + 1.0 - qq) * (ph + pzp) - 0.001 * (gamma * gamma / max(Vi, 1.0e-12)) * (ti * cv - to * cvo)

    beta = be * (Vi / vo) ** ge if vo > 0.0 else 0.0
    fel = -(beta / 2.0) * (ti * ti - to * to)

    ftot = 1000.0 * fo + fbm + fth - ftho + 1000.0 * state.Pi * Vi + fel
    ent = (uth - fth) / ti + beta * ti if ti > 0.0 else 0.0

    k = max(kc + kth, 1.0e-8)
    cp = cv * (1.0 + gamma * gamma * cv * ti / (1000.0 * max(Vi, 1.0e-12) * k))
    alp = 0.001 * gamma * cv / (max(Vi, 1.0e-12) * k)
    ks = k * (1.0 + alp * gamma * ti)

    # Landau contribution pathway follows legacy iltyp switch.
    cplan = 0.0
    slan = 0.0
    vlan = 0.0
    betlan = 0.0
    glan = 0.0
    iltyp = int(getattr(state, "iltyp", 0))
    if iltyp == 1:
        _, _, glan, cplan, _, betlan, slan, vlan = landauqr(ispec, Vi, state)
    elif iltyp == 2:
        _, _, glan, cplan, _, betlan, slan, vlan = landau(ispec, Vi, state)

    cp = cp + cplan
    ent = ent + slan
    ftot = ftot + glan
    vi_out = Vi + vlan
    if vi_out > 0.0 and volnl > 0.0:
        k = vi_out / volnl / (1.0 / max(k, 1.0e-8) + betlan)
        alp = volnl / vi_out * alp
        cv = cp - 1000.0 * ti * vi_out * alp * alp * k
        ks = k * (1.0 + alp * gamma * ti)

    # Conservative shear placeholders until full shear branch is translated.
    gsh = max(apar_value(apar, ispec, 44, 0.0), 0.0)
    dgdt = 0.0

    # Debye velocity proxy from K and G.
    wm = max(apar_value(apar, ispec, 3, 1.0), 1.0e-12)
    rho = wm / max(vi_out, 1.0e-12)
    vpc = ((ks + 4.0 * gsh / 3.0) / max(rho, 1.0e-12)) ** 0.5 if (ks + 4.0 * gsh / 3.0) > 0.0 else 0.0
    vsc = (gsh / max(rho, 1.0e-12)) ** 0.5 if gsh > 0.0 else 0.0
    if vpc > 0.0 and vsc > 0.0:
        vdeb3 = 2.0 / 3.0 / (vsc**3) + 1.0 / 3.0 / (vpc**3)
        vdeb = abs(vdeb3) ** (-1.0 / 3.0)
    else:
        vdeb = 0.0
    gamdeb = gamma

    return ThermResult(
        Cp=cp,
        Cv=cv,
        gamma=gamma,
        K=k,
        Ks=ks,
        alp=alp,
        Ftot=ftot,
        ph=ph,
        ent=ent,
        deltas=0.0,
        tcal=tcal,
        zeta=zeta,
        Gsh=gsh,
        uth=uth,
        uto=uto,
        thet=wd1,
        qq=qq,
        etas=etas,
        dGdT=dgdt,
        pzp=pzp,
        Vdeb=vdeb,
        gamdeb=gamdeb,
    )
