"""Python translation of Fortran Ftotsub.f for solid phases only."""
from __future__ import annotations

from typing import TYPE_CHECKING

from .etherm import Etherm
from .ctherm import Ctherm
from .ftherm import Ftherm
from .gamset import gamset
from .landau import landau
from .landauqr import landauqr
from .therm_common import apar_value

if TYPE_CHECKING:
    from .state import HeFESToState


def Ftotsub(ispec: int, Vi: float, state: HeFESToState) -> float:
    """Compute solid-phase Gibbs free energy for a species."""

    apar = state.apar
    ti = max(state.Ti, 1.0e-12)

    fn = max(apar_value(apar, ispec, 1, 1.0), 1.0e-12)
    to = max(apar_value(apar, ispec, 4, 298.15), 1.0e-12)
    fo = apar_value(apar, ispec, 5, 0.0)
    vo = max(apar_value(apar, ispec, 6, max(Vi, 1.0e-12)), 1.0e-12)
    ko = max(apar_value(apar, ispec, 7, 160.0), 1.0e-12)
    kop = apar_value(apar, ispec, 8, 4.0)
    kopp = apar_value(apar, ispec, 9, 0.0)
    wd1o = max(apar_value(apar, ispec, 16, 800.0), 1.0e-12)
    wd2o = max(apar_value(apar, ispec, 17, 0.0), 0.0)
    wd3o = max(apar_value(apar, ispec, 18, 0.0), 0.0)
    ws1o = max(apar_value(apar, ispec, 19, 0.0), 0.0)
    ws2o = max(apar_value(apar, ispec, 20, 0.0), 0.0)
    ws3o = max(apar_value(apar, ispec, 21, 0.0), 0.0)
    we1o = max(apar_value(apar, ispec, 22, 0.0), 0.0)
    qe1 = apar_value(apar, ispec, 23, 0.0)
    we2o = max(apar_value(apar, ispec, 24, 0.0), 0.0)
    qe2 = apar_value(apar, ispec, 25, 0.0)
    we3o = max(apar_value(apar, ispec, 26, 0.0), 0.0)
    qe3 = apar_value(apar, ispec, 27, 0.0)
    we4o = max(apar_value(apar, ispec, 28, 0.0), 0.0)
    qe4 = apar_value(apar, ispec, 29, 0.0)
    wouo = max(apar_value(apar, ispec, 30, 0.0), 0.0)
    wolo = max(apar_value(apar, ispec, 31, 0.0), 0.0)
    gammo = max(apar_value(apar, ispec, 38, 1.2), 1.0e-12)
    qo = max(apar_value(apar, ispec, 39, 0.3), 1.0e-12)
    be = apar_value(apar, ispec, 35, 0.0)
    ge = apar_value(apar, ispec, 36, 0.0)
    a5 = apar_value(apar, ispec, 43, 0.0)
    got = a5
    q2a2 = apar_value(apar, ispec, 37, 0.0)

    gam = gamset(
        wd1o, wd2o, wd3o, ws1o, ws2o, ws3o, we1o, we2o, we3o, we4o, wouo, wolo,
        gammo, qo, got, Vi, vo,
    )
    gamma = gam.gamma if gam.gamma != 0.0 else gammo
    q = gam.q
    thet = gam.wd1

    f = 0.5 * ((Vi / vo) ** (-2.0 / 3.0) - 1.0)
    a3 = 3.0 * (kop - 4.0)
    a4 = 9.0 * (kopp + kop * (kop - 7.0) + 143.0 / 9.0)
    if kopp == 0.0:
        a4 = 0.0
    fbm = 4500.0 * ko * vo * f * f * (1.0 + (a3 / 3.0) * f + (a4 / 12.0) * f * f + (a5 / 60.0) * f * f * f)

    uth = Etherm(ti, fn, 1.0, gam.wd1, gam.wd2, gam.wd3, gam.ws1, gam.ws2, gam.ws3, gam.wou, gam.wol, gam.we1, gam.we2, gam.we3, gam.we4, qe1, qe2, qe3, qe4)
    uto = Etherm(to, fn, 1.0, gam.wd1, gam.wd2, gam.wd3, gam.ws1, gam.ws2, gam.ws3, gam.wou, gam.wol, gam.we1, gam.we2, gam.we3, gam.we4, qe1, qe2, qe3, qe4)
    cv = Ctherm(ti, fn, 1.0, gam.wd1, gam.wd2, gam.wd3, gam.ws1, gam.ws2, gam.ws3, gam.wou, gam.wol, gam.we1, gam.we2, gam.we3, gam.we4, qe1, qe2, qe3, qe4)
    cvo = Ctherm(to, fn, 1.0, gam.wd1, gam.wd2, gam.wd3, gam.ws1, gam.ws2, gam.ws3, gam.wou, gam.wol, gam.we1, gam.we2, gam.we3, gam.we4, qe1, qe2, qe3, qe4)
    fth = Ftherm(ti, fn, 1.0, gam.wd1, gam.wd2, gam.wd3, gam.ws1, gam.ws2, gam.ws3, gam.wou, gam.wol, gam.we1, gam.we2, gam.we3, gam.we4, qe1, qe2, qe3, qe4)
    ftho = Ftherm(to, fn, 1.0, gam.wd1, gam.wd2, gam.wd3, gam.ws1, gam.ws2, gam.ws3, gam.wou, gam.wol, gam.we1, gam.we2, gam.we3, gam.we4, qe1, qe2, qe3, qe4)

    pzp = 0.0
    izp = int(round(apar_value(apar, ispec, 34, 0.0)))
    if abs(izp) == 1:
        pzp = izp * 0.001 * (9.0 / 8.0) * fn * state.Rgas * thet * gamma / max(Vi, 1.0e-12)

    ph = 0.001 * (gamma / max(Vi, 1.0e-12)) * (uth - uto)
    kth = (gamma + 1.0 - q) * (ph + pzp) - 0.001 * (gamma * gamma / max(Vi, 1.0e-12)) * (ti * cv - to * cvo)
    k = ko * (1.0 + 2.0 * f) ** 2.5 * (1.0 + (7.0 + a3) * f + (4.5 * a3 + 0.5 * a4) * f * f + ((11.0 / 6.0) * a4 + (1.0 / 6.0) * a5) * f * f * f + (13.0 / 24.0) * a5 * f * f * f * f) + kth
    cp = cv * (1.0 + gamma * gamma * cv * ti / (1000.0 * max(Vi, 1.0e-12) * max(k, 1.0e-12)))
    alp = 0.001 * gamma * cv / (max(Vi, 1.0e-12) * max(k, 1.0e-12))

    beta = be * (Vi / vo) ** ge if vo > 0.0 else 0.0
    ftot = 1000.0 * fo + fbm + fth - ftho + 1000.0 * state.Pi * Vi + (-(beta / 2.0) * (ti * ti - to * to))

    glan = 0.0
    iltyp = int(getattr(state, "iltyp", 0))
    if iltyp == 1:
        _, _, glan, _, _, _, _, _ = landauqr(ispec, Vi, state)
    elif iltyp == 2:
        _, _, glan, _, _, _, _, _ = landau(ispec, Vi, state)

    _ = (q2a2, cp, alp)
    return ftot + glan