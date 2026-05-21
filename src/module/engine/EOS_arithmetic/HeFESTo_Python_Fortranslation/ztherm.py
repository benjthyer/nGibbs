"""Exact Python translation of HeFESTo ``Ztherm.f``."""

from __future__ import annotations

from math import exp

from .heat import Heat
from .thermal_kernel_common import mode_state


def Ztherm(
    Ti: float,
    fn: float,
    zu: float,
    wd1: float,
    wd2: float,
    wd3: float,
    ws1: float,
    ws2: float,
    ws3: float,
    wou: float,
    wol: float,
    we1: float,
    we2: float,
    we3: float,
    we4: float,
    qe1: float,
    qe2: float,
    qe3: float,
    qe4: float,
    gamma: float,
) -> float:
    if Ti <= 0.0:
        return 0.0

    state = mode_state(fn, zu, wd1, wd2, wd3, ws1, ws2, ws3, wou, wol, qe1, qe2, qe3, qe4)
    su = state.su
    if state.qo == 0.0:
        su = 1.0 / (1.0 - state.qe)

    cvd1 = Heat(wd1 / Ti, state.d, 1)
    cve1 = Heat(wd1 / Ti, state.d, 2)
    zetad = (1.0 / su) * 3.0 * gamma * (cvd1 - cve1)
    if state.aniso:
        cvd2 = Heat(wd2 / Ti, state.d, 1)
        cve2 = Heat(wd2 / Ti, state.d, 2)
        cvd3 = Heat(wd3 / Ti, state.d, 1)
        cve3 = Heat(wd3 / Ti, state.d, 2)
        zeta1 = zetad / 3.0
        zeta2 = (1.0 / (3.0 * su)) * 3.0 * gamma * (cvd2 - cve2)
        zeta3 = (1.0 / (3.0 * su)) * 3.0 * gamma * (cvd3 - cve3)
        zetad = zeta1 + zeta2 + zeta3

    eps = 1.0e-3
    xs = ws1 / Ti
    x1 = xs - eps
    x2 = xs + eps
    cs1 = Heat(x1, state.d, 3)
    cs2 = Heat(x2, state.d, 3)
    zetas = -xs * gamma * (cs2 - cs1) / (2.0 * eps) / su
    if state.aniso:
        zeta1 = zetas / 3.0
        xs = ws2 / Ti
        x1 = xs - eps
        x2 = xs + eps
        cs1 = Heat(x1, state.d, 3)
        cs2 = Heat(x2, state.d, 3)
        zeta2 = -xs * gamma * (cs2 - cs1) / (2.0 * eps) / (3.0 * su)
        xs = ws3 / Ti
        x1 = xs - eps
        x2 = xs + eps
        cs1 = Heat(x1, state.d, 3)
        cs2 = Heat(x2, state.d, 3)
        zeta3 = -xs * gamma * (cs2 - cs1) / (2.0 * eps) / (3.0 * su)
        zetas = zeta1 + zeta2 + zeta3

    xe1 = we1 / Ti
    xe2 = we2 / Ti
    xe3 = we3 / Ti
    xe4 = we4 / Ti
    ce1 = Heat(xe1, state.d, 2)
    ce2 = Heat(xe2, state.d, 2)
    ce3 = Heat(xe3, state.d, 2)
    ce4 = Heat(xe4, state.d, 2)
    zetae1 = qe1 * ce1 * (xe1 * exp(xe1) / (exp(xe1) - 1.0) - 1.0 - xe1 / 2.0) if xe1 != 0.0 else 0.0
    zetae2 = qe2 * ce2 * (xe2 * exp(xe2) / (exp(xe2) - 1.0) - 1.0 - xe2 / 2.0) if xe2 != 0.0 else 0.0
    zetae3 = qe3 * ce3 * (xe3 * exp(xe3) / (exp(xe3) - 1.0) - 1.0 - xe3 / 2.0) if xe3 != 0.0 else 0.0
    zetae4 = qe4 * ce4 * (xe4 * exp(xe4) / (exp(xe4) - 1.0) - 1.0 - xe4 / 2.0) if xe4 != 0.0 else 0.0
    zetae = 2.0 * gamma * (zetae1 + zetae2 + zetae3 + zetae4)

    xu = wou / Ti
    xl = wol / Ti
    dx = xu - xl
    ceu = Heat(xu, state.d, 2)
    cel = Heat(xl, state.d, 2)
    co = Heat(state.wo / Ti, state.d, 4)
    zetao = state.qo * gamma * (co - xu * ceu / dx + xl * cel / dx) if dx != 0.0 else 0.0

    return zetad + zetas + zetae + zetao
