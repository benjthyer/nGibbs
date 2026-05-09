"""Exact Python translation of HeFESTo ``Ctherm.f``."""

from __future__ import annotations

from .heat import Heat
from .thermal_kernel_common import RGAS_J_MOL_K, mode_state


def Ctherm(
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
) -> float:
    if Ti <= 0.0:
        return 0.0

    state = mode_state(fn, zu, wd1, wd2, wd3, ws1, ws2, ws3, wou, wol, qe1, qe2, qe3, qe4)
    su = state.su
    if state.qo == 0.0:
        denom = 1.0 - state.qe
        su = 1.0 / denom if abs(denom) > 1.0e-12 else 1.0e15

    cd = (1.0 / su) * Heat(wd1 / Ti, state.d, 1)
    if state.aniso:
        cd = cd / 3.0 + (Heat(wd2 / Ti, state.d, 1) + Heat(wd3 / Ti, state.d, 1)) / (3.0 * su)

    cs = (1.0 / su) * Heat(ws1 / Ti, state.d, 3)
    if state.aniso:
        cs = cs / 3.0 + (Heat(ws2 / Ti, state.d, 3) + Heat(ws3 / Ti, state.d, 3)) / (3.0 * su)

    ce = qe1 * Heat(we1 / Ti, state.d, 2) + qe2 * Heat(we2 / Ti, state.d, 2) + qe3 * Heat(we3 / Ti, state.d, 2) + qe4 * Heat(we4 / Ti, state.d, 2)
    co = state.qo * Heat(state.wo / Ti, state.d, 4)
    return 3.0 * fn * RGAS_J_MOL_K * (cd + cs + ce + co)
