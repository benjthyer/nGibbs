"""Exact Python translation of HeFESTo ``Etherm.f``."""

from __future__ import annotations

from .ener import Ener
from .thermal_kernel_common import RGAS_J_MOL_K, mode_state


def Etherm(
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

    ud = (1.0 / su) * Ener(wd1 / Ti, state.d, 1)
    if state.aniso:
        ud = ud / 3.0 + (Ener(wd2 / Ti, state.d, 1) + Ener(wd3 / Ti, state.d, 1)) / (3.0 * su)
    ud = 3.0 * fn * RGAS_J_MOL_K * Ti * ud + 9.0 / 8.0 * fn * RGAS_J_MOL_K * wd1

    us = (1.0 / su) * Ener(ws1 / Ti, state.d, 3)
    if state.aniso:
        us = us / 3.0 + (Ener(ws2 / Ti, state.d, 3) + Ener(ws3 / Ti, state.d, 3)) / (3.0 * su)
    us = 3.0 * fn * RGAS_J_MOL_K * Ti * us

    ue = qe1 * Ener(we1 / Ti, state.d, 2) + qe2 * Ener(we2 / Ti, state.d, 2) + qe3 * Ener(we3 / Ti, state.d, 2) + qe4 * Ener(we4 / Ti, state.d, 2)
    ue = 3.0 * fn * RGAS_J_MOL_K * Ti * ue

    uo = state.qo * Ener(state.wo / Ti, state.d, 4)
    uo = 3.0 * fn * RGAS_J_MOL_K * Ti * uo

    return ud + us + ue + uo
