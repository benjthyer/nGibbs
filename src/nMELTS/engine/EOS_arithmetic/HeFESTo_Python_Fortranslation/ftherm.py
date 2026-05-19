"""Exact Python translation of HeFESTo ``Ftherm.f``."""

from __future__ import annotations

from .helm import Helm
from .thermal_kernel_common import RGAS_J_MOL_K, mode_state


def Ftherm(
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

    fd = (1.0 / su) * Helm(wd1 / Ti, state.d, 1)
    if state.aniso:
        fd = fd / 3.0 + (Helm(wd2 / Ti, state.d, 1) + Helm(wd3 / Ti, state.d, 1)) / (3.0 * su)
    fd = 3.0 * fn * RGAS_J_MOL_K * Ti * fd + 9.0 / 8.0 * fn * RGAS_J_MOL_K * wd1

    fs = (1.0 / su) * Helm(ws1 / Ti, state.d, 3)
    if state.aniso:
        fs = fs / 3.0 + (Helm(ws2 / Ti, state.d, 3) + Helm(ws3 / Ti, state.d, 3)) / (3.0 * su)
    fs = 3.0 * fn * RGAS_J_MOL_K * Ti * fs

    fe = qe1 * Helm(we1 / Ti, state.d, 2) + qe2 * Helm(we2 / Ti, state.d, 2) + qe3 * Helm(we3 / Ti, state.d, 2) + qe4 * Helm(we4 / Ti, state.d, 2)
    fe = 3.0 * fn * RGAS_J_MOL_K * Ti * fe

    fo = state.qo * Helm(state.wo / Ti, state.d, 4)
    fo = 3.0 * fn * RGAS_J_MOL_K * Ti * fo

    return fd + fs + fe + fo
