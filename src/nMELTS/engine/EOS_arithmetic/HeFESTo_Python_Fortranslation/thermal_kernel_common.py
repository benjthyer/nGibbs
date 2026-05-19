"""Shared helpers for exact HeFESTo thermal kernel translations."""

from __future__ import annotations

from dataclasses import dataclass


RGAS_J_MOL_K = 8.314462618


@dataclass(frozen=True)
class HeFESToModeState:
    su: float
    wo: float
    d: float
    wdav: float
    wsav: float
    qe: float
    qo: float
    aniso: bool


def mode_state(
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
    qe1: float,
    qe2: float,
    qe3: float,
    qe4: float,
) -> HeFESToModeState:
    su = fn * zu
    wo = 0.5 * (wou + wol)
    d = 0.0
    if wo != 0.0:
        d = (wou - wol) / wo
    wdav = (wd1 + wd2 + wd3) / 3.0
    wsav = (ws1 + ws2 + ws3) / 3.0
    if wdav == 0.0 and wsav == 0.0:
        su = 1.0e15
    qe = qe1 + qe2 + qe3 + qe4
    qo = 1.0 - 1.0 / su - qe
    if wo == 0.0:
        qo = 0.0
    aniso = wdav != wd1 / 3.0 or wsav != ws1 / 3.0
    if qe == 0.0 and qo == 0.0:
        su = 1.0
    if qo == 0.0:
        denom = 1.0 - qe
        su = 1.0 / denom if abs(denom) > 1.0e-12 else 1.0e15
    return HeFESToModeState(su=su, wo=wo, d=d, wdav=wdav, wsav=wsav, qe=qe, qo=qo, aniso=aniso)
