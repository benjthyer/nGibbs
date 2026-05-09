"""Exact Python translation of HeFESTo ``gamset.f``."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt


@dataclass(frozen=True)
class HeFESToGamsetResult:
    wd1: float
    wd2: float
    wd3: float
    ws1: float
    ws2: float
    ws3: float
    we1: float
    we2: float
    we3: float
    we4: float
    wou: float
    wol: float
    gamma: float
    q: float
    etas: float
    detasdv: float
    qp: float


def gamset(
    wd1o: float,
    wd2o: float,
    wd3o: float,
    ws1o: float,
    ws2o: float,
    ws3o: float,
    we1o: float,
    we2o: float,
    we3o: float,
    we4o: float,
    wouo: float,
    wolo: float,
    gammo: float,
    qo: float,
    Got: float,
    Vi: float,
    Vo: float,
    *,
    ityp: int = 3,
) -> HeFESToGamsetResult:
    """Rescale the thermal frequencies exactly as HeFESTo does in ``gamset.f``."""

    if Vo == 0.0:
        raise ZeroDivisionError("Vo must be non-zero for gamset")

    if ityp == 1:
        E = -0.5 * ((Vi / Vo) ** (-2.0 / 3.0) - 1.0)
        g = -6.0 * gammo
        h = g * (2.0 + 3.0 * qo + g)
        factor = sqrt(1.0 + g * E + 0.5 * h * E * E)
        wd1 = wd1o * factor
        wd2 = wd2o * factor
        wd3 = wd3o * factor
        ws1 = ws1o * factor
        ws2 = ws2o * factor
        ws3 = ws3o * factor
        we1 = we1o * factor
        we2 = we2o * factor
        we3 = we3o * factor
        we4 = we4o * factor
        wou = wouo * factor
        wol = wolo * factor
        gamma = -(1.0 - 2.0 * E) * (g + h * E) / (6.0 * (1.0 + g * E + 0.5 * h * E * E))
        q = 2.0 * gamma - 2.0 / 3.0 + h / 3.0 * (1.0 - 2.0 * E) / (g + h * E)
        etas = Got * gamma / gammo
        return HeFESToGamsetResult(wd1, wd2, wd3, ws1, ws2, ws3, we1, we2, we3, we4, wou, wol, gamma, q, etas, 0.0, 0.0)

    if ityp == 2:
        f = 0.5 * ((Vi / Vo) ** (-2.0 / 3.0) - 1.0)
        a = 6.0 * gammo
        b = gammo * (36.0 * gammo - 18.0 * qo - 12.0)
        bs = -2.0 * gammo - 2.0 * Got
        factor = sqrt(1.0 + a * f + 0.5 * b * f * f)
        wd1 = wd1o * factor
        wd2 = wd2o * factor
        wd3 = wd3o * factor
        ws1 = ws1o * factor
        ws2 = ws2o * factor
        ws3 = ws3o * factor
        we1 = we1o * factor
        we2 = we2o * factor
        we3 = we3o * factor
        we4 = we4o * factor
        wou = wouo * factor
        wol = wolo * factor
        ratio = 1.0 / (1.0 + a * f + 0.5 * b * f * f)
        gamma = (1.0 / 3.0) * 0.5 * ratio * (2.0 * f + 1.0) * (a + b * f)
        q = (1.0 / 9.0) * (18.0 * gamma * gamma - 6.0 * gamma - 0.5 * ratio * (2.0 * f + 1.0) ** 2 * b) / gamma
        qp = (
            (1.0 / 9.0)
            * (
                36.0 * gamma * gamma
                - 6.0 * gamma
                - b * ratio * ratio / (6.0 * q) * (a + b * f) * (2.0 * f + 1.0) ** 3
                + 2.0 * b * ratio / (3.0 * q) * (2.0 * f + 1.0) ** 2
            )
            / gamma
            - q
        )
        etas = -gamma - 0.5 * ratio * (2.0 * f + 1.0) ** 2 * bs
        detasdv = -gamma * q / Vi + 2.0 * bs * (1.0 + 2.0 * f) ** (7.0 / 2.0) * ratio / (3.0 * Vo) - bs * (1.0 + 2.0 * f) ** (9.0 / 2.0) * ratio * ratio / (6.0 * Vo) * (a + b * f)
        return HeFESToGamsetResult(wd1, wd2, wd3, ws1, ws2, ws3, we1, we2, we3, we4, wou, wol, gamma, q, etas, detasdv, qp)

    q = qo
    gamma = gammo * (Vi / Vo) ** q
    scale = exp((gammo - gamma) / q)
    wd1 = wd1o * scale
    wd2 = wd2o * scale
    wd3 = wd3o * scale
    ws1 = ws1o * scale
    ws2 = ws2o * scale
    ws3 = ws3o * scale
    we1 = we1o * scale
    we2 = we2o * scale
    we3 = we3o * scale
    we4 = we4o * scale
    wou = wouo * scale
    wol = wolo * scale
    return HeFESToGamsetResult(wd1, wd2, wd3, ws1, ws2, ws3, we1, we2, we3, we4, wou, wol, gamma, q, 0.0, 0.0, 0.0)