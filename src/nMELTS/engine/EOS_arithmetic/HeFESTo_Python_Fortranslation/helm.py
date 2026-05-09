"""Exact Python translation of HeFESTo ``Helm.f``."""

from __future__ import annotations

from math import exp, log, pi

from .heat import _load_tables, _neville


def Helm(x: float, d: float, idos: int) -> float:
    if x == 0.0:
        return 0.0
    if idos == 5:
        return log(x) - 1.0 / 3.0

    tables = _load_tables()
    xa = tables["xa"]
    xamin = tables["xamin"]
    xamax = tables["xamax"]
    imin = tables["imin"]
    mint = 4
    sfac = (2.0 / pi) ** 3

    from bisect import bisect_right

    def _window(value: float) -> int:
        jlo = bisect_right(xa, value) - 1
        return min(max(jlo - ((mint - 1) // 2), 0), len(xa) - mint)

    if idos != 4:
        klo = _window(x)

    if idos == 1:
        y, _ = _neville(xa[klo : klo + mint], tables["deb3"][klo : klo + mint], x)
        value = y + log(1.0 - exp(-x))
        if x > xamax:
            return -(pi * pi * pi * pi / 15.0) / (x * x * x)
        if x < xamin:
            return log(x) - 4.0 / 3.0 + 1.0 - (1.0 - tables["deb2"][imin]) / xamin * x
        return value

    if idos == 2:
        value = log(1.0 - exp(-x))
        if x > xamax:
            return -exp(-x)
        return value

    if idos == 3:
        y, _ = _neville(xa[klo : klo + mint], tables["sin3"][klo : klo + mint], x)
        value = y + log(1.0 - exp(-x))
        if x > xamax:
            return -(pi * pi * pi * pi / 15.0) / (x * x * x) * sfac - 23.594 * x ** (1.0 - 6.1)
        if x < xamin:
            return log(x) - 4.0 / 3.0 + 1.0 - (1.0 - tables["sin2"][imin]) / xamin * x + 0.188256
        return value

    if idos == 4:
        if d <= 0.4:
            dx = x * d / 2.0
            u = exp(-x)
            v = u / (1.0 - u)
            fein = log(1.0 - u)
            c2 = -2.0 * v - 2.0 * v * v
            c4 = -2.0 * v - 14.0 * v * v - 24.0 * v * v * v - 12.0 * v * v * v * v
            return fein + c2 * dx * dx / 12.0 + c4 * dx * dx * dx * dx / 240.0

        xu = x * (1.0 + d / 2.0)
        xl = x * (1.0 - d / 2.0)
        ku = _window(xu)
        kl = _window(xl)
        yu, _ = _neville(xa[ku : ku + mint], tables["opt3"][ku : ku + mint], xu)
        yl, _ = _neville(xa[kl : kl + mint], tables["opt3"][kl : kl + mint], xl)
        if xu != 0.0:
            yu += log(1.0 - exp(-xu))
        if xl != 0.0:
            yl += log(1.0 - exp(-xl))
        if xu < xamin:
            yu = log(1.0 - exp(-xu)) - (1.0 - (1.0 - tables["opt2"][imin]) / xamin * xu)
        if xl < xamin:
            yl = log(1.0 - exp(-xl)) - (1.0 - (1.0 - tables["opt2"][imin]) / xamin * xl)
        if xu > xamax:
            yu = -(pi * pi / 6.0) / xu
        if xl > xamax:
            yl = -(pi * pi / 6.0) / xl
        if xl == 0.0:
            yl = 0.0
        if xu == 0.0:
            yu = 0.0
        return (xu * yu - xl * yl) / (xu - xl)

    return 0.0
