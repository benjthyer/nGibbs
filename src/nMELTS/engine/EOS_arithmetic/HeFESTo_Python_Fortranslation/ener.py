"""Exact Python translation of HeFESTo ``Ener.f``."""

from __future__ import annotations

from math import exp, pi

from .heat import _load_tables, _neville


def Ener(x: float, d: float, idos: int) -> float:
    if x == 0.0:
        return 0.0
    if idos == 5:
        return 1.0

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
        y, _ = _neville(xa[klo : klo + mint], tables["deb2"][klo : klo + mint], x)
        if x > xamax:
            return (pi * pi * pi * pi / 5.0) / (x * x * x)
        if x < xamin:
            return 1.0 - (1.0 - tables["deb2"][imin]) / xamin * x
        return y

    if idos == 2:
        y, _ = _neville(xa[klo : klo + mint], tables["ein2"][klo : klo + mint], x)
        if x > xamax:
            return x * exp(-x)
        if x < xamin:
            return x / (exp(x) - 1.0)
        return y

    if idos == 3:
        y, _ = _neville(xa[klo : klo + mint], tables["sin2"][klo : klo + mint], x)
        if x > xamax:
            return (pi * pi * pi * pi / 5.0) / (x * x * x) * sfac + 23.594 * (6.1 - 1.0) * x ** (1.0 - 6.1)
        if x < xamin:
            return 1.0 - (1.0 - tables["sin2"][imin]) / xamin * x
        return y

    if idos == 4:
        if d < 0.01:
            return Ener(x, d, 2)
        xu = x * (1.0 + d / 2.0)
        xl = x * (1.0 - d / 2.0)
        ku = _window(xu)
        kl = _window(xl)
        yu, _ = _neville(xa[ku : ku + mint], tables["opt2"][ku : ku + mint], xu)
        yl, _ = _neville(xa[kl : kl + mint], tables["opt2"][kl : kl + mint], xl)
        if xu < xamin:
            yu = 1.0 - (1.0 - tables["opt2"][imin]) / xamin * xu
        if xl < xamin:
            yl = 1.0 - (1.0 - tables["opt2"][imin]) / xamin * xl
        if xu > xamax:
            yu = (pi * pi / 6.0) / xu
        if xl > xamax:
            yl = (pi * pi / 6.0) / xl
        value = (xu * yu - xl * yl) / (xu - xl)
        return value if value >= 0.0 else 0.0

    return 0.0
