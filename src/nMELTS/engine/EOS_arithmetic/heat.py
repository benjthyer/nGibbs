"""Exact Python translation of HeFESTo ``Heat.f``."""

from __future__ import annotations

from bisect import bisect_right
from functools import lru_cache
from math import exp, pi
from pathlib import Path
from typing import Tuple

from .hefesto_thermal_data import OPTIC_CONTINUUM_WIDTH_MIN, SIN_ASYMPTOTE_A, SIN_ASYMPTOTE_B


_DOS_INC_PATH = Path(__file__).resolve().parent / "HeFESToRepository" / "dos.inc"


@lru_cache(maxsize=1)
def _load_tables() -> dict[str, Tuple[float, ...]]:
    text = _DOS_INC_PATH.read_text(encoding="utf-8", errors="ignore")

    def _extract(array_name: str) -> Tuple[float, ...]:
        marker = f"data {array_name}/"
        start = text.find(marker)
        if start < 0:
            raise FileNotFoundError(f"Missing {array_name} in {_DOS_INC_PATH}")
        start += len(marker)
        end = text.find("/", start)
        if end < 0:
            raise ValueError(f"Unterminated data block for {array_name} in {_DOS_INC_PATH}")
        raw = text[start:end]
        values = []
        for chunk in raw.replace("\n", " ").replace("&", " ").split(","):
            piece = chunk.strip()
            if not piece:
                continue
            values.append(float(piece.replace("D", "E").replace("d", "e").split()[0]))
        return tuple(values)

    xa = _extract("xad")
    deb1 = _extract("deb1d")
    deb2 = _extract("deb2d")
    deb3 = _extract("deb3d")
    ein1 = _extract("ein1d")
    ein2 = _extract("ein2d")
    ein3 = _extract("ein3d")
    sin1 = _extract("sin1d")
    sin2 = _extract("sin2d")
    sin3 = _extract("sin3d")
    opt1 = _extract("opt1d")
    opt2 = _extract("opt2d")
    opt3 = _extract("opt3d")
    xamin = min(xa)
    xamax = max(xa)
    imin = xa.index(xamin)
    return {
        "xa": xa,
        "deb1": deb1,
        "deb2": deb2,
        "deb3": deb3,
        "ein1": ein1,
        "ein2": ein2,
        "ein3": ein3,
        "sin1": sin1,
        "sin2": sin2,
        "sin3": sin3,
        "opt1": opt1,
        "opt2": opt2,
        "opt3": opt3,
        "xamin": xamin,
        "xamax": xamax,
        "imin": imin,
    }


def _neville(x_values: Tuple[float, ...], y_values: Tuple[float, ...], x: float) -> Tuple[float, float]:
    n = len(x_values)
    if n != len(y_values):
        raise ValueError("x_values and y_values must have the same length")
    if n == 0:
        return 0.0, 0.0

    c = list(y_values)
    d = list(y_values)
    ns = 0
    dif = abs(x - x_values[0])
    for index in range(n):
        dift = abs(x - x_values[index])
        if dift < dif:
            ns = index
            dif = dift

    y = y_values[ns]
    dy = 0.0
    ns -= 1
    for m in range(1, n):
        for i in range(n - m):
            ho = x_values[i] - x
            hp = x_values[i + m] - x
            w = c[i + 1] - d[i]
            den = ho - hp
            if den == 0.0:
                return y, dy
            den = w / den
            d[i] = hp * den
            c[i] = ho * den
        if 2 * (ns + 1) < n - m:
            dy = c[ns + 1]
        else:
            dy = d[ns]
            ns -= 1
        y += dy
    return y, dy


def Heat(x: float, d: float, idos: int) -> float:
    """Evaluate the HeFESTo vibrational heat function."""

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

    def _window_start(value: float) -> int:
        jlo = bisect_right(xa, value) - 1
        return min(max(jlo - ((mint - 1) // 2), 0), len(xa) - mint)

    if idos == 1:
        klo = _window_start(x)
        y, _ = _neville(xa[klo : klo + mint], tables["deb1"][klo : klo + mint], x)
        if x > xamax:
            return (12.0 * pi * pi * pi * pi / 15.0) / (x * x * x)
        if x < xamin:
            return 1.0 - (1.0 - tables["deb1"][imin]) / (xamin * xamin) * x * x
        return y

    if idos == 2:
        klo = _window_start(x)
        y, _ = _neville(xa[klo : klo + mint], tables["ein1"][klo : klo + mint], x)
        if x > xamax:
            return x * x * exp(-x)
        if x < xamin:
            return 1.0 - (1.0 - tables["ein1"][imin]) / (xamin * xamin) * x * x
        return y

    if idos == 3:
        if x > xamax:
            return (12.0 * pi * pi * pi * pi / 15.0) / (x * x * x) * sfac + SIN_ASYMPTOTE_A * SIN_ASYMPTOTE_B * (SIN_ASYMPTOTE_B - 1.0) * x ** (1.0 - SIN_ASYMPTOTE_B)
        if x < xamin:
            return 1.0 - (1.0 - tables["sin1"][imin]) / (xamin * xamin) * x * x
        klo = _window_start(x)
        y, _ = _neville(xa[klo : klo + mint], tables["sin1"][klo : klo + mint], x)
        return y

    if idos == 4:
        if d < OPTIC_CONTINUUM_WIDTH_MIN:
            return Heat(x, d, 2)
        xu = x * (1.0 + d / 2.0)
        xl = x * (1.0 - d / 2.0)
        ku = _window_start(xu)
        kl = _window_start(xl)
        yu, _ = _neville(xa[ku : ku + mint], tables["opt1"][ku : ku + mint], xu)
        yl, _ = _neville(xa[kl : kl + mint], tables["opt1"][kl : kl + mint], xl)
        if xu < xamin:
            yu = 1.0 - (1.0 - tables["opt1"][imin]) / (xamin * xamin) * xu * xu
        if xl < xamin:
            yl = 1.0 - (1.0 - tables["opt1"][imin]) / (xamin * xamin) * xl * xl
        if xu > xamax:
            yu = (pi * pi / 3.0) / xu
        if xl > xamax:
            yl = (pi * pi / 3.0) / xl
        value = (xu * yu - xl * yl) / (xu - xl)
        return value if value >= 0.0 else 0.0

    return 0.0