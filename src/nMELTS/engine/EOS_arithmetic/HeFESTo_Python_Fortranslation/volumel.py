"""Approximate Python translation of Fortran volumel.f for liquids/vapors."""
from __future__ import annotations

from typing import TYPE_CHECKING

from .therm_common import apar_value, ideal_gas_volume_cm3

if TYPE_CHECKING:
    from .state import HeFESToState


def volumel(ispec: int, x1: float, state: HeFESToState) -> float:
    """
    Liquid/vapor volume estimate.

    Returns positive volume on success, -1.0 for spinodal-like instability,
    -2.0 for numerical/bounds failure.
    """
    apar = state.apar
    rgas = state.Rgas

    vo = apar_value(apar, ispec, 6, default=max(x1, 1.0))
    fn = apar_value(apar, ispec, 1, default=1.0)
    htl = apar_value(apar, ispec, 31, default=1.0)

    vlow = apar_value(apar, ispec, 51, default=1.0)
    vupp = apar_value(apar, ispec, 52, default=1.0e6)
    vsplow = apar_value(apar, ispec, 53, default=vlow)
    vspupp = apar_value(apar, ispec, 54, default=vupp)

    isochor = bool(getattr(state, "isochor", False))
    if isochor:
        return vo

    videal = ideal_gas_volume_cm3(fn, state.Pi, state.Ti, rgas)

    # htl=4 behaves as gas-like liquid branch in legacy code.
    if int(round(htl)) == 4:
        if state.Pi <= 0.0:
            return -1.0
        vlo = max(vspupp, 1.0)
        vhi = max(3.0 * videal, vlo * 1.01)
        vg = videal if vlo <= videal <= vhi else 0.5 * (vlo + vhi)
        return vg

    # htl=1 condensed-liquid branch
    vlo = max(vsplow, 1.0)
    vhi = max(3.0 * vspupp, vlo * 1.01)

    # Start from Murnaghan-like compressed estimate around Vo.
    ko = apar_value(apar, ispec, 7, default=40.0)
    kop = apar_value(apar, ispec, 8, default=4.0)
    if ko <= 0.0:
        return -2.0

    if abs(kop) < 1.0e-12:
        vest = vo
    else:
        base = 1.0 + state.Pi * (kop / ko)
        if base <= 0.0:
            return -1.0
        vest = vo * (base ** (-1.0 / kop))

    if vest != vest or vest <= 0.0:
        return -2.0
    if vest < vlo or vest > vhi:
        return -1.0

    return vest
