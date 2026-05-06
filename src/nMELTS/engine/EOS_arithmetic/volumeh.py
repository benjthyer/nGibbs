"""Approximate Python translation of Fortran volumeh.f (hydrogen EOS spline)."""
from __future__ import annotations

from typing import TYPE_CHECKING

from .therm_common import ideal_gas_volume_cm3

if TYPE_CHECKING:
    from .state import HeFESToState


def volumeh(ispec: int, x1: float, state: HeFESToState) -> float:
    """
    Hydrogen molar volume fallback.

    Fortran uses spline interpolation on hydrogen tables from hydrogen.inc.
    Those tables are not yet exposed in Python, so this fallback returns an
    ideal-gas-like estimate with pressure floor for numerical stability.
    """
    fn = 1.0
    vol = ideal_gas_volume_cm3(fn, max(state.Pi, 1.0e-6), max(state.Ti, 1.0), state.Rgas)

    # Keep values in a practical numeric range for downstream routines.
    return max(0.5, min(vol, 1.0e7))
