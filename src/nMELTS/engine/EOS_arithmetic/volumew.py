"""Approximate Python translation of Fortran volumew.f (water EOS wrapper)."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .state import HeFESToState


def volumew(ispec: int, rhog: float, state: HeFESToState) -> float:
    """
    Approximate H2O molar volume (cm^3/mol).

    Fortran calls IAPWS routines (CALPRE). Those are not translated yet, so we
    use a physically bounded density model that preserves the same failure mode
    below 273.15 K.
    """
    tmin = 273.15
    if state.Ti < tmin:
        return -1.0

    # Molar mass of water (kg/mol).
    xmcapw = 0.018015268

    # Reference at ~298 K, 0.1 MPa.
    t_ref = 298.15
    rho_ref = 997.0  # kg/m^3
    alpha = 2.57e-4  # 1/K thermal expansion near ambient
    beta = 4.5e-10   # 1/Pa isothermal compressibility near ambient

    p_pa = max(state.Pi, 0.0) * 1.0e9
    drho_t = rho_ref * alpha * (state.Ti - t_ref)
    rho_t = max(100.0, rho_ref - drho_t)
    rho = rho_t * (1.0 + beta * p_pa)

    # cm^3/mol
    return 1.0e6 * xmcapw / rho
