"""
Landau phase transition free-energy contributions, translated from Fortran landau.f.

Computes order parameter and thermodynamic properties for a Landau-type
phase transition (e.g., alpha-beta quartz at ~850 K).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .state import HeFESToState


def landau(ispec: int, Vi: float, state: HeFESToState) -> tuple[float, float, float, float, float, float, float, float]:
    """
    Compute Landau transition properties for a single species at given conditions.
    
    Args:
        ispec: Species index (0-based).
        Vi: Molar volume (cm^3/mol).
        state: HeFESToState object containing:
            - apar[ispec, 38]: Critical temperature offset (Tco, K).
            - apar[ispec, 39]: Entropy max (smax, J/(mol*K)).
            - apar[ispec, 40]: Volume max (vmax, cm^3/mol).
            - Pi: Pressure (GPa).
            - Ti: Temperature (K).
    
    Returns:
        Tuple (qorder, Tc, glan, cplan, alplan, betlan, slan, vlan):
            - qorder: Order parameter.
            - Tc: Transition temperature (K).
            - glan: Gibbs free-energy contribution (J/mol).
            - cplan: Heat capacity contribution (J/(mol*K)).
            - alplan: Thermal expansivity contribution (1/K).
            - betlan: Isothermal compressibility contribution (1/GPa).
            - slan: Entropy contribution (J/(mol*K)).
            - vlan: Volume contribution (cm^3/mol).
    """
    # Extract Landau parameters from apar
    # Note: Fortran 1-based indexing; Python is 0-based.
    # apar(ispec, 38/39/40) -> apar[ispec, 37/38/39] in 0-based Python
    Tco = state.apar[ispec, 37] if state.apar.shape[1] > 37 else 0.0
    smax = state.apar[ispec, 38] if state.apar.shape[1] > 38 else 0.0
    vmax = state.apar[ispec, 39] if state.apar.shape[1] > 39 else 0.0
    
    # Default return values
    glan = 0.0
    slan = 0.0
    vlan = 0.0
    cplan = 0.0
    alplan = 0.0
    betlan = 0.0
    qorder = 0.0
    Tc = 0.0
    
    # If no Landau transition parameters, return zeros
    if Tco <= 0.0:
        return qorder, Tc, glan, cplan, alplan, betlan, slan, vlan
    
    # Compute transition temperature: Tc = Tco + vmax/(0.001*smax)*P
    Tc = Tco + vmax / (0.001 * smax) * state.Pi
    
    # If above transition temperature, return zeros
    if state.Ti >= Tc:
        return qorder, Tc, glan, cplan, alplan, betlan, slan, vlan
    
    # Compute order parameter: q^4 = (Tc - T) / Tc
    qorder = ((Tc - state.Ti) / Tc) ** (1.0 / 4.0)
    
    # Compute thermodynamic contributions
    glan = smax * ((state.Ti - Tc) * qorder**2 + (1.0 / 3.0) * Tco * qorder**6)
    slan = -smax * qorder**2 * (1.5 - 0.5 * Tco / Tc)
    vlan = -vmax * qorder**2 * (1.0 + 0.5 * state.Ti / Tc * (1.0 - Tco / Tc))
    cplan = smax * state.Ti / (2.0 * Tc * qorder**2) * (1.5 - 0.5 * Tco / Tc)
    
    # Thermal expansivity (three attempts at correct formula in Fortran; using the latest)
    alplan = 0.5 * vmax / Vi / Tc * (
        1.0 / qorder**2 * (1.0 + 0.5 * state.Ti / Tc * (1.0 - Tco / Tc))
        - qorder**2 * (1.0 - Tco / Tc)
    )
    
    # Isothermal compressibility
    betlan = vmax / Vi * vmax / (0.001 * Tc * smax) * state.Ti / Tc * (
        0.5 / qorder**2 * (1.0 + 0.5 * state.Ti / Tc * (1.0 - Tco / Tc))
        + qorder**2 * (Tco / Tc - 0.5)
    )
    
    return qorder, Tc, glan, cplan, alplan, betlan, slan, vlan
