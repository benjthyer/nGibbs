"""
Landau phase transition (Q-reference state), translated from Fortran landauqr.f.

Computes order parameter and thermodynamic properties for a Landau-type
phase transition with Q-referenced (ordered) reference state.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .state import HeFESToState


def _hillert_stub(Ti: float, Tco: float, p: float, smax: float) -> tuple[float, float, float]:
    """
    Stub for Hillert's magnetic/structural transition model.
    
    Used for special phases (BCC iron, magnetite, wustite).
    Returns (glan, slan, cplan) contributions.
    
    TODO: Implement full Hillert model if needed.
    """
    # Placeholder: return zero contributions
    return 0.0, 0.0, 0.0


def landauqr(ispec: int, Vi: float, state: HeFESToState, lphase: dict | None = None) -> tuple[float, float, float, float, float, float, float, float]:
    """
    Compute Landau transition properties (Q-referenced) for a single species.
    
    This variant assumes the reference state is ORDERED (q=1), unlike the
    standard Landau formulation.
    
    Args:
        ispec: Species index (0-based).
        Vi: Molar volume (cm^3/mol).
        state: HeFESToState object containing:
            - apar[ispec, 38]: Critical temperature offset (Tco, K).
            - apar[ispec, 39]: Entropy max (smax, J/(mol*K)).
            - apar[ispec, 40]: Volume max (vmax, cm^3/mol).
            - Pi: Pressure (GPa).
            - Ti: Temperature (K).
            - f, n, s: (Optional) Phase/species membership and composition matrices.
        lphase: (Optional) Mapping from species index to phase index.
    
    Returns:
        Tuple (qorder, Tc, glan, cplan, alplan, betlan, slan, vlan):
            - qorder: Order parameter (may exceed 1.0).
            - Tc: Transition temperature (K).
            - glan: Gibbs free-energy contribution (J/mol).
            - cplan: Heat capacity contribution (J/(mol*K)).
            - alplan: Thermal expansivity contribution (1/K).
            - betlan: Isothermal compressibility contribution (1/GPa).
            - slan: Entropy contribution (J/(mol*K)).
            - vlan: Volume contribution (cm^3/mol).
    
    Notes:
        - Q^4 = (T_C - T) / T_C0 (not normalized by T_C as in landau.py).
        - Transition parameters are limited to prevent runaway at high P.
        - Special handling for BCC iron, magnetite, wustite (currently stubbed).
    """
    # Extract Landau parameters from apar (0-based indexing)
    Tco = state.apar[ispec, 37] if state.apar.shape[1] > 37 else 0.0
    smax = state.apar[ispec, 38] if state.apar.shape[1] > 38 else 0.0
    vmax = state.apar[ispec, 39] if state.apar.shape[1] > 39 else 0.0
    
    # Parameters
    qmax = 1.5  # Maximum order parameter (limited as of April 2022)
    Tcmax = 10.0  # Maximum Tc/Tco ratio (limited as of July 2024)
    
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
    
    # Check for special phases (BCC iron, magnetite, wustite)
    # BCC iron: smax ~ 9.46028, Tco ~ 1043.01
    if abs(smax - 9.46028) <= 1e-5 and abs(Tco - 1043.01) <= 1e-5:
        p = 0.40
        glan_h, slan_h, cplan_h = _hillert_stub(state.Ti, Tco, p, smax)
        return 0.0, Tco, glan_h, cplan_h, 0.0, 0.0, slan_h, 0.0
    
    # Magnetite: Tco ~ 845.5
    if abs(Tco - 845.5) <= 1.0:
        p = 0.40
        glan_h, slan_h, cplan_h = _hillert_stub(state.Ti, Tco, p, smax)
        return 0.0, Tco, glan_h, cplan_h, 0.0, 0.0, slan_h, 0.0
    
    # Wustite: Tco ~ 191.0 (currently commented out in Fortran)
    # Could be activated if needed
    
    # Compute transition temperature: Tc = Tco + vmax/(0.001*smax)*P
    Tc = Tco + vmax / (0.001 * smax) * state.Pi
    
    # Limit Tc to prevent runaway at high pressure
    if Tc > Tcmax * Tco:
        Tc = Tcmax * Tco
    
    # Compute order parameter and contributions below Tc
    qorder = 0.0
    if state.Ti <= Tc:
        qorder = ((Tc - state.Ti) / Tco) ** (1.0 / 4.0)
        # Note: Could limit qorder here, but current Fortran limits Tc instead
        # if qorder > qmax:
        #     qorder = qmax
        
        cplan = smax * state.Ti / (2.0 * Tco * qorder**2)
        alplan = vmax / Vi / (2.0 * Tco * qorder**2)
        betlan = vmax / Vi * vmax / (0.001 * smax) / (2.0 * Tco * qorder**2)
    
    # Free energy and volume contributions (normalized differently from landau.f)
    glan = smax * ((state.Ti - Tc) * (qorder**2 - 1.0) + (1.0 / 3.0) * Tco * (qorder**6 - 1.0))
    slan = -smax * (qorder**2 - 1.0)
    vlan = -vmax * (qorder**2 - 1.0)
    
    return qorder, Tc, glan, cplan, alplan, betlan, slan, vlan
