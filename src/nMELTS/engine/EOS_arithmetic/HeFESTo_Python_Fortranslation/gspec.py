"""
Single-species thermodynamic properties, translated from Fortran gspec.f.

Computes all thermodynamic properties (volume, Gibbs free energy, entropy, etc.)
for a single species at given P,T conditions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .state import HeFESToState

from .Ftotsub import Ftotsub
from .therm import therm
from .therml import therml
from .thermg import thermg
from .thermw import thermw
from .thermh import thermh
from .volume import volume
from .volumel import volumel
from .volumew import volumew
from .volumeh import volumeh
from .therm_common import apar_value, ideal_gas_volume_cm3


@dataclass(frozen=True)
class GspecResult:
    """Result structure for gspec computation."""
    vol: float  # Molar volume (cm^3/mol)
    Cp: float  # Heat capacity at constant pressure (J/(mol*K))
    Cv: float  # Heat capacity at constant volume (J/(mol*K))
    gamma: float  # Gruneisen parameter
    K: float  # Bulk modulus (GPa)
    Ks: float  # Adiabatic bulk modulus (GPa)
    alp: float  # Thermal expansivity (1/K)
    Ftot: float  # Gibbs free energy (J/mol)
    ph: float  # Enthalpy-related property
    ent: float  # Entropy (J/(mol*K))
    deltas: float  # Entropy excess
    tcal: float  # Calibrated Debye temperature (K)
    zeta: float  # Order parameter or similar
    Gsh: float  # Shear modulus (GPa)
    uth: float  # Thermal internal energy property
    uto: float  # Reference internal energy property
    thet: float  # Theta parameter
    qq: float  # Q parameter
    etas: float  # Eta parameter
    dGdT: float  # Gibbs free energy derivative wrt T
    pzp: float  # Pressure-zero-point correction
    Vdeb: float  # Debye volume contribution
    gamdeb: float  # Debye gamma
    spinodal: bool  # Spinodal instability flag


def gspec(ispec: int, state: HeFESToState) -> tuple[GspecResult, bool]:
    """
    Compute all thermodynamic properties for a single species.
    
    Args:
        ispec: Species index (0-based).
        state: HeFESToState containing:
            - apar: Parameter matrix for species.
            - Ti, Pi: Temperature (K) and pressure (GPa).
            - etc.
    
    Returns:
        Tuple (result, spinodal):
            - result: GspecResult with all computed properties.
            - spinodal: True if species is spinodally unstable.
    
    Notes:
        - High-temperature limit (htl) determines which therm variant to use.
        - Volume solver (volume/volumel/volumew/volumeh) called based on htl.
        - Spinodal flag set if volume solver fails.
        - Thermal properties computed from Heat() function tables.
    
    TODO:
        - Implement volume solver functions (volume, volumel, volumew, etc).
        - Integrate therm/therml/thermg/thermw/thermh calls.
        - Integrate Ftotsub* calls for final free energy.
    """
    apar = state.apar
    
    # Retrieve key parameters for this species
    fn = apar_value(apar, ispec, 1, default=1.0)
    htl = int(round(apar_value(apar, ispec, 31, default=0.0)))
    
    # Initialize result structure
    result = GspecResult(
        vol=0.0, Cp=0.0, Cv=0.0, gamma=0.0, K=0.0, Ks=0.0, alp=0.0,
        Ftot=0.0, ph=0.0, ent=0.0, deltas=0.0, tcal=0.0, zeta=0.0,
        Gsh=0.0, uth=0.0, uto=0.0, thet=0.0, qq=0.0, etas=0.0,
        dGdT=0.0, pzp=0.0, Vdeb=0.0, gamdeb=0.0, spinodal=False
    )
    
    spinodal = False
    
    # Guard against very low temperature
    Tsmall = 1e-5
    if state.Ti < Tsmall:
        return result, spinodal
    
    # Compute reference volume
    Vo = apar_value(apar, ispec, 6, default=40.0)
    x1 = Vo
    
    # For ideal-gas branch seed with ideal-gas volume.
    if htl == 4:
        x1 = ideal_gas_volume_cm3(fn, state.Pi, state.Ti, state.Rgas)

    # Volume dispatch (matches Fortran gspec.f branch table).
    if htl == 0:
        vol = volume(ispec, x1, state)
    elif htl in (1, 4):
        vol = volumel(ispec, x1, state)
    elif htl == 3:
        vol = volumew(ispec, x1, state)
    elif htl == 5:
        vol = volumeh(ispec, x1, state)
    else:
        vol = Vo
    
    # Volume failed for physical/numerical reasons: mark spinodal and continue.
    if vol in (-1.0, -2.0) or vol <= 0.0:
        spinodal = True
        vol = x1 if x1 > 0.0 else Vo

    volnl = vol

    # Thermodynamic dispatch (matches Fortran gspec.f branch table).
    if htl == 0:
        tr = therm(ispec, vol, volnl, state)
    elif htl in (1, 4):
        tr = therml(ispec, vol, volnl, state)
    elif htl == 2:
        tr = thermg(ispec, vol, volnl, state)
    elif htl == 3:
        tr = thermw(ispec, vol, volnl, state)
    elif htl == 5:
        tr = thermh(ispec, vol, volnl, state)
    else:
        tr = therm(ispec, vol, volnl, state)
    
    if htl == 0:
        ftot = Ftotsub(ispec, volnl, state)
    else:
        ftot = tr.Ftot

    # Build result
    result_dict = {
        'vol': vol,
        'Cp': tr.Cp,
        'Cv': tr.Cv,
        'gamma': tr.gamma,
        'K': tr.K,
        'Ks': tr.Ks,
        'alp': tr.alp,
        'Ftot': ftot,
        'ph': tr.ph,
        'ent': tr.ent,
        'deltas': tr.deltas,
        'tcal': tr.tcal,
        'zeta': tr.zeta,
        'Gsh': tr.Gsh,
        'uth': tr.uth,
        'uto': tr.uto,
        'thet': tr.thet,
        'qq': tr.qq,
        'etas': tr.etas,
        'dGdT': tr.dGdT,
        'pzp': tr.pzp,
        'Vdeb': tr.Vdeb,
        'gamdeb': tr.gamdeb,
        'spinodal': spinodal,
    }
    result = GspecResult(**result_dict)
    
    return result, spinodal
