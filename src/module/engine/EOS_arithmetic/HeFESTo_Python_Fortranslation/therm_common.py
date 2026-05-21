"""Shared result types and helpers for therm/volume translations."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ThermResult:
    """Thermodynamic outputs used by gspec dispatch routines."""

    Cp: float
    Cv: float
    gamma: float
    K: float
    Ks: float
    alp: float
    Ftot: float
    ph: float
    ent: float
    deltas: float
    tcal: float
    zeta: float
    Gsh: float
    uth: float
    uto: float
    thet: float
    qq: float
    etas: float
    dGdT: float
    pzp: float
    Vdeb: float
    gamdeb: float


def apar_value(apar: np.ndarray, ispec: int, one_based_index: int, default: float = 0.0) -> float:
    """Read one-based Fortran-like apar index safely from a 2D numpy array."""
    idx = one_based_index - 1
    if apar.ndim != 2:
        return default
    if ispec < 0 or ispec >= apar.shape[0]:
        return default
    if idx < 0 or idx >= apar.shape[1]:
        return default
    return float(apar[ispec, idx])


def ideal_gas_volume_cm3(fn: float, pressure_gpa: float, temperature_k: float, rgas: float) -> float:
    """Ideal-gas molar volume in cm^3/mol for pressure in GPa."""
    p_pa = max(pressure_gpa, 1.0e-12) * 1.0e9
    vol_m3 = max(fn, 1.0) * rgas * max(temperature_k, 1.0e-12) / p_pa
    return vol_m3 * 1.0e6
