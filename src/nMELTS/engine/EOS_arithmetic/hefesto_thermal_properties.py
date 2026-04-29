"""Temperature-dependent thermodynamic property calculations for HeFESTo.

This module computes species-level thermodynamic properties from HeFESTo
parameters, mirroring the Fortran physub/Ctherm implementation. It uses
Debye and Einstein models with component-level aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass
from bisect import bisect_right
from functools import lru_cache
from math import exp, pi
from pathlib import Path
from typing import Tuple

import numpy as np

# Constants
RGAS_J_MOL_K = 8.314462618  # J/(mol·K)
HCOK = 1.4387769  # hc/k in cm·K (converts cm^-1 to Kelvin)

# HeFESTo Heat.f constants for Sin high-T tail and optic continuum handling.
_SIN_A = 23.594
_SIN_B = 6.1
_OPTIC_DMIN = 0.01
_DOS_INC_PATH = Path(__file__).resolve().parent / "HeFESToRepository" / "dos.inc"


@dataclass(frozen=True)
class ComponentThermodynamicState:
    heat_capacity_p: float
    heat_capacity_v: float
    thermal_expansivity: float
    gruneisen_parameter: float
    entropy: float


@lru_cache(maxsize=1)
def _load_heat_tables() -> dict:
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
        token = ""
        for chunk in raw.replace("\n", " ").replace("&", " ").split(","):
            piece = chunk.strip()
            if not piece:
                continue
            token = piece.replace("D", "E").replace("d", "e")
            values.append(float(token.split()[0]))
        return tuple(values)

    xa = _extract("xad")
    deb1 = _extract("deb1d")
    ein1 = _extract("ein1d")
    sin1 = _extract("sin1d")
    opt1 = _extract("opt1d")
    xamin = min(xa)
    xamax = max(xa)
    imin = xa.index(xamin)
    return {
        "xa": xa,
        "deb1": deb1,
        "ein1": ein1,
        "sin1": sin1,
        "opt1": opt1,
        "xamin": xamin,
        "xamax": xamax,
        "imin": imin,
    }


def _neville_interpolate(x_values: Tuple[float, ...], y_values: Tuple[float, ...], x: float) -> float:
    n = len(x_values)
    if n != len(y_values):
        raise ValueError("x_values and y_values must have the same length")
    if n == 0:
        return 0.0

    c = list(y_values)
    d = list(y_values)
    ns = 0
    dif = abs(x - x_values[0])
    for i in range(n):
        dift = abs(x - x_values[i])
        if dift < dif:
            ns = i
            dif = dift

    y = y_values[ns]
    ns -= 1
    for m in range(1, n):
        for i in range(n - m):
            ho = x_values[i] - x
            hp = x_values[i + m] - x
            w = c[i + 1] - d[i]
            den = ho - hp
            if den == 0.0:
                return y
            den = w / den
            d[i] = hp * den
            c[i] = ho * den
        if 2 * (ns + 1) < n - m:
            dy = c[ns + 1]
        else:
            dy = d[ns]
            ns -= 1
        y += dy
    return y


def _debye_function(x: float) -> float:
    tables = _load_heat_tables()
    xa = tables["xa"]
    deb1 = tables["deb1"]
    xamin = tables["xamin"]
    xamax = tables["xamax"]
    imin = tables["imin"]
    if x == 0.0:
        return 0.0
    if x > xamax:
        return (12.0 * pi**4 / 15.0) / (x * x * x)
    if x < xamin:
        return 1.0 - (1.0 - deb1[imin]) / (xamin * xamin) * x * x
    klo = min(max(bisect_right(xa, x) - 2, 0), len(xa) - 4)
    return _neville_interpolate(xa[klo : klo + 4], deb1[klo : klo + 4], x)


def _einstein_function(x: float) -> float:
    tables = _load_heat_tables()
    xa = tables["xa"]
    ein1 = tables["ein1"]
    xamin = tables["xamin"]
    xamax = tables["xamax"]
    imin = tables["imin"]
    if x == 0.0:
        return 0.0
    if x > xamax:
        return x * x * exp(-x)
    if x < xamin:
        return 1.0 - (1.0 - ein1[imin]) / (xamin * xamin) * x * x
    klo = min(max(bisect_right(xa, x) - 2, 0), len(xa) - 4)
    return _neville_interpolate(xa[klo : klo + 4], ein1[klo : klo + 4], x)


def _heat_function(x: float, d: float, idos: int) -> float:
    """Fortran-style table interpolation for HeFESTo Heat(x,d,idos)."""
    tables = _load_heat_tables()
    xa = tables["xa"]
    xamin = tables["xamin"]
    xamax = tables["xamax"]
    imin = tables["imin"]
    sfac = (2.0 / pi) ** 3

    if x == 0.0:
        return 0.0
    if idos == 5:
        return 1.0

    if idos == 1:
        return _debye_function(x)

    if idos == 2:
        return _einstein_function(x)

    if idos == 3:
        if x > 100.0:
            return (12.0 * pi**4 / 15.0) / (x**3) * sfac + _SIN_A * _SIN_B * (_SIN_B - 1.0) * x ** (1.0 - _SIN_B)
        if x < xamin:
            return 1.0 - (1.0 - tables["sin1"][imin]) / (xamin * xamin) * x * x
        klo = min(max(bisect_right(xa, x) - 2, 0), len(xa) - 4)
        return _neville_interpolate(xa[klo : klo + 4], tables["sin1"][klo : klo + 4], x)

    if idos == 4:
        if d < _OPTIC_DMIN:
            return _einstein_function(x)

        xu = x * (1.0 + d / 2.0)
        xl = x * (1.0 - d / 2.0)
        if abs(xu - xl) < 1.0e-12:
            return _einstein_function(x)

        opt1 = tables["opt1"]
        ku = min(max(bisect_right(xa, xu) - 2, 0), len(xa) - 4)
        kl = min(max(bisect_right(xa, xl) - 2, 0), len(xa) - 4)
        yu = _neville_interpolate(xa[ku : ku + 4], opt1[ku : ku + 4], xu)
        yl = _neville_interpolate(xa[kl : kl + 4], opt1[kl : kl + 4], xl)
        if xu < xamin:
            yu = 1.0 - (1.0 - opt1[imin]) / (xamin * xamin) * xu * xu
        if xl < xamin:
            yl = 1.0 - (1.0 - opt1[imin]) / (xamin * xamin) * xl * xl
        if xu > xamax:
            yu = (pi * pi / 3.0) / xu
        if xl > xamax:
            yl = (pi * pi / 3.0) / xl
        value = (xu * yu - xl * yl) / (xu - xl)
        return max(value, 0.0)

    return 0.0


def rescale_component_thermal_modes(
    *,
    reference_volume: float,
    current_volume: float,
    gamma_0: float,
    q_0: float,
    got: float,
    debye_temp: float,
    debye_temps_2_3: Tuple[float, float] = (0.0, 0.0),
    sin_temps: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    optic_continuum_upper: float = 0.0,
    optic_continuum_lower: float = 0.0,
    einstein_temps: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    ityp: int = 3,
) -> dict:
    """Rescale HeFESTo thermal mode temperatures using the Fortran gamset branches."""

    wd1o = max(debye_temp, 0.0)
    wd2o, wd3o = debye_temps_2_3
    ws1o, ws2o, ws3o = sin_temps
    we1o, we2o, we3o, we4o = einstein_temps
    wouo = optic_continuum_upper
    wolo = optic_continuum_lower

    if current_volume <= 0.0 or reference_volume <= 0.0:
        return {
            "wd1": wd1o,
            "wd2": wd2o,
            "wd3": wd3o,
            "ws1": ws1o,
            "ws2": ws2o,
            "ws3": ws3o,
            "we1": we1o,
            "we2": we2o,
            "we3": we3o,
            "we4": we4o,
            "wou": wouo,
            "wol": wolo,
            "gamma": gamma_0,
            "q": q_0,
            "qp": 0.0,
            "etas": 0.0,
            "detasdv": 0.0,
        }

    ratio = current_volume / reference_volume

    if ityp == 1:
        e = -0.5 * (ratio ** (-2.0 / 3.0) - 1.0)
        g = -6.0 * gamma_0
        h = g * (2.0 + 3.0 * q_0 + g)
        factor = max(0.0, 1.0 + g * e + 0.5 * h * e * e) ** 0.5

        wd1 = wd1o * factor
        wd2 = wd2o * factor
        wd3 = wd3o * factor
        ws1 = ws1o * factor
        ws2 = ws2o * factor
        ws3 = ws3o * factor
        we1 = we1o * factor
        we2 = we2o * factor
        we3 = we3o * factor
        we4 = we4o * factor
        wou = wouo * factor
        wol = wolo * factor

        denom = 1.0 + g * e + 0.5 * h * e * e
        gamma = -(1.0 - 2.0 * e) * (g + h * e) / (6.0 * denom) if denom != 0.0 else gamma_0
        q = 2.0 * gamma - 2.0 / 3.0 + h / 3.0 * (1.0 - 2.0 * e) / (g + h * e) if (g + h * e) != 0.0 else q_0
        qp = 0.0
        etas = got * gamma / gamma_0 if gamma_0 != 0.0 else 0.0
        etas = -gamma - 0.5 * ratio ** 0.0 * (2.0 * e + 1.0) ** 2 * (-gamma_0 - got)
        detasdv = 0.0
    elif ityp == 2:
        f = 0.5 * (ratio ** (-2.0 / 3.0) - 1.0)
        a = 3.0 * gamma_0
        b = gamma_0 * (9.0 * gamma_0 - 9.0 * q_0 - 6.0)
        bs = -gamma_0 - got
        factor = max(0.0, 1.0 + a * f + 0.5 * b * f * f) ** 0.5

        wd1 = wd1o * factor
        wd2 = wd2o * factor
        wd3 = wd3o * factor
        ws1 = ws1o * factor
        ws2 = ws2o * factor
        ws3 = ws3o * factor
        we1 = we1o * factor
        we2 = we2o * factor
        we3 = we3o * factor
        we4 = we4o * factor
        wou = wouo * factor
        wol = wolo * factor

        gamma = (1.0 / 3.0) * (wd1o / wd1) * (2.0 * f + 1.0) * (a + b * f) if wd1 != 0.0 else gamma_0
        q = (
            (1.0 / 9.0) * (9.0 * gamma * gamma - 6.0 * gamma - (wd1o / wd1) * (2.0 * f + 1.0) ** 2 * b) / gamma
            if gamma != 0.0 and wd1 != 0.0
            else q_0
        )
        qp = 0.0
        etas = -gamma - (wd1o / wd1) * (2.0 * f + 1.0) ** 2 * bs if wd1 != 0.0 else 0.0
        detasdv = 0.0
    else:
        q = q_0
        if abs(q) > 0.0:
            gamma = gamma_0 * (ratio**q)
            scale = exp((gamma_0 - gamma) / q)
        else:
            gamma = gamma_0
            scale = 1.0

        wd1 = wd1o * scale
        wd2 = wd2o * scale
        wd3 = wd3o * scale
        ws1 = ws1o * scale
        ws2 = ws2o * scale
        ws3 = ws3o * scale
        we1 = we1o * scale
        we2 = we2o * scale
        we3 = we3o * scale
        we4 = we4o * scale
        wou = wouo * scale
        wol = wolo * scale
        qp = 0.0
        etas = 0.0
        detasdv = 0.0

    return {
        "wd1": wd1,
        "wd2": wd2,
        "wd3": wd3,
        "ws1": ws1,
        "ws2": ws2,
        "ws3": ws3,
        "we1": we1,
        "we2": we2,
        "we3": we3,
        "we4": we4,
        "wou": wou,
        "wol": wol,
        "gamma": gamma,
        "q": q,
        "qp": qp,
        "etas": etas,
        "detasdv": detasdv,
    }


def _compute_cv_from_ctherm(
    *,
    temperature: float,
    atoms_per_formula: float,
    formula_units_per_cell: float,
    wd1: float,
    wd2: float,
    wd3: float,
    ws1: float,
    ws2: float,
    ws3: float,
    wou: float,
    wol: float,
    we1: float,
    we2: float,
    we3: float,
    we4: float,
    qe1: float,
    qe2: float,
    qe3: float,
    qe4: float,
) -> float:
    su = atoms_per_formula * formula_units_per_cell
    wo = (wou + wol) / 2.0
    do = 0.0
    if wo != 0.0:
        do = (wou - wol) / wo
    wdav = (wd1 + wd2 + wd3) / 3.0
    wsav = (ws1 + ws2 + ws3) / 3.0
    if wdav == 0.0 and wsav == 0.0:
        su = 1.0e15
    qe = qe1 + qe2 + qe3 + qe4
    qo = 1.0 - 1.0 / su - qe
    if wo == 0.0:
        qo = 0.0
    if wdav != wd1 / 3.0:
        cd = (1.0 / su) * _heat_function(wd1 / temperature if wd1 > 0.0 else 0.0, do, 1)
        cd = cd / 3.0 + (
            _heat_function(wd2 / temperature if wd2 > 0.0 else 0.0, do, 1)
            + _heat_function(wd3 / temperature if wd3 > 0.0 else 0.0, do, 1)
        ) / (3.0 * su)
    else:
        cd = (1.0 / su) * _heat_function(wd1 / temperature if wd1 > 0.0 else 0.0, do, 1)

    if wsav != ws1 / 3.0:
        cs = (1.0 / su) * _heat_function(ws1 / temperature if ws1 > 0.0 else 0.0, do, 3)
        cs = cs / 3.0 + (
            _heat_function(ws2 / temperature if ws2 > 0.0 else 0.0, do, 3)
            + _heat_function(ws3 / temperature if ws3 > 0.0 else 0.0, do, 3)
        ) / (3.0 * su)
    else:
        cs = (1.0 / su) * _heat_function(ws1 / temperature if ws1 > 0.0 else 0.0, do, 3)

    ce = 0.0
    for theta_e, weight in ((we1, qe1), (we2, qe2), (we3, qe3), (we4, qe4)):
        if theta_e > 0.0 and weight > 0.0:
            ce += weight * _heat_function(theta_e / temperature, do, 2)

    co = qo * _heat_function(wo / temperature, do, 4) if qo > 0.0 and wo > 0.0 else 0.0
    return 3.0 * atoms_per_formula * RGAS_J_MOL_K * (cd + cs + ce + co)


def compute_component_thermodynamic_state(
    *,
    temperature: float,
    volume: float,
    reference_volume: float,
    bulk_modulus: float,
    atoms_per_formula: float,
    formula_units_per_cell: float,
    debye_temp: float,
    debye_temps_2_3: Tuple[float, float] = (0.0, 0.0),
    sin_temps: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    optic_continuum_upper: float = 0.0,
    optic_continuum_lower: float = 0.0,
    einstein_temps: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    einstein_weights: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    gamma_0: float = 0.0,
    q_0: float = 0.0,
    got: float = 0.0,
    reference_entropy_j_mol_k: float = 0.0,
    ityp: int = 3,
) -> ComponentThermodynamicState:
    """Return a Fortran-style thermal state for one component at a given volume."""

    scaled = rescale_component_thermal_modes(
        reference_volume=reference_volume,
        current_volume=volume,
        gamma_0=gamma_0,
        q_0=q_0,
        got=got,
        debye_temp=debye_temp,
        debye_temps_2_3=debye_temps_2_3,
        sin_temps=sin_temps,
        optic_continuum_upper=optic_continuum_upper,
        optic_continuum_lower=optic_continuum_lower,
        einstein_temps=einstein_temps,
        ityp=ityp,
    )

    cv = _compute_cv_from_ctherm(
        temperature=temperature,
        atoms_per_formula=atoms_per_formula,
        formula_units_per_cell=formula_units_per_cell,
        wd1=scaled["wd1"],
        wd2=scaled["wd2"],
        wd3=scaled["wd3"],
        ws1=scaled["ws1"],
        ws2=scaled["ws2"],
        ws3=scaled["ws3"],
        wou=scaled["wou"],
        wol=scaled["wol"],
        we1=scaled["we1"],
        we2=scaled["we2"],
        we3=scaled["we3"],
        we4=scaled["we4"],
        qe1=einstein_weights[0] if len(einstein_weights) > 0 else 0.0,
        qe2=einstein_weights[1] if len(einstein_weights) > 1 else 0.0,
        qe3=einstein_weights[2] if len(einstein_weights) > 2 else 0.0,
        qe4=einstein_weights[3] if len(einstein_weights) > 3 else 0.0,
    )

    gamma = scaled["gamma"]
    if temperature <= 0.0 or volume <= 0.0 or bulk_modulus <= 0.0:
        cp = cv
        alpha = 0.0
    else:
        alpha = 0.001 * gamma * cv / (volume * bulk_modulus)
        cp = cv * (1.0 + gamma * gamma * cv * temperature / (1000.0 * volume * bulk_modulus))

    entropy = compute_component_entropy(
        temperature=temperature,
        atoms_per_formula=atoms_per_formula,
        debye_temp=scaled["wd1"],
        formula_units_per_cell=formula_units_per_cell,
        debye_temps_2_3=(scaled["wd2"], scaled["wd3"]),
        sin_temps=(scaled["ws1"], scaled["ws2"], scaled["ws3"]),
        optic_continuum_upper=scaled["wou"],
        optic_continuum_lower=scaled["wol"],
        einstein_temps=(scaled["we1"], scaled["we2"], scaled["we3"], scaled["we4"]),
        einstein_weights=einstein_weights,
        reference_entropy_j_mol_k=reference_entropy_j_mol_k,
    )

    return ComponentThermodynamicState(
        heat_capacity_p=cp,
        heat_capacity_v=cv,
        thermal_expansivity=alpha,
        gruneisen_parameter=gamma,
        entropy=entropy,
    )


def compute_component_heat_capacity_p(
    temperature: float,
    atoms_per_formula: float,
    debye_temp: float,
    formula_units_per_cell: float = 1.0,
    debye_temps_2_3: Tuple[float, float] = (0.0, 0.0),
    sin_temps: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    optic_continuum_upper: float = 0.0,
    optic_continuum_lower: float = 0.0,
    einstein_temps: Tuple[float, ...] = (),
    einstein_weights: Tuple[float, ...] = (),
) -> float:
    """Compute Cp for a single component using Debye + Einstein models.
    
    Mirrors the Fortran Ctherm() function for a simplified case (acoustic
    Debye + Einstein oscillators, no optic continuum or Sin model).
    
    Parameters
    ----------
    temperature : float
        Temperature in Kelvin
    atoms_per_formula : float
        Number of atoms per formula unit (fn in Fortran)
    debye_temp : float
        Debye temperature in Kelvin
    einstein_temps : tuple of float, optional
        Einstein temperatures (in K) for optical modes
    einstein_weights : tuple of float, optional
        Weights (fractions) for each Einstein mode; sum should be <= 1
        
    Returns
    -------
    float
        Heat capacity Cp in J/(mol·K) = R * (Debye contrib + Einstein contrib)
    """
    if temperature <= 0.0:
        return 0.0

    wd1 = max(debye_temp, 0.0)
    wd2, wd3 = debye_temps_2_3
    ws1, ws2, ws3 = sin_temps

    we = tuple(float(v) for v in einstein_temps[:4])
    qe = tuple(float(v) for v in einstein_weights[:4])
    if len(we) < 4:
        we = we + (0.0,) * (4 - len(we))
    if len(qe) < 4:
        qe = qe + (0.0,) * (4 - len(qe))

    su = atoms_per_formula * formula_units_per_cell
    wo = 0.5 * (optic_continuum_upper + optic_continuum_lower)
    d_optic = (optic_continuum_upper - optic_continuum_lower) / wo if wo != 0.0 else 0.0

    wdav = (wd1 + wd2 + wd3) / 3.0
    wsav = (ws1 + ws2 + ws3) / 3.0
    if wdav == 0.0 and wsav == 0.0:
        su = 1.0e15

    qe_sum = qe[0] + qe[1] + qe[2] + qe[3]
    qo = 1.0 - 1.0 / su - qe_sum if su != 0.0 else 0.0
    if wo == 0.0:
        qo = 0.0

    aniso = (wdav != wd1 / 3.0) or (wsav != ws1 / 3.0)

    if qe_sum == 0.0 and qo == 0.0:
        su = 1.0
    if qo == 0.0 and (1.0 - qe_sum) > 1.0e-12:
        su = 1.0 / (1.0 - qe_sum)

    cd = (1.0 / su) * _heat_function(wd1 / temperature if wd1 > 0.0 else 0.0, d_optic, 1)
    if aniso:
        cd = cd / 3.0 + (
            _heat_function(wd2 / temperature if wd2 > 0.0 else 0.0, d_optic, 1)
            + _heat_function(wd3 / temperature if wd3 > 0.0 else 0.0, d_optic, 1)
        ) / (3.0 * su)

    cs = (1.0 / su) * _heat_function(ws1 / temperature if ws1 > 0.0 else 0.0, d_optic, 3)
    if aniso:
        cs = cs / 3.0 + (
            _heat_function(ws2 / temperature if ws2 > 0.0 else 0.0, d_optic, 3)
            + _heat_function(ws3 / temperature if ws3 > 0.0 else 0.0, d_optic, 3)
        ) / (3.0 * su)

    ce = 0.0
    for theta_e, weight in zip(we, qe):
        if theta_e > 0.0 and weight > 0.0:
            ce += weight * _heat_function(theta_e / temperature, d_optic, 2)

    co = 0.0
    if qo > 0.0 and wo > 0.0:
        co = qo * _heat_function(wo / temperature, d_optic, 4)

    return 3.0 * atoms_per_formula * RGAS_J_MOL_K * (cd + cs + ce + co)


def compute_component_heat_capacity_v(
    temperature: float,
    cp: float,
    thermal_expansivity: float,
    volume: float,
    bulk_modulus: float,
) -> float:
    """Compute Cv from Cp using thermodynamic relation.
    
    Cv = Cp - T * V * alpha^2 * K_T
    
    Parameters
    ----------
    temperature : float
        Temperature in Kelvin
    cp : float
        Heat capacity at constant pressure, J/(mol·K)
    thermal_expansivity : float
        Thermal expansivity alpha, 1/K
    volume : float
        Molar volume, cm^3/mol
    bulk_modulus : float
        Isothermal bulk modulus K_T, GPa
        
    Returns
    -------
    float
        Cv in J/(mol·K)
    """
    if temperature <= 0.0 or bulk_modulus <= 0.0:
        return cp

    delta_cp = temperature * thermal_expansivity * thermal_expansivity * volume * bulk_modulus * 1000.0
    return cp - delta_cp


def compute_component_entropy(
    temperature: float,
    atoms_per_formula: float,
    debye_temp: float,
    formula_units_per_cell: float = 1.0,
    debye_temps_2_3: Tuple[float, float] = (0.0, 0.0),
    sin_temps: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    optic_continuum_upper: float = 0.0,
    optic_continuum_lower: float = 0.0,
    einstein_temps: Tuple[float, ...] = (),
    einstein_weights: Tuple[float, ...] = (),
    reference_entropy_j_mol_k: float = 0.0,
) -> float:
    """Compute entropy from Debye + Einstein models.
    
    S(T) = S_0 + integral_0^T (Cp / T_) dT_
    
    Uses numerical integration and analytical limits.
    
    Parameters
    ----------
    temperature : float
        Temperature in Kelvin
    atoms_per_formula : float
        Atoms per formula unit
    debye_temp : float
        Debye temperature, K
    einstein_temps : tuple of float, optional
        Einstein temperatures, K
    einstein_weights : tuple of float, optional
        Einstein weights
    reference_entropy_j_mol_k : float, optional
        Reference entropy at T=0 (usually 0 for standard state)
        
    Returns
    -------
    float
        Entropy S(T) in J/(mol·K)
    """
    if temperature <= 0.0:
        return reference_entropy_j_mol_k

    n_steps = 256
    t_grid = np.linspace(1.0, float(temperature), n_steps, dtype=np.float64)
    cp_over_t = np.zeros_like(t_grid)

    for i, t in enumerate(t_grid):
        cp_t = compute_component_heat_capacity_p(
            temperature=float(t),
            atoms_per_formula=atoms_per_formula,
            debye_temp=debye_temp,
            formula_units_per_cell=formula_units_per_cell,
            debye_temps_2_3=debye_temps_2_3,
            sin_temps=sin_temps,
            optic_continuum_upper=optic_continuum_upper,
            optic_continuum_lower=optic_continuum_lower,
            einstein_temps=einstein_temps,
            einstein_weights=einstein_weights,
        )
        cp_over_t[i] = cp_t / max(float(t), 1.0e-12)

    integral = float(np.trapz(cp_over_t, t_grid))
    return integral + reference_entropy_j_mol_k


def compute_component_thermal_expansivity(
    temperature: float,
    gruneisen_parameter: float,
    heat_capacity_v: float,
    volume: float,
    bulk_modulus: float,
) -> float:
    """Compute thermal expansivity alpha using Grüneisen model.
    
    alpha = (gamma * Cv) / (V * K_T)
    
    where gamma is the Grüneisen parameter.
    
    Parameters
    ----------
    temperature : float
        Temperature, K (used for Cv evaluation if needed)
    gruneisen_parameter : float
        Grüneisen parameter gamma (dimensionless)
    heat_capacity_v : float
        Heat capacity at constant volume, J/(mol·K)
    volume : float
        Molar volume, cm^3/mol
    bulk_modulus : float
        Isothermal bulk modulus, GPa
        
    Returns
    -------
    float
        Thermal expansivity alpha in 1/K
    """
    if volume <= 0.0 or bulk_modulus <= 0.0 or heat_capacity_v <= 0.0:
        return 0.0

    return 0.001 * gruneisen_parameter * heat_capacity_v / (volume * bulk_modulus)
