"""
Attenuation quality factor Q computation, translated from Fortran qr19.f.

Computes seismic quality factors (Qs, Qp) at a given depth and temperature
using the Romanowicz attenuation model with lithospheric age and temperature
corrections.
"""
from __future__ import annotations

from functools import lru_cache
from math import exp
from pathlib import Path
from typing import Tuple

import numpy as np

#from .bserch import bserch
#from .heat import _neville


_ADQREF_INC_PATH = Path(__file__).resolve().parent / "HeFESToRepository" / "adqref.inc"


@lru_cache(maxsize=1)
def _load_adqref_tables() -> dict[str, Tuple[float, ...]]:
    """Load depth (dad) and temperature (tad) arrays from adqref.inc."""
    text = _ADQREF_INC_PATH.read_text(encoding="utf-8", errors="ignore")
    
    def _extract(array_name: str) -> Tuple[float, ...]:
        marker = f"data {array_name}/"
        start = text.find(marker)
        if start < 0:
            raise FileNotFoundError(f"Missing {array_name} in {_ADQREF_INC_PATH}")
        start += len(marker)
        end = text.find("/", start)
        if end < 0:
            raise ValueError(f"Unterminated data block for {array_name}")
        raw = text[start:end]
        values = []
        for chunk in raw.replace("\n", " ").replace("&", " ").split(","):
            piece = chunk.strip()
            if not piece:
                continue
            values.append(float(piece.replace("D", "E").replace("d", "e").split()[0]))
        return tuple(values)
    
    dad = _extract("dad")
    tad = _extract("tad")
    return {"dad": dad, "tad": tad}


#def qr19(depth: float, Ti: float) -> Tuple[float, float]:
    """
    Compute seismic quality factors (Qs, Qp) at given depth and temperature.
    
    Uses the Romanowicz (1995) attenuation model with:
    - Depth-dependent reference Q from a 13-layer PREM-like model.
    - Adiabatic temperature reference from inv251010 pyrolite adiabat.
    - Lithospheric age correction (100 Ma Pacific average).
    - Temperature dependence correction.
    
    Args:
        depth: Depth in kilometers.
        Ti: Temperature in Kelvin.
    
    Returns:
        Tuple (Qs, Qp): Shear and compressional quality factors.
    
    References:
        - Romanowicz, B. (1995). A global tomographic model of shear 
          attenuation in the upper mantle. JGR, 100(B7), 12375-12394.
        - Temperature-dependence added 31 July 2007.
    """
    """# Local Q model: depth layers and reference Q values
    # Based on PREM with modifications
    d_vals = (2891.0, 648.0, 647.0, 511.0, 422.0, 370.0, 310.0, 250.0, 200.0, 160.0, 100.0, 81.5, 0.0)
    q_vals = (32.1, 32.1, 67.1, 66.2, 65.1, 65.2, 67.8, 76.6, 145.3, 154.9, 153.5, 17.1, 17.1)
    
    # Parameters
    qlarge = 1e15
    age = 100.0  # Lithospheric age in Ma
    Eact = 424000.0  # Activation energy (J/mol)
    alpha = 0.26  # Exponential factor
    qslarge = 9999.0
    Rgas = 8.314462618  # Gas constant (J/(mol*K))
    mint = 2  # Neville interpolation order
    
    # Step 1: Interpolate reference Q from depth layers
    jlo_d = bserch(d_vals, depth)
    if jlo_d < 0:
        jlo_d = 0
    klo_d = min(max(jlo_d - (mint - 1) // 2, 0), len(d_vals) - mint)
    qs00, _ = _neville(
        d_vals[klo_d : klo_d + mint],
        q_vals[klo_d : klo_d + mint],
        depth
    )
    qs00 = 10000.0 / qs00
    
    # Step 2: Interpolate reference temperature from adiabat
    tables = _load_adqref_tables()
    dad = tables["dad"]
    tad = tables["tad"]
    
    jlo_ad = bserch(dad, depth)
    if jlo_ad < 0:
        jlo_ad = 0
    klo_ad = min(max(jlo_ad - (mint - 1) // 2, 0), len(dad) - mint)
    tref, _ = _neville(
        dad[klo_ad : klo_ad + mint],
        tad[klo_ad : klo_ad + mint],
        depth
    )
    
    # Step 3: Apply lithospheric age correction (Stixrude & Lithgow-Bertelloni 2005)
    qs = 1000.0 / (1000.0 / qs00 + 3.0 * (1.0 - age / 100.0))
    
    # Step 4: Apply temperature-dependent correction
    if Ti > 0.0:
        qs = qs * exp(alpha * Eact / Rgas * (1.0 / Ti - 1.0 / tref))
    else:
        qs = qslarge
    
    qs = min(qs, qslarge)
    
    # Step 5: Assume Poisson solid (no bulk attenuation)
    qp = 9.0 / 4.0 * qs

    return qs, qp

"""
# ---------------------------------------------------------------------------
# PREM depth-pressure table embedded from const.inc (ndepth = 57)
# ---------------------------------------------------------------------------

_PREM_DEPTH_KM = np.array([
     0.00000,  3.00000,  3.00000, 15.00000, 15.00000, 24.40000,
    24.40000, 40.00000, 60.00000, 80.00000, 80.00000,115.00000,
   150.00000,185.00000,220.00000,220.00000,265.00000,310.00000,
   355.00000,400.00000,400.00000,450.00000,500.00000,550.00000,
   600.00000,600.00000,635.00000,670.00000,670.00000,721.00000,
   771.00000,771.00000,871.00000,971.00000,1071.0000,1171.0000,
  1271.0000,1371.0000,1471.0000,1571.0000,1671.0000,1771.0000,
  1871.0000,1971.0000,2071.0000,2171.0000,2271.0000,2371.0000,
  2471.0000,2571.0000,2671.0000,2741.0000,2741.0000,2771.0000,
  2871.0000,2891.0000,2891.0000,
])

_PREM_PRESS_GPA = np.array([
    0.00000,  0.02990,  0.03030,  0.33640,  0.33700,  0.60400,
    0.60430,  1.12390,  1.78910,  2.45390,  2.45460,  3.61830,
    4.78240,  5.94660,  7.11080,  7.11500,  8.64970, 10.20270,
   11.77020, 13.35200, 13.35270, 15.22510, 17.13110, 19.07030,
   21.04250, 21.04260, 22.43640, 23.83340, 23.83420, 26.07830,
   28.29270, 28.29280, 32.76230, 37.28520, 41.86060, 46.48820,
   51.16760, 55.89910, 60.68300, 65.52020, 70.41190, 75.35980,
   80.36600, 85.43320, 90.56460, 95.76410,101.03630,106.38640,
  111.82070,117.34650,122.97190,126.97410,126.97420,128.70670,
  134.56190,135.75090,135.75100,
])


@lru_cache(maxsize=1)
def _get_prem_interp() -> Tuple[np.ndarray, np.ndarray]:
    """Deduplicated PREM (pressure_GPa, depth_km) arrays for interpolation.

    Fortran depth.f skips rows where depth equals the previous depth value,
    producing a strictly increasing depth axis.  We replicate that here and
    return (pressure_GPa, depth_km) sorted by ascending pressure so that
    np.interp can use pressure as the look-up axis.
    """
    p_out, d_out = [], []
    prev_d = -999.0
    for d, p in zip(_PREM_DEPTH_KM.tolist(), _PREM_PRESS_GPA.tolist()):
        if d == prev_d:
            continue
        prev_d = d
        p_out.append(p)
        d_out.append(d)
    return np.array(p_out), np.array(d_out)


def depth_from_pressure(P) -> np.ndarray:
    """Convert pressure (GPa) to depth (km) via PREM linear interpolation.

    Replicates the behaviour of the Fortran ``depth(Pi)`` function in
    ``depth.f``, which uses the PREM table from ``const.inc``.

    Args:
        P: Scalar or array-like pressure in GPa.

    Returns:
        Depth in km (same shape as *P*, clipped to >= 0).
    """
    p_prem, d_prem = _get_prem_interp()
    depth = np.interp(np.asarray(P, dtype=float), p_prem, d_prem,
                      left=0.0, right=float(d_prem[-1]))
    return np.maximum(depth, 0.0)


def qr19_vec(P, Ti) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized quality-factor computation as a function of pressure.

    Equivalent to calling :func:`qr19` element-wise but operates on NumPy
    arrays without Python loops.

    Args:
        P:  Pressure in GPa — scalar or array-like.
        Ti: Temperature in Kelvin — scalar or array-like, broadcastable with *P*.

    Returns:
        Tuple ``(Qs, Qp)`` of shear and compressional quality factors,
        same shape as the broadcast of *P* and *Ti*.
    """
    P  = np.asarray(P,  dtype=float)
    Ti = np.asarray(Ti, dtype=float)

    depth = depth_from_pressure(P)

    # Local Q model in ascending-depth order for np.interp
    d_asc = np.array([   0.0,  81.5, 100.0, 160.0, 200.0, 250.0,
                       310.0, 370.0, 422.0, 511.0, 647.0, 648.0, 2891.0])
    q_asc = np.array([  17.1,  17.1, 153.5, 154.9, 145.3,  76.6,
                        67.8,  65.2,  65.1,  66.2,  67.1,  32.1,   32.1])

    qs00 = np.interp(depth, d_asc, q_asc)
    qs00 = 10000.0 / qs00

    # Reference adiabat temperature
    tables = _load_adqref_tables()
    dad = np.array(tables["dad"])
    tad = np.array(tables["tad"])
    tref = np.interp(depth, dad, tad)

    # Lithospheric age correction (100 Ma Pacific average → term vanishes)
    age = 100.0
    qs = 1000.0 / (1000.0 / qs00 + 3.0 * (1.0 - age / 100.0))

    # Temperature-dependent correction
    Eact    = 424000.0
    alpha   = 0.26
    Rgas    = 8.314462618
    qslarge = 9999.0

    safe_Ti  = np.where(Ti > 0.0, Ti, 1.0)  # avoid 1/0; masked below
    qs_corr  = qs * np.exp(alpha * Eact / Rgas * (1.0 / safe_Ti - 1.0 / tref))
    qs = np.where(Ti > 0.0, qs_corr, qslarge)
    qs = np.minimum(qs, qslarge)

    qp = 9.0 / 4.0 * qs
    return qs, qp
