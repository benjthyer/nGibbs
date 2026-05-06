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

from .bserch import bserch
from .heat import _neville


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


def qr19(depth: float, Ti: float) -> Tuple[float, float]:
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
    # Local Q model: depth layers and reference Q values
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
