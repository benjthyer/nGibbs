"""
Theta calibration function, translated from Fortran thetacal.f.

Computes the temperature at which a given Debye/Einstein heat capacity is achieved,
using table inversion and interpolation.
"""
from __future__ import annotations

from .heat import Heat


def thetacal(cvn: float) -> float:
    """
    Compute calibrated Debye temperature (Theta_D) given a normalized heat capacity.
    
    This function inverts the Heat() function to find the temperature (scaled by
    reference theta) at which the heat capacity equals cvn.
    
    Args:
        cvn: Normalized heat capacity (dimensionless, typically 0 < cvn < 1).
    
    Returns:
        Calibrated Theta (in Kelvin, relative to reference Theta_D).
    
    Notes:
        - Uses bisection on Heat() tables.
        - Fortran thetacal.f uses idos=1 (Debye model).
        - Returns 0 for edge cases (cvn <= 0 or cvn >= 1).
    """
    if cvn <= 0.0:
        return 0.0
    if cvn >= 1.0:
        return 1.0
    
    # Use bisection to find x such that Heat(x, 0.0, idos=1) == cvn
    # Heat function is monotonically increasing with x
    idos = 1
    d = 0.0  # Not used for Debye model
    
    x_min = 0.0
    x_max = 10.0  # Search range
    
    # Expand search range if needed
    while Heat(x_max, d, idos) < cvn:
        x_max *= 2.0
        if x_max > 1e6:  # Safety limit
            return 1.0
    
    # Bisection
    tol = 1e-8
    for _ in range(100):  # Max iterations
        x_mid = 0.5 * (x_min + x_max)
        h_mid = Heat(x_mid, d, idos)
        
        if abs(h_mid - cvn) < tol:
            return x_mid
        
        if h_mid < cvn:
            x_min = x_mid
        else:
            x_max = x_mid
    
    return 0.5 * (x_min + x_max)
