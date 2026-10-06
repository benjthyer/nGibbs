"""
Leucite (leucite-analcime-na-leucite) solid-solution mixing model --
vectorized translation of MAGMA `sources/leucite.c`: a K-Na regular solution
on the large-cation site coupled to an H2O / vacancy exchange (analcime), no
ordering parameter (NS = 0).

Composition: r0 = x_leucite, r1 = x_analcime, x_na-leucite = 1 - r0 - r1
(conLeu, FR0 / FR1). Site fractions (xKL, xNaL, xH2OL, xKS, xNaS, xVcS) are
clipped at DBL_EPSILON exactly as in gmixLeu. hmix = G + T*S, smix = S,
vmix = V = 0 and cpmix = 0, as in leucite.c.
"""
from __future__ import annotations
import numpy as np

from .constants import Rgas
from .solution_model import darken_activities

ENDMEMBERS = ["leucite", "analcime", "na-leucite"]

_WNAK, _WNAH, _DGR = 7000.0, 7000.0, 53000.0
_G0, _GX, _GY, _GXX, _GYY, _GXY = 0.0, _WNAK, _WNAH, -_WNAK, -_WNAH, _DGR
_EPS = np.finfo(float).eps
_C = (3.0/2.0)*np.log(2.0/3.0) + (1.0/2.0)*np.log(1.0/2.0)


def solution_thermo(X, T, P):
    """Mixing properties per mole of leucite for (B, 3) mole fractions X in
    ENDMEMBERS order: G_mix, H_mix, S_mix, V_mix, Cp_mix, dCpdT_mix, dVdT_mix,
    dVdP_mix, mu, activities."""
    X = np.asarray(X, dtype=np.float64)
    t = np.asarray(T, dtype=np.float64)
    r0, r1 = X[:, 0], X[:, 1]
    xKL, xNaL, xH2OL, xKS, xNaS, xVcS = np.maximum(np.stack([
        r0*(1.0 - r1), (1.0 - r0)*(1.0 - r1), r1, 2.0*r0*r1/3.0, 2.0*(1.0 - r0)*r1/3.0, 1.0 - 2.0*r1/3.0]), _EPS)
    R = Rgas
    S = -R*(xKL*np.log(xKL) + xNaL*np.log(xNaL) + xH2OL*np.log(xH2OL)
            + 3.0*xKS*np.log(xKS)/2.0 + 3.0*xNaS*np.log(xNaS)/2.0 + 3.0*xVcS*np.log(xVcS)/2.0
            - r1*_C)
    H = _G0 + _GX*r0 + _GY*r1 + _GXX*r0*r0 + _GYY*r1*r1 + _GXY*r0*r1
    G = H - t*S
    dgdr0 = (R*t*((1.0 - r1)*np.log(xKL) - (1.0 - r1)*np.log(xNaL) + r1*np.log(xKS) - r1*np.log(xNaS))
             + _GX + 2.0*_GXX*r0 + _GXY*r1)
    dgdr1 = (R*t*(-r0*np.log(xKL) - (1.0 - r0)*np.log(xNaL) + np.log(xH2OL)
                  + r0*np.log(xKS) + (1.0 - r0)*np.log(xNaS) - np.log(xVcS) - _C)
             + _GY + 2.0*_GYY*r1 + _GXY*r0)
    # FR0(i) = (i == 0) ? 1 - x[0] : -x[0] ; FR1(i) = (i == 1) ? 1 - x[1] : -x[1]
    fr = np.stack([np.stack([1.0 - r0, -r1], -1), np.stack([-r0, 1.0 - r1], -1),
                   np.stack([-r0, -r1], -1)], -2)
    mu, a = darken_activities(G, np.stack([dgdr0, dgdr1], -1), fr, Rgas, t)
    z = np.zeros_like(G)
    return dict(G_mix=G, H_mix=G + t*S, S_mix=S, V_mix=z, Cp_mix=z, dCpdT_mix=z, dVdT_mix=z, dVdP_mix=z,
                mu=mu, activities=a)
