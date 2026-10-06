"""
Hornblende (pargasite-ferropargasite-magnesiohastingsite) reciprocal
solid-solution mixing model -- vectorized translation of MAGMA
`sources/hornblende.c`: Fe2+-Mg mixing on four M1,2 sites and Fe3+-Al on M3,
no ordering parameter (NS = 0).

Composition (conHrn): r0 = x_ferropargasite, r1 = x_magnesiohastingsite,
x_pargasite = 1 - r0 - r1; site fractions xFe2M12 = r0, xMgM12 = 1 - r0,
xFe3M3 = r1, xAlM3 = 1 - r1. hornblende.c does not clip these; MELTS's own
element-to-component conversion keeps Fe3+ >= 1% of Fe, so it never meets
x = 0. Here x ln x takes its limit 0 at x = 0 (G, H, S stay finite for a
pure endmember; the corresponding chemical potential is -inf, activity 0).
hmix = G + T*S, V = 0, cpmix = 0.
"""
from __future__ import annotations
import numpy as np

from .constants import Rgas
from .solution_model import darken_activities

ENDMEMBERS = ["pargasite", "ferropargasite", "magnesiohastingsite"]

_WFEMG = 4.0*1.68*4.184*1000.0
_WFEAL = 16.78*1000.0
_DGR = 0.0
_G0, _GX, _GY, _GXX, _GYY, _GXY = 0.0, _WFEMG, _WFEAL, -_WFEMG, -_WFEAL, _DGR


def solution_thermo(X, T, P):
    """Mixing properties per mole of hornblende for (B, 3) mole fractions X in
    ENDMEMBERS order: G_mix, H_mix, S_mix, V_mix, Cp_mix, dCpdT_mix, dVdT_mix,
    dVdP_mix, mu, activities."""
    X = np.asarray(X, dtype=np.float64)
    t = np.asarray(T, dtype=np.float64)
    r0, r1 = X[:, 1], X[:, 2]
    xMgM12, xFe2M12, xAlM3, xFe3M3 = 1.0 - r0, r0, 1.0 - r1, r1
    R = Rgas
    xs = np.stack([xMgM12, xFe2M12, xFe3M3, xAlM3])
    xl = np.where(xs > 0.0, xs*np.log(np.where(xs > 0.0, xs, 1.0)), 0.0)
    S = -R*(4.0*xl[0] + 4.0*xl[1] + xl[2] + xl[3])
    H = _G0 + _GX*r0 + _GY*r1 + _GXX*r0*r0 + _GYY*r1*r1 + _GXY*r0*r1
    G = H - t*S
    with np.errstate(divide='ignore'):
        dgdr0 = R*t*(4.0*np.log(xFe2M12) - 4.0*np.log(xMgM12)) + _GX + 2.0*_GXX*r0 + _GXY*r1
        dgdr1 = R*t*(np.log(xFe3M3) - np.log(xAlM3)) + _GY + 2.0*_GYY*r1 + _GXY*r0
    # FR0(i) = (i == 1) ? 1 - x[0] : -x[0] ; FR1(i) = (i == 2) ? 1 - x[1] : -x[1], x = r
    fr = np.stack([np.stack([-r0, -r1], -1), np.stack([1.0 - r0, -r1], -1),
                   np.stack([-r0, 1.0 - r1], -1)], -2)
    with np.errstate(invalid='ignore', over='ignore'):
        mu, a = darken_activities(G, np.stack([dgdr0, dgdr1], -1), fr, Rgas, t)
    z = np.zeros_like(G)
    return dict(G_mix=G, H_mix=G + t*S, S_mix=S, V_mix=z, Cp_mix=z, dCpdT_mix=z, dVdT_mix=z, dVdP_mix=z,
                mu=mu, activities=a)
