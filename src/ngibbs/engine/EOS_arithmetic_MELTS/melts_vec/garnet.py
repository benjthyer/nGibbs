"""
Garnet (almandine-grossular-pyrope) solid-solution mixing model --
vectorized translation of MAGMA `sources/garnet.c` (Berman 1990; Berman &
Koziol 1991): an asymmetric ternary Margules model on three sites (hence
the 3 R T sum x ln x), no ordering parameter (NS = 0).

Composition: r0 = x_almandine, r1 = x_grossular, x_pyrope = 1 - r0 - r1
(garnet.c conGrn / FR0 / FR1). Site fractions are clipped at DBL_EPSILON
exactly as in gmixGrn. Note G carries p*V, not (p - 1)*V, and WG includes
p*WV (both verbatim).

MELTS's bulk recipe (read_write.c) adds hmix = G + T*S, smix = S, vmix = V
and cpmix = -T d2G/dT2 = 0 to the endmember sums; dV/dT and dV/dP of mixing
are identically zero (D2GDTDP = D2GDP2 = 0).
"""
from __future__ import annotations
import numpy as np

from .constants import Rgas
from .solution_model import darken_activities

ENDMEMBERS = ["almandine", "grossular", "pyrope"]

# 1 == grossular, 2 == pyrope, 3 == almandine (garnet.c)
_WH = dict(W112=21560.0, W122=69200.0, W113=20320.0, W133=2620.0, W223=230.0, W233=3720.0, W123=0.0)
_WS = dict(W112=18.79, W122=18.79, W113=5.08, W133=5.08, W223=0.0, W233=0.0, W123=0.0)
_WV = dict(W112=0.10, W122=0.10, W113=0.17, W133=0.09, W223=0.01, W233=0.06, W123=0.00)
_EPS = np.finfo(float).eps


def _excess(W, xgr, xpy, xal):
    """The H, S or V macro: sum of the seven asymmetric Margules terms."""
    return (W['W112']*xgr*xpy*(xgr + xal/2.0) + W['W122']*xgr*xpy*(xpy + xal/2.0)
            + W['W113']*xgr*xal*(xgr + xpy/2.0) + W['W133']*xgr*xal*(xal + xpy/2.0)
            + W['W223']*xpy*xal*(xpy + xgr/2.0) + W['W233']*xpy*xal*(xal + xgr/2.0)
            + W['W123']*xgr*xpy*xal)


def solution_thermo(X, T, P):
    """Mixing properties per mole of garnet for (B, 3) mole fractions X in
    ENDMEMBERS order. Returns G_mix, H_mix, S_mix, V_mix, Cp_mix, dCpdT_mix,
    dVdT_mix, dVdP_mix, mu, activities (gmixGrn/hmixGrn/smixGrn/vmixGrn/actGrn)."""
    X = np.asarray(X, dtype=np.float64)
    t = np.asarray(T, dtype=np.float64)
    p = np.asarray(P, dtype=np.float64)
    xal = np.where(X[:, 0] > _EPS, X[:, 0], _EPS)
    xgr = np.where(X[:, 1] > _EPS, X[:, 1], _EPS)
    xpy0 = 1.0 - X[:, 0] - X[:, 1]
    xpy = np.where(xpy0 > _EPS, xpy0, _EPS)
    R = Rgas
    S = -3.0*R*(xgr*np.log(xgr) + xpy*np.log(xpy) + xal*np.log(xal)) + _excess(_WS, xgr, xpy, xal)
    H = _excess(_WH, xgr, xpy, xal)
    V = _excess(_WV, xgr, xpy, xal)
    G = H - t*S + p*V
    WG = {k: _WH[k] - t*_WS[k] + p*_WV[k] for k in _WH}
    dgdr0 = (3.0*R*t*(np.log(xal) - np.log(xpy))
             + WG['W112']*(xgr*xpy/2.0 - xgr*(xgr + xal/2.0))
             + WG['W122']*(-xgr*xpy/2.0 - xgr*(xpy + xal/2.0))
             + WG['W113']*(-xgr*xal/2.0 + xgr*(xgr + xpy/2.0))
             + WG['W133']*(xgr*xal/2.0 + xgr*(xal + xpy/2.0))
             + WG['W223']*(-xpy*xal + (xpy - xal)*(xpy + xgr/2.0))
             + WG['W233']*(xpy*xal + (xpy - xal)*(xal + xgr/2.0))
             + WG['W123']*xgr*(xpy - xal))
    dgdr1 = (3.0*R*t*(np.log(xgr) - np.log(xpy))
             + WG['W112']*(xgr*xpy + (xpy - xgr)*(xgr + xal/2.0))
             + WG['W122']*(-xgr*xpy + (xpy - xgr)*(xpy + xal/2.0))
             + WG['W113']*(xgr*xal/2.0 + xal*(xgr + xpy/2.0))
             + WG['W133']*(-xgr*xal/2.0 + xal*(xal + xpy/2.0))
             + WG['W223']*(-xpy*xal/2.0 - xal*(xpy + xgr/2.0))
             + WG['W233']*(xpy*xal/2.0 - xal*(xal + xgr/2.0))
             + WG['W123']*(xpy - xgr)*xal)
    # FR0(i) = (i == 0) ? 1 - xal : -xal ; FR1(i) = (i == 1) ? 1 - xgr : -xgr
    fr = np.stack([np.stack([1.0 - xal, -xgr], -1), np.stack([-xal, 1.0 - xgr], -1),
                   np.stack([-xal, -xgr], -1)], -2)
    mu, a = darken_activities(G, np.stack([dgdr0, dgdr1], -1), fr, Rgas, t)
    z = np.zeros_like(G)
    return dict(G_mix=G, H_mix=G + t*S, S_mix=S, V_mix=V, Cp_mix=z, dCpdT_mix=z, dVdT_mix=z, dVdP_mix=z,
                mu=mu, activities=a)
