"""
Biotite (annite-phlogopite) solid-solution mixing model -- vectorized
translation of MAGMA `sources/biotite.c` (binary regular solution, Sack &
Ghiorso 1989), the model the rhyolite-MELTS solid tables actually use for
"biotite" (conBio/gmixBio in sol_struct_data.h).

`sources/biotiteTaj.c` (the 5-endmember fbiTaj/tbiTaj/eastTaj/annTaj/phlTaj
model) is NOT translated: no MELTS solid table references it (no conBioTaj
entry anywhere in sol_struct_data.h), so alphaMELTS never evaluates it and
there are no endmember records for it.

Composition: r0 = x_annite, x_phlogopite = 1 - r0; both clipped at
DBL_EPSILON as in gmixBio. hmix = G + T*S, V = 0, cpmix = 0.
"""
from __future__ import annotations
import numpy as np

from .constants import Rgas
from .solution_model import darken_activities

ENDMEMBERS = ["annite", "phlogopite"]

_WHANPH = 4.184*1000.0*3.21
_EPS = np.finfo(float).eps


def solution_thermo(X, T, P):
    """Mixing properties per mole of biotite for (B, 2) mole fractions X in
    ENDMEMBERS order: G_mix, H_mix, S_mix, V_mix, Cp_mix, dCpdT_mix, dVdT_mix,
    dVdP_mix, mu, activities."""
    X = np.asarray(X, dtype=np.float64)
    t = np.asarray(T, dtype=np.float64)
    xan = np.where(X[:, 0] > _EPS, X[:, 0], _EPS)
    xph = np.where(1.0 - X[:, 0] > _EPS, 1.0 - X[:, 0], _EPS)
    R = Rgas
    S = -R*(xan*np.log(xan) + xph*np.log(xph))
    H = _WHANPH*xan*xph
    G = H - t*S
    dgdr0 = R*t*(np.log(xan) - np.log(xph)) + _WHANPH*(xph - xan)
    # FR0(i) = (i == 0) ? 1 - xan : -xan
    fr = np.stack([1.0 - xan, -xan], -1)[..., None]
    mu, a = darken_activities(G, dgdr0[:, None], fr, Rgas, t)
    z = np.zeros_like(G)
    return dict(G_mix=G, H_mix=G + t*S, S_mix=S, V_mix=z, Cp_mix=z, dCpdT_mix=z, dVdT_mix=z, dVdP_mix=z,
                mu=mu, activities=a)
