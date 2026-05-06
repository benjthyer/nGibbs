"""Python translation scaffold for Fortran thermw.f (water branch)."""
from __future__ import annotations

from typing import TYPE_CHECKING

from .therm_common import ThermResult

if TYPE_CHECKING:
    from .state import HeFESToState


def thermw(ispec: int, Vi: float, volnl: float, state: HeFESToState) -> ThermResult:
    """Approximate water thermodynamics consistent with volumew fallback."""
    ti = max(state.Ti, 1.0e-12)

    # Mildly temperature-dependent liquid water heat capacities.
    cp = 75.3 + 0.01 * (ti - 298.15)
    cv = max(cp - 1.5, 1.0)

    # Compressibility-based K estimate around ambient water.
    beta = 4.5e-10  # 1/Pa
    k = 1.0 / beta / 1.0e9  # GPa
    alp = 2.57e-4
    gamma = max((cp - cv) / cv, 0.01)
    ks = k * (1.0 + alp * gamma * ti)

    s_const = 86.808 - 23.46179376313335
    f_const = -236.839 - (-19.0987)
    ent = cp * (ti - 298.15) / ti + s_const
    e = cv * (ti - 298.15)
    ftot = 1000.0 * f_const + e - ti * ent

    return ThermResult(
        Cp=cp,
        Cv=cv,
        gamma=gamma,
        K=k,
        Ks=ks,
        alp=alp,
        Ftot=ftot,
        ph=0.0,
        ent=ent,
        deltas=0.0,
        tcal=0.0,
        zeta=0.0,
        Gsh=0.0,
        uth=e,
        uto=0.0,
        thet=0.0,
        qq=0.0,
        etas=0.0,
        dGdT=0.0,
        pzp=0.0,
        Vdeb=0.0,
        gamdeb=gamma,
    )
