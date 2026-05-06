"""Python translation scaffold for Fortran thermh.f (hydrogen branch)."""
from __future__ import annotations

from math import log
from typing import TYPE_CHECKING

from .therm_common import ThermResult

if TYPE_CHECKING:
    from .state import HeFESToState


def thermh(ispec: int, Vi: float, volnl: float, state: HeFESToState) -> ThermResult:
    """Approximate hydrogen-fluid thermodynamics."""
    ti = max(state.Ti, 1.0)
    p_pa = max(state.Pi, 1.0e-8) * 1.0e9

    # H2 ideal gas with mild non-ideal stiffness for high-P numerics.
    cp = 28.8 + 0.002 * (ti - 300.0)
    cv = max(cp - state.Rgas, 1.0)
    gamma = cp / cv - 1.0

    # kappa_T~1/P for ideal gas -> K~P (GPa).
    k = max(state.Pi, 1.0e-6)
    alp = 1.0 / ti
    ks = k * (1.0 + alp * gamma * ti)

    # Relative free energy (reference shifted).
    ent = cp * log(ti / 300.0) - state.Rgas * log(p_pa / 1.0e5)
    uth = cv * (ti - 300.0)
    ftot = uth - ti * ent + 1000.0 * state.Pi * Vi

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
        uth=uth,
        uto=0.0,
        thet=0.0,
        qq=0.0,
        etas=0.0,
        dGdT=0.0,
        pzp=0.0,
        Vdeb=0.0,
        gamdeb=gamma,
    )
