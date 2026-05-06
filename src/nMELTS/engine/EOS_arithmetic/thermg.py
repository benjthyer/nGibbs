"""Python translation of Fortran thermg.f (ideal gas branch)."""
from __future__ import annotations

from math import log, sqrt
from typing import TYPE_CHECKING

from .therm_common import ThermResult, apar_value

if TYPE_CHECKING:
    from .state import HeFESToState


def thermg(ispec: int, Vi: float, volnl: float, state: HeFESToState) -> ThermResult:
    """Ideal-gas thermodynamic properties for species ispec."""
    apar = state.apar
    rgas = state.Rgas
    ti = max(state.Ti, 1.0e-12)

    fn = max(apar_value(apar, ispec, 1, 1.0), 1.0e-12)
    to = max(apar_value(apar, ispec, 4, 298.15), 1.0e-12)
    so = apar_value(apar, ispec, 10, 0.0)
    acp = apar_value(apar, ispec, 11, 0.0)
    bcp = apar_value(apar, ispec, 12, 0.0)
    ccp = apar_value(apar, ispec, 13, 0.0)
    dcp = apar_value(apar, ispec, 14, 0.0)
    ecp = apar_value(apar, ispec, 15, 0.0)

    p1bar = 1.0e-4

    cp = acp + bcp * ti + ccp / (ti * ti) + dcp / sqrt(ti) + ecp * ti * ti
    cv = max(cp - fn * rgas, 1.0e-8)

    s = acp * log(ti) + bcp * ti - 0.5 * ccp * ti ** (-2.0) - 2.0 * dcp * ti ** (-0.5) + 0.5 * ecp * ti * ti
    sref = acp * log(to) + bcp * to - 0.5 * ccp * to ** (-2.0) - 2.0 * dcp * to ** (-0.5) + 0.5 * ecp * to * to
    ent = s - sref + so

    h = acp * ti + 0.5 * bcp * ti * ti - ccp / ti + 2.0 * dcp * sqrt(ti) + (ecp / 3.0) * ti * ti * ti
    href = acp * to + 0.5 * bcp * to * to - ccp / to + 2.0 * dcp * sqrt(to) + (ecp / 3.0) * to * to * to
    h = h - href

    ftot = h - ti * ent + to * so + rgas * ti * log(max(state.Pi, p1bar) / p1bar)

    p = 0.001 * fn * rgas * ti / max(Vi, 1.0e-12)
    k = max(p, 1.0e-8)
    alp = (0.001 * fn * rgas / max(Vi, 1.0e-12)) / k

    gamma = cp / cv - 1.0
    ks = k * (1.0 + alp * gamma * ti)

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
        uth=cv * ti,
        uto=cv * to,
        thet=0.0,
        qq=0.0,
        etas=0.0,
        dGdT=0.0,
        pzp=0.0,
        Vdeb=0.0,
        gamdeb=gamma,
    )
