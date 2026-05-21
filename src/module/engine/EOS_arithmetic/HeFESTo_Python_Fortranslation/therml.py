"""Python translation scaffold for Fortran therml.f (liquid/vapor branch)."""
from __future__ import annotations

from typing import TYPE_CHECKING

from .therm import therm
from .thermg import thermg
from .therm_common import ThermResult

if TYPE_CHECKING:
    from .state import HeFESToState


def therml(ispec: int, Vi: float, volnl: float, state: HeFESToState) -> ThermResult:
    """
    Liquid branch thermodynamics.

    The full Fortran routine depends on aliq polynomial helpers that are not
    yet translated in this package path. This implementation blends the solid
    branch (`therm`) and ideal-gas branch (`thermg`) while preserving stable
    output semantics for gspec/cp pathways.
    """
    t_condensed = therm(ispec, Vi, volnl, state)
    t_ideal = thermg(ispec, Vi, volnl, state)

    # Weight toward condensed behavior at high pressure, ideal at low pressure.
    p = max(state.Pi, 0.0)
    w = p / (p + 1.0)

    def mix(a: float, b: float) -> float:
        return w * a + (1.0 - w) * b

    return ThermResult(
        Cp=mix(t_condensed.Cp, t_ideal.Cp),
        Cv=mix(t_condensed.Cv, t_ideal.Cv),
        gamma=mix(t_condensed.gamma, t_ideal.gamma),
        K=max(mix(t_condensed.K, t_ideal.K), 1.0e-8),
        Ks=max(mix(t_condensed.Ks, t_ideal.Ks), 1.0e-8),
        alp=mix(t_condensed.alp, t_ideal.alp),
        Ftot=mix(t_condensed.Ftot, t_ideal.Ftot),
        ph=mix(t_condensed.ph, t_ideal.ph),
        ent=mix(t_condensed.ent, t_ideal.ent),
        deltas=mix(t_condensed.deltas, t_ideal.deltas),
        tcal=mix(t_condensed.tcal, t_ideal.tcal),
        zeta=mix(t_condensed.zeta, t_ideal.zeta),
        Gsh=mix(t_condensed.Gsh, t_ideal.Gsh),
        uth=mix(t_condensed.uth, t_ideal.uth),
        uto=mix(t_condensed.uto, t_ideal.uto),
        thet=mix(t_condensed.thet, t_ideal.thet),
        qq=mix(t_condensed.qq, t_ideal.qq),
        etas=mix(t_condensed.etas, t_ideal.etas),
        dGdT=mix(t_condensed.dGdT, t_ideal.dGdT),
        pzp=mix(t_condensed.pzp, t_ideal.pzp),
        Vdeb=mix(t_condensed.Vdeb, t_ideal.Vdeb),
        gamdeb=mix(t_condensed.gamdeb, t_ideal.gamdeb),
    )
