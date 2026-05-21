"""Python translation of Fortran volume.f for solid phases."""
from __future__ import annotations

from math import exp
from typing import TYPE_CHECKING

from .cage import cage
from .etherm import Etherm
from .gamset import gamset
from .nlmin_V import nlmin_V
from .therm_common import apar_value
from .zeroin import zeroin

if TYPE_CHECKING:
    from .state import HeFESToState


def _pressure_residual(ispec: int, Vi: float, state: HeFESToState) -> float:
    apar = state.apar
    ti = max(state.Ti, 1.0e-12)

    fn = max(apar_value(apar, ispec, 1, 1.0), 1.0e-12)
    to = max(apar_value(apar, ispec, 4, 298.15), 1.0e-12)
    vo = max(apar_value(apar, ispec, 6, max(Vi, 1.0e-12)), 1.0e-12)
    ko = max(apar_value(apar, ispec, 7, 160.0), 1.0e-12)
    kop = apar_value(apar, ispec, 8, 4.0)
    kopp = apar_value(apar, ispec, 9, 0.0)
    wd1o = max(apar_value(apar, ispec, 16, 800.0), 1.0e-12)
    wd2o = max(apar_value(apar, ispec, 17, 0.0), 0.0)
    wd3o = max(apar_value(apar, ispec, 18, 0.0), 0.0)
    ws1o = max(apar_value(apar, ispec, 19, 0.0), 0.0)
    ws2o = max(apar_value(apar, ispec, 20, 0.0), 0.0)
    ws3o = max(apar_value(apar, ispec, 21, 0.0), 0.0)
    we1o = max(apar_value(apar, ispec, 22, 0.0), 0.0)
    qe1 = apar_value(apar, ispec, 23, 0.0)
    we2o = max(apar_value(apar, ispec, 24, 0.0), 0.0)
    qe2 = apar_value(apar, ispec, 25, 0.0)
    we3o = max(apar_value(apar, ispec, 26, 0.0), 0.0)
    qe3 = apar_value(apar, ispec, 27, 0.0)
    we4o = max(apar_value(apar, ispec, 28, 0.0), 0.0)
    qe4 = apar_value(apar, ispec, 29, 0.0)
    wouo = max(apar_value(apar, ispec, 30, 0.0), 0.0)
    wolo = max(apar_value(apar, ispec, 31, 0.0), 0.0)
    gammo = max(apar_value(apar, ispec, 38, 1.2), 1.0e-12)
    qo = max(apar_value(apar, ispec, 39, 0.3), 1.0e-12)
    be = apar_value(apar, ispec, 35, 0.0)
    ge = apar_value(apar, ispec, 36, 0.0)
    anh = apar_value(apar, ispec, 37, 0.0)
    izp = int(round(apar_value(apar, ispec, 34, 0.0)))
    ibv = int(round(apar_value(apar, ispec, 32, 0.0)))
    got = apar_value(apar, ispec, 43, 0.0)

    gam = gamset(
        wd1o, wd2o, wd3o, ws1o, ws2o, ws3o, we1o, we2o, we3o, we4o, wouo, wolo,
        gammo, qo, got, Vi, vo,
    )
    gamma = gam.gamma if gam.gamma != 0.0 else gammo

    uth = Etherm(ti, fn, 1.0, gam.wd1, gam.wd2, gam.wd3, gam.ws1, gam.ws2, gam.ws3, gam.wou, gam.wol, gam.we1, gam.we2, gam.we3, gam.we4, qe1, qe2, qe3, qe4)
    uto = Etherm(to, fn, 1.0, gam.wd1, gam.wd2, gam.wd3, gam.ws1, gam.ws2, gam.ws3, gam.wou, gam.wol, gam.we1, gam.we2, gam.we3, gam.we4, qe1, qe2, qe3, qe4)
    ph = 0.001 * (gamma / max(Vi, 1.0e-12)) * (uth - uto)
    pa = 0.001 * 3.0 * fn * state.Rgas * anh * (ti * ti - to * to) / max(Vi, 1.0e-12)
    pzp = 0.0
    if abs(izp) == 1:
        pzp = izp * 0.001 * (9.0 / 8.0) * fn * state.Rgas * gam.wd1 * gamma / max(Vi, 1.0e-12)

    beta = be * (Vi / vo) ** ge if vo > 0.0 else 0.0
    pel = 0.001 * 0.5 * ge * beta * (ti * ti - to * to) / max(Vi, 1.0e-12)

    if ibv == 2:
        g = -0.5 * ((Vi / vo) ** (2.0 / 3.0) - 1.0)
        pc = 3.0 * ko * g * (1.0 - 2.0 * g) ** (-0.5) * (1.0 + 1.5 * kop * g)
    elif ibv == 1:
        eta = 3.0 * (kop - 1.0) / 2.0
        xv = (Vi / vo) ** (1.0 / 3.0)
        pc = 3.0 * ko * (1.0 - xv) * exp(eta * (1.0 - xv)) / (xv * xv)
    else:
        f = 0.5 * ((Vi / vo) ** (-2.0 / 3.0) - 1.0)
        a3 = 3.0 * (kop - 4.0)
        a4 = 9.0 * (kopp + kop * (kop - 7.0) + 143.0 / 9.0)
        a5 = apar_value(apar, ispec, 43, 0.0)
        if kopp == 0.0:
            a4 = 0.0
        pc = 3.0 * ko * f * (1.0 + 2.0 * f) ** 2.5 * (1.0 + 0.5 * a3 * f + (1.0 / 6.0) * a4 * f * f + (1.0 / 24.0) * a5 * f * f * f)

    return pc + ph + pa + pel + pzp - state.Pi


def volume(ispec: int, x1: float, state: HeFESToState) -> float:
    """Solve the solid-phase volume for the current P,T point.

    Raise ValueError for unrecoverable/physical failures (zero seed volume,
    bracketing/solver failure) so callers can abort instead of propagating
    invalid values.
    """

    apar = state.apar
    EPS = 1.0e-12
    vo = max(apar_value(apar, ispec, 6, max(x1, EPS)), EPS)
    # If seed/reference volume is effectively zero, this assemblage is invalid
    if vo <= EPS:
        sname = None
        try:
            sname = state.sname[ispec]
        except Exception:
            pass
        raise ValueError(f"zero or near-zero seed volume for species {ispec} ({sname}); apar v0 missing or zero")

    vlan = max(apar_value(apar, ispec, 40, 0.0), 0.0)
    vlow = max(apar_value(apar, ispec, 51, 1.0), EPS)
    vupp = max(apar_value(apar, ispec, 52, 1.0e6), vlow)
    vsplow = max(apar_value(apar, ispec, 53, vlow), vlow)
    vspupp = max(apar_value(apar, ispec, 54, vupp), vsplow)
    vlo = max(vlow, vsplow)
    vhi = min(vupp, vspupp)
    if vhi <= vlo:
        raise ValueError(f"invalid volume bounds for species {ispec}: vlo={vlo}, vhi={vhi}")

    if bool(getattr(state, "isochor", False)):
        if vo <= 0.0:
            raise ValueError("isochoric state but seed volume is zero")
        return vo

    x1 = min(max(x1, vlo), vhi)
    x2 = min(vhi, x1 * (1.0 + 1.0e-12) if x1 > 0.0 else x1 + 1.0e-12)
    if x2 <= x1:
        x2 = min(vhi, x1 + max(EPS, 1.0e-6 * max(abs(x1), 1.0)))

    # Attempt to bracket the root
    a, b, ires = cage(lambda v: _pressure_residual(ispec, v, state), x1, x2, vlo, vhi)
    if ires == 1:
        val = zeroin(a, b, lambda v: _pressure_residual(ispec, v, state), 1.0e-15)
        if val <= EPS:
            raise ValueError(f"root finder returned non-positive volume {val} for species {ispec}")
        return val

    # If bracketing failed, fail fast rather than continuing
    raise ValueError(f"failed to bracket root for species {ispec} (cage returned ires={ires}); x1={x1}, x2={x2}, vlo={vlo}, vhi={vhi}")

    # The remainder of the original legacy fallback is intentionally removed so
    # failures are explicit and do not propagate zeros/NaNs silently.
