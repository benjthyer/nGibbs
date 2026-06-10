"""
Vectorized pressure function (Birch-Murnaghan + thermal) and its volume
derivative, matching pressure.f and therm.f.

For the standard mantle case (izp=0, be=0, iltyp=0/inactive, ibv=0):
    P = pc + ph - Pi

where:
    pc = BM cold pressure  (GPa)
    ph = 0.001 * (γ/V) * (Uth - Uto)  (GPa)
    Pi = target pressure  (GPa)

Additional terms that can be present but are zero for most mantle minerals:
    pa   = 0.001 * 3*fn*R*q2A2*(Ti²-To²)/V   (anharmonic, usually q2A2=0)
    pel  = 0.001 * 0.5*ge*be*(Ti²-To²)/V      (electronic, usually be=0)
    pzp  = 0.001 * (9/8)*fn*R*wd1*gamma/V     (zero-point, izp≠0)
    pn   = Landau contribution (after landauqr, small and applied post-convergence)

Inelastic corrections to pressure (Hessian terms) are NOT included here;
they require the full chemical potential Hessian and are irrelevant for the
fixed-composition Vp/Vs path.

Units throughout: GPa, cm³/mol, J/mol (use 0.001 factor to convert J/cm³→GPa)
"""

from __future__ import annotations
import numpy as np

from .constants import Rgas


def pressure_vec(
    Vi       : np.ndarray,   # (...) current volume (cm³/mol)
    Ti       : np.ndarray,   # (...) or (B,) temperature (K)
    Pi_target: np.ndarray,   # (...) or (B,) target pressure (GPa)
    # Parameters
    Vo: np.ndarray, Ko: np.ndarray, Kop: np.ndarray, Kopp: np.ndarray,
    gamma: np.ndarray, q: np.ndarray,
    fn: np.ndarray, wd1: np.ndarray,
    a5: np.ndarray,           # apar[:,42] = C44' (5th-order BM coefficient)
    izp: np.ndarray,          # zero-point flag
    be : np.ndarray,          # electronic beta
    ge : np.ndarray,          # electronic Grüneisen
    q2A2: np.ndarray,         # anharmonic coefficient
    To : np.ndarray,          # reference temperature (K)
    # Precomputed thermal energies at Vi (updated every Newton step)
    Uth: np.ndarray,          # Etherm(Ti)
    Uto: np.ndarray,          # Etherm(To) at same volume
    # Optional: currently scaled Debye frequency (for pzp)
    wd1_Vi: np.ndarray | None = None,
    ibv: np.ndarray | None = None,   # EOS type flag (0=BM, 1=Vinet, 2=Lag)
) -> np.ndarray:
    """Compute pressure residual P(Vi) - Pi_target.

    Works for any broadcastable array shapes, including flat 1D (N,) arrays
    (used by the optimised flat-active-only NR solver).

    Returns array of residuals in GPa.
    """
    # ── Cold pressure (BM unless ibv overridden) ──────────────────────────────
    pc = cold_pressure_vec(Vi, Vo, Ko, Kop, Kopp, a5, ibv=ibv)

    # ── Thermal pressure ──────────────────────────────────────────────────────
    ph = 1.0e-3 * (gamma / Vi) * (Uth - Uto)

    # ── Anharmonic pressure pa = 0.001*3*fn*R*q2A2*(Ti²-To²)/V ──────────────
    Ti_b = np.broadcast_to(Ti, Vi.shape)
    To_b = np.broadcast_to(To, Vi.shape)
    pa = 1.0e-3 * 3.0 * fn * Rgas * q2A2 * (Ti_b**2 - To_b**2) / Vi

    # ── Electronic pressure ───────────────────────────────────────────────────
    beta = be * (Vi / Vo) ** ge
    pel = 1.0e-3 * 0.5 * ge * beta * (Ti_b**2 - To_b**2) / Vi

    # ── Zero-point pressure ───────────────────────────────────────────────────
    w1 = wd1_Vi if wd1_Vi is not None else wd1
    pzp = np.sign(izp) * np.where(
        np.abs(izp) == 1,
        1.0e-3 * (9.0 / 8.0) * fn * Rgas * w1 * gamma / Vi,
        0.0
    )

    # ── Target pressure broadcast ─────────────────────────────────────────────
    # Handles both 2D case (Pi_target shape (B,), Vi shape (B,S)) and
    # flat 1D case (both shape (N,)).
    pi = np.asarray(Pi_target)
    if pi.ndim == 1 and Vi.ndim == 2:
        Pi_b = np.broadcast_to(pi[:, np.newaxis], Vi.shape)
    else:
        Pi_b = np.broadcast_to(pi, Vi.shape)

    return pc + ph + pa + pel + pzp - Pi_b


def cold_pressure_vec(
    Vi : np.ndarray,
    Vo : np.ndarray,
    Ko : np.ndarray,
    Kop: np.ndarray,
    Kopp: np.ndarray,
    a5 : np.ndarray,  # apar[:,42] 0-indexed = C44' (5th-order term)
    ibv: np.ndarray | None = None,  # EOS type (0=BM default)
) -> np.ndarray:
    """Cold (static) pressure in GPa — Birch-Murnaghan 3rd/4th/5th order.

    Matches the BM branch in therm.f (and pressure.f):
        f = 0.5*((V/V0)^(-2/3) - 1)
        a3 = 3*(Kop-4)
        a4 = 9*(Kopp + Kop*(Kop-7) + 143/9)  [if Kopp≠0, else 0]
        pc = 3*Ko*f*(1+2f)^2.5*(1 + a3/2*f + a4/6*f² + a5/24*f³)

    Note: therm.f uses a slightly different a4 definition in the Kc formula
    vs the pc formula.  Here we implement the PRESSURE formula:
        a4_pc = 1.5*(Ko*Kopp + Kop*(Kop-7) + 143/9)  (from pc line in therm.f)
    which matches pressure.f.  The Kc (bulk modulus) formula uses different
    coefficients — see therm_props.py.
    """
    # Birch-Murnaghan (default, ibv==0 or None)
    f  = 0.5 * ((Vi / Vo) ** (-2.0 / 3.0) - 1.0)
    a3 = 1.5 * (Kop - 4.0)
    # In therm.f pc uses:  a4 = 1.5*(Ko*Kopp + Kop*(Kop-7) + 143/9)
    a4_p = np.where(Kopp != 0.0,
                    1.5 * (Ko * Kopp + Kop * (Kop - 7.0) + 143.0 / 9.0),
                    0.0)

    pc_bm = (3.0 * Ko * f * (1.0 + 2.0 * f) ** 2.5
             * (1.0 + (1.5 * Kop - 6.0) * f + a4_p * f**2))

    if ibv is None:
        return pc_bm

    # Vinet EOS
    xV   = (Vi / Vo) ** (1.0 / 3.0)
    eta  = 1.5 * (Kop - 1.0)
    pc_v = Ko * (2.0 + (eta - 1.0) * xV - eta * xV**2) * np.exp(eta * (1.0 - xV)) / xV**2

    # Lagrangian EOS
    g_l   = -0.5 * ((Vi / Vo) ** (2.0 / 3.0) - 1.0)
    pc_l  = Ko * (1.0 - 2.0 * g_l) ** (-1.0 / 2.0) \
            * (1.0 + g_l * (3.0 * Kop - 1.0) - 4.5 * Kop * g_l**2)

    out = np.where(ibv == 1, pc_v, np.where(ibv == 2, pc_l, pc_bm))
    return out


def cold_bulk_modulus_vec(
    Vi  : np.ndarray,
    Vo  : np.ndarray,
    Ko  : np.ndarray,
    Kop : np.ndarray,
    Kopp: np.ndarray,
    a5  : np.ndarray,
    ibv : np.ndarray | None = None,
) -> np.ndarray:
    """Cold (isothermal) bulk modulus Kc in GPa, matching therm.f BM formula.

    This is the MODULUS formula (different from the pressure formula coefficients):
        Kc = Ko*(1+2f)^2.5*(1 + (7+a3)*f + (9/2*a3+a4/2)*f² + (11/6*a4+a5/6)*f³
             + 13/24*a5*f⁴)
    where a3=3*(Kop-4), a4=9*(Kopp+Kop*(Kop-7)+143/9) [if Kopp≠0].
    """
    f   = 0.5 * ((Vi / Vo) ** (-2.0 / 3.0) - 1.0)
    a3  = 3.0 * (Kop - 4.0)
    a4  = np.where(Kopp != 0.0,
                   9.0 * (Kopp + Kop * (Kop - 7.0) + 143.0 / 9.0),
                   0.0)

    Kc_bm = Ko * (1.0 + 2.0 * f) ** 2.5 * (
        1.0
        + (7.0 + a3) * f
        + (4.5 * a3 + 0.5 * a4) * f**2
        + (11.0 / 6.0 * a4 + a5 / 6.0) * f**3
        + 13.0 / 24.0 * a5 * f**4
    )

    if ibv is None:
        return Kc_bm

    # Vinet
    xV  = (Vi / Vo) ** (1.0 / 3.0)
    eta = 1.5 * (Kop - 1.0)
    Kc_v = Ko * (2.0 + (eta - 1.0) * xV - eta * xV**2) \
           * np.exp(eta * (1.0 - xV)) / xV**2
    # (This is pc_v, not Kc_v — for Vinet use the full Kc from therm.f which
    #  computes it differently.  For now use pc_v as approximation.)

    # Lagrangian
    g_l  = -0.5 * ((Vi / Vo) ** (2.0 / 3.0) - 1.0)
    Kc_l = Ko * (1.0 - 2.0 * g_l) ** (-1.0 / 2.0) \
           * (1.0 + g_l * (3.0 * Kop - 1.0) - 4.5 * Kop * g_l**2)

    return np.where(ibv == 1, Kc_v, np.where(ibv == 2, Kc_l, Kc_bm))


# ---------------------------------------------------------------------------
# Torch-native versions (GPU-compatible)
# ---------------------------------------------------------------------------

def cold_pressure_torch(Vi, Vo, Ko, Kop, Kopp, a5, ibv=None):
    """Torch equivalent of cold_pressure_vec."""
    import torch
    f    = 0.5 * ((Vi / Vo) ** (-2.0 / 3.0) - 1.0)
    a3   = 1.5 * (Kop - 4.0)
    a4_p = torch.where(Kopp != 0.0,
                       1.5 * (Ko * Kopp + Kop * (Kop - 7.0) + 143.0 / 9.0),
                       torch.zeros_like(Kopp))
    pc_bm = (3.0 * Ko * f * (1.0 + 2.0 * f) ** 2.5
             * (1.0 + (1.5 * Kop - 6.0) * f + a4_p * f**2))
    if ibv is None:
        return pc_bm
    xV   = (Vi / Vo) ** (1.0 / 3.0)
    eta  = 1.5 * (Kop - 1.0)
    pc_v = Ko * (2.0 + (eta - 1.0) * xV - eta * xV**2) * torch.exp(eta * (1.0 - xV)) / xV**2
    g_l  = -0.5 * ((Vi / Vo) ** (2.0 / 3.0) - 1.0)
    pc_l = Ko * (1.0 - 2.0 * g_l) ** (-0.5) * (1.0 + g_l * (3.0*Kop - 1.0) - 4.5*Kop*g_l**2)
    return torch.where(ibv == 1, pc_v, torch.where(ibv == 2, pc_l, pc_bm))


def cold_bulk_modulus_torch(Vi, Vo, Ko, Kop, Kopp, a5, ibv=None):
    """Torch equivalent of cold_bulk_modulus_vec."""
    import torch
    f   = 0.5 * ((Vi / Vo) ** (-2.0 / 3.0) - 1.0)
    a3  = 3.0 * (Kop - 4.0)
    a4  = torch.where(Kopp != 0.0,
                      9.0 * (Kopp + Kop * (Kop - 7.0) + 143.0 / 9.0),
                      torch.zeros_like(Kopp))
    Kc_bm = Ko * (1.0 + 2.0 * f) ** 2.5 * (
        1.0
        + (7.0 + a3) * f
        + (4.5 * a3 + 0.5 * a4) * f**2
        + (11.0 / 6.0 * a4 + a5 / 6.0) * f**3
        + 13.0 / 24.0 * a5 * f**4
    )
    if ibv is None:
        return Kc_bm
    xV   = (Vi / Vo) ** (1.0 / 3.0)
    eta  = 1.5 * (Kop - 1.0)
    Kc_v = Ko * (2.0 + (eta - 1.0) * xV - eta * xV**2) * torch.exp(eta*(1.0-xV)) / xV**2
    g_l  = -0.5 * ((Vi / Vo) ** (2.0 / 3.0) - 1.0)
    Kc_l = Ko * (1.0 - 2.0*g_l)**(-0.5) * (1.0 + g_l*(3.0*Kop-1.0) - 4.5*Kop*g_l**2)
    return torch.where(ibv == 1, Kc_v, torch.where(ibv == 2, Kc_l, Kc_bm))


def pressure_torch(Vi, Ti, Pi_target, Vo, Ko, Kop, Kopp,
                   gamma, q, fn, wd1, a5, izp, be, ge, q2A2,
                   To, Uth, Uto, wd1_Vi=None, ibv=None):
    """Torch equivalent of pressure_vec."""
    import torch
    from .constants import Rgas
    pc  = cold_pressure_torch(Vi, Vo, Ko, Kop, Kopp, a5, ibv=ibv)
    ph  = 1.0e-3 * (gamma / Vi) * (Uth - Uto)
    Ti_b = Ti.expand_as(Vi) if Ti.shape != Vi.shape else Ti
    To_b = To.expand_as(Vi) if To.shape != Vi.shape else To
    pa   = 1.0e-3 * 3.0 * fn * Rgas * q2A2 * (Ti_b**2 - To_b**2) / Vi
    beta = be * (Vi / Vo) ** ge
    pel  = 1.0e-3 * 0.5 * ge * beta * (Ti_b**2 - To_b**2) / Vi
    w1   = wd1_Vi if wd1_Vi is not None else wd1
    pzp  = torch.sign(izp.float()) * torch.where(
        izp.abs() == 1,
        1.0e-3 * (9.0 / 8.0) * fn * Rgas * w1 * gamma / Vi,
        torch.zeros_like(Vi)
    )
    Pi_b = Pi_target.expand_as(Vi) if Pi_target.shape != Vi.shape else Pi_target
    return pc + ph + pa + pel + pzp - Pi_b
