"""
Vectorized gamset: volume-dependent Grüneisen parameter and mode frequencies.

Implements ityp=1 (full finite-strain expansion in θ², Davies 1974,
Stixrude & Lithgow-Bertelloni 2005) which is the benchmark configuration.

For ityp=1, gamset.f runs two blocks of code (E-based then f-based) and
the f-based block overwrites all results.  Only the f-based formulas are
implemented here since ityp=1 always returns after the f-based block.

All modes scale by the SAME factor sqrt(1 + a*f + 0.5*b*f²):
    wd1, wd2, wd3, ws1, ws2, ws3, we1, we2, we3, we4, wou, wol

Shapes
------
All array arguments are broadcast-compatible:
  V    : (B, S) or (N,) flat  volumes at which to evaluate
  All parameter arrays: matching shape or broadcastable

The function returns same-shape arrays for γ, q, η_S, and scaled mode
frequencies.
"""

from __future__ import annotations
import numpy as np


def gamset_vec(
    # Reference mode frequencies (K) — broadcastable to V's shape
    wd1o: np.ndarray, wd2o: np.ndarray, wd3o: np.ndarray,
    ws1o: np.ndarray, ws2o: np.ndarray, ws3o: np.ndarray,
    we1o: np.ndarray, we2o: np.ndarray, we3o: np.ndarray, we4o: np.ndarray,
    wouo: np.ndarray, wolo: np.ndarray,
    # Grüneisen parameters
    gammo: np.ndarray, qo: np.ndarray, Got: np.ndarray,
    # Volumes
    Vi: np.ndarray, Vo: np.ndarray,
) -> dict:
    """Evaluate gamset (ityp=1) for all (batch, species) pairs simultaneously.

    Parameters
    ----------
    wd1o .. wolo : reference mode frequencies (K), shape broadcastable to Vi
    gammo        : γ_0, shape broadcastable to Vi
    qo           : q_0, shape broadcastable to Vi
    Got          : η_S0, shape broadcastable to Vi
    Vi           : current volume (cm³/mol)
    Vo           : reference volume (cm³/mol)

    Returns
    -------
    dict with keys:
        gamma         : Grüneisen parameter at Vi
        q             : logarithmic volume derivative of γ
        qp            : additional derivative term
        etas          : shear thermal parameter η_S
        detasdv       : dη_S/dV
        scale         : mode scale factor sqrt(1 + a*f + 0.5*b*f²)
        wd1..wol      : scaled mode frequencies (K)
    """
    # ── f-based finite-strain variable (SLB2005) ─────────────────────────────
    x = Vi / Vo                          # V/V0
    f = 0.5 * (x ** (-2.0 / 3.0) - 1.0)  # Eulerian finite strain

    # ── Expansion coefficients ────────────────────────────────────────────────
    a  = 6.0 * gammo
    b  = gammo * (36.0 * gammo - 18.0 * qo - 12.0)
    bs = -2.0 * gammo - 2.0 * Got

    # ── Mode scale factor ─────────────────────────────────────────────────────
    # Guard against arg <= 0 (can occur for highly-expanded minerals at extreme
    # V during NR transients).  Clipping to 1e-12 prevents sqrt(NaN) and allows
    # the solver to recover rather than propagating NaN forever.
    arg      = 1.0 + a * f + 0.5 * b * f**2
    arg_safe = np.maximum(arg, 1.0e-12)
    scale    = np.sqrt(arg_safe)
    ratio    = 1.0 / arg_safe

    # ── Grüneisen parameter γ ─────────────────────────────────────────────────
    gamma = (1.0 / 3.0) * 0.5 * ratio * (2.0 * f + 1.0) * (a + b * f)

    # ── q = d ln γ / d ln V ───────────────────────────────────────────────────
    # Guard against γ=0 (mineral with gammo=0 and qo=0)
    safe_gamma = np.where(np.abs(gamma) < 1e-30, 1e-30, gamma)
    q = (1.0 / 9.0) * (
        18.0 * gamma**2
        - 6.0 * gamma
        - 0.5 * ratio * (2.0 * f + 1.0)**2 * b
    ) / safe_gamma

    # Special case: gammo=0 and qo=0  →  q = qo = 0 (matches gamset.f line)
    zero_mask = (gammo == 0.0) & (qo == 0.0)
    q = np.where(zero_mask, qo, q)

    # ── qp (additional derivative, used in Kpth) ─────────────────────────────
    qp = (1.0 / 9.0) * (
        36.0 * gamma**2
        - 6.0 * gamma
        - b * ratio**2 / (6.0 * np.where(np.abs(q) < 1e-30, 1e-30, q))
          * (a + b * f) * (2.0 * f + 1.0)**3
        + 2.0 * b * ratio / (3.0 * np.where(np.abs(q) < 1e-30, 1e-30, q))
          * (2.0 * f + 1.0)**2
    ) / safe_gamma - q

    # ── η_S: shear thermal Grüneisen parameter ────────────────────────────────
    # Note: gamset.f sets etas=Got*gamma/gammo then immediately overwrites:
    etas = -gamma - 0.5 * ratio * (2.0 * f + 1.0)**2 * bs

    # ── dη_S/dV ───────────────────────────────────────────────────────────────
    detasdv = (
        -gamma * q / Vi
        + 2.0 * bs * (1.0 + 2.0 * f)**(7.0 / 2.0) * ratio / (3.0 * Vo)
        - bs * (1.0 + 2.0 * f)**(9.0 / 2.0) * ratio**2 / (6.0 * Vo) * (a + b * f)
    )

    # ── Scaled mode frequencies ───────────────────────────────────────────────
    wd1 = wd1o * scale;  wd2 = wd2o * scale;  wd3 = wd3o * scale
    ws1 = ws1o * scale;  ws2 = ws2o * scale;  ws3 = ws3o * scale
    we1 = we1o * scale;  we2 = we2o * scale
    we3 = we3o * scale;  we4 = we4o * scale
    wou = wouo * scale;  wol = wolo * scale

    return dict(
        gamma=gamma, q=q, qp=qp, etas=etas, detasdv=detasdv,
        scale=scale,
        wd1=wd1, wd2=wd2, wd3=wd3,
        ws1=ws1, ws2=ws2, ws3=ws3,
        we1=we1, we2=we2, we3=we3, we4=we4,
        wou=wou, wol=wol,
    )


# ---------------------------------------------------------------------------
# Torch-native version (GPU-compatible) — identical math, torch ops
# ---------------------------------------------------------------------------

def gamset_torch(
    wd1o, wd2o, wd3o, ws1o, ws2o, ws3o,
    we1o, we2o, we3o, we4o, wouo, wolo,
    gammo, qo, Got,
    Vi, Vo,
) -> dict:
    """Torch equivalent of gamset_vec — accepts and returns torch tensors."""
    import torch

    x = Vi / Vo
    f = 0.5 * (x ** (-2.0 / 3.0) - 1.0)

    a  = 6.0 * gammo
    b  = gammo * (36.0 * gammo - 18.0 * qo - 12.0)
    bs = -2.0 * gammo - 2.0 * Got

    arg      = 1.0 + a * f + 0.5 * b * f**2
    arg_safe = torch.clamp(arg, min=1.0e-12)
    scale    = torch.sqrt(arg_safe)
    ratio    = 1.0 / arg_safe

    gamma = (1.0 / 3.0) * 0.5 * ratio * (2.0 * f + 1.0) * (a + b * f)

    safe_gamma = torch.where(gamma.abs() < 1e-30,
                             torch.full_like(gamma, 1e-30), gamma)
    q = ((1.0 / 9.0) * (
        18.0 * gamma**2
        - 6.0 * gamma
        - 0.5 * ratio * (2.0 * f + 1.0)**2 * b
    ) / safe_gamma)

    zero_mask = (gammo == 0.0) & (qo == 0.0)
    q = torch.where(zero_mask, qo, q)

    safe_q = torch.where(q.abs() < 1e-30, torch.full_like(q, 1e-30), q)
    qp = ((1.0 / 9.0) * (
        36.0 * gamma**2
        - 6.0 * gamma
        - b * ratio**2 / (6.0 * safe_q) * (a + b * f) * (2.0 * f + 1.0)**3
        + 2.0 * b * ratio / (3.0 * safe_q) * (2.0 * f + 1.0)**2
    ) / safe_gamma - q)

    etas    = -gamma - 0.5 * ratio * (2.0 * f + 1.0)**2 * bs
    detasdv = (
        -gamma * q / Vi
        + 2.0 * bs * (1.0 + 2.0 * f)**(7.0 / 2.0) * ratio / (3.0 * Vo)
        - bs * (1.0 + 2.0 * f)**(9.0 / 2.0) * ratio**2 / (6.0 * Vo) * (a + b * f)
    )

    wd1 = wd1o * scale;  wd2 = wd2o * scale;  wd3 = wd3o * scale
    ws1 = ws1o * scale;  ws2 = ws2o * scale;  ws3 = ws3o * scale
    we1 = we1o * scale;  we2 = we2o * scale
    we3 = we3o * scale;  we4 = we4o * scale
    wou = wouo * scale;  wol = wolo * scale

    return dict(
        gamma=gamma, q=q, qp=qp, etas=etas, detasdv=detasdv,
        scale=scale,
        wd1=wd1, wd2=wd2, wd3=wd3,
        ws1=ws1, ws2=ws2, ws3=ws3,
        we1=we1, we2=we2, we3=we3, we4=we4,
        wou=wou, wol=wol,
    )
