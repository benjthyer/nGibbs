"""
Vectorized thermal kernel functions: Etherm, Ctherm, Ftherm.

These match Etherm.f, Ctherm.f, and Ftherm.f exactly, operating on
(B, S) arrays (batch × species) simultaneously.

Design notes
------------
- The "aniso" flag in the Fortran is True only when wd2≠wd1 or wd3≠wd1
  (checked via the average).  For most mantle minerals wd2=wd3=0, so
  aniso=False and only wd1 is used for the Debye branch.
- The zero-point energy term 9/8 * fn * R * wd1 is always included in
  Etherm and Ftherm (it is absent from Ctherm).
- su (mode normalisation) and qo (optic weight) follow the precedence
  rules from the Fortran exactly.
- All input frequency arrays must already be at the current volume V
  (i.e. post-gamset scaling).
"""

from __future__ import annotations
import numpy as np

from .constants import Rgas
from .dos_tables import Heat_vec, Ener_vec, Helm_vec, Heat_torch, Ener_torch, Helm_torch

# torch is optional: every *_torch routine below imports it locally, matching the
# lazy-import convention used by the rest of this package.  Keeping it optional
# here means the numpy path (and the metamorphic module, which is pure numpy)
# works on installs without torch.
try:                                    # pragma: no cover
    import torch
except ImportError:                     # pragma: no cover
    torch = None

# ─────────────────────────────────────────────────────────────────────────────
# Internal helper: compute mode weights (su, qo, aniso mask)
# ─────────────────────────────────────────────────────────────────────────────

def _mode_weights(fn, zu,
                  wd1, wd2, wd3,
                  ws1, ws2, ws3,
                  wou, wol,
                  we1, we2, we3, we4,
                  qe1, qe2, qe3, qe4):
    """Compute su, qo, do, aniso matching Etherm.f / Ctherm.f / Ftherm.f.

    All inputs are (B, S) arrays (or broadcastable).
    Returns su, qo, do, aniso — all (B, S).
    """
    su   = fn * zu                           # total mode count per formula unit
    wo   = (wou + wol) / 2.0
    do_  = np.where(wo != 0.0, (wou - wol) / wo, 0.0)

    wdav = (wd1 + wd2 + wd3) / 3.0
    wsav = (ws1 + ws2 + ws3) / 3.0

    # If no acoustic modes at all: su → ∞ (all weight goes to optic/Einstein)
    no_acoustic = (wdav == 0.0) & (wsav == 0.0)
    su = np.where(no_acoustic, 1.0e15, su)

    qe = qe1 + qe2 + qe3 + qe4
    qo = 1.0 - 1.0 / su - qe
    # If optic absent: qo = 0
    qo = np.where(wo == 0.0, 0.0, qo)

    # aniso: True when debye or sin branches are not all equal (via averages)
    aniso_deb = wdav != (wd1 / 3.0)
    aniso_sin = wsav != (ws1 / 3.0)
    aniso = aniso_deb | aniso_sin

    # If no Einstein and no optic: su=1 (assign all weight to acoustic band)
    no_ein_opt = (qe == 0.0) & (qo == 0.0)
    su = np.where(no_ein_opt, 1.0, su)

    # If qo=0: redistribute all non-Einstein modes to acoustic band
    qo_zero = (qo == 0.0)
    su_redist = 1.0 / (1.0 - qe)
    # Protect against qe=1 (division by zero)
    su_redist = np.where(qe >= 1.0, 1.0e15, su_redist)
    su = np.where(qo_zero, su_redist, su)

    return su, qo, do_, aniso


# ─────────────────────────────────────────────────────────────────────────────
# Batch DOS evaluations
# ─────────────────────────────────────────────────────────────────────────────

def _batch_ener(w_over_T: np.ndarray, do_: np.ndarray,
                idos: int, flat_only=False) -> np.ndarray:
    """Evaluate Ener_vec on a (B,S) array of x=w/T values.

    idos=4 (optic): uses do_ as width parameter d.
    For idos≠4, do_ is ignored (matches Fortran: d is passed but not used).
    """
    shape = w_over_T.shape
    x_flat = w_over_T.ravel()
    if idos == 4:
        d_flat = do_.ravel()
    else:
        d_flat = 0.0
    result = Ener_vec(x_flat, idos=idos, d=d_flat)
    return result.reshape(shape)


def _batch_heat(w_over_T: np.ndarray, do_: np.ndarray, idos: int) -> np.ndarray:
    shape = w_over_T.shape
    x_flat = w_over_T.ravel()
    d_flat = do_.ravel() if idos == 4 else 0.0
    result = Heat_vec(x_flat, idos=idos, d=d_flat)
    return result.reshape(shape)


def _batch_helm(w_over_T: np.ndarray, do_: np.ndarray, idos: int) -> np.ndarray:
    shape = w_over_T.shape
    x_flat = w_over_T.ravel()
    d_flat = do_.ravel() if idos == 4 else 0.0
    result = Helm_vec(x_flat, idos=idos, d=d_flat)
    return result.reshape(shape)


def _safe_x(w, T):
    """x = w/T with zero result when w=0 (absent mode)."""
    return np.where(w == 0.0, 0.0, w / T)


# ─────────────────────────────────────────────────────────────────────────────
# Public functions
# ─────────────────────────────────────────────────────────────────────────────

def Etherm_vec(Ti,
               fn, zu,
               wd1, wd2, wd3,
               ws1, ws2, ws3,
               wou, wol,
               we1, we2, we3, we4,
               qe1, qe2, qe3, qe4) -> np.ndarray:
    """Vectorized Etherm: internal thermal energy Uth (J/mol).

    Matches Etherm.f exactly:
        Etherm = Ud + Us + Ue + Uo
    where Ud includes the zero-point energy 9/8*fn*R*wd1.

    Parameters
    ----------
    Ti   : (B, S) or scalar — temperature (K)
    fn   : (S,) or (B, S)  — atoms in formula unit
    zu   : (S,) or (B, S)  — formula units per unit cell
    wd1..wol : (B, S)      — scaled mode frequencies (K) at current volume
    qe1..qe4 : (S,) or (B,S) — Einstein oscillator weights

    Returns
    -------
    (B, S)  thermal energy in J/mol
    """
    Ti_arr = np.asarray(Ti, dtype=np.float64)

    su, qo, do_, aniso = _mode_weights(
        fn, zu, wd1, wd2, wd3, ws1, ws2, ws3,
        wou, wol, we1, we2, we3, we4, qe1, qe2, qe3, qe4
    )

    # Guard: Ti=0 → Etherm=0
    Ti_safe = np.where(Ti_arr <= 0.0, 1.0, Ti_arr)   # avoids x/0

    # ── Debye (idos=1) ────────────────────────────────────────────────────────
    x1 = _safe_x(wd1, Ti_safe)
    ud = (1.0 / su) * np.where(wd1 == 0.0, 0.0,
                                _batch_ener(x1, do_, idos=1))
    if np.any(aniso):
        x2 = _safe_x(wd2, Ti_safe)
        x3 = _safe_x(wd3, Ti_safe)
        e2 = np.where(wd2 == 0.0, 0.0, _batch_ener(x2, do_, idos=1))
        e3 = np.where(wd3 == 0.0, 0.0, _batch_ener(x3, do_, idos=1))
        ud = np.where(aniso,
                      ud / 3.0 + (e2 + e3) / (3.0 * su),
                      ud)
    # Ud = 3*fn*R*Ti*ud  +  zero-point energy
    Ud = 3.0 * fn * Rgas * Ti_safe * ud + (9.0 / 8.0) * fn * Rgas * wd1

    # ── Sin (idos=3) ──────────────────────────────────────────────────────────
    xs1 = _safe_x(ws1, Ti_safe)
    us = (1.0 / su) * np.where(ws1 == 0.0, 0.0,
                                _batch_ener(xs1, do_, idos=3))
    if np.any(aniso):
        xs2 = _safe_x(ws2, Ti_safe)
        xs3 = _safe_x(ws3, Ti_safe)
        s2 = np.where(ws2 == 0.0, 0.0, _batch_ener(xs2, do_, idos=3))
        s3 = np.where(ws3 == 0.0, 0.0, _batch_ener(xs3, do_, idos=3))
        us = np.where(aniso,
                      us / 3.0 + (s2 + s3) / (3.0 * su),
                      us)
    Us = 3.0 * fn * Rgas * Ti_safe * us

    # ── Einstein (idos=2) ─────────────────────────────────────────────────────
    xe1 = _safe_x(we1, Ti_safe)
    xe2 = _safe_x(we2, Ti_safe)
    xe3 = _safe_x(we3, Ti_safe)
    xe4 = _safe_x(we4, Ti_safe)
    ue = (qe1 * np.where(we1 == 0.0, 0.0, _batch_ener(xe1, do_, idos=2))
        + qe2 * np.where(we2 == 0.0, 0.0, _batch_ener(xe2, do_, idos=2))
        + qe3 * np.where(we3 == 0.0, 0.0, _batch_ener(xe3, do_, idos=2))
        + qe4 * np.where(we4 == 0.0, 0.0, _batch_ener(xe4, do_, idos=2)))
    Ue = 3.0 * fn * Rgas * Ti_safe * ue

    # ── Optic continuum (idos=4) ──────────────────────────────────────────────
    wo = (wou + wol) / 2.0
    xo = _safe_x(wo, Ti_safe)
    uo = qo * np.where(wo == 0.0, 0.0,
                        _batch_ener(xo, do_, idos=4))
    Uo = 3.0 * fn * Rgas * Ti_safe * uo

    result = Ud + Us + Ue + Uo
    # Mask out Ti<=0 rows
    result = np.where(Ti_arr <= 0.0, 0.0, result)
    return result


def Ctherm_vec(Ti,
               fn, zu,
               wd1, wd2, wd3,
               ws1, ws2, ws3,
               wou, wol,
               we1, we2, we3, we4,
               qe1, qe2, qe3, qe4) -> np.ndarray:
    """Vectorized Ctherm: isochoric heat capacity Cv (J/mol/K).

    Matches Ctherm.f exactly:
        Ctherm = 3*fn*R*(Cd + Cs + Ce + Co)
    (no zero-point term in Cv).
    """
    Ti_arr = np.asarray(Ti, dtype=np.float64)

    su, qo, do_, aniso = _mode_weights(
        fn, zu, wd1, wd2, wd3, ws1, ws2, ws3,
        wou, wol, we1, we2, we3, we4, qe1, qe2, qe3, qe4
    )

    Ti_safe = np.where(Ti_arr <= 0.0, 1.0, Ti_arr)

    # Debye
    x1 = _safe_x(wd1, Ti_safe)
    Cd = (1.0 / su) * np.where(wd1 == 0.0, 0.0,
                                _batch_heat(x1, do_, idos=1))
    if np.any(aniso):
        x2 = _safe_x(wd2, Ti_safe); x3 = _safe_x(wd3, Ti_safe)
        h2 = np.where(wd2 == 0.0, 0.0, _batch_heat(x2, do_, idos=1))
        h3 = np.where(wd3 == 0.0, 0.0, _batch_heat(x3, do_, idos=1))
        Cd = np.where(aniso, Cd / 3.0 + (h2 + h3) / (3.0 * su), Cd)

    # Sin
    xs1 = _safe_x(ws1, Ti_safe)
    Cs = (1.0 / su) * np.where(ws1 == 0.0, 0.0,
                                _batch_heat(xs1, do_, idos=3))
    if np.any(aniso):
        xs2 = _safe_x(ws2, Ti_safe); xs3 = _safe_x(ws3, Ti_safe)
        hs2 = np.where(ws2 == 0.0, 0.0, _batch_heat(xs2, do_, idos=3))
        hs3 = np.where(ws3 == 0.0, 0.0, _batch_heat(xs3, do_, idos=3))
        Cs = np.where(aniso, Cs / 3.0 + (hs2 + hs3) / (3.0 * su), Cs)

    # Einstein
    xe1 = _safe_x(we1, Ti_safe); xe2 = _safe_x(we2, Ti_safe)
    xe3 = _safe_x(we3, Ti_safe); xe4 = _safe_x(we4, Ti_safe)
    Ce = (qe1 * np.where(we1 == 0.0, 0.0, _batch_heat(xe1, do_, idos=2))
        + qe2 * np.where(we2 == 0.0, 0.0, _batch_heat(xe2, do_, idos=2))
        + qe3 * np.where(we3 == 0.0, 0.0, _batch_heat(xe3, do_, idos=2))
        + qe4 * np.where(we4 == 0.0, 0.0, _batch_heat(xe4, do_, idos=2)))

    # Optic
    wo = (wou + wol) / 2.0
    xo = _safe_x(wo, Ti_safe)
    Co = qo * np.where(wo == 0.0, 0.0,
                        _batch_heat(xo, do_, idos=4))

    result = 3.0 * fn * Rgas * (Cd + Cs + Ce + Co)
    result = np.where(Ti_arr <= 0.0, 0.0, result)
    return result


def Ftherm_vec(Ti,
               fn, zu,
               wd1, wd2, wd3,
               ws1, ws2, ws3,
               wou, wol,
               we1, we2, we3, we4,
               qe1, qe2, qe3, qe4) -> np.ndarray:
    """Vectorized Ftherm: vibrational Helmholtz free energy (J/mol).

    Matches Ftherm.f exactly:
        Ftherm = Fd + Fs + Fe + Fo
    where Fd includes the zero-point energy 9/8*fn*R*wd1.
    """
    Ti_arr = np.asarray(Ti, dtype=np.float64)

    su, qo, do_, aniso = _mode_weights(
        fn, zu, wd1, wd2, wd3, ws1, ws2, ws3,
        wou, wol, we1, we2, we3, we4, qe1, qe2, qe3, qe4
    )

    Ti_safe = np.where(Ti_arr <= 0.0, 1.0, Ti_arr)

    # Debye
    x1 = _safe_x(wd1, Ti_safe)
    Fd_norm = (1.0 / su) * np.where(wd1 == 0.0, 0.0,
                                     _batch_helm(x1, do_, idos=1))
    if np.any(aniso):
        x2 = _safe_x(wd2, Ti_safe); x3 = _safe_x(wd3, Ti_safe)
        h2 = np.where(wd2 == 0.0, 0.0, _batch_helm(x2, do_, idos=1))
        h3 = np.where(wd3 == 0.0, 0.0, _batch_helm(x3, do_, idos=1))
        Fd_norm = np.where(aniso,
                           Fd_norm / 3.0 + (h2 + h3) / (3.0 * su),
                           Fd_norm)
    Fd = 3.0 * fn * Rgas * Ti_safe * Fd_norm + (9.0 / 8.0) * fn * Rgas * wd1

    # Sin
    xs1 = _safe_x(ws1, Ti_safe)
    Fs_norm = (1.0 / su) * np.where(ws1 == 0.0, 0.0,
                                     _batch_helm(xs1, do_, idos=3))
    if np.any(aniso):
        xs2 = _safe_x(ws2, Ti_safe); xs3 = _safe_x(ws3, Ti_safe)
        hs2 = np.where(ws2 == 0.0, 0.0, _batch_helm(xs2, do_, idos=3))
        hs3 = np.where(ws3 == 0.0, 0.0, _batch_helm(xs3, do_, idos=3))
        Fs_norm = np.where(aniso,
                           Fs_norm / 3.0 + (hs2 + hs3) / (3.0 * su),
                           Fs_norm)
    Fs = 3.0 * fn * Rgas * Ti_safe * Fs_norm

    # Einstein
    xe1 = _safe_x(we1, Ti_safe); xe2 = _safe_x(we2, Ti_safe)
    xe3 = _safe_x(we3, Ti_safe); xe4 = _safe_x(we4, Ti_safe)
    Fe_norm = (qe1 * np.where(we1 == 0.0, 0.0, _batch_helm(xe1, do_, idos=2))
             + qe2 * np.where(we2 == 0.0, 0.0, _batch_helm(xe2, do_, idos=2))
             + qe3 * np.where(we3 == 0.0, 0.0, _batch_helm(xe3, do_, idos=2))
             + qe4 * np.where(we4 == 0.0, 0.0, _batch_helm(xe4, do_, idos=2)))
    Fe = 3.0 * fn * Rgas * Ti_safe * Fe_norm

    # Optic
    wo = (wou + wol) / 2.0
    xo = _safe_x(wo, Ti_safe)
    Fo_norm = qo * np.where(wo == 0.0, 0.0,
                             _batch_helm(xo, do_, idos=4))
    Fo = 3.0 * fn * Rgas * Ti_safe * Fo_norm

    result = Fd + Fs + Fe + Fo
    result = np.where(Ti_arr <= 0.0, 0.0, result)
    return result


# ---------------------------------------------------------------------------
# Torch-native thermal functions (GPU-compatible)
# ---------------------------------------------------------------------------

def _mode_weights_torch(fn, zu, wd1, wd2, wd3, ws1, ws2, ws3,
                        wou, wol, we1, we2, we3, we4, qe1, qe2, qe3, qe4):
    import torch
    su   = fn * zu
    wo   = (wou + wol) / 2.0
    wdav = (wd1 + wd2 + wd3) / 3.0
    wsav = (ws1 + ws2 + ws3) / 3.0
    no_acoustic = (wdav == 0.0) & (wsav == 0.0)
    su = torch.where(no_acoustic, torch.full_like(su, 1.0e15), su)
    qe = qe1 + qe2 + qe3 + qe4
    qo = 1.0 - 1.0 / su - qe
    qo = torch.where(wo == 0.0, torch.zeros_like(qo), qo)
    do_ = torch.where(wo != 0.0, (wou - wol) / wo.clamp(min=1e-30), torch.zeros_like(wo))
    aniso_deb = wdav != (wd1 / 3.0)
    aniso_sin = wsav != (ws1 / 3.0)
    aniso = aniso_deb | aniso_sin
    no_ein_opt = (qe == 0.0) & (qo == 0.0)
    su = torch.where(no_ein_opt, torch.ones_like(su), su)
    qo_zero = qo == 0.0
    su_redist = torch.where(qe >= 1.0, torch.full_like(su, 1.0e15), 1.0 / (1.0 - qe).clamp(min=1e-30))
    su = torch.where(qo_zero, su_redist, su)
    return su, qo, do_, aniso


def _sx(w, T):
    """Safe x = w/T, zero where w==0."""
    return torch.where(w == 0.0, torch.zeros_like(w), w / T)


def _bener(w_over_T, do_, idos):
    shape = w_over_T.shape
    x_f   = w_over_T.reshape(-1)
    d_f   = do_.reshape(-1) if idos == 4 else 0.0
    return Ener_torch(x_f, idos=idos, d=d_f).reshape(shape)


def _bheat(w_over_T, do_, idos):
    shape = w_over_T.shape
    x_f   = w_over_T.reshape(-1)
    d_f   = do_.reshape(-1) if idos == 4 else 0.0
    return Heat_torch(x_f, idos=idos, d=d_f).reshape(shape)


def _bhelm(w_over_T, do_, idos):
    shape = w_over_T.shape
    x_f   = w_over_T.reshape(-1)
    d_f   = do_.reshape(-1) if idos == 4 else 0.0
    return Helm_torch(x_f, idos=idos, d=d_f).reshape(shape)


def _has_aniso_from_modes(wd2, wd3, ws2, ws3) -> bool:
    """Return True if any species has anisotropic modes (one GPU→CPU sync)."""
    return bool((wd2.abs() + wd3.abs() + ws2.abs() + ws3.abs()).max().item() > 0)


def Etherm_torch(Ti, fn, zu, wd1, wd2, wd3, ws1, ws2, ws3,
                 wou, wol, we1, we2, we3, we4, qe1, qe2, qe3, qe4,
                 has_aniso: 'bool | None' = None):
    """Torch Etherm: vibrational internal energy (J/mol).

    has_aniso : pre-computed Python bool to avoid GPU→CPU sync on every call.
        Pass False when wd2=wd3=ws2=ws3=0 for all active species (typical).
        None (default) falls back to a single .any() check inside the call.
    """
    import torch
    Ti_t    = Ti if torch.is_tensor(Ti) else torch.as_tensor(Ti, dtype=torch.float64, device=fn.device)
    Ti_safe = torch.where(Ti_t <= 0.0, torch.ones_like(Ti_t), Ti_t)
    su, qo, do_, aniso = _mode_weights_torch(fn, zu, wd1, wd2, wd3, ws1, ws2, ws3,
                                              wou, wol, we1, we2, we3, we4,
                                              qe1, qe2, qe3, qe4)
    # Resolve has_aniso once (Python bool → no further GPU→CPU syncs in the loop)
    _ha = bool(aniso.any()) if has_aniso is None else has_aniso

    x1 = _sx(wd1, Ti_safe)
    ud = (1.0 / su) * torch.where(wd1 == 0.0, torch.zeros_like(x1), _bener(x1, do_, 1))
    if _ha:
        x2 = _sx(wd2, Ti_safe);  x3 = _sx(wd3, Ti_safe)
        e2 = torch.where(wd2 == 0.0, torch.zeros_like(x2), _bener(x2, do_, 1))
        e3 = torch.where(wd3 == 0.0, torch.zeros_like(x3), _bener(x3, do_, 1))
        ud = torch.where(aniso, ud / 3.0 + (e2 + e3) / (3.0 * su), ud)
    Ud = 3.0 * fn * Rgas * Ti_safe * ud + (9.0 / 8.0) * fn * Rgas * wd1

    xs1 = _sx(ws1, Ti_safe)
    us = (1.0 / su) * torch.where(ws1 == 0.0, torch.zeros_like(xs1), _bener(xs1, do_, 3))
    if _ha:
        xs2 = _sx(ws2, Ti_safe);  xs3 = _sx(ws3, Ti_safe)
        s2  = torch.where(ws2 == 0.0, torch.zeros_like(xs2), _bener(xs2, do_, 3))
        s3  = torch.where(ws3 == 0.0, torch.zeros_like(xs3), _bener(xs3, do_, 3))
        us  = torch.where(aniso, us / 3.0 + (s2 + s3) / (3.0 * su), us)
    Us = 3.0 * fn * Rgas * Ti_safe * us

    xe1 = _sx(we1, Ti_safe);  xe2 = _sx(we2, Ti_safe)
    xe3 = _sx(we3, Ti_safe);  xe4 = _sx(we4, Ti_safe)
    ue  = (qe1 * torch.where(we1 == 0.0, torch.zeros_like(xe1), _bener(xe1, do_, 2))
         + qe2 * torch.where(we2 == 0.0, torch.zeros_like(xe2), _bener(xe2, do_, 2))
         + qe3 * torch.where(we3 == 0.0, torch.zeros_like(xe3), _bener(xe3, do_, 2))
         + qe4 * torch.where(we4 == 0.0, torch.zeros_like(xe4), _bener(xe4, do_, 2)))
    Ue = 3.0 * fn * Rgas * Ti_safe * ue

    wo  = (wou + wol) / 2.0
    xo  = _sx(wo, Ti_safe)
    uo  = qo * torch.where(wo == 0.0, torch.zeros_like(xo), _bener(xo, do_, 4))
    Uo  = 3.0 * fn * Rgas * Ti_safe * uo

    return torch.where(Ti_t <= 0.0, torch.zeros_like(Ud), Ud + Us + Ue + Uo)


def Ctherm_torch(Ti, fn, zu, wd1, wd2, wd3, ws1, ws2, ws3,
                 wou, wol, we1, we2, we3, we4, qe1, qe2, qe3, qe4,
                 has_aniso: 'bool | None' = None):
    """Torch Ctherm: isochoric heat capacity (J/mol/K).

    has_aniso : see Etherm_torch.
    """
    import torch
    Ti_t    = Ti if torch.is_tensor(Ti) else torch.as_tensor(Ti, dtype=torch.float64, device=fn.device)
    Ti_safe = torch.where(Ti_t <= 0.0, torch.ones_like(Ti_t), Ti_t)
    su, qo, do_, aniso = _mode_weights_torch(fn, zu, wd1, wd2, wd3, ws1, ws2, ws3,
                                              wou, wol, we1, we2, we3, we4,
                                              qe1, qe2, qe3, qe4)
    _ha = bool(aniso.any()) if has_aniso is None else has_aniso

    x1  = _sx(wd1, Ti_safe)
    Cd  = (1.0 / su) * torch.where(wd1 == 0.0, torch.zeros_like(x1), _bheat(x1, do_, 1))
    if _ha:
        x2 = _sx(wd2, Ti_safe);  x3 = _sx(wd3, Ti_safe)
        h2 = torch.where(wd2 == 0.0, torch.zeros_like(x2), _bheat(x2, do_, 1))
        h3 = torch.where(wd3 == 0.0, torch.zeros_like(x3), _bheat(x3, do_, 1))
        Cd = torch.where(aniso, Cd / 3.0 + (h2 + h3) / (3.0 * su), Cd)

    xs1 = _sx(ws1, Ti_safe)
    Cs  = (1.0 / su) * torch.where(ws1 == 0.0, torch.zeros_like(xs1), _bheat(xs1, do_, 3))
    if _ha:
        xs2 = _sx(ws2, Ti_safe);  xs3 = _sx(ws3, Ti_safe)
        hs2 = torch.where(ws2 == 0.0, torch.zeros_like(xs2), _bheat(xs2, do_, 3))
        hs3 = torch.where(ws3 == 0.0, torch.zeros_like(xs3), _bheat(xs3, do_, 3))
        Cs  = torch.where(aniso, Cs / 3.0 + (hs2 + hs3) / (3.0 * su), Cs)

    xe1 = _sx(we1, Ti_safe);  xe2 = _sx(we2, Ti_safe)
    xe3 = _sx(we3, Ti_safe);  xe4 = _sx(we4, Ti_safe)
    Ce  = (qe1 * torch.where(we1 == 0.0, torch.zeros_like(xe1), _bheat(xe1, do_, 2))
         + qe2 * torch.where(we2 == 0.0, torch.zeros_like(xe2), _bheat(xe2, do_, 2))
         + qe3 * torch.where(we3 == 0.0, torch.zeros_like(xe3), _bheat(xe3, do_, 2))
         + qe4 * torch.where(we4 == 0.0, torch.zeros_like(xe4), _bheat(xe4, do_, 2)))

    wo  = (wou + wol) / 2.0
    xo  = _sx(wo, Ti_safe)
    Co  = qo * torch.where(wo == 0.0, torch.zeros_like(xo), _bheat(xo, do_, 4))

    result = 3.0 * fn * Rgas * (Cd + Cs + Ce + Co)
    return torch.where(Ti_t <= 0.0, torch.zeros_like(result), result)


def Ftherm_torch(Ti, fn, zu, wd1, wd2, wd3, ws1, ws2, ws3,
                 wou, wol, we1, we2, we3, we4, qe1, qe2, qe3, qe4,
                 has_aniso: 'bool | None' = None):
    """Torch Ftherm: vibrational Helmholtz free energy (J/mol).

    has_aniso : see Etherm_torch.
    """
    import torch
    Ti_t    = Ti if torch.is_tensor(Ti) else torch.as_tensor(Ti, dtype=torch.float64, device=fn.device)
    Ti_safe = torch.where(Ti_t <= 0.0, torch.ones_like(Ti_t), Ti_t)
    su, qo, do_, aniso = _mode_weights_torch(fn, zu, wd1, wd2, wd3, ws1, ws2, ws3,
                                              wou, wol, we1, we2, we3, we4,
                                              qe1, qe2, qe3, qe4)
    _ha = bool(aniso.any()) if has_aniso is None else has_aniso

    x1   = _sx(wd1, Ti_safe)
    Fd_n = (1.0 / su) * torch.where(wd1 == 0.0, torch.zeros_like(x1), _bhelm(x1, do_, 1))
    if _ha:
        x2 = _sx(wd2, Ti_safe);  x3 = _sx(wd3, Ti_safe)
        h2 = torch.where(wd2 == 0.0, torch.zeros_like(x2), _bhelm(x2, do_, 1))
        h3 = torch.where(wd3 == 0.0, torch.zeros_like(x3), _bhelm(x3, do_, 1))
        Fd_n = torch.where(aniso, Fd_n / 3.0 + (h2 + h3) / (3.0 * su), Fd_n)
    Fd = 3.0 * fn * Rgas * Ti_safe * Fd_n + (9.0 / 8.0) * fn * Rgas * wd1

    xs1  = _sx(ws1, Ti_safe)
    Fs_n = (1.0 / su) * torch.where(ws1 == 0.0, torch.zeros_like(xs1), _bhelm(xs1, do_, 3))
    if _ha:
        xs2 = _sx(ws2, Ti_safe);  xs3 = _sx(ws3, Ti_safe)
        hs2 = torch.where(ws2 == 0.0, torch.zeros_like(xs2), _bhelm(xs2, do_, 3))
        hs3 = torch.where(ws3 == 0.0, torch.zeros_like(xs3), _bhelm(xs3, do_, 3))
        Fs_n = torch.where(aniso, Fs_n / 3.0 + (hs2 + hs3) / (3.0 * su), Fs_n)
    Fs = 3.0 * fn * Rgas * Ti_safe * Fs_n

    xe1 = _sx(we1, Ti_safe);  xe2 = _sx(we2, Ti_safe)
    xe3 = _sx(we3, Ti_safe);  xe4 = _sx(we4, Ti_safe)
    Fe_n = (qe1 * torch.where(we1 == 0.0, torch.zeros_like(xe1), _bhelm(xe1, do_, 2))
          + qe2 * torch.where(we2 == 0.0, torch.zeros_like(xe2), _bhelm(xe2, do_, 2))
          + qe3 * torch.where(we3 == 0.0, torch.zeros_like(xe3), _bhelm(xe3, do_, 2))
          + qe4 * torch.where(we4 == 0.0, torch.zeros_like(xe4), _bhelm(xe4, do_, 2)))
    Fe = 3.0 * fn * Rgas * Ti_safe * Fe_n

    wo   = (wou + wol) / 2.0
    xo   = _sx(wo, Ti_safe)
    Fo_n = qo * torch.where(wo == 0.0, torch.zeros_like(xo), _bhelm(xo, do_, 4))
    Fo   = 3.0 * fn * Rgas * Ti_safe * Fo_n

    result = Fd + Fs + Fe + Fo
    return torch.where(Ti_t <= 0.0, torch.zeros_like(result), result)
