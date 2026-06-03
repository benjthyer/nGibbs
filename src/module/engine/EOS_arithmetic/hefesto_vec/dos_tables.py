"""
DOS table management: load dos.inc tables, apply dossetup corrections, and
provide vectorized Heat / Ener / Helm lookup functions matching HeFESTo.

Background
----------
HeFESTo stores 5000-point tabulated functions in dos.inc covering four
vibrational DOS shapes:
  idos=1  Debye acoustic      (deb1, deb2, deb3)
  idos=2  Einstein            (ein1, ein2, ein3)
  idos=3  Sinusoidal acoustic (sin1, sin2, sin3)
  idos=4  Optic continuum     (opt1, opt2, opt3)

Suffix meaning: 1=Heat (Cv/3NkB), 2=Ener (U/3NkBT), 3=Helm (F/3NkBT, log-reduced)

dossetup.f subtracts log(1-exp(-x)) from the *3 (Helmholtz) arrays before
interpolation; Helm.f adds it back after.  We reproduce this exactly.

Interpolation: 4-point Lagrange (Neville) matching the Fortran neville() call.
Extrapolation: exact analytical formulas from Heat.f / Ener.f / Helm.f.

Performance notes
-----------------
The abscissa xa is perfectly uniform (xa[0]=0.01, h=0.01, N=5000).
_find_klo uses O(1) floor arithmetic instead of binary search (7x faster).
_neville4_vec uses the uniform-grid Lagrange formula, avoiding fancy xa
indexing and Python loops (2.4x faster).
_interp_two interpolates two tables in a single pass (same klo and basis).
"""

import math
import os
import numpy as np

from .constants import pirad

# Module-level cache
_tables = None


def _ensure_loaded(npz_path=None):
    global _tables
    if _tables is not None:
        return

    if npz_path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        npz_path = os.path.normpath(os.path.join(here, '..', 'dos_tables.npz'))

    raw = np.load(npz_path)
    xa   = raw['xad'].astype(np.float64)
    xamin = xa[0]
    xamax = xa[-1]

    deb1 = raw['deb1d'].astype(np.float64)
    deb2 = raw['deb2d'].astype(np.float64)
    deb3 = raw['deb3d'].astype(np.float64)
    ein1 = raw['ein1d'].astype(np.float64)
    ein2 = raw['ein2d'].astype(np.float64)
    ein3 = raw['ein3d'].astype(np.float64)
    sin1 = raw['sin1d'].astype(np.float64)
    sin2 = raw['sin2d'].astype(np.float64)
    sin3 = raw['sin3d'].astype(np.float64)
    opt1 = raw['opt1d'].astype(np.float64)
    opt2 = raw['opt2d'].astype(np.float64)
    opt3 = raw['opt3d'].astype(np.float64)

    # dossetup correction: subtract log(1-exp(-x)) from *3 arrays
    log_term = np.log(1.0 - np.exp(-xa))
    deb3 = deb3 - log_term
    ein3 = ein3 - log_term
    sin3 = sin3 - log_term
    opt3 = opt3 - log_term

    _tables = dict(
        xa=xa, xamin=xamin, xamax=xamax,
        xa0=xa[0], h=xa[1]-xa[0],
        deb1=deb1, deb2=deb2, deb3=deb3,
        ein1=ein1, ein2=ein2, ein3=ein3,
        sin1=sin1, sin2=sin2, sin3=sin3,
        opt1=opt1, opt2=opt2, opt3=opt3,
        deb1_min=deb1[0], deb2_min=deb2[0],
        ein1_min=ein1[0], ein2_min=ein2[0],
        opt1_min=opt1[0], opt2_min=opt2[0],
        sin1_min=sin1[0], sin2_min=sin2[0],
    )


# ---------------------------------------------------------------------------
# Uniform-grid helpers
# ---------------------------------------------------------------------------

def _find_klo(xa, x, xamin, xamax):
    """4-point window start index using O(1) arithmetic (uniform grid)."""
    na  = len(xa)
    h   = xa[1] - xa[0]
    xa0 = xa[0]
    jlo = np.floor((x - xa0) / h).astype(np.intp)
    jlo = np.clip(jlo, 0, na - 1)
    klo = np.clip(jlo - 1, 0, na - 4)
    return klo


def _neville4_vec(xa, ya, x, klo):
    """4-point Lagrange on uniform grid — no xa fancy-index, no Python loops."""
    h   = xa[1] - xa[0]
    xa0 = xa[0]
    idx = klo[:, None] + np.arange(4, dtype=np.intp)[None, :]  # (M,4)
    ya4 = ya[idx]
    u  = (x - (xa0 + klo * h)) / h
    u1 = u - 1.0; u2 = u - 2.0; u3 = u - 3.0
    l0 = -u1 * u2 * u3 / 6.0
    l1 =  u  * u2 * u3 / 2.0
    l2 = -u  * u1 * u3 / 2.0
    l3 =  u  * u1 * u2 / 6.0
    return ya4[:, 0]*l0 + ya4[:, 1]*l1 + ya4[:, 2]*l2 + ya4[:, 3]*l3


def _interp_two(xa, ya, yb, x, klo):
    """Interpolate two tables at the same x/klo — single basis computation."""
    h   = xa[1] - xa[0]
    xa0 = xa[0]
    idx = klo[:, None] + np.arange(4, dtype=np.intp)[None, :]
    ya4 = ya[idx]; yb4 = yb[idx]
    u  = (x - (xa0 + klo * h)) / h
    u1 = u - 1.0; u2 = u - 2.0; u3 = u - 3.0
    l0 = -u1 * u2 * u3 / 6.0
    l1 =  u  * u2 * u3 / 2.0
    l2 = -u  * u1 * u3 / 2.0
    l3 =  u  * u1 * u2 / 6.0
    va = ya4[:, 0]*l0 + ya4[:, 1]*l1 + ya4[:, 2]*l2 + ya4[:, 3]*l3
    vb = yb4[:, 0]*l0 + yb4[:, 1]*l1 + yb4[:, 2]*l2 + yb4[:, 3]*l3
    return va, vb


# Sinusoidal empirical constants (Heat.f / Ener.f / Helm.f)
_ASIN = 23.594
_BSIN = 6.1
_CSIN = 0.188256
_SFAC = (2.0 / math.pi) ** 3


# ---------------------------------------------------------------------------
# Public vectorised functions
# ---------------------------------------------------------------------------

def Heat_vec(x, idos, d=0.0, npz_path=None):
    """Cv/(3NkB) — vectorized.  idos: 1=Debye, 2=Einstein, 3=Sin, 4=Optic."""
    _ensure_loaded(npz_path)
    t = _tables
    xa, xamin, xamax = t['xa'], t['xamin'], t['xamax']
    x = np.asarray(x, dtype=np.float64)
    scalar_input = x.ndim == 0
    x = np.atleast_1d(x)
    out = np.zeros(len(x), dtype=np.float64)

    if idos == 5:
        out[:] = 1.0
        return out.item() if scalar_input else out

    if idos == 4:
        d = np.broadcast_to(np.asarray(d, dtype=np.float64), x.shape).copy()
        dmin = 0.01
        narrow = d < dmin; wide = ~narrow
        if np.any(narrow):
            out[narrow] = Heat_vec(x[narrow], idos=2, d=0.0)
        if np.any(wide):
            xn = x[wide]; dn = d[wide]
            xu = xn*(1.0+dn/2.0); xl = xn*(1.0-dn/2.0)
            yu = Heat_vec(xu, idos=2); yl = Heat_vec(xl, idos=2)
            below_u = xu < xamin; below_l = xl < xamin
            above_u = xu > xamax; above_l = xl > xamax
            yu[below_u] = 1.0-(1.0-t['opt1_min'])/xamin**2*xu[below_u]**2
            yl[below_l] = 1.0-(1.0-t['opt1_min'])/xamin**2*xl[below_l]**2
            yu[above_u] = (math.pi**2/3.0)/xu[above_u]
            yl[above_l] = (math.pi**2/3.0)/xl[above_l]
            out[wide] = np.maximum((xu*yu - xl*yl)/(xu - xl), 0.0)
        return out.item() if scalar_input else out

    if   idos == 1: ya = t['deb1']; y_min = t['deb1_min']
    elif idos == 2: ya = t['ein1']; y_min = t['ein1_min']
    else:           ya = t['sin1']; y_min = t['sin1_min']

    interior = (x >= xamin) & (x <= xamax)
    below    = x < xamin
    above    = x > xamax

    if np.any(interior):
        xi = x[interior]
        klo = _find_klo(xa, xi, xamin, xamax)
        out[interior] = _neville4_vec(xa, ya, xi, klo)
    if np.any(above):
        xa_ = x[above]
        if   idos == 1: out[above] = (12.0*math.pi**4/15.0)/xa_**3
        elif idos == 2: out[above] = xa_**2 * np.exp(-xa_)
        else:           out[above] = (12.0*math.pi**4/15.0)/xa_**3*_SFAC + _ASIN*_BSIN*(_BSIN-1.0)*xa_**(1.0-_BSIN)
    if np.any(below):
        xb = x[below]
        out[below] = 1.0 - (1.0 - y_min)/xamin**2 * xb**2

    return out.item() if scalar_input else out


def Ener_vec(x, idos, d=0.0, npz_path=None):
    """U/(3NkBT) — vectorized.  idos: 1=Debye, 2=Einstein, 3=Sin, 4=Optic."""
    _ensure_loaded(npz_path)
    t = _tables
    xa, xamin, xamax = t['xa'], t['xamin'], t['xamax']
    x = np.asarray(x, dtype=np.float64)
    scalar_input = x.ndim == 0
    x = np.atleast_1d(x)
    out = np.zeros(len(x), dtype=np.float64)

    if idos == 5:
        out[:] = 1.0
        return out.item() if scalar_input else out

    if idos == 4:
        d = np.broadcast_to(np.asarray(d, dtype=np.float64), x.shape).copy()
        dmin = 0.01
        narrow = d < dmin; wide = ~narrow
        if np.any(narrow):
            out[narrow] = Ener_vec(x[narrow], idos=2)
        if np.any(wide):
            xn = x[wide]; dn = d[wide]
            xu = xn*(1.0+dn/2.0); xl = xn*(1.0-dn/2.0)
            yu = Ener_vec(xu, idos=2); yl = Ener_vec(xl, idos=2)
            below_u = xu < xamin; below_l = xl < xamin
            above_u = xu > xamax; above_l = xl > xamax
            y2_min = t['opt2_min']
            yu[below_u] = 1.0-(1.0-y2_min)/xamin*xu[below_u]
            yl[below_l] = 1.0-(1.0-y2_min)/xamin*xl[below_l]
            yu[above_u] = (math.pi**2/6.0)/xu[above_u]
            yl[above_l] = (math.pi**2/6.0)/xl[above_l]
            out[wide] = np.maximum((xu*yu - xl*yl)/(xu - xl), 0.0)
        return out.item() if scalar_input else out

    if   idos == 1: ya = t['deb2']; y_min = t['deb2_min']
    elif idos == 2: ya = t['ein2']; y_min = t['ein2_min']
    else:           ya = t['sin2']; y_min = t['sin2_min']

    interior = (x >= xamin) & (x <= xamax)
    below    = x < xamin
    above    = x > xamax

    if np.any(interior):
        xi = x[interior]
        klo = _find_klo(xa, xi, xamin, xamax)
        out[interior] = _neville4_vec(xa, ya, xi, klo)
    if np.any(above):
        xa_ = x[above]
        if   idos == 1: out[above] = (math.pi**4/5.0)/xa_**3
        elif idos == 2: out[above] = xa_ * np.exp(-xa_)
        else:           out[above] = (math.pi**4/5.0)/xa_**3*_SFAC + _ASIN*(_BSIN-1.0)*xa_**(1.0-_BSIN)
    if np.any(below):
        xb = x[below]
        if idos == 2:
            out[below] = xb / np.expm1(xb)
        else:
            out[below] = 1.0 - (1.0 - y_min)/xamin * xb

    return out.item() if scalar_input else out


def Helm_vec(x, idos, d=0.0, npz_path=None):
    """F/(3NkBT) — vectorized.  idos: 1=Debye, 2=Einstein, 3=Sin, 4=Optic."""
    _ensure_loaded(npz_path)
    t = _tables
    xa, xamin, xamax = t['xa'], t['xamin'], t['xamax']
    x = np.asarray(x, dtype=np.float64)
    scalar_input = x.ndim == 0
    x = np.atleast_1d(x)
    out = np.zeros(len(x), dtype=np.float64)

    if idos == 5:
        out = np.log(x) - 1.0/3.0
        return out.item() if scalar_input else out

    if idos == 2:
        out = np.log(1.0 - np.exp(-x))
        above = x > xamax
        out[above] = -np.exp(-x[above])
        return out.item() if scalar_input else out

    if idos == 4:
        d = np.broadcast_to(np.asarray(d, dtype=np.float64), x.shape).copy()
        dmin = 0.4
        narrow = d <= dmin; wide = ~narrow

        if np.any(narrow):
            xn = x[narrow]; dn = d[narrow]
            dx = xn*dn/2.0
            u  = np.exp(-xn); v = u/(1.0-u)
            fein = np.log(1.0-u)
            c2 = -2.0*v - 2.0*v**2
            c4 = -2.0*v - 14.0*v**2 - 24.0*v**3 - 12.0*v**4
            out[narrow] = fein + c2*dx**2/(2.0*6.0) + c4*dx**4/(2.0*120.0)

        if np.any(wide):
            xw = x[wide]; dw = d[wide]
            xu = xw*(1.0+dw/2.0); xl = xw*(1.0-dw/2.0)
            yu = _helm_table_lookup(xa, t['opt3'], xu, xamin, xamax, idos=4)
            yl = _helm_table_lookup(xa, t['opt3'], xl, xamin, xamax, idos=4)
            mask_u_int = (xu >= xamin) & (xu <= xamax)
            mask_l_int = (xl >= xamin) & (xl <= xamax)
            yu[mask_u_int] += np.log(1.0 - np.exp(-xu[mask_u_int]))
            yl[mask_l_int] += np.log(1.0 - np.exp(-xl[mask_l_int]))
            below_u = xu < xamin; below_l = xl < xamin
            above_u = xu > xamax; above_l = xl > xamax
            yu[below_u] = (np.log(1.0 - np.exp(-xu[below_u]))
                           - (1.0 - (1.0-t['opt2_min'])/xamin * xu[below_u]))
            yl[below_l] = (np.log(1.0 - np.exp(-xl[below_l]))
                           - (1.0 - (1.0-t['opt2_min'])/xamin * xl[below_l]))
            yu[above_u] = -(math.pi**2/6.0)/xu[above_u]
            yl[above_l] = -(math.pi**2/6.0)/xl[above_l]
            zero_u = xu == 0.0; zero_l = xl == 0.0
            yu[zero_u] = 0.0; yl[zero_l] = 0.0
            out[wide] = (xu*yu - xl*yl)/(xu - xl)

        return out.item() if scalar_input else out

    # idos 1 or 3: Debye or Sin
    ya    = t['deb3'] if idos == 1 else t['sin3']
    y2min = t['deb2_min'] if idos == 1 else t['sin2_min']

    interior = (x >= xamin) & (x <= xamax)
    below    = x < xamin
    above    = x > xamax

    if np.any(interior):
        xi = x[interior]
        klo = _find_klo(xa, xi, xamin, xamax)
        out[interior] = _neville4_vec(xa, ya, xi, klo) + np.log(1.0 - np.exp(-xi))
    if np.any(above):
        xa_ = x[above]
        if idos == 1:
            out[above] = -(math.pi**4/15.0)/xa_**3
        else:
            out[above] = -(math.pi**4/15.0)/xa_**3*_SFAC - _ASIN*xa_**(1.0-_BSIN)
    if np.any(below):
        xb = x[below]
        if idos == 1:
            out[below] = np.log(xb) - 4.0/3.0 + 1.0 - (1.0-t['deb2_min'])/xamin*xb
        else:
            out[below] = np.log(xb) - 4.0/3.0 + 1.0 - (1.0-t['sin2_min'])/xamin*xb + _CSIN

    return out.item() if scalar_input else out


def _helm_table_lookup(xa, ya, x, xamin, xamax, idos):
    """Raw interpolated opt3 values (log term not added back)."""
    out = np.zeros(len(x), dtype=np.float64)
    interior = (x >= xamin) & (x <= xamax)
    if np.any(interior):
        xi = x[interior]
        klo = _find_klo(xa, xi, xamin, xamax)
        out[interior] = _neville4_vec(xa, ya, xi, klo)
    return out


def load_tables(npz_path=None):
    """Pre-load DOS tables (optional; loaded automatically on first use)."""
    _ensure_loaded(npz_path)


# ---------------------------------------------------------------------------
# Torch-native DOS table functions (GPU-compatible)
# ---------------------------------------------------------------------------

_tables_torch: dict = {}   # str(device) -> dict of torch tensors


def _ensure_loaded_torch(device) -> dict:
    import torch
    key = str(device)
    if key in _tables_torch:
        return _tables_torch[key]
    _ensure_loaded()
    t = _tables
    tt = {}
    for k, v in t.items():
        if isinstance(v, np.ndarray):
            tt[k] = torch.tensor(v, dtype=torch.float64, device=device)
        else:
            tt[k] = float(v)
    _tables_torch[key] = tt
    return tt


def _find_klo_torch(xa: 'torch.Tensor', x: 'torch.Tensor', n: int) -> 'torch.Tensor':
    h   = xa[1] - xa[0]
    xa0 = xa[0]
    jlo = ((x - xa0) / h).floor().long().clamp(0, n - 1)   # no bare torch.* needed
    return (jlo - 1).clamp(0, n - 4)


def _neville4_torch(xa: 'torch.Tensor', ya: 'torch.Tensor',
                    x: 'torch.Tensor', klo: 'torch.Tensor') -> 'torch.Tensor':
    h   = xa[1] - xa[0]
    xa0 = xa[0]
    offsets = klo.new_tensor([0, 1, 2, 3])     # same device & dtype (long) as klo
    idx  = klo.unsqueeze(1) + offsets          # (M, 4)
    ya4  = ya[idx]                              # (M, 4)
    u    = (x - (xa0 + klo.to(dtype=x.dtype) * h)) / h
    u1   = u - 1.0;  u2 = u - 2.0;  u3 = u - 3.0
    l0   = -u1 * u2 * u3 / 6.0
    l1   =  u  * u2 * u3 / 2.0
    l2   = -u  * u1 * u3 / 2.0
    l3   =  u  * u1 * u2 / 6.0
    return ya4[:, 0]*l0 + ya4[:, 1]*l1 + ya4[:, 2]*l2 + ya4[:, 3]*l3


def _interp_torch(xa: 'torch.Tensor', ya: 'torch.Tensor',
                  x: 'torch.Tensor') -> 'torch.Tensor':
    """Interpolate ya on uniform grid xa at positions x (clamped internally)."""
    n     = xa.shape[0]
    x_cl  = x.clamp(xa[0].item(), xa[-1].item())
    klo   = _find_klo_torch(xa, x_cl, n)
    return _neville4_torch(xa, ya, x_cl, klo)


def Heat_torch(x: 'torch.Tensor', idos: int, d=0.0) -> 'torch.Tensor':
    """Cv/(3NkB) — torch vectorised. idos: 1=Debye 2=Einstein 3=Sin 4=Optic 5=Classic."""
    import torch

    if idos == 5:
        return torch.ones_like(x)

    # ── Einstein: exact closed-form, no table needed ──────────────────────────
    # C_Ein/3NkB = (x/2 / sinh(x/2))^2   [numerically stable for all x ≥ 0]
    if idos == 2:
        hx = (x / 2.0).clamp(min=1e-15)
        return (hx / torch.sinh(hx)) ** 2

    # ── Optic: finite-diff average of Einstein values at x*(1 ± d/2) ─────────
    # Both Einstein evaluations are now purely arithmetic — no table reads.
    if idos == 4:
        d_t  = torch.as_tensor(d, dtype=x.dtype, device=x.device).expand_as(x)
        xu   = x * (1.0 + d_t / 2.0);  xl = x * (1.0 - d_t / 2.0)
        hxu  = (xu / 2.0).clamp(min=1e-15)
        hxl  = (xl / 2.0).clamp(min=1e-15)
        yu   = (hxu / torch.sinh(hxu)) ** 2
        yl   = (hxl / torch.sinh(hxl)) ** 2
        den  = (xu - xl).abs().clamp(min=1e-30) * torch.sign(xu - xl + 1e-60)
        return ((xu * yu - xl * yl) / den).clamp(min=0.0)

    # ── Debye (idos=1) and Sin (idos=3): table lookup ────────────────────────
    tt    = _ensure_loaded_torch(x.device)
    xa    = tt['xa'];  xamin = tt['xamin'];  xamax = tt['xamax']
    ya    = tt['deb1'] if idos == 1 else tt['sin1']
    y_min = tt['deb1_min'] if idos == 1 else tt['sin1_min']

    ri    = _interp_torch(xa, ya, x)
    xs    = x.clamp(min=1e-30)
    if idos == 1:
        ra = (12.0 * math.pi**4 / 15.0) / xs**3
    else:
        ra = (12.0*math.pi**4/15.0)/xs**3 * _SFAC + _ASIN*_BSIN*(_BSIN-1.0)*xs**(1.0-_BSIN)
    rb = 1.0 - (1.0 - y_min) / xamin**2 * x**2
    return torch.where((x >= xamin) & (x <= xamax), ri,
                       torch.where(x > xamax, ra, rb))


def Ener_torch(x: 'torch.Tensor', idos: int, d=0.0) -> 'torch.Tensor':
    """U/(3NkBT) — torch vectorised."""
    import torch

    if idos == 5:
        return torch.ones_like(x)

    # ── Einstein: exact closed-form  U/3NkBT = x / (e^x - 1) ────────────────
    if idos == 2:
        xs = x.clamp(min=1e-15)
        return xs / torch.expm1(xs)

    # ── Optic: finite-diff average of Einstein at x*(1 ± d/2) ───────────────
    if idos == 4:
        d_t  = torch.as_tensor(d, dtype=x.dtype, device=x.device).expand_as(x)
        xu   = x * (1.0 + d_t / 2.0);  xl = x * (1.0 - d_t / 2.0)
        xu_s = xu.clamp(min=1e-15);    xl_s = xl.clamp(min=1e-15)
        yu   = xu_s / torch.expm1(xu_s)
        yl   = xl_s / torch.expm1(xl_s)
        den  = (xu - xl).abs().clamp(min=1e-30) * torch.sign(xu - xl + 1e-60)
        return ((xu * yu - xl * yl) / den).clamp(min=0.0)

    # ── Debye (idos=1) and Sin (idos=3): table lookup ────────────────────────
    tt    = _ensure_loaded_torch(x.device)
    xa    = tt['xa'];  xamin = tt['xamin'];  xamax = tt['xamax']
    ya    = tt['deb2'] if idos == 1 else tt['sin2']
    y_min = tt['deb2_min'] if idos == 1 else tt['sin2_min']

    ri    = _interp_torch(xa, ya, x)
    xs    = x.clamp(min=1e-30)
    if idos == 1:
        ra = (math.pi**4 / 5.0) / xs**3
    else:
        ra = (math.pi**4/5.0)/xs**3 * _SFAC + _ASIN*(_BSIN-1.0)*xs**(1.0-_BSIN)
    rb = 1.0 - (1.0 - y_min) / xamin * x
    return torch.where((x >= xamin) & (x <= xamax), ri,
                       torch.where(x > xamax, ra, rb))


def Helm_torch(x: 'torch.Tensor', idos: int, d=0.0) -> 'torch.Tensor':
    """F/(3NkBT) — torch vectorised."""
    import torch
    tt   = _ensure_loaded_torch(x.device)
    xa   = tt['xa'];  xamin = tt['xamin'];  xamax = tt['xamax']

    if idos == 5:
        return torch.log(x.clamp(min=1e-30)) - 1.0 / 3.0

    if idos == 2:
        xs = x.clamp(min=1e-30)
        r  = torch.log1p(-torch.exp(-xs))
        return torch.where(x > xamax, -torch.exp(-xs), r)

    if idos == 4:
        d_t   = torch.as_tensor(d, dtype=x.dtype, device=x.device).expand_as(x)
        narrow = d_t <= 0.4

        # narrow branch: Taylor expansion
        xn = x.clamp(min=1e-30)
        u  = torch.exp(-xn.clamp(max=30.0))
        u  = u.clamp(max=1.0 - 1e-10)
        v  = u / (1.0 - u)
        fein = torch.log1p(-u)
        dx   = xn * d_t / 2.0
        c2   = -2.0*v - 2.0*v**2
        c4   = -2.0*v - 14.0*v**2 - 24.0*v**3 - 12.0*v**4
        r_narrow = fein + c2*dx**2/12.0 + c4*dx**4/240.0

        # wide branch
        xu   = x * (1.0 + d_t / 2.0);  xl = x * (1.0 - d_t / 2.0)
        ya_o = tt['opt3']
        yu_r = _interp_torch(xa, ya_o, xu)
        yl_r = _interp_torch(xa, ya_o, xl)
        xu_s = xu.clamp(min=1e-30);    xl_s = xl.clamp(min=1e-30)
        om2  = float(tt['opt2_min'])
        # add log(1-exp(-x)) where interior
        log_xu = torch.where((xu >= xamin) & (xu <= xamax),
                             torch.log1p(-torch.exp(-xu_s)), torch.zeros_like(xu))
        log_xl = torch.where((xl >= xamin) & (xl <= xamax),
                             torch.log1p(-torch.exp(-xl_s)), torch.zeros_like(xl))
        yu = yu_r + log_xu
        yl = yl_r + log_xl
        yu = torch.where(xu < xamin,
                         torch.log1p(-torch.exp(-xu_s)) - (1.0 - (1.0-om2)/xamin*xu), yu)
        yl = torch.where(xl < xamin,
                         torch.log1p(-torch.exp(-xl_s)) - (1.0 - (1.0-om2)/xamin*xl), yl)
        yu = torch.where(xu > xamax, -(math.pi**2/6.0)/xu_s, yu)
        yl = torch.where(xl > xamax, -(math.pi**2/6.0)/xl_s, yl)
        yu = torch.where(xu == 0.0, torch.zeros_like(yu), yu)
        yl = torch.where(xl == 0.0, torch.zeros_like(yl), yl)
        den  = (xu - xl).abs().clamp(min=1e-30) * torch.sign(xu - xl + 1e-60)
        r_wide = (xu * yu - xl * yl) / den
        return torch.where(narrow, r_narrow, r_wide)

    # idos 1 or 3
    ya    = tt['deb3'] if idos == 1 else tt['sin3']
    y2min = float(tt['deb2_min'] if idos == 1 else tt['sin2_min'])
    xs    = x.clamp(min=1e-30)
    ri    = _interp_torch(xa, ya, x) + torch.log1p(-torch.exp(-xs))
    if idos == 1:
        ra = -(math.pi**4/15.0) / xs**3
        rb = torch.log(xs) - 4.0/3.0 + 1.0 - (1.0 - y2min)/xamin * x
    else:
        ra = -(math.pi**4/15.0)/xs**3 * _SFAC - _ASIN * xs**(1.0 - _BSIN)
        rb = torch.log(xs) - 4.0/3.0 + 1.0 - (1.0 - y2min)/xamin * x + _CSIN
    return torch.where((x >= xamin) & (x <= xamax), ri,
                       torch.where(x > xamax, ra, rb))
