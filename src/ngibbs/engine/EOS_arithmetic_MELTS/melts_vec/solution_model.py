"""
Generic machinery shared by MELTS's solid-solution mixing models
(feldspar.py, olivine.py, and future pyroxene/spinel/rhombohedral-oxide
modules) -- a batched Newton solver for internal ordering parameters, and
the "Darken equation" activity/chemical-potential construction MELTS uses
identically across every solution phase (`actFld`, `actOlv`, `actCpx`,
`actSpn`, `actMsg` in the MAGMA source all follow this exact same pattern:
mu_i = g + sum_j fr[i][j] * dg/dr_j, a_i = exp(mu_i / R T)).

Design note on derivative strategy (why some things are analytic and some
are numerical)
---------------------------------------------------------------------------
Every one of MELTS's ordered solid-solution models works like this: a
Gibbs energy G(r, s, T, P) is a Taylor/Landau polynomial in composition
variables `r` (independent mole fractions) and internal ordering
variables `s` (site-occupancy asymmetry), and at fixed (r, T, P) the
equilibrium `s* (r,T,P)` is the value satisfying dG/ds = 0 (found by
Newton iteration in the original C code, `order()`).

Once `s*` is known, the *envelope theorem* means:
  - dG/dr_j at fixed s = s*  EQUALS  the true total dG/dr_j (activities'
    "Darken" gradient) -- because dG/ds = 0 at s* kills the ds*/dr_j
    cross-term in the chain rule. So we use the phase's own analytic
    dG/dr_j macros (transcribed verbatim from source) evaluated at s*.
  - Likewise dG/dT at fixed s = s* equals the true dG/dT, so entropy
    S = -dG/dT can be read directly off the phase's own analytic
    (purely-configurational, for olivine) entropy formula evaluated at
    s* -- no finite differencing needed for G, H, S, or the activities.
  - This does NOT extend to second derivatives: d(dG/dP)/dT (needed for
    dV/dT) and d^2G/dT^2 (needed for Cp) pick up real ds*/dT, ds*/dP
    contributions (this is exactly the physical "excess heat capacity
    of ordering" effect near an order-disorder transition) that a purely
    analytic evaluation at fixed s* would miss. Re-deriving those second-
    derivative macros by hand from the C source is exactly the kind of
    large, error-prone algebra this project is trying to avoid, so
    instead we get dV/dT, dV/dP, Cp by finite-differencing V(T,P) and
    G(T,P) with s* *re-solved* at each stencil point -- this is both
    lower-transcription-risk AND more physically complete (it captures
    the ordering contribution to Cp/thermal expansion correctly, rather
    than silently dropping it as a fixed-s-only analytic formula would).
"""
from __future__ import annotations
import numpy as np


def newton_solve_ordering(dgds_func, s0, r, T, P, n_iter=60, jac_eps=1e-4,
                            s_min=None, s_max=None):
    """Batched Newton solve for internal ordering parameter(s) s such that
    dgds_func(s, r, T, P) == 0, using a numerically-estimated Jacobian
    (central differences) rather than a hand-transcribed analytic Hessian
    -- see module docstring.

    Parameters
    ----------
    dgds_func : callable(s, r, T, P) -> (B, NS) array, the NS first
        partial derivatives dG/ds_i evaluated at the given state.
    s0 : (B, NS) initial guess.
    r : (B, NR) bulk composition variables (held fixed during the solve).
    T, P : (B,) or (B,1).
    n_iter : fixed iteration count (no convergence-based early exit --
        cheap to over-iterate with numpy, same approach as the Vinet EOS
        solve in solid_eos.py).
    jac_eps : step size for the central-difference Jacobian.
    s_min, s_max : optional (NS,) or scalar clip bounds applied each step
        (MELTS's own order() also clips at physical bounds in places).

    Returns
    -------
    s : (B, NS) converged ordering parameters.
    """
    s = np.array(s0, dtype=np.float64, copy=True)
    B, NS = s.shape

    for _ in range(n_iter):
        f0 = dgds_func(s, r, T, P)               # (B, NS)

        # Numerical Jacobian J[..., i, j] = d f_i / d s_j (central diff).
        J = np.empty((B, NS, NS), dtype=np.float64)
        for j in range(NS):
            s_plus = s.copy();  s_plus[:, j]  += jac_eps
            s_minus = s.copy(); s_minus[:, j] -= jac_eps
            f_plus  = dgds_func(s_plus,  r, T, P)
            f_minus = dgds_func(s_minus, r, T, P)
            J[:, :, j] = (f_plus - f_minus) / (2.0*jac_eps)

        # Guard against a singular/near-singular Jacobian (e.g. a
        # component at exactly zero mole fraction, where several s_i
        # become physically meaningless) by regularizing the diagonal.
        J = J + 1e-10*np.eye(NS)[None, :, :]
        try:
            step = np.linalg.solve(J, f0[:, :, None])[:, :, 0]
        except np.linalg.LinAlgError:
            step = np.zeros_like(f0)

        s = s - step
        if s_min is not None or s_max is not None:
            s = np.clip(s, s_min, s_max)

    return s


def darken_activities(g, dgdr, fr, R, T):
    """MELTS's shared "Darken equation" activity/chemical-potential
    formula, vectorized: mu_i = g + sum_j fr[i,j]*dgdr[j];  a_i =
    exp(mu_i / (R T)).

    Parameters
    ----------
    g : (B,) Gibbs energy of mixing (per formula unit of the solution).
    dgdr : (B, NR) dG/dr_j.
    fr : (NA, NR) or (B, NA, NR) -- the FRj(i) coefficient matrix
        (component i's stoichiometric coefficient against composition
        variable r_j; phase-specific, defined in each phase's `conXxx`/
        `actXxx` translation, e.g. feldspar.py / olivine.py).
    R : gas constant.
    T : (B,) temperature.

    Returns
    -------
    mu : (B, NA) chemical potentials of mixing.
    a  : (B, NA) activities.
    """
    g = np.asarray(g, dtype=np.float64)
    dgdr = np.asarray(dgdr, dtype=np.float64)
    fr = np.asarray(fr, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)

    if fr.ndim == 2:
        # (NA, NR) . (B, NR) -> (B, NA)
        mu = g[:, None] + np.einsum('ij,bj->bi', fr, dgdr)
    else:
        mu = g[:, None] + np.einsum('bij,bj->bi', fr, dgdr)
    a = np.exp(mu / (R*T[:, None]))
    return mu, a


def central_diff(f, x0, h, *args):
    """Simple (f(x0+h)-f(x0-h))/(2h) helper for the T,P finite differences
    described in the module docstring. `f` must accept `x0`'s shape and
    broadcast normally against `*args`."""
    return (f(x0 + h, *args) - f(x0 - h, *args)) / (2.0*h)
