"""
Hessian matrix calculation, translated from Fortran hessian.f.

Computes the Hessian (matrix of second derivatives of Gibbs free energy
with respect to composition) for a single species.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .state import HeFESToState


def dkron(i: int, j: int) -> float:
    """Kronecker delta: 1.0 if i == j, else 0.0."""
    return 1.0 if i == j else 0.0


def hessian(ispec: int, ncp: np.ndarray, state: HeFESToState) -> np.ndarray:
    """
    Compute Hessian matrix row for species ispec.
    
    The Hessian is the matrix of second derivatives of Gibbs free energy
    with respect to compositional variables. This is used in the equilibrium
    solver to estimate the Jacobian of the constraint equations.
    
    Args:
        ispec: Species index (0-based).
        ncp: Molar amounts of components (ncompp,) array.
        state: HeFESToState containing:
            - apar: Parameter matrix (species x parameters).
            - Ti, Pi: Temperature and pressure.
            - f, r: Phase membership and site occupancy matrices.
            - n: Total species amounts.
            - Rgas: Gas constant.
            - etc. (wreg, vreg, iastate, lphase, nsite, nco, nspec, etc.)
    
    Returns:
        hess: (nspec, nspec) Hessian matrix for species ispec (row ispec).
    
    Notes:
        - Depends heavily on site occupancy (r), phase membership (f), and
          order-disorder state (iastate).
        - Regexing parameters (wreg, vreg) model site interaction energies.
        - Only first site (kst=1) contributes to van Laar asymmetry correction.
    """
    nspec = state.nspec
    ncompp = state.nco if hasattr(state, 'nco') else 0
    Rgas = state.Rgas if hasattr(state, 'Rgas') else 8.314462618
    
    # Retrieve site occupancy matrix r and phase membership f
    r = getattr(state, 'r_site', None)  # Will be (nco, nspec, nsite)
    f = getattr(state, 'f_phase', None)  # Will be (nphase, nspec)
    apar = getattr(state, 'apar', np.zeros((nspec, 0)))

    # If these are not available, return zero Hessian
    if r is None or f is None:
        return np.zeros((nspec, nspec))
    
    # Determine phase for ispec
    lphase = getattr(state, 'lphase', np.zeros(state.nspecp, dtype=int))
    iph = lphase[ispec] if ispec < len(lphase) else 0
    
    nsite_ph = getattr(state, 'nsite', np.ones(1, dtype=int))
    nsitecp = nsite_ph[iph] if iph < len(nsite_ph) else 1
    
    wreg = getattr(state, 'wreg', None)
    vreg = getattr(state, 'vreg', None)
    iastate = getattr(state, 'iastate', None)
    
    hess = np.zeros((nspec, nspec))
    rsuma = np.zeros(nspec)
    
    for kst in range(nsitecp):
        nkp = 0.0
        nkpr = 0.0
        sum1 = 0.0
        sum1a = np.zeros(nspec)
        sum2a = np.zeros(nspec)
        nikp = np.zeros(ncompp) if ncompp > 0 else np.array([0.0])

        for ic in range(ncompp):
            nikp[ic] = 0.0
            for jsp in range(nspec):
                if iastate is not None and iastate[ispec, jsp, kst]:
                    continue
                nkp += f[iph, jsp] * r[ic, jsp, kst] * ncp[jsp]
                nikp[ic] += f[iph, jsp] * r[ic, jsp, kst] * ncp[jsp]

            nkpr += nikp[ic]
            sum1 += r[ic, ispec, kst]

            for jspec in range(nspec):
                if iastate is not None and iastate[ispec, jspec, kst]:
                    continue
                sum1a[jspec] += r[ic, jspec, kst] * f[iph, jspec]
                if nikp[ic] > 0.0:
                    sum2a[jspec] += r[ic, ispec, kst] * r[ic, jspec, kst] * f[iph, jspec] / nikp[ic]

        EPS = 1.0e-12
        for jspec in range(nspec):
            hess[ispec, jspec] -= sum2a[jspec]
            if nkp > EPS:
                hess[ispec, jspec] += sum1 * sum1a[jspec] / nkp
            else:
                # If nkp is effectively zero but sum1 is non-zero, this indicates
                # inconsistent occupancy/composition leading to a singular Hessian.
                if abs(sum1) > EPS and abs(sum1a[jspec]) > EPS:
                    raise ValueError(f"singular site occupancy in hessian for species {ispec}: nkp~0, sum1={sum1}")
                # Otherwise leave contribution as-is (no additive term)

        # Van Laar asymmetry correction (only on first site)
        if kst == 0 and wreg is not None and vreg is not None:
            nsum = 0.0
            for ia in range(nspec):
                nsum += f[iph, ia] * apar[ia, 40] * ncp[ia]

            if nsum > EPS:
                apar = state.apar
                for ia in range(nspec - 1):
                    for ib in range(ia + 1, nspec):
                        siza = apar[ia, 40]
                        sizb = apar[ib, 40]
                        sizi = apar[ispec, 40]
                        qa = siza * ncp[ia] / nsum
                        qb = sizb * ncp[ib] / nsum

                        wregsave = wreg[iph, kst, ia, ib]
                        oregsave = wreg[iph, kst, ib, ia]

                        wreg[iph, kst, ia, ib] = wregsave + state.Pi * vreg[iph, kst, ia, ib]
                        wreg[iph, kst, ib, ia] = oregsave + state.Pi * vreg[iph, kst, ib, ia]

                        wregsz = (wreg[iph, kst, ia, ib] + wreg[iph, kst, ib, ia] * (qb - qa)) * 2.0 * abs(sizi) / (abs(siza) + abs(sizb))
                        oregsz = wreg[iph, kst, ib, ia] * 2.0 * abs(sizi) / (abs(siza) + abs(sizb))

                        wreg[iph, kst, ia, ib] = wregsave
                        wreg[iph, kst, ib, ia] = oregsave

                        for jspec in range(nspec):
                            dkron_ia_j = dkron(jspec, ia)
                            dkron_ib_j = dkron(jspec, ib)
                            dqadnj = f[iph, jspec] * apar[ia, 40] * (dkron_ia_j * nsum - ncp[ia] * apar[jspec, 40]) / (nsum * nsum)
                            dqbdnj = f[iph, jspec] * apar[ib, 40] * (dkron_ib_j * nsum - ncp[ib] * apar[jspec, 40]) / (nsum * nsum)

                            term1 = ((-dqbdnj * (dkron(ispec, ia) - qa) - dqadnj * (dkron(ispec, ib) - qb)) * wregsz
                                    + (dkron(ispec, ia) - qa) * (dkron(ispec, ib) - qb) * wreg[iph, kst, ib, ia] * (dqbdnj - dqadnj) * 2.0 * abs(sizi) / (abs(siza) + abs(sizb)))
                            term2 = (oregsz * ((dqadnj * qb + dqbdnj * qa) * (dkron(ispec, ib) - qb - dkron(ispec, ia) + qa)
                                            + qa * qb * (dqadnj - dqbdnj)))

                            rsuma[jspec] -= f[iph, ia] * f[iph, ib] * term1
                            rsuma[jspec] += f[iph, ia] * f[iph, ib] * term2

    # Scale by temperature and gas constant
    for jspec in range(nspec):
        hess[ispec, jspec] = -state.Ti * Rgas * hess[ispec, jspec] / 1000.0 + rsuma[jspec]

    return hess
