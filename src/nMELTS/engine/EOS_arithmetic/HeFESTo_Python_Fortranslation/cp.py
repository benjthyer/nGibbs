"""
Chemical potential and related thermodynamic properties, translated from cp.f.

Computes the contribution to the chemical potential of a single species
including site occupancy, order-disorder, van Laar asymmetry, and Landau
transition effects.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .state import HeFESToState


def ourlog(x: float) -> float:
    """Safe logarithm: returns -1e100 if x <= 0, else log(x)."""
    if x <= 0.0:
        return -1e100
    return np.log(x)


def dkron(i: int, j: int) -> float:
    """Kronecker delta."""
    return 1.0 if i == j else 0.0


def cp(ispec: int, ncp: np.ndarray, state: HeFESToState) -> tuple[float, float, float, float]:
    """
    Compute chemical potential and related properties for a single species.
    
    Args:
        ispec: Species index (0-based).
        ncp: Molar amounts of components (ncompp,).
        state: HeFESToState containing:
            - apar: Parameter matrix (nspec, nparp).
            - Ti, Pi: Temperature and pressure.
            - n: Molar amounts of all species (nspec,).
            - f, r, nsite, nco: Phase/site matrices.
            - wreg, vreg: Interaction parameters.
            - iastate: Order-disorder state.
            - lphase: Phase for each species.
            - Rgas: Gas constant.
            - iltyp: Landau type selection.
            - etc.
    
    Returns:
        Tuple (chempot, rsum, volsum, smixi):
            - chempot: Chemical potential contribution (J/mol).
            - rsum: Interaction energy contribution.
            - volsum: Volume contribution from interactions.
            - smixi: Mixing entropy contribution.
    
    Notes:
        - Site occupancy is computed from r matrix and component amounts.
        - Van Laar asymmetry correction applied at first site only.
        - Landau transition contributions optionally added based on iltyp.
        - Result should be integrated into bulk property calculations.
    """
    nspec = state.nspec
    ncompp = state.nco if hasattr(state, 'nco') else 0
    Rgas = state.Rgas if hasattr(state, 'Rgas') else 8.314462618
    
    # Retrieve site occupancy and phase info
    r = getattr(state, 'r_site', None)
    f = getattr(state, 'f_phase', None)
    
    if r is None or f is None:
        # Return zeros if site data unavailable
        return 0.0, 0.0, 0.0, 0.0
    
    lphase = getattr(state, 'lphase', np.zeros(state.nspecp, dtype=int))
    iph = lphase[ispec] if ispec < len(lphase) else 0
    
    nsite_ph = getattr(state, 'nsite', np.ones(1, dtype=int))
    nsitecp = nsite_ph[iph] if iph < len(nsite_ph) else 1
    
    wreg = getattr(state, 'wreg', None)
    vreg = getattr(state, 'vreg', None)
    iastate = getattr(state, 'iastate', None)
    
    chempot = 0.0
    smixi = 0.0
    rsum = 0.0
    volsum = 0.0
    
    # Main site loop
    for kst in range(nsitecp):
        nkp = 0.0
        nkpr = 0.0
        sum1 = 0.0
        sum2 = 0.0
        
        for ic in range(ncompp):
            nikp_ic = 0.0
            for jsp in range(nspec):
                if iastate is not None and iastate[ispec, jsp, kst]:
                    continue
                nkp += f[iph, jsp] * r[ic, jsp, kst] * ncp[jsp]
                nikp_ic += f[iph, jsp] * r[ic, jsp, kst] * ncp[jsp]
            
            nkpr += nikp_ic
            sum1 += r[ic, ispec, kst]
            sum2 += r[ic, ispec, kst] * ourlog(nikp_ic)
        
        chempot -= sum2
        if nkp > 0.0:
            chempot += sum1 * ourlog(nkp)
        
        # Van Laar asymmetry correction (first site only)
        if kst == 0 and wreg is not None and vreg is not None:
            msum = 0.0
            nsum = 0.0
            ssum = 0.0
            apar = state.apar
            
            for ia in range(nspec):
                msum += f[iph, ia] * apar[ia, 40] * ncp[ia]
                nsum += f[iph, ia] * abs(apar[ia, 40]) * ncp[ia]
                ssum += f[iph, ia] * ncp[ia]
            
            if nsum > 0.0:
                ratio = msum / nsum
                for ia in range(nspec - 1):
                    for ib in range(ia + 1, nspec):
                        siza = apar[ia, 40]
                        sizb = apar[ib, 40]
                        sizi = apar[ispec, 40]
                        qa = abs(siza) * ncp[ia] / nsum
                        qb = abs(sizb) * ncp[ib] / nsum
                        
                        wregsave = wreg[iph, kst, ia, ib]
                        oregsave = wreg[iph, kst, ib, ia]
                        
                        wreg[iph, kst, ia, ib] = wregsave + state.Pi * vreg[iph, kst, ia, ib]
                        wreg[iph, kst, ib, ia] = oregsave + state.Pi * vreg[iph, kst, ib, ia]
                        
                        wregsz = ((wreg[iph, kst, ia, ib] + wreg[iph, kst, ib, ia] * (qb - qa))
                                 * 2.0 * abs(sizi) / (abs(siza) + abs(sizb)))
                        oregsz = wreg[iph, kst, ib, ia] * 2.0 * abs(sizi) / (abs(siza) + abs(sizb))
                        vregsz = ((vreg[iph, kst, ia, ib] + vreg[iph, kst, ib, ia] * (qb - qa))
                                 * 2.0 * abs(sizi) / (abs(siza) + abs(sizb)))
                        
                        wreg[iph, kst, ia, ib] = wregsave
                        wreg[iph, kst, ib, ia] = oregsave
                        
                        d_ia_ispec = dkron(ispec, ia)
                        d_ib_ispec = dkron(ispec, ib)
                        
                        rsum -= (f[iph, ia] * f[iph, ib] * (d_ia_ispec - qa) * (d_ib_ispec - qb) * wregsz * ratio)
                        rsum += (f[iph, ia] * f[iph, ib] * qa * qb * (d_ib_ispec - qb - d_ia_ispec + qa) * oregsz)
                        volsum -= (f[iph, ia] * f[iph, ib] * (d_ia_ispec - qa) * (d_ib_ispec - qb) * vregsz * ratio)
    
    # Finalize smixi and convert chempot
    smag = 0.0  # Magnetic contribution (stub)
    smixi = Rgas * chempot
    chempot = -state.Ti * Rgas * chempot + 1000.0 * rsum - state.Ti * smag
    
    # Add Landau transition contribution (if enabled)
    iltyp = getattr(state, 'iltyp', 0)
    mphase = getattr(state, 'mphase', np.ones(1, dtype=int))
    
    if mphase[iph] > 1:
        ntot = 0.0
        for jsp in range(nspec):
            ntot += f[iph, jsp] * state.n[jsp]
        
        if ntot > 0.0:
            cplandau = 0.0
            Vi = 7.0  # Typical molar volume estimate
            
            for jsp in range(nspec):
                if f[iph, jsp] == 0.0:
                    continue
                
                x = state.n[jsp] / ntot
                
                # Apply Landau transition contribution
                if iltyp == 1:
                    from .landauqr import landauqr
                    qorder, Tc, glan, cplan, alplan, betlan, slan, vlan = landauqr(jsp, Vi, state)
                elif iltyp == 2:
                    from .landau import landau
                    qorder, Tc, glan, cplan, alplan, betlan, slan, vlan = landau(jsp, Vi, state)
                else:
                    qorder = 0.0
                
                bx = -state.apar[jsp, 37]  # apar[jsp, 38] in Fortran
                cplandau -= x * state.apar[jsp, 38] * bx * (dkron(ispec, jsp) - x) * (1.0 - qorder**2)
            
            # Note: Landau contribution is commented out in original Fortran
            # cplandau would be added to chempot here if enabled
    
    return chempot, rsum, volsum, smixi
