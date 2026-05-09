"""
Projected Hessian computation, translated from Fortran hessfunc.f.

Computes the Hessian projected onto the null space (subspace of allowed
degree of freedom in the system).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from .state import HeFESToState

from .hessian import hessian as compute_hessian


def hessfunc(nnew: np.ndarray, state: HeFESToState, extern: Optional[object] = None) -> np.ndarray:
    """
    Compute the projected Hessian matrix.
    
    The Hessian is first computed for all species pairs, then projected onto
    the null space defined by the stoichiometric constraints using the q2 matrix.
    
    Args:
        nnew: Solution vector in null space (nnull,).
        state: HeFESToState containing:
            - n: Molar amounts of all species (nspec,).
            - nspec: Number of species.
            - nnull: Dimension of null space.
            - q2: Null space basis matrix (nspec, nnull).
            - etc.
        extern: Unused; for API compatibility.
    
    Returns:
        hespro: Projected Hessian in null space (nnull, nnull).
    
    Notes:
        - Projection formula: H_projected = q2^T @ H @ q2
        - Uses pure NumPy for all matrix operations.
    """
    nspec = state.nspec
    nnull = state.nnull
    
    # Initialize Hessian matrix
    hess = np.zeros((nspec, nspec))
    
    # Compute Hessian row by row for each species
    for ispec in range(nspec):
        hess_row = compute_hessian(ispec, state.n, state)
        hess[ispec, :] = hess_row[ispec, :]
    
    # Project Hessian: qh = q2^T @ hess
    qh = np.zeros((nnull, nspec))
    for i in range(nnull):
        for j in range(nspec):
            for k in range(nspec):
                qh[i, j] += state.q2[k, i] * hess[k, j]
    
    # Project further: hespro = qh @ q2
    hespro = np.zeros((nnull, nnull))
    for i in range(nnull):
        for j in range(nnull):
            for k in range(nspec):
                hespro[i, j] += qh[i, k] * state.q2[k, j]
    
    return hespro
