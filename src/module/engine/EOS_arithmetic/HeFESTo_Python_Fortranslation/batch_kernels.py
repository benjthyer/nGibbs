"""Vectorized batch kernels for fast processing of multiple assemblages.

These kernels wrap scalar functions with NumPy vectorization, enabling
efficient batch computation for ~60K+ rows of data. Each function accepts
(N,) shaped arrays and returns (N,) shaped results.
"""
from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Tuple

import numpy as np

if TYPE_CHECKING:
    from .state import HeFESToState


@lru_cache(maxsize=16)
def _get_vectorized_volume():
    """Cache vectorized volume function to avoid repeated compilation."""
    from .volume import volume as volume_scalar
    return np.vectorize(volume_scalar, excluded=['state'], otypes=[float])


@lru_cache(maxsize=16)
def _get_vectorized_gspec():
    """Cache vectorized gspec function to avoid repeated compilation."""
    from . import gspec as gspec_mod
    return np.vectorize(gspec_mod.gspec, excluded=['state'], otypes=[object])


def volume_batch(
    ispec: np.ndarray,
    x1: np.ndarray,
    state: HeFESToState
) -> np.ndarray:
    """Compute volumes for a batch of species/conditions.

    Parameters
    ----------
    ispec : np.ndarray
        Shape (N,), species indices (typically all 0 for single species)
    x1 : np.ndarray
        Shape (N,), initial guesses for volume
    state : HeFESToState
        Shared state object (P, T already set per row in batch loop)

    Returns
    -------
    np.ndarray
        Shape (N,), computed volumes
    """
    ispec = np.asarray(ispec, dtype=int).ravel()
    x1 = np.asarray(x1, dtype=float).ravel()
    
    if ispec.shape[0] != x1.shape[0]:
        raise ValueError(f"ispec and x1 must have same length, got {ispec.shape[0]} and {x1.shape[0]}")
    
    vfunc = _get_vectorized_volume()
    return vfunc(ispec, x1, state)


def gspec_batch(
    ispec: np.ndarray,
    state: HeFESToState
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Gibbs free energies for a batch of species/conditions.

    Parameters
    ----------
    ispec : np.ndarray
        Shape (N,), species indices
    state : HeFESToState
        Shared state object (P, T already set per row in batch loop)

    Returns
    -------
    tuple of np.ndarray
        (freae_energies, spinodal_flags) both shape (N,)
    """
    ispec = np.asarray(ispec, dtype=int).ravel()
    
    gfunc = _get_vectorized_gspec()
    results = gfunc(ispec, state)
    
    # Unpack results (each element is a tuple from gspec)
    results = np.asarray(results, dtype=object)
    if results.ndim == 0:
        results = np.array([results])
    
    frees = np.array([r[0] if isinstance(r, tuple) else r for r in results], dtype=float)
    spinodals = np.array([r[1] if isinstance(r, tuple) and len(r) > 1 else False for r in results], dtype=bool)
    
    return frees, spinodals


def extract_properties_batch(
    state_list: list,
    indices: np.ndarray
) -> np.ndarray:
    """Extract bulk properties from a list of HeFESToState objects.

    Parameters
    ----------
    state_list : list of HeFESToState
        State objects with computed properties
    indices : np.ndarray
        Shape (N,), indices into state_list (typically just range(N))

    Returns
    -------
    np.ndarray
        Shape (N, 6) with properties: density, Cp, Cv, alpha, K_S, gamma
    """
    indices = np.asarray(indices, dtype=int).ravel()
    output = np.zeros((len(indices), 6), dtype=float)
    
    for i, idx in enumerate(indices):
        state = state_list[idx]
        output[i, 0] = state.vol
        output[i, 1] = state.Cap
        output[i, 2] = state.Cv
        output[i, 3] = state.alp
        output[i, 4] = state.Ks
        output[i, 5] = state.gamma
    
    return output
