"""
Binary search helper function, translated from Fortran bserch.f.

Replicates Fortran's BSERCH (binary search in sorted array) semantics.
Given a sorted array xx and a search value x, returns the index jlo such that
xx[jlo] <= x < xx[jlo+1] (using 1-based Fortran indexing in mind, but 0-based
Python arrays here).
"""
from __future__ import annotations

from typing import Sequence


def bserch(xx: Sequence[float], x: float) -> int:
    """
    Binary search for index jlo such that xx[jlo] <= x < xx[jlo+1].
    
    Args:
        xx: Sorted sequence of floats (ascending order).
        x: Search value.
    
    Returns:
        Index jlo (0-based) such that xx[jlo] <= x < xx[jlo+1].
        If x < xx[0], returns -1.
        If x >= xx[-1], returns len(xx) - 1.
    
    Notes:
        - Matches Fortran behavior for edge cases.
        - Caller is responsible for bounds-checking when accessing xx[jlo+1].
    """
    n = len(xx)
    if n == 0:
        return -1
    if x < xx[0]:
        return -1
    if x >= xx[n - 1]:
        return n - 1
    
    # Standard binary search
    jlo = 0
    jhi = n - 1
    while jhi - jlo > 1:
        jmid = (jhi + jlo) // 2
        if x < xx[jmid]:
            jhi = jmid
        else:
            jlo = jmid
    
    return jlo
