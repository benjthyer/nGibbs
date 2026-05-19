"""Python wrapper for HeFESTo ``entrop.f``."""

from __future__ import annotations

from typing import Sequence

from .param_state import HeFESToPhaseState, compute_physub_properties


def entrop(
    Ttry: float,
    phases: Sequence[HeFESToPhaseState],
    pressure_gpa: float,
    starg: float = 0.0,
    depth_km: float = 0.0,
) -> float:
    bulk = compute_physub_properties(phases, pressure_gpa, Ttry, depth_km)
    return bulk.entropy - starg
