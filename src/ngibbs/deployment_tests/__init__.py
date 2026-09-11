"""Deployable self-tests for the shipped nGibbs emulators.

These modules run against ground-truth data that ships inside the package
(this directory) so an end user can call ``HeFESToEmulatorCPU.test()`` and get
a quality-metrics report without any external data or the training-side
``builder`` package.

Nothing here imports ``builder`` or ``scripts``; the research-side equivalents
live in ``scripts/`` and may.
"""

from .emulator_quality import (
    evaluate_emulator_quality,
    legend_text,
    METRIC_DESCRIPTIONS,
)
from .meltstable_comparison import (
    run_meltstable_phase_comparison,
    run_meltstable_property_comparison,
)

__all__ = [
    "evaluate_emulator_quality",
    "legend_text",
    "METRIC_DESCRIPTIONS",
    "run_meltstable_phase_comparison",
    "run_meltstable_property_comparison",
]
