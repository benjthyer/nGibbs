"""Deployable self-tests for the shipped nGibbs emulators.

These modules run against ground-truth data that ships with each model (its
validation datasets and its standards/ folder), so an end user can call
``make_ngibbs('HeFESToEarth').test()`` and get a quality-metrics report without
any other data or the training-side ``builder`` package.

Nothing here imports ``builder`` or ``scripts``; the research-side equivalents
live in ``scripts/`` and may.
"""

from .emulator_quality import (
    evaluate_emulator_quality,
    legend_text,
    METRIC_DESCRIPTIONS,
)
from .meltstable_comparison import hefesto_run_comparison
from .melts_comparison import melts_run_comparison
from .workspace_benchmark import classify_path, evaluate_workspace
from .benchmark import (
    BenchmarkRegressionError,
    BenchmarkTolerances,
    check_benchmark,
    compare_benchmarks,
    generate_benchmark,
)
from .quality_thresholds import (
    check_melts_quality,
    check_meltstable_quality,
    check_phase_quality,
    EmulatorQualityError,
    QualityFailure,
)

__all__ = [
    "evaluate_emulator_quality",
    "legend_text",
    "METRIC_DESCRIPTIONS",
    "hefesto_run_comparison",
    "melts_run_comparison",
    "classify_path",
    "evaluate_workspace",
    "generate_benchmark",
    "compare_benchmarks",
    "check_benchmark",
    "BenchmarkTolerances",
    "BenchmarkRegressionError",
    "check_melts_quality",
    "check_meltstable_quality",
    "check_phase_quality",
    "EmulatorQualityError",
    "QualityFailure",
]
