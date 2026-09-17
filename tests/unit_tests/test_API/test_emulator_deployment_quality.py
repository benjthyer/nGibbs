"""Deployment quality gate for every instantiated EmulatorAPI singleton.

Runs each model's `.assert_quality()` (test() + the fixed tolerances in
`ngibbs.deployment_tests.quality_thresholds`) against its shipped ground-truth
bundles. Real model weights and real bundles -- no mocking -- so this is closer
to an integration test than the rest of `tests/unit_tests`; it loads a
checkpoint and runs a full quality pass per model (tens of seconds each).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / 'src'
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ngibbs.deployment_tests import EmulatorQualityError
from ngibbs.engine.models import CPU_MODELS


@pytest.mark.parametrize('name', sorted(CPU_MODELS))
def test_model_passes_deployment_quality_gate(name):
    model = CPU_MODELS[name]
    try:
        model.assert_quality(write_outputs=False, verbose=False)
    except EmulatorQualityError as exc:
        pytest.fail(str(exc))
