"""Path classification and output routing for evaluate_workspace (no model weights)."""
from pathlib import Path

import numpy as np
import pytest

from ngibbs.deployment_tests.workspace_benchmark import (
    CUSTOM_WORKSPACE, STANDARD_WORKSPACE, classify_path, default_output_root, model_label,
    standard_workspace,
)


@pytest.mark.parametrize('P, kind', [
    ([1500.0] * 5, 'isobar'),
    ([0.1, 1.0, 2.0, 5.0], 'transect'),
    ([5.0, 1.0, 3.0], 'transect'),                       # order does not matter
    ([1.0, 1.0, 2.0], 'unplottable'),                    # repeats but not constant
    ([1.0, 2.0, 2.0, 3.0], 'unplottable'),
    ([1.0], 'unplottable'),                              # too few rows
    ([np.nan, 2.0, np.nan], 'unplottable'),
    ([1500.0, 1500.0 * (1 + 1e-12)], 'isobar'),          # float noise is one pressure
])
def test_classify_path(P, kind):
    got, reason = classify_path(np.array(P))
    assert got == kind
    assert (reason == '') == (kind != 'unplottable')


class _FakeAPI:
    def __init__(self, home, name=None, standard=None):
        self.home_dir = Path(home)
        if name:
            self.benchmark_name = name
        self.standard_workspace_dir = standard


def test_output_root_uses_model_name_in_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert default_output_root(_FakeAPI('/x/HeFESToEarth', 'HeFESToEmulator')) == \
        tmp_path / 'ngibbs_deployment_test' / 'HeFESToEmulator'
    assert model_label(_FakeAPI('/x/HeFESToEarth')) == 'HeFESToEarth'


def test_standard_workspace_attribute(tmp_path):
    assert standard_workspace(_FakeAPI(tmp_path)) is None
    assert standard_workspace(_FakeAPI(tmp_path, standard=str(tmp_path))) == tmp_path
    assert (STANDARD_WORKSPACE, CUSTOM_WORKSPACE) == ('standardWorkspace', 'customWorkspace')


def test_checkout_models_ship_standards_and_baselines():
    """Each working copy carries its standard runs inside its folder, and has a
    CPU regression baseline under benchmarks/ (outside the package)."""
    from ngibbs import registry
    from ngibbs.engine.API import _standards_dir
    models = registry.checkout_models()
    if not models:
        pytest.skip('not a source checkout')
    for name, d in models.items():
        assert _standards_dir(d) is not None, name
        assert (registry.baseline_dir(name, 'cpu') / 'benchmark_index.json').is_file(), name
