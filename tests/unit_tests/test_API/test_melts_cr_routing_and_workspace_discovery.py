"""MELTSAPI Cr/NoCr routing (an all-zero Cr2O3 column selects NoCr) and raw
workspace discovery for ``workspace_benchmark``. No model weights are loaded."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / 'src'
for p in (REPO_ROOT, SRC_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ngibbs.engine.API import MELTSAPI
from ngibbs.deployment_tests.workspace_benchmark import find_hefesto_runs, find_melts_runs

HEADERS = ['Pressure(System_main)', 'Temperature(System_main)', 'SiO2', 'MgO', 'Cr2O3']


@pytest.fixture
def api():
    obj = MELTSAPI.__new__(MELTSAPI)   # routing only needs .cr / .nocr
    obj.cr, obj.nocr = object(), object()
    return obj


def _rows(cr):
    return np.array([[1000.0, 1200.0, 50.0, 10.0, cr[0]],
                     [1000.0, 1100.0, 50.0, 10.0, cr[1]]])


@pytest.mark.parametrize('make', [
    lambda a: a,
    lambda a: torch.tensor(a),
    lambda a: pd.DataFrame(a, columns=HEADERS),
    lambda a: dict(zip(HEADERS, a.T)),
])
def test_zero_cr_column_routes_to_nocr(api, make):
    table = make(_rows([0.0, 0.0]))
    headers = None if isinstance(table, (pd.DataFrame, dict)) else HEADERS
    assert api._route(table, headers) is api.nocr


@pytest.mark.parametrize('make', [
    lambda a: a,
    lambda a: pd.DataFrame(a, columns=HEADERS),
    lambda a: dict(zip(HEADERS, a.T)),
])
def test_any_nonzero_cr_routes_to_cr(api, make):
    table = make(_rows([0.0, 0.3]))
    headers = None if isinstance(table, (pd.DataFrame, dict)) else HEADERS
    assert api._route(table, headers) is api.cr


def test_nan_cr_counts_as_zero(api):
    assert api._route(_rows([np.nan, 0.0]), HEADERS) is api.nocr


def test_no_cr_column_routes_to_nocr(api):
    assert api._route(_rows([0.0, 0.0])[:, :4], HEADERS[:4]) is api.nocr


def test_headers_only_keep_presence_rule(api):
    assert api._route(headers=HEADERS) is api.cr
    assert api._route(headers=HEADERS[:4]) is api.nocr


def test_scalar_composition_dict(api):
    assert api._route({'SiO2': 50.0, 'Cr2O3': 0.0}, ['SiO2', 'Cr2O3']) is api.nocr
    assert api._route({'SiO2': 50.0, 'Cr2O3': 0.2}, ['SiO2', 'Cr2O3']) is api.cr


def test_element_cr_column(api):
    headers = ['P(GPa)(System_main)', 'T(K)(System_main)', 'Si', 'Cr']
    assert api._route(np.array([[1.0, 1500.0, 0.4, 0.0]]), headers) is api.nocr
    assert api._route(np.array([[1.0, 1500.0, 0.4, 0.01]]), headers) is api.cr


def test_find_melts_runs_any_melts_name(tmp_path):
    (tmp_path / 'a' / 'Simulation10').mkdir(parents=True)
    (tmp_path / 'a' / 'Simulation10' / 'basalt.melts').write_text('')
    (tmp_path / 'a' / 'Simulation2').mkdir(parents=True)
    (tmp_path / 'a' / 'Simulation2' / 'input.melts').write_text('')
    (tmp_path / 'b').mkdir()
    (tmp_path / 'b' / 'input.melts_bak').write_text('')   # not a .melts file
    runs = find_melts_runs(tmp_path)
    assert [r.name for r in runs] == ['Simulation2', 'Simulation10']


def test_find_hefesto_runs(tmp_path):
    for name in ('Simulation3', 'Simulation20'):
        (tmp_path / 'Batch0001' / name).mkdir(parents=True)
        (tmp_path / 'Batch0001' / name / 'control').write_text('')
    (tmp_path / 'Batch0001' / 'notes').mkdir()
    assert [r.name for r in find_hefesto_runs(tmp_path)] == ['Simulation3', 'Simulation20']


def test_package_level_model_names_match_models_py():
    """ngibbs.__init__ lists the model names so `import ngibbs` stays light;
    keep that list in step with engine.models._MODEL_SPECS."""
    import ngibbs
    from ngibbs.engine import models
    assert set(ngibbs._MODEL_NAMES) == set(models._MODEL_SPECS)
    for name in models._MODEL_SPECS:
        assert getattr(ngibbs, f'{name}CPU') is models.CPU_MODELS[name]
        assert getattr(ngibbs, f'{name}GPU') is models.GPU_MODELS.get(name)
