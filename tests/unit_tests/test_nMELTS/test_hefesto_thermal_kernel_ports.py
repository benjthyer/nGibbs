from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
for path in (str(SRC_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)


import importlib

from nMELTS.engine.EOS_arithmetic import Ctherm, Ener, Etherm, Ftherm, Helm, Ztherm

entrop_module = importlib.import_module("nMELTS.engine.EOS_arithmetic.entrop")


def test_energy_and_helm_exact_limits() -> None:
    assert Ener(0.0, 0.0, 1) == 0.0
    assert Ener(1.25, 0.0, 5) == 1.0
    assert Helm(0.0, 0.0, 1) == 0.0
    assert math.isclose(Helm(2.0, 0.0, 5), math.log(2.0) - (1.0 / 3.0), rel_tol=0.0, abs_tol=1.0e-15)


@pytest.mark.parametrize(
    "kernel",
    [Ctherm, Etherm, Ftherm, Ztherm],
)
def test_thermal_kernels_return_zero_at_nonpositive_temperature(kernel) -> None:
    if kernel is Ztherm:
        result = kernel(0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    else:
        result = kernel(0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert result == 0.0


def test_entrop_wrapper_uses_bulk_entropy(monkeypatch) -> None:
    class DummyBulk:
        entropy = 123.4

    def fake_compute_physub_properties(phases, pressure_gpa, temperature_k, depth_km):
        return DummyBulk()

    monkeypatch.setattr(entrop_module, "compute_physub_properties", fake_compute_physub_properties)
    assert entrop_module.entrop(1000.0, [], 3.0, starg=23.4, depth_km=10.0) == pytest.approx(100.0)
