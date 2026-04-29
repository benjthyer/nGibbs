"""Benchmark comparison test for the Python physub translation.

This test reconstructs a simplified species state from HeFESTo BENCHMARK
tables, evaluates the Python implementation, and compares key bulk outputs
against ground truth values from fort.58.
"""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
BUILDER_ROOT = SRC_ROOT / "builder"
if str(BUILDER_ROOT) not in sys.path:
    sys.path.insert(0, str(BUILDER_ROOT))

from nMELTS.engine.EOS_arithmetic.hefesto_physub import (  # noqa: E402
    PHYSUB_BULK_ATTRIBUTE_NAMES,
    compute_physub_bulk_matrix,
    compare_physub_against_benchmark_directory,
    get_hefesto_physub_context,
    load_hefesto_parameter_directory,
)
from nMELTS.engine.EOS_arithmetic import hefesto_physub as physub_module  # noqa: E402
from nMELTS.engine.emulator import NN_MELTS  # noqa: E402


def extract_bulk_properties_from_simulation_dir(*args, **kwargs):
    module = import_module("HeFESTo.HeFESTo_functions")
    return module.extract_bulk_properties_from_simulation_dir(*args, **kwargs)


def _resolve_benchmark_dir() -> Path:
    default_dir = SRC_ROOT / "nMELTS" / "engine" / "EOS_arithmetic" / "BENCHMARK"
    required = ("control", "fort.42", "fort.58", "fort.61", "fort.62", "fort.63", "fort.99")
    if all((default_dir / name).exists() for name in required):
        return default_dir

    hefesto_root = REPO_ROOT / "data" / "HeFESToWorkspace"
    if hefesto_root.is_dir():
        for candidate in sorted(hefesto_root.glob("**/Simulation*")):
            if all((candidate / name).exists() for name in required):
                return candidate
    return default_dir


BENCHMARK_DIR = _resolve_benchmark_dir()


VERBOSE_TEST_OUTPUT = True


def _vprint(message: str) -> None:
    if VERBOSE_TEST_OUTPUT:
        print(message)


def _read_table(path: Path) -> Tuple[List[str], List[Dict[str, float]]]:
    lines = [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines()]
    header = None
    rows: List[Dict[str, float]] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        tokens = stripped.split()
        if header is None:
            header = tokens
            continue

        if len(tokens) < len(header):
            continue

        try:
            values = [float(token) for token in tokens[: len(header)]]
        except ValueError:
            continue

        rows.append({header[i]: values[i] for i in range(len(header))})

    if header is None:
        raise ValueError(f"No header found in table: {path}")

    return header, rows


def _parse_control_species_map(path: Path) -> Tuple[List[str], Dict[str, str], Dict[str, List[str]]]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    species_order: List[str] = []
    species_to_phase: Dict[str, str] = {}
    phase_to_species: Dict[str, List[str]] = {}

    current_phase = ""
    skip_next = False
    for line in lines:
        if line.lower().startswith("phase "):
            current_phase = line.split()[1]
            phase_to_species[current_phase] = []
            skip_next = True
            continue

        if skip_next:
            skip_next = False
            continue

        if current_phase:
            token = line.split()[0]
            if token.lower() == "phase":
                continue
            species_order.append(token)
            species_to_phase[token] = current_phase
            phase_to_species[current_phase].append(token)

    return species_order, species_to_phase, phase_to_species


def _parse_fort42_blocks(path: Path, expected_species: List[str]) -> List[Dict[str, Tuple[float, float]]]:
    lines = [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines()]
    blocks: List[Dict[str, Tuple[float, float]]] = []
    current: Dict[str, Tuple[float, float]] = {}

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("dndt and dndp by species"):
            if current:
                blocks.append(current)
                current = {}
            continue

        parts = stripped.split()
        if len(parts) < 4:
            continue

        try:
            int(parts[0])
            species = parts[1]
            dndt = float(parts[2])
            dndp = float(parts[3])
        except ValueError:
            continue

        if species in expected_species:
            current[species] = (dndt, dndp)

    return blocks


def test_physub_python_matches_benchmark_tables():
    _vprint("=" * 88)
    _vprint("Starting benchmark comparison for Python physub translation")
    _vprint(f"Benchmark directory: {BENCHMARK_DIR}")
    _vprint("=" * 88)

    required = ["control", "fort.42", "fort.58", "fort.61", "fort.62", "fort.63", "fort.99"]
    if not all((BENCHMARK_DIR / name).exists() for name in required):
        pytest.skip(f"Benchmark tables are unavailable at {BENCHMARK_DIR}")

    param_records = load_hefesto_parameter_directory(use_cache=True)
    hefesto_context = get_hefesto_physub_context()
    result = compare_physub_against_benchmark_directory(
        BENCHMARK_DIR,
        param_records=param_records,
        hefesto_context=hefesto_context,
        verbose=VERBOSE_TEST_OUTPUT,
    )

    _vprint("-" * 88)
    _vprint("Benchmark comparison summary")
    for name, value in result.mean_errors.items():
        _vprint(f"{name:<7} mean={value:.4e} max={result.max_errors[name]:.4e}")
    _vprint("=" * 88)

    assert result.passed, f"benchmark comparison failed: {result.mean_errors}"


if __name__ == "__main__":
    _vprint("Running benchmark test as a standalone script")
    test_physub_python_matches_benchmark_tables()
    _vprint("Standalone run completed successfully")


def test_physub_singleton_context_is_deterministic_and_cached(monkeypatch):
    context_a = get_hefesto_physub_context()
    context_b = get_hefesto_physub_context()
    assert context_a is context_b
    assert len(context_a.parameter_records) > 0
    assert context_a.parse_count == len(context_a.parameter_records)

    def _should_not_reparse(_):
        raise AssertionError("unexpected uncached parameter parse")

    monkeypatch.setattr(physub_module, "_load_hefesto_parameter_directory_uncached", _should_not_reparse)
    cached = load_hefesto_parameter_directory(use_cache=True)
    assert len(cached) == len(context_a.parameter_records)


def test_physub_matrix_selectors_and_batched_shape_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bsz = 3
    ncomps = 4

    component_moles = torch.tensor(
        [[0.9, 0.1, 0.0, 0.3], [0.2, 0.4, 0.5, 0.1], [0.6, 0.0, 0.2, 0.7]],
        dtype=torch.float32,
        device=device,
    )
    molar_mass = torch.tensor([50.0, 55.0, 60.0, 45.0], dtype=torch.float32, device=device)
    attrs = {
        "molar_volume": torch.full((bsz, ncomps), 1.0, dtype=torch.float32, device=device),
        "bulk_modulus": torch.full((bsz, ncomps), 130.0, dtype=torch.float32, device=device),
        "shear_modulus": torch.full((bsz, ncomps), 80.0, dtype=torch.float32, device=device),
        "heat_capacity_p": torch.full((bsz, ncomps), 1.2, dtype=torch.float32, device=device),
        "heat_capacity_v": torch.full((bsz, ncomps), 0.9, dtype=torch.float32, device=device),
        "thermal_expansivity": torch.full((bsz, ncomps), 3.0e-5, dtype=torch.float32, device=device),
        "entropy": torch.full((bsz, ncomps), 5.0, dtype=torch.float32, device=device),
        "enthalpy": torch.full((bsz, ncomps), 10.0, dtype=torch.float32, device=device),
        "gibbs": torch.full((bsz, ncomps), 8.0, dtype=torch.float32, device=device),
    }

    full, full_names = compute_physub_bulk_matrix(component_moles, molar_mass, attrs)
    assert full.shape == (bsz, len(PHYSUB_BULK_ATTRIBUTE_NAMES))
    assert full.device.type == device.type
    assert full_names == PHYSUB_BULK_ATTRIBUTE_NAMES

    subset_names = ["density", "K_Hill", "Vd", "Gibbs"]
    subset, resolved = compute_physub_bulk_matrix(
        component_moles,
        molar_mass,
        attrs,
        selectors=subset_names,
    )
    assert subset.shape == (bsz, len(subset_names))
    assert subset.device.type == device.type
    assert resolved == tuple(subset_names)


def test_polish_masses_selector_behavior_and_default_compatibility():
    class FakeEmulator:
        pass

    fake = FakeEmulator()
    fake.feature_offset = 0
    fake.compToEl = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    fake.compToOx = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    fake.phaseToCompMap = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    fake.MM = torch.eye(2, dtype=torch.float32)
    fake.label_indices = {"orthopyroxene": 0}
    fake._table_calls = 0

    def _make_phase_tables(newComps, compToOx, MM, compPhaseMap, features, out='oxides', eps=1e-12):
        fake._table_calls += 1
        return torch.ones((newComps.shape[0], 2, 2), dtype=torch.float32), torch.ones((newComps.shape[0], 2), dtype=torch.float32)

    fake.make_phase_tables = _make_phase_tables

    phase_moles = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    recon_bulk = torch.tensor([[0.3, 0.7]], dtype=torch.float32)
    component_moles = torch.tensor([[0.3, 0.7]], dtype=torch.float32)
    phase_proportions = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
    features = torch.tensor([[0.3, 0.7]], dtype=torch.float32)

    legacy = NN_MELTS.polish_masses(
        fake,
        phase_moles,
        recon_bulk,
        component_moles,
        phase_proportions,
        features,
        optimize_masses=False,
        output_componentMoles=True,
    )
    assert isinstance(legacy, tuple)
    assert len(legacy) == 3
    assert fake._table_calls == 1

    fake._table_calls = 0
    selected = NN_MELTS.polish_masses(
        fake,
        phase_moles,
        recon_bulk,
        component_moles,
        phase_proportions,
        features,
        optimize_masses=False,
        outputs=["component_moles"],
    )
    assert "component_moles" in selected
    assert "phase_tables" not in selected
    assert fake._table_calls == 0


def test_forwardmb_selector_behavior_and_default_compatibility():
    class IdentityNorm:
        @staticmethod
        def norm(x):
            return x

    class FakeModel:
        @staticmethod
        def forward(features, detailed=True):
            bsz = features.shape[0]
            likelihoods = torch.ones((bsz, 2), dtype=torch.float32)
            chem_out = torch.zeros((bsz, 2), dtype=torch.float32)
            log_moles = torch.zeros((bsz, 2), dtype=torch.float32)
            recon_bulk = torch.ones((bsz, 2), dtype=torch.float32)
            component_moles = torch.ones((bsz, 2), dtype=torch.float32)
            phase_proportions = torch.ones((bsz, 2), dtype=torch.float32)
            phase_moles = torch.ones((bsz, 2), dtype=torch.float32)
            return likelihoods, chem_out, log_moles, recon_bulk, component_moles, phase_proportions, phase_moles

    class FakeEmulator:
        pass

    fake = FakeEmulator()
    fake.norm_features = IdentityNorm()
    fake.convertOxToMol = lambda x, convert=True: x
    fake.model = FakeModel()
    fake._polish_calls = 0

    def _polish(*args, **kwargs):
        fake._polish_calls += 1
        requested = kwargs.get("outputs")
        if requested is None or requested == ["phase_tables"]:
            return (torch.zeros((1, 2, 2), dtype=torch.float32), torch.zeros((1, 2), dtype=torch.float32))
        out = {}
        req = requested or []
        if "chem_out" in req:
            out["chem_out"] = torch.zeros((1, 2), dtype=torch.float32)
        if "phase_tables" in req:
            out["phase_tables"] = (torch.zeros((1, 2, 2), dtype=torch.float32), torch.zeros((1, 2), dtype=torch.float32))
        if "component_moles" in req:
            out["component_moles"] = torch.zeros((1, 2), dtype=torch.float32)
        if "wt_del_component_moles" in req:
            out["wt_del_component_moles"] = torch.zeros((1, 2), dtype=torch.float32)
        return out

    fake.polish_masses = _polish

    features = torch.ones((1, 2), dtype=torch.float32)

    legacy = NN_MELTS.forwardMB(fake, features)
    assert isinstance(legacy, tuple)
    assert len(legacy) == 2
    assert fake._polish_calls == 1

    fake._polish_calls = 0
    selected = NN_MELTS.forwardMB(fake, features, outputs=["chem_out"])
    assert "chem_out" in selected
    assert fake._polish_calls == 1


def test_extract_bulk_properties_includes_entropy_cp_cv_alpha_gamma():
    required = ["control", "fort.56", "fort.61", "fort.68", "fort.99"]
    if not all((BENCHMARK_DIR / name).exists() for name in required):
        pytest.skip(f"Reader input tables unavailable at {BENCHMARK_DIR}")

    extracted = extract_bulk_properties_from_simulation_dir(str(BENCHMARK_DIR))

    assert "fort56_bulk" in extracted
    fort56_bulk = extracted["fort56_bulk"]

    fort56_df = pd.read_csv(BENCHMARK_DIR / "fort.56", sep=r"\s+", engine="python", skiprows=1)
    nrows = min(len(fort56_df), len(extracted["P(GPa)"]))
    fort56_df = fort56_df.iloc[:nrows].reset_index(drop=True)

    # Properties directly provided by fort.56 should match reader output closely.
    direct_map = {
        "entropy": "S(J/g/K)",
        "cp": "cp(J/g/K)",
        "alpha": "alpha(1e5_K^-1)",
    }
    for prop_name, fort56_col in direct_map.items():
        assert fort56_col in fort56_bulk, f"{fort56_col} missing from reader output"
        observed = np.asarray(fort56_bulk[fort56_col], dtype=np.float64)[:nrows]
        expected = pd.to_numeric(fort56_df[fort56_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        np.testing.assert_allclose(
            observed,
            expected,
            rtol=1.0e-6,
            atol=1.0e-8,
            err_msg=f"Mismatch for {prop_name} ({fort56_col})",
        )

    # cv and gamma may come from fort.56 (if present) or from physub matrix output.
    cv_vec = None
    gamma_vec = None

    if "Cv" in fort56_bulk:
        cv_vec = np.asarray(fort56_bulk["Cv"], dtype=np.float64)
    if "gamma" in fort56_bulk:
        gamma_vec = np.asarray(fort56_bulk["gamma"], dtype=np.float64)

    if (cv_vec is None or gamma_vec is None) and extracted.get("bulk_properties") is not None:
        bulk_names = tuple(extracted.get("bulk_property_names", tuple()))
        bulk_vals = np.asarray(extracted["bulk_properties"], dtype=np.float64)
        if cv_vec is None and "Cv" in bulk_names:
            cv_vec = bulk_vals[:, bulk_names.index("Cv")]
        if gamma_vec is None and "gamma" in bulk_names:
            gamma_vec = bulk_vals[:, bulk_names.index("gamma")]

    assert cv_vec is not None, "Cv not available from fort.56 or physub bulk output"
    assert gamma_vec is not None, "gamma not available from fort.56 or physub bulk output"
    assert cv_vec.shape[0] >= nrows
    assert gamma_vec.shape[0] >= nrows
    assert np.isfinite(cv_vec[:nrows]).all(), "Cv contains non-finite values"
    assert np.isfinite(gamma_vec[:nrows]).all(), "gamma contains non-finite values"
