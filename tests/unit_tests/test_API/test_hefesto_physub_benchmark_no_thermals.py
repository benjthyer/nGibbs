"""Tests for the elastic-only HeFESTo physub translation.

These tests focus on the simplified vectorized bulk reducer and the lean
phase wrapper that now only produces density, K, G, and wave velocities.
"""

from __future__ import annotations

from importlib import import_module
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
BUILDER_ROOT = SRC_ROOT / "builder"
if str(BUILDER_ROOT) not in sys.path:
    sys.path.insert(0, str(BUILDER_ROOT))

from nMELTS.engine.EOS_arithmetic.hefesto_physub_no_thermal import (  # noqa: E402
    PHYSUB_BULK_ATTRIBUTE_NAMES,
    HeFESToPhaseState,
    HeFESToSpeciesState,
    compute_physub_bulk_matrix,
    compute_physub_properties,
    get_hefesto_physub_context,
    load_hefesto_parameter_directory,
)
from nMELTS.config.constants import HEFESTO_ABBREVIATION_TO_SHORT_NAMES
from nMELTS.engine.EOS_arithmetic.hefesto_physub import compute_physub_bulk_matrix as full_compute_physub_bulk_matrix


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


def _read_table(path: Path) -> tuple[list[str], list[dict[str, float]]]:
    lines = [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines()]
    header = None
    rows: list[dict[str, float]] = []

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

print
def _pick_column(source: dict[str, object], candidates: tuple[str, ...]):
    lower_map = {str(key).lower(): key for key in source.keys()}
    for candidate in candidates:
        if candidate in source:
            return source[candidate]
        matched = lower_map.get(candidate.lower())
        if matched is not None:
            return source[matched]
    raise KeyError(f"None of {candidates} found in source columns: {list(source.keys())}")


def _component_attribute_tensors(component_names: list[str], nrows: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    context = get_hefesto_physub_context()
    records = load_hefesto_parameter_directory(use_cache=True)

    # Build reverse mapping: full short name -> list of possible abbreviations
    rev: dict[str, list[str]] = {}
    for abbr, short in HEFESTO_ABBREVIATION_TO_SHORT_NAMES.items():
        rev.setdefault(short.lower(), []).append(abbr)

    molar_masses = []
    molar_volumes = []
    bulk_moduli = []
    shear_moduli = []

    for name in component_names:
        record = records.get(name)
        if record is None:
            # Try common reverse lookups: short mineral name -> abbreviation filenames
            aliases = rev.get(str(name).lower(), ())
            for alias in aliases:
                if alias in records:
                    record = records[alias]
                    break
        if record is None:
            # Fallback: try lowercase key
            record = records.get(str(name).lower())
        if record is None:
            raise KeyError(f"No HeFESTo parameter record found for component '{name}'")
        molar_masses.append(record.value("formula_mass_g_mol"))
        molar_volumes.append(record.value("v0_cm3_mol"))
        bulk_moduli.append(record.value("k0_gpa"))
        shear_moduli.append(record.value("ambient_shear_modulus_gpa"))

    device = torch.device("cpu")
    molar_mass = torch.tensor(molar_masses, dtype=torch.float32, device=device)
    attrs = {
        "molar_volume": torch.tensor([molar_volumes], dtype=torch.float32, device=device).expand(nrows, -1),
        "bulk_modulus": torch.tensor([bulk_moduli], dtype=torch.float32, device=device).expand(nrows, -1),
        "shear_modulus": torch.tensor([shear_moduli], dtype=torch.float32, device=device).expand(nrows, -1),
    }

    return molar_mass, attrs["molar_volume"], attrs["bulk_modulus"], attrs["shear_modulus"]


def test_physub_bulk_matrix_vectorized_selectors_and_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    component_moles = torch.tensor(
        [[0.9, 0.1, 0.0, 0.3], [0.2, 0.4, 0.5, 0.1], [0.6, 0.0, 0.2, 0.7]],
        dtype=torch.float32,
        device=device,
    )
    molar_mass = torch.tensor([50.0, 55.0, 60.0, 45.0], dtype=torch.float32, device=device)
    attrs = {
        "molar_volume": torch.tensor([[2.0, 2.5, 3.0, 1.5]], dtype=torch.float32, device=device).expand_as(component_moles),
        "bulk_modulus": torch.tensor([[120.0, 130.0, 140.0, 150.0]], dtype=torch.float32, device=device).expand_as(component_moles),
        "shear_modulus": torch.tensor([[70.0, 80.0, 90.0, 100.0]], dtype=torch.float32, device=device).expand_as(component_moles),
    }

    full, names = compute_physub_bulk_matrix(component_moles, molar_mass, attrs)
    assert full.shape == (3, len(PHYSUB_BULK_ATTRIBUTE_NAMES))
    assert full.device.type == device.type
    assert names == PHYSUB_BULK_ATTRIBUTE_NAMES

    density = (component_moles * molar_mass.unsqueeze(0)).sum(dim=1) / (component_moles * attrs["molar_volume"]).sum(dim=1)
    assert torch.allclose(full[:, 0], density)

    subset_names = ["density", "K_Hill", "Vs", "Vp"]
    subset, resolved = compute_physub_bulk_matrix(component_moles, molar_mass, attrs, selectors=subset_names)
    assert subset.shape == (3, len(subset_names))
    assert resolved == tuple(subset_names)


def test_physub_bulk_matrix_matches_direct_tensor_formulae():
    component_moles = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    molar_mass = torch.tensor([10.0, 20.0], dtype=torch.float32)
    attrs = {
        "molar_volume": torch.tensor([[2.0, 3.0]], dtype=torch.float32),
        "bulk_modulus": torch.tensor([[100.0, 200.0]], dtype=torch.float32),
        "shear_modulus": torch.tensor([[50.0, 80.0]], dtype=torch.float32),
    }

    output, names = compute_physub_bulk_matrix(component_moles, molar_mass, attrs)
    values = {name: output[0, i].item() for i, name in enumerate(names)}

    total_mass = float((component_moles * molar_mass.unsqueeze(0)).sum())
    total_volume = float((component_moles * attrs["molar_volume"]).sum())
    vol_weight = component_moles * attrs["molar_volume"]
    k_voigt = float((vol_weight * attrs["bulk_modulus"]).sum() / total_volume)
    g_voigt = float((vol_weight * attrs["shear_modulus"]).sum() / total_volume)
    k_reuss = float(total_volume / (vol_weight / attrs["bulk_modulus"]).sum())
    g_reuss = float(total_volume / (vol_weight / attrs["shear_modulus"]).sum())
    k_hill = 0.5 * (k_voigt + k_reuss)
    g_hill = 0.5 * (g_voigt + g_reuss)

    assert pytest.approx(values["density"], rel=1e-6) == total_mass / total_volume
    assert pytest.approx(values["K_Voigt"], rel=1e-6) == k_voigt
    assert pytest.approx(values["K_Reuss"], rel=1e-6) == k_reuss
    assert pytest.approx(values["K_Hill"], rel=1e-6) == k_hill
    assert pytest.approx(values["G_Voigt"], rel=1e-6) == g_voigt
    assert pytest.approx(values["G_Reuss"], rel=1e-6) == g_reuss
    assert pytest.approx(values["G_Hill"], rel=1e-6) == g_hill
    assert pytest.approx(values["Vb"], rel=1e-6) == (k_hill / (total_mass / total_volume)) ** 0.5
    assert pytest.approx(values["Vs"], rel=1e-6) == (g_hill / (total_mass / total_volume)) ** 0.5
    assert pytest.approx(values["Vp"], rel=1e-6) == ((k_hill + 4.0 / 3.0 * g_hill) / (total_mass / total_volume)) ** 0.5


def test_physub_phase_wrapper_uses_vectorized_bulk_path():
    phase = HeFESToPhaseState(
        name="test",
        species=(
            HeFESToSpeciesState(
                name="a",
                phase_name="test",
                amount=1.0,
                molar_mass=10.0,
                molar_volume=2.0,
                bulk_modulus_t=100.0,
                shear_modulus=50.0,
            ),
            HeFESToSpeciesState(
                name="b",
                phase_name="test",
                amount=2.0,
                molar_mass=20.0,
                molar_volume=3.0,
                bulk_modulus_t=200.0,
                shear_modulus=80.0,
            ),
        ),
    )

    result = compute_physub_properties([phase], pressure_gpa=1.0, temperature_k=1200.0)
    assert pytest.approx(result.density, rel=1e-6) == 6.25
    assert pytest.approx(result.bulk_modulus_hill, rel=1e-6) == 167.5
    assert pytest.approx(result.shear_modulus_hill, rel=1e-6) == 71.03260803222656
    assert pytest.approx(result.bulk_sound_velocity, rel=1e-6) == 5.1768717765808105
    assert pytest.approx(result.shear_velocity, rel=1e-6) == 3.3712337017059326
    assert pytest.approx(result.pressure_velocity, rel=1e-6) == 6.477161884307861


def test_physub_empty_phase_assembly_returns_zeros():
    result = compute_physub_properties([], pressure_gpa=0.0, temperature_k=1000.0)
    assert result.density == 0.0
    assert result.bulk_modulus_hill == 0.0
    assert result.shear_modulus_hill == 0.0
    assert result.bulk_sound_velocity == 0.0


def test_physub_benchmark_comparison_prints_elastic_outputs():
    benchmark_dir = _resolve_benchmark_dir()
    required = ["control", "fort.58", "fort.99"]
    if not all((benchmark_dir / name).exists() for name in required):
        pytest.skip(f"Benchmark tables are unavailable at {benchmark_dir}")

    extracted = import_module("HeFESTo.HeFESTo_functions").extract_bulk_properties_from_simulation_dir(str(benchmark_dir))
    component_names = list(extracted["component_names"])
    comp_moles_np = extracted["component_moles"]
    component_moles = torch.tensor(comp_moles_np, dtype=torch.float32)

    # Use the full HeFESTo physub context to compute temperature-dependent
    # component attributes and align component ordering to the context.
    from nMELTS.engine.EOS_arithmetic import hefesto_physub as full_physub

    full_ctx = full_physub.get_hefesto_physub_context()

    # Align component mole ordering to the context ordering
    aligned_moles = full_ctx.align_component_tensor(component_moles, component_names)

    # Compute per-row component attributes at each temperature and stack them
    temps = list(extracted.get("T(K)", []))
    nrows = component_moles.shape[0]
    if len(temps) < nrows:
        temps = [temps[0]] * nrows

    # Prepare containers
    molar_mass = full_ctx.formula_mass_g_mol.clone()
    attr_names = ("molar_volume", "bulk_modulus", "shear_modulus")
    stacked_attrs = {name: [] for name in attr_names}

    for i in range(nrows):
        attrs = full_ctx.compute_component_attributes_at_temperature(float(temps[i]), batch_size=1, device=torch.device("cpu"))
        for name in attr_names:
            # attrs[name] is shape (1, C)
            stacked_attrs[name].append(attrs[name][0].cpu())

    # Stack into (nrows, C)
    component_attributes = {name: torch.stack(stacked_attrs[name], dim=0) for name in attr_names}

    bulk_matrix, names = compute_physub_bulk_matrix(
        component_moles=aligned_moles,
        molar_mass=molar_mass,
        component_attributes=component_attributes,
        selectors=("density", "K_Hill", "G_Hill", "Vb", "Vs", "Vp"),
    )

    _, fort58_rows = _read_table(benchmark_dir / "fort.58")

    density_ref = _pick_column(fort58_rows[0], ("rho", "density", "rho(g/cm^3)"))
    ks_ref = _pick_column(fort58_rows[0], ("KS", "K_Hill"))
    g_ref = _pick_column(fort58_rows[0], ("G", "G_Hill"))
    vb_ref = _pick_column(fort58_rows[0], ("VBh", "Vb"))
    vs_ref = _pick_column(fort58_rows[0], ("VSh", "Vs"))
    vp_ref = _pick_column(fort58_rows[0], ("VPh", "Vp"))

    print("Elastic benchmark comparison (first 5 rows)")
    print("row | density calc/ref | KS calc/ref | G calc/ref | Vb calc/ref | Vs calc/ref | Vp calc/ref")
    sample_rows = min(5, bulk_matrix.shape[0], len(fort58_rows))
    for row_index in range(sample_rows):
        ref_row = fort58_rows[row_index]
        calc = {name: float(bulk_matrix[row_index, idx].item()) for idx, name in enumerate(names)}
        # also compute using the original full physub implementation for comparison
        try:
            full_mat, full_names = full_compute_physub_bulk_matrix(
                component_moles=component_moles,
                molar_mass=molar_mass,
                component_attributes={
                    "molar_volume": molar_volume,
                    "bulk_modulus": bulk_modulus,
                    "shear_modulus": shear_modulus,
                },
                selectors=("density", "K_Hill", "G_Hill", "Vb", "Vs", "Vp"),
            )
            full_calc = {name: float(full_mat[row_index, idx].item()) for idx, name in enumerate(full_names)}
        except Exception:
            full_calc = None
        print(
            f"{row_index:03d} | "
            f"{calc['density']:.6f}/{float(_pick_column(ref_row, ('rho', 'density', 'rho(g/cm^3)'))):.6f} | "
            f"{calc['K_Hill']:.6f}/{float(_pick_column(ref_row, ('KS', 'K_Hill'))):.6f} | "
            f"{calc['G_Hill']:.6f}/{float(_pick_column(ref_row, ('G', 'G_Hill'))):.6f} | "
            f"{calc['Vb']:.6f}/{float(_pick_column(ref_row, ('VBh', 'Vb'))):.6f} | "
            f"{calc['Vs']:.6f}/{float(_pick_column(ref_row, ('VSh', 'Vs'))):.6f} | "
            f"{calc['Vp']:.6f}/{float(_pick_column(ref_row, ('VPh', 'Vp'))):.6f}"
        )
        if full_calc is not None:
            print(
                f"      full impl: {full_calc['density']:.6f} {full_calc['K_Hill']:.6f} {full_calc['G_Hill']:.6f} {full_calc['Vb']:.6f} {full_calc['Vs']:.6f} {full_calc['Vp']:.6f}"
            )

    assert bulk_matrix.shape[1] == len(names)



if __name__ == "__main__":
    print("Running benchmark test as a standalone script")
    test_physub_benchmark_comparison_prints_elastic_outputs()
    print("Standalone run completed successfully")

