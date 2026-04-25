"""Benchmark comparison test for the Python physub translation.

This test reconstructs a simplified species state from HeFESTo BENCHMARK
tables, evaluates the Python implementation, and compares key bulk outputs
against ground truth values from fort.58.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nMELTS.engine.EOS_arithmetic.hefesto_physub import (  # noqa: E402
    HeFESToPhaseState,
    HeFESToSpeciesState,
    compute_physub_properties,
    load_hefesto_parameter_directory,
)


BENCHMARK_DIR = SRC_ROOT / "nMELTS" / "engine" / "EOS_arithmetic" / "BENCHMARK"


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

    if current:
        blocks.append(current)

    return blocks


def _build_phase_states_for_row(
    row_index: int,
    phase_to_species: Dict[str, List[str]],
    species_moles_rows: List[Dict[str, float]],
    phase_density_rows: List[Dict[str, float]],
    fort63_rows: List[Dict[str, float]],
    fort64_rows: List[Dict[str, float]],
    dndt_dndp_blocks: List[Dict[str, Tuple[float, float]]],
):
    param_records = load_hefesto_parameter_directory()

    species_row = species_moles_rows[row_index]
    density_row = phase_density_rows[row_index]
    vb_row = fort63_rows[row_index]
    vs_row = fort64_rows[row_index]
    derivative_block = dndt_dndp_blocks[row_index]

    phase_states: List[HeFESToPhaseState] = []

    for phase_name, species_names in phase_to_species.items():
        phase_moles = sum(species_row.get(species, 0.0) for species in species_names)
        if phase_moles <= 0.0:
            continue

        phase_mass = 0.0
        for species in species_names:
            amount = species_row.get(species, 0.0)
            if amount <= 0.0:
                continue
            phase_mass += amount * param_records[species].value("formula_mass_g_mol")

        if phase_mass <= 0.0:
            continue

        phase_density = density_row.get(f"rh{phase_name}", 0.0)
        if phase_density <= 0.0:
            continue

        phase_volume = phase_mass / phase_density
        vb = vb_row.get(f"vb{phase_name}", 0.0)
        vs = vs_row.get(f"vs{phase_name}", 0.0)

        if vb == 0.0 or vs == 0.0:
            continue

        bulk_modulus = phase_density * vb * vb
        shear_modulus = phase_density * vs * vs
        molar_volume = phase_volume / phase_moles

        species_states: List[HeFESToSpeciesState] = []
        for species in species_names:
            amount = species_row.get(species, 0.0)
            if amount <= 0.0:
                continue

            dndt, dndp = derivative_block.get(species, (0.0, 0.0))
            species_states.append(
                HeFESToSpeciesState(
                    name=species,
                    phase_name=phase_name,
                    amount=amount,
                    molar_mass=param_records[species].value("formula_mass_g_mol"),
                    molar_volume=molar_volume,
                    density=phase_density,
                    bulk_modulus_t=bulk_modulus,
                    bulk_modulus_s=bulk_modulus,
                    shear_modulus=shear_modulus,
                    heat_capacity_p=0.0,
                    heat_capacity_v=0.0,
                    thermal_expansivity=0.0,
                    entropy=0.0,
                    enthalpy=0.0,
                    gibbs=0.0,
                    dndt=dndt,
                    dndp=dndp,
                )
            )

        if species_states:
            phase_states.append(HeFESToPhaseState(name=phase_name, species=tuple(species_states)))

    return phase_states


def test_physub_python_matches_benchmark_tables():
    _vprint("=" * 88)
    _vprint("Starting benchmark comparison for Python physub translation")
    _vprint(f"Benchmark directory: {BENCHMARK_DIR}")
    _vprint("=" * 88)

    species_order, _, phase_to_species = _parse_control_species_map(BENCHMARK_DIR / "control")
    _vprint(f"Loaded control species map: {len(species_order)} species across {len(phase_to_species)} phases")

    fort58_header, fort58_rows = _read_table(BENCHMARK_DIR / "fort.58")
    fort61_header, fort61_rows = _read_table(BENCHMARK_DIR / "fort.61")
    fort62_header, fort62_rows = _read_table(BENCHMARK_DIR / "fort.62")
    fort63_header, fort63_rows = _read_table(BENCHMARK_DIR / "fort.63")
    fort99_header, fort99_rows = _read_table(BENCHMARK_DIR / "fort.99")
    dndt_dndp_blocks = _parse_fort42_blocks(BENCHMARK_DIR / "fort.42", species_order)

    _vprint(f"fort.58 columns={len(fort58_header)} rows={len(fort58_rows)}")
    _vprint(f"fort.61 columns={len(fort61_header)} rows={len(fort61_rows)}")
    _vprint(f"fort.62 columns={len(fort62_header)} rows={len(fort62_rows)}")
    _vprint(f"fort.63 columns={len(fort63_header)} rows={len(fort63_rows)}")
    _vprint(f"fort.99 columns={len(fort99_header)} rows={len(fort99_rows)}")
    _vprint(f"fort.42 derivative blocks={len(dndt_dndp_blocks)}")

    n_rows = min(
        len(fort58_rows),
        len(fort61_rows),
        len(fort62_rows),
        len(fort63_rows),
        len(fort99_rows),
        len(dndt_dndp_blocks),
    )
    _vprint(f"Rows used for comparison (min common rows): {n_rows}")

    assert n_rows > 0, "No common benchmark rows were found across required fort.* files"

    density_rel_err: List[float] = []
    ks_rel_err: List[float] = []
    g_rel_err: List[float] = []
    vb_rel_err: List[float] = []
    vs_rel_err: List[float] = []
    vp_rel_err: List[float] = []

    for row_index in range(n_rows):
        target = fort58_rows[row_index]
        phase_states = _build_phase_states_for_row(
            row_index=row_index,
            phase_to_species=phase_to_species,
            species_moles_rows=fort99_rows,
            phase_density_rows=fort61_rows,
            fort63_rows=fort62_rows,
            fort64_rows=fort63_rows,
            dndt_dndp_blocks=dndt_dndp_blocks,
        )

        phase_count = len(phase_states)
        species_count = sum(len(phase.species) for phase in phase_states)

        output = compute_physub_properties(
            phases=phase_states,
            pressure_gpa=target["Pi"],
            temperature_k=target["Ti"],
            depth_km=target["depth"],
        )

        d_err = abs(output.density - target["rho"]) / max(abs(target["rho"]), 1.0e-12)
        ks_err = abs(output.bulk_modulus_hill - target["KS"]) / max(abs(target["KS"]), 1.0e-12)
        g_err = abs(output.shear_modulus_hill - target["G"]) / max(abs(target["G"]), 1.0e-12)
        vb_err = abs(output.bulk_sound_velocity - target["VBh"]) / max(abs(target["VBh"]), 1.0e-12)
        vs_err = abs(output.shear_velocity - target["VSh"]) / max(abs(target["VSh"]), 1.0e-12)
        vp_err = abs(output.pressure_velocity - target["VPh"]) / max(abs(target["VPh"]), 1.0e-12)

        density_rel_err.append(d_err)
        ks_rel_err.append(ks_err)
        g_rel_err.append(g_err)
        vb_rel_err.append(vb_err)
        vs_rel_err.append(vs_err)
        vp_rel_err.append(vp_err)

        _vprint(
            " | ".join(
                [
                    f"row={row_index:03d}",
                    f"P={target['Pi']:.3f} GPa",
                    f"T={target['Ti']:.2f} K",
                    f"depth={target['depth']:.2f} km",
                    f"phases={phase_count}",
                    f"species={species_count}",
                ]
            )
        )
        _vprint(
            "    "
            + " | ".join(
                [
                    f"rho py={output.density:.6f} ref={target['rho']:.6f} rel={d_err:.4e}",
                    f"KS py={output.bulk_modulus_hill:.6f} ref={target['KS']:.6f} rel={ks_err:.4e}",
                    f"G py={output.shear_modulus_hill:.6f} ref={target['G']:.6f} rel={g_err:.4e}",
                ]
            )
        )
        _vprint(
            "    "
            + " | ".join(
                [
                    f"VBh py={output.bulk_sound_velocity:.6f} ref={target['VBh']:.6f} rel={vb_err:.4e}",
                    f"VSh py={output.shear_velocity:.6f} ref={target['VSh']:.6f} rel={vs_err:.4e}",
                    f"VPh py={output.pressure_velocity:.6f} ref={target['VPh']:.6f} rel={vp_err:.4e}",
                ]
            )
        )

    mean_density = float(np.mean(density_rel_err))
    mean_ks = float(np.mean(ks_rel_err))
    mean_g = float(np.mean(g_rel_err))
    mean_vb = float(np.mean(vb_rel_err))
    mean_vs = float(np.mean(vs_rel_err))
    mean_vp = float(np.mean(vp_rel_err))

    max_density = float(np.max(density_rel_err))
    max_ks = float(np.max(ks_rel_err))
    max_g = float(np.max(g_rel_err))
    max_vb = float(np.max(vb_rel_err))
    max_vs = float(np.max(vs_rel_err))
    max_vp = float(np.max(vp_rel_err))

    worst_density_idx = int(np.argmax(density_rel_err))
    worst_ks_idx = int(np.argmax(ks_rel_err))
    worst_g_idx = int(np.argmax(g_rel_err))
    worst_vb_idx = int(np.argmax(vb_rel_err))
    worst_vs_idx = int(np.argmax(vs_rel_err))
    worst_vp_idx = int(np.argmax(vp_rel_err))

    _vprint("-" * 88)
    _vprint("Benchmark comparison summary")
    _vprint(
        f"density mean={mean_density:.4e} max={max_density:.4e} worst_row={worst_density_idx}"
    )
    _vprint(f"KS      mean={mean_ks:.4e} max={max_ks:.4e} worst_row={worst_ks_idx}")
    _vprint(f"G       mean={mean_g:.4e} max={max_g:.4e} worst_row={worst_g_idx}")
    _vprint(f"VBh     mean={mean_vb:.4e} max={max_vb:.4e} worst_row={worst_vb_idx}")
    _vprint(f"VSh     mean={mean_vs:.4e} max={max_vs:.4e} worst_row={worst_vs_idx}")
    _vprint(f"VPh     mean={mean_vp:.4e} max={max_vp:.4e} worst_row={worst_vp_idx}")
    _vprint("Thresholds: density<0.12 KS<0.20 G<0.20 VBh<0.10 VSh<0.10 VPh<0.10 (means)")
    _vprint("=" * 88)

    assert mean_density < 0.12, f"density mean relative error too high: {mean_density:.4e}"
    assert mean_ks < 0.20, f"KS mean relative error too high: {mean_ks:.4e}"
    assert mean_g < 0.20, f"G mean relative error too high: {mean_g:.4e}"
    assert mean_vb < 0.10, f"VBh mean relative error too high: {mean_vb:.4e}"
    assert mean_vs < 0.10, f"VSh mean relative error too high: {mean_vs:.4e}"
    assert mean_vp < 0.10, f"VPh mean relative error too high: {mean_vp:.4e}"


if __name__ == "__main__":
    _vprint("Running benchmark test as a standalone script")
    test_physub_python_matches_benchmark_tables()
    _vprint("Standalone run completed successfully")
