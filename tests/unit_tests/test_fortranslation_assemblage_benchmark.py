"""Benchmark Fortranslation-computed properties against HeFESTo ground truth.

Usage (from repository root):
python -m tests.unit_tests.test_fortranslation_assemblage_benchmark --sim-dir PATH/TO/SimulationDir

The script:
- loads fort.99 via `load_fort99_component_moles_and_labels`
- loads ground-truth bulk properties via `extract_bulk_properties_from_simulation_dir`
- computes Fortranslation properties using the HeFESToAPI.get_property_fortranslation_from_assemblage() method
- compares rho, VP, VS, S when available (unit-converted) and prints errors
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import numpy as np
import torch
import importlib.util
import sys
import pandas as pd

src_path = str(Path(__file__).parent.parent.parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)
file_path = str(Path(__file__).parent)
if file_path not in sys.path:
    sys.path.insert(0, file_path)
base_path = str(Path(__file__).parent.parent.parent.parent)
if base_path not in sys.path:
    sys.path.insert(0, base_path)


def load_hefesto_module():
    """Load HeFESTo_functions module."""
    repo_root = Path(__file__).resolve().parents[2]
    mod_path = repo_root / 'src' / 'builder' / 'HeFESTo' / 'HeFESTo_functions.py'
    if not mod_path.exists():
        raise FileNotFoundError(f'HeFESTo functions module not found: {mod_path}')
    spec = importlib.util.spec_from_file_location('hefesto_funcs', str(mod_path))
    module = importlib.util.module_from_spec(spec)
    loader = spec.loader
    assert loader is not None
    loader.exec_module(module)
    return module


"""""density",
    "K_Reuss",
    "K_Voigt",
    "K_Hill",
    "G_Reuss",
    "G_Voigt",
    "G_Hill",
    "Vb",
    "Vs",
    "Vp",
    "Vd",
    "Cp",
    "Cv",
    "alpha",
    "gamma",
    "entropy",
    "enthalpy",
    "Gibbs","""

def compare_and_report(physub_out, physub_names, hef_result):
    """Compare Fortranslation outputs against HeFESTo ground truth with unit conversion."""
    fort56 = hef_result.get('fort56_bulk', {})
    mapping = []  # list of tuples (label, fortranslation_col_index, gt_array, conversion_fn)

    # S: fortran 'S(J/g/K)'
    if 'S(J/g/K)' in fort56 and 'entropy' in physub_names:
        idx = physub_names.index('entropy')
        #MM_idx = physub_names.index('molar_mass') if 'molar_mass' in physub_names else None
        #physub_out[:,idx] /= physub_out[:,MM_idx] # (J/mol/K) -> (J/kg/K)
        mapping.append(('S(J/g/K)', idx, fort56['S(J/g/K)'], lambda x: x))

    # rho: burnman 'rho(g/cm^3)'
    if 'rho(g/cm^3)' in fort56 and 'density' in physub_names:
        idx = physub_names.index('density')
        mapping.append(('rho(g/cm^3)', idx, fort56['rho(g/cm^3)'], lambda x: x))

    # VP/VS: burnman 'v_p'/'v_s' (m/s) -> 'VP(km/s)'/'VS(km/s)' (km/s = m/s / 1000)
    if 'VP(km/s)' in fort56 and 'Vp' in physub_names:
        idx = physub_names.index('Vp')
        mapping.append(('VP(km/s)', idx, fort56['VP(km/s)'], lambda x: x))
    if 'VS(km/s)' in fort56 and 'Vs' in physub_names:
        idx = physub_names.index('Vs')
        mapping.append(('VS(km/s)', idx, fort56['VS(km/s)'], lambda x:x))

    """if 'KT(GPa)' in fort56 and 'K_T' in physub_names:
        idx = physub_names.index('K_T')
        mapping.append(('KT(GPa)', idx, fort56['KT(GPa)'], lambda x: x/1E9))"""

    if len(mapping) == 0:
        print('No overlapping properties found between Burnman outputs and HeFESTo fort.56 ground truth.')
        return

    report = []
    for (gt_label, b_idx, gt_arr, conv) in mapping:
        gt = np.asarray(gt_arr, dtype=float)
        pred_raw = physub_out[:, b_idx]
        pred = conv(pred_raw)
        mask = np.isfinite(pred) & np.isfinite(gt)
        if not np.any(mask):
            continue
        diff = pred[mask] - gt[mask]
        mae = np.mean(np.abs(diff))
        rmse = math.sqrt(np.mean(diff ** 2))
        mre = np.mean(np.abs(diff / np.clip(gt[mask], 1e-12, None))) * 100.0
        report.append((gt_label, mae, rmse, mre, np.sum(mask)))

    print('\nBenchmark results:')
    print('Property | MAE | RMSE | MeanRelErr% | N')
    print('-' * 60)
    for row in report:
        print(f"{row[0]:10s} | {row[1]:.6g} | {row[2]:.6g} | {row[3]:.3f}% | {row[4]}")


def main():
    p = argparse.ArgumentParser(description='Benchmark Physub vs HeFESTo outputs via HeFESToAPI')
    p.add_argument('--sim-dir', required=True, help='Path to a HeFESTo Simulation directory (contains fort.99 etc)')
    args = p.parse_args()

    # Import API and HeFESTo functions
    try:
        from src.module.engine.API import HeFESToEmulatorCPU
        if HeFESToEmulatorCPU is None:
            raise RuntimeError('HeFESToEmulatorCPU not available (models not found at import)')
    except ImportError as exc:
        raise RuntimeError(f'Failed to import HeFESToEmulatorCPU: {exc}')

    hef = load_hefesto_module()

    sim_dir = Path(args.sim_dir)
    if not sim_dir.exists():
        print('Simulation directory not found:', sim_dir)
        sys.exit(2)

    # Load ground truth
    gt = hef.extract_bulk_properties_from_simulation_dir(str(sim_dir))

    # Get indexer from emulator
    indexer = HeFESToEmulatorCPU.isentropic_emulator.ml_indexer
    comp_names = gt['component_names']

    # Load P, VC using the helper function
    componentMoles = hef.load_fort99_componentMoles(str(sim_dir), indexer)

    # Build PT matrix from fort56
    fort56 = gt['fort56_bulk']
    P_gpa = fort56['P(GPa)']
    T_k = fort56['T(K)']
    PT = np.stack([P_gpa, T_k], axis=1)


    # Use the API method to compute Fortranslation properties
    print('Computing Fortranslation properties via HeFESToAPI.get_property_fortranslation_from_assemblage()...')
    #property_names = ['S', 'rho', 'v_p', 'v_s', 'molar_mass', 'K_T']
    fort_out = HeFESToEmulatorCPU.get_property_fortranslation_from_assemblage(
        componentMoles=componentMoles,
        PT=PT,
    )

    fort_tables = pd.DataFrame(fort_out).to_numpy()

    compare_and_report(fort_tables, list(fort_out.keys()), gt)


if __name__ == '__main__':
    main()
