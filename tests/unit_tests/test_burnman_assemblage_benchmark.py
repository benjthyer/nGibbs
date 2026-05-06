"""Benchmark Burnman-computed properties against HeFESTo ground truth.

Usage (from repository root):
python -m tests.unit_tests.test_burnman_assemblage_benchmark --sim-dir PATH/TO/SimulationDir

The script:
- loads fort.99 via `load_fort99_component_moles_and_labels`
- loads ground-truth bulk properties via `extract_bulk_properties_from_simulation_dir`
- computes Burnman properties using the HeFESToAPI.get_property_burnman_from_assemblage() method
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

src_path = str(Path(__file__).parent.parent.parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)
"""file_path = str(Path(__file__).parent)
if file_path not in sys.path:
    sys.path.insert(0, file_path)
base_path = str(Path(__file__).parent.parent.parent.parent)
if base_path not in sys.path:
    sys.path.insert(0, base_path)"""

# Import API and HeFESTo functions

from src.nMELTS.engine.API import HeFESToEmulatorGPU

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


def compare_and_report(burnman_out, burnman_names, hef_result):
    """Compare Burnman outputs against HeFESTo ground truth with unit conversion."""
    fort56 = hef_result.get('fort56_bulk', {})
    is_dict_output = isinstance(burnman_out, dict)
    burnman_array = None
    burnman_name_list = None

    if is_dict_output:
        for key, values in burnman_out.items():
            arr = np.asarray(values)
            if arr.ndim != 1:
                raise ValueError(
                    f"Expected 1D vector for burnman_out['{key}'], got shape {arr.shape}"
                )
    else:
        burnman_array = np.asarray(burnman_out)
        if burnman_array.ndim != 2:
            raise ValueError(
                f'Expected 2D burnman_out array, got shape {burnman_array.shape}'
            )
        if burnman_names is None:
            raise ValueError('burnman_names is required when burnman_out is an array.')
        burnman_name_list = list(burnman_names)

    def has_property(name: str) -> bool:
        if is_dict_output:
            return name in burnman_out
        return name in burnman_name_list

    def get_property(name: str) -> np.ndarray:
        if is_dict_output:
            return np.asarray(burnman_out[name], dtype=float)
        idx = burnman_name_list.index(name)
        return np.asarray(burnman_array[:, idx], dtype=float)

    mapping = []  # list of tuples (gt_label, burnman_prop_name, gt_array, converter_fn)

    # S: burnman 'S' (J/mol/K) -> J/kg/K via molar_mass, then -> J/g/K
    #if 'S(J/g/K)' in fort56 and has_property('entropy_by_mass'):

    mapping.append(
        ('S(J/g/K)', 'entropy_by_mass', fort56['S(J/g/K)'], lambda x: x / 1000.0)
    )

    # rho: burnman 'rho' (kg/m3) -> 'rho(g/cm^3)' (g/cm3 = kg/m3 / 1000)
    #if 'rho(g/cm^3)' in fort56 and has_property('density'):
    mapping.append(('rho(g/cm^3)', 'density', fort56['rho(g/cm^3)'], lambda x: x / 1000.0))

    # VP/VS: burnman 'v_p'/'v_s' (m/s) -> 'VP(km/s)'/'VS(km/s)' (km/s = m/s / 1000)
    #if 'VP(km/s)' in fort56 and has_property('p_wave_velocity'):
    mapping.append(('VP(km/s)', 'p_wave_velocity', fort56['VP(km/s)'], lambda x: x / 1000.0))
    #if 'VS(km/s)' in fort56 and has_property('s_wave_velocity'):
    mapping.append(('VS(km/s)', 's_wave_velocity', fort56['VS(km/s)'], lambda x: x / 1000.0))

    #if 'KT(GPa)' in fort56 and has_property('isothermal_bulk_modulus_reuss'):
    mapping.append(('KS(GPa)', 'isentropic_bulk_modulus_reuss', fort56['KS(GPa)'], lambda x : x / 1E9))

    if len(mapping) == 0:
        print('No overlapping properties found between Burnman outputs and HeFESTo fort.56 ground truth.')
        return

    report = []
    print(mapping)
    for (gt_label, burnman_prop, gt_arr, conv) in mapping:
        gt = np.asarray(gt_arr, dtype=float)
        pred_raw = get_property(burnman_prop)
        pred = conv(pred_raw)
        mask = np.isfinite(pred) & np.isfinite(gt)
        if not np.any(mask):
            print(f'No valid data for {gt_label}!')
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
    p = argparse.ArgumentParser(description='Benchmark Burnman vs HeFESTo outputs via HeFESToAPI')
    p.add_argument('--sim-dir', required=True, help='Path to a HeFESTo Simulation directory (contains fort.99 etc)')
    args = p.parse_args()



    hef = load_hefesto_module()

    sim_dir = Path(args.sim_dir)
    if not sim_dir.exists():
        print('Simulation directory not found:', sim_dir)
        sys.exit(2)

    # Load ground truth
    gt = hef.extract_bulk_properties_from_simulation_dir(str(sim_dir), repeat=10)

    # Get indexer from emulator
    indexer = HeFESToEmulatorGPU.isentropic_emulator.ml_indexer
    comp_names = gt['component_names']

    # Load P, VC using the helper function
    #VC, P = hef.load_fort99_component_moles_and_labels(str(sim_dir), indexer)
    componentMoles = hef.load_fort99_componentMoles(str(sim_dir), indexer)
    componentMoles = np.repeat(componentMoles, 10, axis=0)
    print(componentMoles.shape)
    # Build PT matrix from fort56
    fort56 = gt['fort56_bulk']
    P_gpa = fort56['P(GPa)']
    T_k = fort56['T(K)']
    PT = np.stack([P_gpa*1E9, T_k], axis=1)
    print(PT.shape)

    # Use the API method to compute Burnman properties
    print('Computing Burnman properties via HeFESToAPI.get_property_burnman_vectorized_from_assemblage()...')
    property_names = ['entropy_by_mass', 'density', 'p_wave_velocity', 's_wave_velocity', 'isentropic_bulk_modulus_reuss']

    burnman_out, returnednames = HeFESToEmulatorGPU.get_property_burnman_vectorized_from_assemblage(
       torch.tensor(componentMoles, dtype=torch.float64, device='cuda'), torch.tensor(PT, dtype=torch.float64, device='cuda'), 
       property_names= property_names
    )

    """# Use the API method to compute Burnman properties
    print('Computing Burnman properties via HeFESToAPI.get_property_burnman_from_assemblage()...')
    property_names = ['S', 'rho', 'v_p', 'v_s', 'molar_mass', 'K_T']
    burnman_out, returnednames = HeFESToEmulatorCPU.get_property_burnman_from_assemblage(
       VC, P, PT, property_names=property_names
    )"""

    compare_and_report(burnman_out, returnednames, gt)


if __name__ == '__main__':
    main()
