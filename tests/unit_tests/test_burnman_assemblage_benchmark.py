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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

from src.nMELTS.engine.API import HeFESToEmulatorCPU, HeFESToEmulatorGPU

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


def compare_and_report(burnman_out, hef_result, save_dir: "Path | None" = None):
    """Compare Burnman outputs against HeFESTo ground truth with unit conversion."""
    fort56 = hef_result.get('fort56_bulk', {})
    is_dict_output = isinstance(burnman_out, dict)
    burnman_array = None
    burnman_name_list = None
    burnman_names = list(burnman_out.keys())

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

    # Cp: burnman heat_capacity_p_by_mass (J/kg/K) -> J/g/K (÷1000) vs fort56 cp(J/g/K)
    mapping.append(('cp(J/g/K)', 'heat_capacity_p_by_mass', fort56['cp(J/g/K)'], lambda x: x / 1000.0))

    # Cv: Burnman heat_capacity_v_by_mass (J/kg/K) -> J/g/K (÷1000)
    # GT Cv derived from fort56 via the exact identity: Cv = Cp² / (Cp + T*V*alpha²*Ks)
    # (all per-mass, consistent units; avoids needing K_T from fort56)
    _cp_gt  = np.asarray(fort56['cp(J/g/K)'],        dtype=float) * 1e3   # J/kg/K
    _T_gt   = np.asarray(fort56['T(K)'],              dtype=float)
    _rho_gt = np.asarray(fort56['rho(g/cm^3)'],       dtype=float) * 1e3   # kg/m³
    _a_gt   = np.asarray(fort56['alpha(1e5_K^-1)'],   dtype=float) * 1e-5  # 1/K
    _Ks_gt  = np.asarray(fort56['KS(GPa)'],           dtype=float) * 1e9   # Pa
    _cv_gt  = _cp_gt ** 2 / (_cp_gt + _T_gt * (_a_gt ** 2) * _Ks_gt / _rho_gt) / 1e3  # J/g/K
    mapping.append(('cv(J/g/K)', 'heat_capacity_v_by_mass', _cv_gt, lambda x: x / 1000.0))

    if len(mapping) == 0:
        print('No overlapping properties found between Burnman outputs and HeFESTo fort.56 ground truth.')
        return

    report = []
    print(mapping)
    for (gt_label, burnman_prop, gt_arr, conv) in mapping:
        gt = np.asarray(gt_arr, dtype=float)
        pred_raw = burnman_out[burnman_prop]
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

    if save_dir is None:
        return

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Collect arrays and labels for the valid properties
    P_gpa = np.asarray(fort56['P(GPa)'], dtype=float)
    S_jgk = np.asarray(fort56['S(J/g/K)'], dtype=float)

    # Round entropy to 4 dp to group isentropes, then pick 4 evenly spaced
    S_rounded = np.round(S_jgk, 4)
    unique_S = np.unique(S_rounded)
    iso_indices = np.linspace(0, len(unique_S) - 1, 4, dtype=int)
    selected_S = unique_S[iso_indices]
    iso_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    valid_rows = []  # (gt_label, gt_arr_converted, pred_arr_converted)
    for (gt_label, burnman_prop, gt_arr, conv) in mapping:
        gt = np.asarray(gt_arr, dtype=float)
        pred = conv(np.asarray(burnman_out[burnman_prop], dtype=float))
        mask = np.isfinite(pred) & np.isfinite(gt)
        if np.any(mask):
            valid_rows.append((gt_label, gt, pred, mask))

    n_props = len(valid_rows)
    if n_props == 0:
        return

    # ── 1. Pressure transects ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, n_props, figsize=(4 * n_props, 5), squeeze=False)
    axes = axes[0]

    for ax, (gt_label, gt, pred, mask) in zip(axes, valid_rows):
        for color, s_val in zip(iso_colors, selected_S):
            rows = np.where(S_rounded == s_val)[0]
            rows = rows[np.argsort(P_gpa[rows])]
            ax.plot(P_gpa[rows], gt[rows],   color=color, lw=1.8,
                    label=f'GT  S={s_val:.3f}')
            ax.plot(P_gpa[rows], pred[rows], color=color, lw=1.2,
                    ls='--', label=f'BM  S={s_val:.3f}')
        ax.set_xlabel('P (GPa)')
        ax.set_ylabel(gt_label)
        ax.set_title(gt_label)
        ax.grid(True, alpha=0.25)

    # Single shared legend outside the last axis
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',
               ncol=4, fontsize=7, bbox_to_anchor=(0.5, -0.12))
    fig.suptitle('Pressure transects — GT (solid) vs Burnman (dashed)', fontsize=11)
    fig.tight_layout()
    out_path = save_dir / 'transects.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved transects plot → {out_path}')

    # ── 2. 1:1 plots ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, n_props, figsize=(4 * n_props, 4), squeeze=False)
    axes = axes[0]

    for ax, (gt_label, gt, pred, mask) in zip(axes, valid_rows):
        ax.scatter(gt[mask], pred[mask], s=4, alpha=0.4, rasterized=True)
        lims = [min(gt[mask].min(), pred[mask].min()),
                max(gt[mask].max(), pred[mask].max())]
        ax.plot(lims, lims, 'k--', lw=1, label='1:1')
        ax.set_xlabel(f'GT  {gt_label}')
        ax.set_ylabel(f'Burnman  {gt_label}')
        ax.set_title(gt_label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    fig.suptitle('1:1 comparison — GT vs Burnman', fontsize=11)
    fig.tight_layout()
    out_path = save_dir / 'one_to_one.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved 1:1 plot      → {out_path}')


def main():
    p = argparse.ArgumentParser(description='Benchmark Burnman vs HeFESTo outputs via HeFESToAPI')
    p.add_argument('--sim-dir', required=True, help='Path to a HeFESTo Simulation directory (contains fort.99 etc)')
    p.add_argument('--save-dir', default=None,
                   help='Directory to save plots (default: <sim-dir>/plots)')
    args = p.parse_args()



    hef = load_hefesto_module()

    sim_dir = Path(args.sim_dir)
    if not sim_dir.exists():
        print('Simulation directory not found:', sim_dir)
        sys.exit(2)

    # Load ground truth
    gt = hef.extract_bulk_properties_from_simulation_dir(str(sim_dir), repeat=1)

    # Get indexer from emulator
    indexer = HeFESToEmulatorGPU.isentropic_emulator.ml_indexer
    comp_names = gt['component_names']

    # Load P, VC using the helper function
    VC, P = hef.load_fort99_component_moles_and_labels(str(sim_dir), indexer)
    #componentMoles = hef.load_fort99_componentMoles(str(sim_dir), indexer)
    #componentMoles = hef.load_fort99_(str(sim_dir), indexer)

    #componentMoles = np.repeat(componentMoles, 1, axis=0)
    #print(componentMoles.shape)
    # Build PT matrix from fort56
    fort56 = gt['fort56_bulk']
    P_gpa = fort56['P(GPa)']
    #T_k = HeFESToEmulatorGPU.get_T
    T_k = fort56['T(K)']
    PT = np.stack([P_gpa*1E9, T_k], axis=1)
    print(PT.shape)

    # Use the API method to compute Burnman properties
    print('Computing Burnman properties via HeFESToAPI.get_property_burnman_vectorized_from_assemblage()...')
    property_names = [
        'entropy_by_mass', 'density', 'p_wave_velocity', 's_wave_velocity',
        'isentropic_bulk_modulus_reuss', 'isothermal_bulk_modulus_reuss',
        'heat_capacity_p_by_mass', 'heat_capacity_v_by_mass',
    ]

    burnman_out = HeFESToEmulatorCPU.get_property_burnman_from_assemblage(VC, P, PT, property_names=property_names)

    """# Use the API method to compute Burnman properties
    print('Computing Burnman properties via HeFESToAPI.get_property_burnman_from_assemblage()...')
    property_names = ['S', 'rho', 'v_p', 'v_s', 'molar_mass', 'K_T']
    burnman_out, returnednames = HeFESToEmulatorCPU.get_property_burnman_from_assemblage(
       VC, P, PT, property_names=property_names
    )"""

    save_dir = Path(args.save_dir) if args.save_dir else sim_dir / 'plots'
    compare_and_report(burnman_out, gt, save_dir=save_dir)


if __name__ == '__main__':
    main()
