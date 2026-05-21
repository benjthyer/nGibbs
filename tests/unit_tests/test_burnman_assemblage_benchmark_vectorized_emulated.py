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

from src.module.engine.API import HeFESToEmulatorCPU, HeFESToEmulatorGPU

compositions = {
    'htz_PS': {
    'Si': 3.57887,
    'Mg': 5.64233,
    'Fe': 0.58052,
    'Ca': 0.07969,
    'Al': 0.09740,
    'Na': 0.00160,
    'Cr': 0.01960,
    'O': 13.63947
    },

    'DMM_PS': {
    'Si': 3.79222,
    'Mg': 4.88475,
    'Fe': 0.57874,
    'Ca': 0.28734,
    'Al': 0.39685,
    'Na': 0.02197,
    'Cr': 0.03813,
    'O': 14.00450
    },

    'basalt_PS': {
    'Si': 4.61116,
    'Mg': 1.33197,
    'Fe': 0.61701,
    'Ca': 1.22855,
    'Al': 1.81090,
    'Na': 0.39533,
    'Cr': 0.00508,
    'O': 15.35841
    }
}

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


def compare_and_report(pred_out: dict, ref_out: dict, save_dir: "Path | None" = None,
                       P_gpa=None, S_jgk=None):
    """Compare two Burnman property output dicts (pred vs ref), both in native Burnman SI units.

    P_gpa and S_jgk are optional 1-D arrays used as axes for pressure-transect plots.
    """
    prop_map = [
        ('S (J/kg/K)',  'entropy_by_mass'),
        ('rho (kg/m³)', 'density'),
        ('VP (m/s)',    'p_wave_velocity'),
        ('VS (m/s)',    's_wave_velocity'),
        ('KS (Pa)',     'isentropic_bulk_modulus_reuss'),
        ('cp (J/kg/K)', 'heat_capacity_p_by_mass'),
        ('cv (J/kg/K)', 'heat_capacity_v_by_mass'),
    ]

    valid_rows = []  # (display_label, ref_arr, pred_arr, mask)
    for display_label, prop_name in prop_map:
        if prop_name not in pred_out or prop_name not in ref_out:
            continue
        pred = np.asarray(pred_out[prop_name], dtype=float)
        ref  = np.asarray(ref_out[prop_name],  dtype=float)
        mask = np.isfinite(pred) & np.isfinite(ref)
        if np.any(mask):
            valid_rows.append((display_label, ref, pred, mask))

    if not valid_rows:
        print('No overlapping properties found between the two Burnman outputs.')
        return

    report = []
    for display_label, ref, pred, mask in valid_rows:
        diff = pred[mask] - ref[mask]
        mae  = np.mean(np.abs(diff))
        rmse = math.sqrt(np.mean(diff ** 2))
        mre  = np.mean(np.abs(diff / np.clip(ref[mask], 1e-12, None))) * 100.0
        report.append((display_label, mae, rmse, mre, int(np.sum(mask))))

    print('\nBenchmark results (pred vs ref, both in Burnman SI units):')
    print('Property      | MAE        | RMSE       | MeanRelErr% | N')
    print('-' * 65)
    for row in report:
        print(f'{row[0]:13s} | {row[1]:.6g} | {row[2]:.6g} | {row[3]:.3f}% | {row[4]}')

    if save_dir is None:
        return

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    n_props = len(valid_rows)

    # ── 1. Pressure transects ────────────────────────────────────────────────
    if P_gpa is not None and S_jgk is not None:
        P_arr     = np.asarray(P_gpa, dtype=float)
        S_arr     = np.asarray(S_jgk, dtype=float)
        S_rounded = np.round(S_arr, 4)
        unique_S  = np.unique(S_rounded)
        iso_indices = np.linspace(0, len(unique_S) - 1, 4, dtype=int)
        selected_S  = unique_S[iso_indices]
        iso_colors  = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        fig, axes = plt.subplots(1, n_props, figsize=(4 * n_props, 5), squeeze=False)
        axes = axes[0]

        for ax, (display_label, ref, pred, mask) in zip(axes, valid_rows):
            for color, s_val in zip(iso_colors, selected_S):
                rows = np.where(S_rounded == s_val)[0]
                rows = rows[np.argsort(P_arr[rows])]
                ax.plot(P_arr[rows], ref[rows],  color=color, lw=1.8,
                        label=f'Ref  S={s_val:.3f}')
                ax.plot(P_arr[rows], pred[rows], color=color, lw=1.2,
                        ls='--', label=f'Pred S={s_val:.3f}')
            ax.set_xlabel('P (GPa)')
            ax.set_ylabel(display_label)
            ax.set_title(display_label)
            ax.grid(True, alpha=0.25)

        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center',
                   ncol=4, fontsize=7, bbox_to_anchor=(0.5, -0.12))
        fig.suptitle('Pressure transects — Ref (solid) vs Pred (dashed)', fontsize=11)
        fig.tight_layout()
        out_path = save_dir / 'transects.png'
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved transects plot → {out_path}')

    # ── 2. 1:1 plots ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, n_props, figsize=(4 * n_props, 4), squeeze=False)
    axes = axes[0]

    for ax, (display_label, ref, pred, mask) in zip(axes, valid_rows):
        ax.scatter(ref[mask], pred[mask], s=4, alpha=0.4, rasterized=True)
        lims = [min(ref[mask].min(), pred[mask].min()),
                max(ref[mask].max(), pred[mask].max())]
        ax.plot(lims, lims, 'k--', lw=1, label='1:1')
        ax.set_xlabel(f'Ref  {display_label}')
        ax.set_ylabel(f'Pred  {display_label}')
        ax.set_title(display_label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    fig.suptitle('1:1 comparison — Ref vs Pred', fontsize=11)
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
    try:
        comp = compositions[sim_dir.name.split('/')[-1]]
    except NameError:
        
        comp = compositions[sim_dir.name.split('/')[-2]]

            


    if not sim_dir.exists():
        print('Simulation directory not found:', sim_dir)
        sys.exit(2)

    # Load ground truth
    gt = hef.extract_bulk_properties_from_simulation_dir(str(sim_dir), repeat=1)

    # Get indexer from emulator
    indexer = HeFESToEmulatorGPU.isentropic_emulator.ml_indexer
    comp_names = gt['component_names']

    # Load P, VC using the helper function
    #VC, P = hef.load_fort99_component_moles_and_labels(str(sim_dir), indexer)
    gt_componentMoles = hef.load_fort99_componentMoles(str(sim_dir), indexer)
    #componentMoles = hef.load_fort99_(str(sim_dir), indexer)
    
   
    # Build PT matrix from fort56
    fort56 = gt['fort56_bulk']
    P_gpa = fort56['P(GPa)']
    #T_k = HeFESToEmulatorGPU.get_T
    T_k = fort56['T(K)']
    S_K = fort56['S(J/g/K)']
    PT = np.stack([P_gpa*1E9, T_k], axis=1)

    comp_vec = np.zeros(len(comp)).reshape(1, -1)
    headers = []
    for i, (name, value) in enumerate(comp.items()):
        comp_vec[0, i] = value
        headers.append(name)

    features = np.concatenate([P_gpa.reshape(-1, 1), S_K.reshape(-1, 1), np.repeat(comp_vec, PT.shape[0], axis=0)], axis=1)
    print(PT.shape)
    headers = ['P(GPa)(System_main)', 'S(J/g/K)(System_main)'] + headers

    output = HeFESToEmulatorCPU.ForwardMB(features,headers=headers,outputs=['component_moles', 'temperature'])

    componentMoles = np.repeat(output['component_moles'], 1, axis=0)

    Em_PT = np.concatenate([PT, output['temperature'].detach().cpu().numpy().reshape(-1, 1)], axis=1)
    # Use the API method to compute Burnman properties
    print('Computing Burnman properties via HeFESToAPI.get_property_burnman_vectorized_from_assemblage()...')
    property_names = [
        'entropy_by_mass', 'density', 'p_wave_velocity', 's_wave_velocity',
        'isentropic_bulk_modulus_reuss', 'isothermal_bulk_modulus_reuss',
        'heat_capacity_p_by_mass', 'heat_capacity_v_by_mass',
    ]

    burnman_out = HeFESToEmulatorGPU.get_property_burnman_vectorized_from_assemblage(
       torch.tensor(componentMoles, dtype=torch.float64, device='cuda'), torch.tensor(Em_PT, dtype=torch.float64, device='cuda'), 
       property_names= property_names
    )

    gt_out = HeFESToEmulatorGPU.get_property_burnman_vectorized_from_assemblage(
       torch.tensor(gt_componentMoles, dtype=torch.float64, device='cuda'), torch.tensor(PT, dtype=torch.float64, device='cuda'), 
       property_names= property_names
    )

    """# Use the API method to compute Burnman properties
    print('Computing Burnman properties via HeFESToAPI.get_property_burnman_from_assemblage()...')
    property_names = ['S', 'rho', 'v_p', 'v_s', 'molar_mass', 'K_T']
    burnman_out, returnednames = HeFESToEmulatorCPU.get_property_burnman_from_assemblage(
       VC, P, PT, property_names=property_names
    )"""

    save_dir = Path(args.save_dir) if args.save_dir else sim_dir / 'plots'
    compare_and_report(burnman_out, gt_out, save_dir=save_dir, P_gpa=P_gpa, S_jgk=S_K)


if __name__ == '__main__':
    main()
