"""Benchmark HeFESTo-vec EOS against HeFESTo ground truth.

Usage (from repository root):
    python -m tests.unit_tests.test_hefesto_assemblage_benchmark_vectorized \\
        --sim-dir PATH/TO/SimulationDir \\
        --control-path PATH/TO/control \\
        --param-dir PATH/TO/HeFESTo_Parameters_010123

The script:
- loads fort.99 via `load_fort99_componentMoles`
- loads ground-truth bulk properties via `extract_bulk_properties_from_simulation_dir`
- computes EOS properties using HeFESToAPI.get_property_hefesto_vectorized_from_assemblage()
- compares rho, Vp, Vs, S directly in native HeFESTo units (no conversion needed)
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

from src.module.engine.API import HeFESToEmulatorCPU, HeFESToEmulatorGPU
from src.module.engine.EOS_arithmetic.hefesto_vec import load_control


def load_hefesto_module():
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


def compare_and_report(hefesto_out: dict, hef_result: dict, save_dir: 'Path | None' = None):
    """Compare HeFESTo-vec outputs against fort.56 ground truth.

    Both sides are in native HeFESTo units, so no unit conversion is applied.

    Native units:
        S     J/g/K
        rho   g/cm³
        Vp    km/s
        Vs    km/s
        Kh    GPa   (VRH Hill adiabatic bulk modulus)
    """
    fort56 = hef_result.get('fort56_bulk', {})

    # (fort56_key, hefesto_key)
    # NOTE: fort.56 KS(GPa) = bstot includes metamorphic phase-change softening
    # (bmet = sum(dndp*dmdp) from Gibbs minimisation).  Our Kh is the isomorphic
    # Hill adiabatic modulus and is NOT directly comparable.  Omit Kh from the
    # mapping to avoid a misleading metric; use rho/Vp/Vs/S as the benchmarks.
    mapping = [
        ('S(J/g/K)',    'S'),
        ('rho(g/cm^3)', 'rho'),
        ('VP(km/s)',    'Vp'),
        ('VS(km/s)',    'Vs'),
    ]

    valid_rows = []
    for gt_label, prop in mapping:
        if prop not in hefesto_out or gt_label not in fort56:
            continue
        pred = np.asarray(hefesto_out[prop], dtype=float)
        gt   = np.asarray(fort56[gt_label],  dtype=float)
        mask = np.isfinite(pred) & np.isfinite(gt)
        if np.any(mask):
            valid_rows.append((gt_label, prop, gt, pred, mask))

    if not valid_rows:
        print('No overlapping properties found between HeFESTo-vec output and fort.56 ground truth.')
        return

    report = []
    for gt_label, prop, gt, pred, mask in valid_rows:
        diff = pred[mask] - gt[mask]
        mae  = np.mean(np.abs(diff))
        rmse = math.sqrt(np.mean(diff ** 2))
        mre  = np.mean(np.abs(diff / np.clip(gt[mask], 1e-12, None))) * 100.0
        report.append((gt_label, mae, rmse, mre, int(np.sum(mask))))

    print('\nBenchmark results (HeFESTo-vec vs fort.56 ground truth, native units):')
    print('Property      | MAE        | RMSE       | MeanRelErr% | N')
    print('-' * 65)
    for row in report:
        print(f'{row[0]:13s} | {row[1]:.6g} | {row[2]:.6g} | {row[3]:.3f}% | {row[4]}')

    if save_dir is None:
        return

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fort56 = hef_result['fort56_bulk']
    P_gpa   = np.asarray(fort56['P(GPa)'],    dtype=float)
    S_jgk   = np.asarray(fort56['S(J/g/K)'],  dtype=float)
    S_rounded = np.round(S_jgk, 4)
    unique_S  = np.unique(S_rounded)
    iso_indices = np.linspace(0, len(unique_S) - 1, 4, dtype=int)
    selected_S  = unique_S[iso_indices]
    iso_colors  = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    n_props = len(valid_rows)

    # ── 1. Pressure transects ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, n_props, figsize=(4 * n_props, 5), squeeze=False)
    axes = axes[0]

    for ax, (gt_label, prop, gt, pred, mask) in zip(axes, valid_rows):
        for color, s_val in zip(iso_colors, selected_S):
            rows = np.where(S_rounded == s_val)[0]
            rows = rows[np.argsort(P_gpa[rows])]
            ax.plot(P_gpa[rows], gt[rows],   color=color, lw=1.8, label=f'GT  S={s_val:.3f}')
            ax.plot(P_gpa[rows], pred[rows], color=color, lw=1.2, ls='--', label=f'HEF S={s_val:.3f}')
        ax.set_xlabel('P (GPa)')
        ax.set_ylabel(gt_label)
        ax.set_title(gt_label)
        ax.grid(True, alpha=0.25)

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=7, bbox_to_anchor=(0.5, -0.12))
    fig.suptitle('Pressure transects — GT (solid) vs HeFESTo-vec (dashed)', fontsize=11)
    fig.tight_layout()
    out_path = save_dir / 'transects.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved transects plot → {out_path}')

    # ── 2. 1:1 plots ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, n_props, figsize=(4 * n_props, 4), squeeze=False)
    axes = axes[0]

    for ax, (gt_label, prop, gt, pred, mask) in zip(axes, valid_rows):
        ax.scatter(gt[mask], pred[mask], s=4, alpha=0.4, rasterized=True)
        lims = [min(gt[mask].min(), pred[mask].min()),
                max(gt[mask].max(), pred[mask].max())]
        ax.plot(lims, lims, 'k--', lw=1, label='1:1')
        ax.set_xlabel(f'GT  {gt_label}')
        ax.set_ylabel(f'HeFESTo-vec  {gt_label}')
        ax.set_title(gt_label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    fig.suptitle('1:1 comparison — GT vs HeFESTo-vec', fontsize=11)
    fig.tight_layout()
    out_path = save_dir / 'one_to_one.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved 1:1 plot → {out_path}')


def main():
    p = argparse.ArgumentParser(
        description='Benchmark HeFESTo-vec EOS against fort.56 ground truth via HeFESToAPI'
    )
    p.add_argument('--sim-dir',   required=True, help='HeFESTo Simulation directory (contains fort.99 etc)')
    p.add_argument('--param-dir', default=None,  help='Override parameter directory (e.g. HeFESTo_Parameters_010123/)')
    p.add_argument('--save-dir',  default=None,  help='Directory to save plots (default: <sim-dir>/plots)')
    args = p.parse_args()

    hef = load_hefesto_module()

    sim_dir = Path(args.sim_dir)
    if not sim_dir.exists():
        print('Simulation directory not found:', sim_dir)
        sys.exit(2)

    control_path = sim_dir / 'control'

    # Wire up HeFESTo params on the global emulator
    emulator = HeFESToEmulatorGPU if HeFESToEmulatorGPU is not None else HeFESToEmulatorCPU
    emulator.hefesto_params = load_control(control_path, param_dir_override=args.param_dir)
    emulator.hefesto_npz_path = None
    print(f'Loaded HeFESTo params: {emulator.hefesto_params.nspec} species')

    # Load ground truth
    gt = hef.extract_bulk_properties_from_simulation_dir(str(sim_dir), repeat=1)
    fort56 = gt['fort56_bulk']

    # Load ground-truth component moles from fort.99
    indexer = emulator.isentropic_emulator.ml_indexer
    component_moles = hef.load_fort99_componentMoles(str(sim_dir), indexer)
    component_moles = np.repeat(component_moles, 1, axis=0)
    print(f'component_moles shape: {component_moles.shape}')

    # PT in GPa / K — no unit conversion (HeFESTo EOS expects GPa, not Pa)
    P_gpa = fort56['P(GPa)']
    T_k   = fort56['T(K)']
    PT    = np.stack([P_gpa, T_k], axis=1)
    print(f'PT shape: {PT.shape}')

    property_names = ['S', 'rho', 'Vp', 'Vs']
    print('Computing HeFESTo-vec EOS properties...')
    hefesto_out = emulator.get_property_hefesto_vectorized_from_assemblage(
        torch.tensor(component_moles, dtype=torch.float64),
        torch.tensor(PT,              dtype=torch.float64),
        property_names=property_names,
    )

    save_dir = Path(args.save_dir) if args.save_dir else sim_dir / 'plots_hefesto_vec'
    compare_and_report(hefesto_out, gt, save_dir=save_dir)


if __name__ == '__main__':
    main()
