"""Benchmark HeFESTo-vec EOS on emulated assemblages against HeFESTo ground truth.

Usage (from repository root):
    python -m tests.unit_tests.test_hefesto_assemblage_benchmark_vectorized_emulated \\
        --sim-dir PATH/TO/SimulationDir \\
        --control-path PATH/TO/control \\
        --param-dir PATH/TO/HeFESTo_Parameters_010123

The script:
- reconstructs isentropic emulator inputs from fort.56 (P, S, composition)
- runs ForwardMB to obtain emulated component moles and predicted temperature
- computes EOS properties using HeFESToAPI.get_property_hefesto_vectorized_from_assemblage()
  at (P_fort56, T_emulator)
- compares rho, Vp, Vs, S directly in native HeFESTo units against the bulk properties
  read directly from fort.56 (no EOS re-evaluation on the GT assemblage)
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

from src.ngibbs.engine.API import HeFESToEmulatorCPU, HeFESToEmulatorGPU
from src.ngibbs.engine.EOS_arithmetic.hefesto_vec import load_control

compositions = {
    'htz_PS': {
        'Si': 3.57887, 'Mg': 5.64233, 'Fe': 0.58052, 'Ca': 0.07969,
        'Al': 0.09740, 'Na': 0.00160, 'Cr': 0.01960, 'O': 13.63947,
    },
    'DMM_PS': {
        'Si': 3.79222, 'Mg': 4.88475, 'Fe': 0.57874, 'Ca': 0.28734,
        'Al': 0.39685, 'Na': 0.02197, 'Cr': 0.03813, 'O': 14.00450,
    },
    'basalt_PS': {
        'Si': 4.61116, 'Mg': 1.33197, 'Fe': 0.61701, 'Ca': 1.22855,
        'Al': 1.81090, 'Na': 0.39533, 'Cr': 0.00508, 'O': 15.35841,
    },
}


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


def compare_and_report(
    pred_out: dict,
    ref_out:  dict,
    save_dir: 'Path | None' = None,
    P_gpa=None,
    S_jgk=None,
):
    """Compare two HeFESTo-vec output dicts (pred vs ref), both in native HeFESTo units.

    P_gpa and S_jgk are optional 1-D arrays used as axes for pressure-transect plots.
    """
    # (display_label, key) — native units, no conversion
    prop_map = [
        ('T (K)',       'T'),
        ('S (J/g/K)',   'S'),
        ('rho (g/cm³)', 'rho'),
        ('VP (km/s)',   'Vp'),
        ('VS (km/s)',   'Vs'),
        ('Kh (GPa)',    'Kh'),
        ('Gh (GPa)',    'Gh'),
    ]

    valid_rows = []
    for display_label, prop_name in prop_map:
        if prop_name not in pred_out or prop_name not in ref_out:
            continue
        pred = np.asarray(pred_out[prop_name], dtype=float)
        ref  = np.asarray(ref_out[prop_name],  dtype=float)
        mask = np.isfinite(pred) & np.isfinite(ref)
        if np.any(mask):
            valid_rows.append((display_label, ref, pred, mask))

    if not valid_rows:
        print('No overlapping properties found between the two HeFESTo-vec outputs.')
        return

    report = []
    for display_label, ref, pred, mask in valid_rows:
        diff = pred[mask] - ref[mask]
        mae  = np.mean(np.abs(diff))
        rmse = math.sqrt(np.mean(diff ** 2))
        mre  = np.mean(np.abs(diff / np.clip(ref[mask], 1e-12, None))) * 100.0
        report.append((display_label, mae, rmse, mre, int(np.sum(mask))))

    print('\nBenchmark results (emulated vs GT, both in native HeFESTo units):')
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
                ax.plot(P_arr[rows], ref[rows],  color=color, lw=1.8, label=f'GT   S={s_val:.3f}')
                ax.plot(P_arr[rows], pred[rows], color=color, lw=1.2, ls='--', label=f'Emul S={s_val:.3f}')
            ax.set_xlabel('P (GPa)')
            ax.set_ylabel(display_label)
            ax.set_title(display_label)
            ax.grid(True, alpha=0.25)

        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=7, bbox_to_anchor=(0.5, -0.12))
        fig.suptitle('Pressure transects — GT (solid) vs Emulated (dashed)', fontsize=11)
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
        ax.set_xlabel(f'GT  {display_label}')
        ax.set_ylabel(f'Emul  {display_label}')
        ax.set_title(display_label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    fig.suptitle('1:1 comparison — GT vs Emulated', fontsize=11)
    fig.tight_layout()
    out_path = save_dir / 'one_to_one.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved 1:1 plot → {out_path}')


def main():
    p = argparse.ArgumentParser(
        description='Benchmark HeFESTo-vec EOS on emulated assemblages vs fort.56 ground truth'
    )
    p.add_argument('--sim-dir',   required=True, help='HeFESTo Simulation directory (contains fort.99 etc)')
    p.add_argument('--param-dir', default=None,  help='Override parameter directory (e.g. HeFESTo_Parameters_010123/)')
    p.add_argument('--save-dir',  default=None,  help='Directory to save plots (default: <sim-dir>/plots_hefesto_emul)')
    args = p.parse_args()

    hef = load_hefesto_module()

    sim_dir = Path(args.sim_dir)
    if not sim_dir.exists():
        print('Simulation directory not found:', sim_dir)
        sys.exit(2)

    control_path = sim_dir / 'control'

    # Resolve composition from sim_dir name
    sim_name = sim_dir.name
    comp = compositions.get(sim_name) or compositions.get(sim_dir.parent.name)
    if comp is None:
        print(f'Warning: composition not found for sim_dir "{sim_name}"; using DMM_PS')
        comp = compositions['DMM_PS']

    # Wire up HeFESTo params on the global emulator
    emulator = HeFESToEmulatorGPU if HeFESToEmulatorGPU is not None else HeFESToEmulatorCPU
    emulator.hefesto_params = load_control(control_path, param_dir_override=args.param_dir)
    emulator.hefesto_npz_path = None
    print(f'Loaded HeFESTo params: {emulator.hefesto_params.nspec} species')

    # Load ground truth
    gt = hef.extract_bulk_properties_from_simulation_dir(str(sim_dir), repeat=1)
    fort56 = gt['fort56_bulk']

    P_gpa = fort56['P(GPa)']
    T_k   = fort56['T(K)']
    S_K   = fort56['S(J/g/K)']

    # ── Step 1: build isentropic emulator features from fort.56 ──────────────
    comp_vec = np.zeros((1, len(comp)), dtype=np.float32)
    headers  = []
    for i, (name, value) in enumerate(comp.items()):
        comp_vec[0, i] = value
        headers.append(name)

    features = np.concatenate(
        [P_gpa.reshape(-1, 1), S_K.reshape(-1, 1),
         np.repeat(comp_vec, P_gpa.shape[0], axis=0)],
        axis=1,
    )
    headers = ['P(GPa)(System_main)', 'S(J/g/K)(System_main)'] + headers
    print(f'Emulator input shape: {features.shape}')

    # ── Step 2: run isentropic emulator → emulated component moles + T ───────
    output = emulator.ForwardMB(
        features,
        headers=headers,
        outputs=['component_moles', 'temperature'],
    )
    component_moles = output['component_moles'].detach().cpu().numpy()
    T_emul          = output['temperature'].detach().cpu().numpy().reshape(-1)
    print(f'Emulated component_moles shape: {component_moles.shape}')
    print(f'Emulated T range: {T_emul.min():.1f} – {T_emul.max():.1f} K')

    # ── Step 3: EOS at (P_fort56, T_emulator) using emulated assemblage ───────
    # Using emulator-predicted T rather than fort.56 T tests both the emulator's
    # assemblage prediction and its temperature prediction simultaneously.
    PT_emul = np.stack([P_gpa, T_emul], axis=1)   # (N, 2): GPa, K

    property_names = ['S', 'rho', 'Vp', 'Vs', 'Kh', 'Gh']
    print('Computing HeFESTo-vec EOS properties on emulated assemblage...')
    emul_out = emulator.get_property_hefesto_vectorized_from_assemblage(
        torch.tensor(component_moles, dtype=torch.float64),
        torch.tensor(PT_emul,         dtype=torch.float64),
        property_names=property_names,
    )
    emul_out['T'] = T_emul   # emulator-predicted temperature

    # ── Step 4: read ground-truth bulk properties directly from fort.56 ─────────
    # Build the reference dict using the same short-key convention as emul_out.
    # fort.56 does not contain Hill-averaged Kh/Gh, so those keys are absent and
    # compare_and_report will skip them automatically.
    _f56 = fort56
    gt_out: dict = {'T': T_k}
    _fort56_key_map = {
        'S':   'S(J/g/K)',
        'rho': 'rho(g/cm^3)',
        'Vp':  'VP(km/s)',
        'Vs':  'VS(km/s)',
    }
    for short_key, col in _fort56_key_map.items():
        if col in _f56:
            gt_out[short_key] = np.asarray(_f56[col], dtype=float)

    save_dir = Path(args.save_dir) if args.save_dir else sim_dir / 'plots_hefesto_emul'
    compare_and_report(emul_out, gt_out, save_dir=save_dir, P_gpa=P_gpa, S_jgk=S_K)


if __name__ == '__main__':
    main()
