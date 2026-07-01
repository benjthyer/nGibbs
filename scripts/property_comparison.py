"""Static thermodynamic property comparison — emulator vs HeFESTo ground truth.

Picks N random, complete simulation directories from a HeFESTo workspace (e.g.
data/HeFESToWorkspace/JiChingSims/), reads each simulation's composition
(params.json) and (P, T, S) profile, and compares three ways of computing
bulk properties (density, VP, VS, entropy) at the same conditions:

  1. HeFESTo ground truth   — read directly from fort.56 (the real simulation output).
  2. Emulator                — phase equilibria (component moles, and for the
                                isentropic pathway also temperature) predicted by
                                the NN emulator, then run through the same internal
                                HeFESTo-vec property calculator.
  3. GT assemblage, internal calc — the *real* HeFESTo phase equilibria (component
                                moles read from fort.99 via load_fort99_componentMoles)
                                run through the internal property calculator.

Line 3 isolates whether any emulator/GT mismatch comes from the property
calculator itself or from the phase-equilibria (assemblage) prediction: if
line 3 already deviates from line 1, the internal calculator explains some of
the gap; whatever gap remains between line 2 and line 3 is attributable to the
phase-equilibria emulator (and, for the isentropic pathway, the temperature
emulator).

Two variants are produced:
  - Isothermal pathway: emulator takes (P, T) from the ground truth directly.
  - Isentropic pathway: emulator takes (P, S) from the ground truth and must
    also predict T; the predicted T is used for property evaluation and is
    plotted against the ground-truth T.

This is a single static comparison — no parameter is swept and no animation
is produced.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / 'src'
for _p in (str(REPO_ROOT), str(SRC_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.ngibbs.engine.API import HeFESToMarsEmulatorCPU as HeFESToEmulatorCPU
from src.ngibbs.engine.EOS_arithmetic.hefesto_vec import load_control
from builder.HeFESTo.HeFESTo_functions import (
    extract_bulk_properties_from_simulation_dir,
    load_fort99_componentMoles,
)

REQUIRED_FILES = ['control', 'fort.56', 'fort.61', 'fort.68', 'fort.99', 'params.json']
ELEMENT_KEYS = ['Si', 'Mg', 'Fe', 'Ca', 'Al', 'Na', 'Cr', 'O']

# HeFESTo-vec EOS property key -> (fort.56 column, axis label).
# Kh/Gh (Hill-averaged moduli from the vectorized EOS) are intentionally
# excluded: fort.56's KS(GPa) uses a different aggregation scheme, so the two
# are not directly comparable (see test_hefesto_assemblage_benchmark_vectorized_emulated.py).
PROPERTY_MAP = {
    'rho': ('rho(g/cm^3)', 'rho (g/cm3)'),
    'Vp':  ('VP(km/s)',    'VP (km/s)'),
    'Vs':  ('VS(km/s)',    'VS (km/s)'),
    'S':   ('S(J/g/K)',    'S (J/g/K)'),
}

# HeFESTo_Parameters_010123 embedded in the control files points at the
# original run machine (macOS) — override with the local copy.
PARAM_DIR = REPO_ROOT / 'recipes' / 'HeFESTo_Parameters_010123'


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, 'detach'):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def find_complete_sim_dirs(workspace_dir: Path) -> List[Path]:
    return sorted(
        d for d in workspace_dir.glob('model_*')
        if all((d / f).exists() for f in REQUIRED_FILES)
    )


def select_sim_dirs(workspace_dir: Path, sims: List[str] = None, n: int = 3, seed: int = None) -> List[Path]:
    """Resolve the simulations to compare.

    If ``sims`` is given (explicit directory names, e.g. 'model_000167'), use
    those directly — this is the only way to guarantee the same simulations
    are compared across separate script invocations (e.g. property_comparison.py
    and phase_comparison.py), since the workspace directory can grow between
    runs and shift what a given --seed selects. Otherwise, sample ``n`` random
    complete simulations using ``seed``.
    """
    if sims:
        chosen = [workspace_dir / name for name in sims]
        for d in chosen:
            if not all((d / f).exists() for f in REQUIRED_FILES):
                raise ValueError(f'{d} is missing required files {REQUIRED_FILES}')
        return chosen

    sim_dirs = find_complete_sim_dirs(workspace_dir)
    if len(sim_dirs) < n:
        raise ValueError(f'Only found {len(sim_dirs)} complete simulations in {workspace_dir}, need {n}')
    rng = np.random.default_rng(seed)
    return [sim_dirs[i] for i in rng.choice(len(sim_dirs), size=n, replace=False)]


def load_composition(sim_dir: Path) -> Dict[str, float]:
    with open(sim_dir / 'params.json') as f:
        params = json.load(f)
    return {k: float(params[k]) for k in ELEMENT_KEYS}


def get_prop_array(result: dict, source: str, prop: str) -> np.ndarray:
    """Fetch a property array from a comparison result by source name.

    source is one of 'fort56', 'gt_internal', 'isothermal_emulator',
    'isentropic_emulator'.
    """
    if source == 'fort56':
        fort56_col, _ = PROPERTY_MAP[prop]
        return np.asarray(result['fort56'][fort56_col], dtype=np.float64)
    return np.asarray(result[source][prop], dtype=np.float64)


def compute_error_stats(gt_vals: np.ndarray, pred_vals: np.ndarray):
    finite = np.isfinite(gt_vals) & np.isfinite(pred_vals)
    abs_err = np.abs(pred_vals[finite] - gt_vals[finite])
    rel_err = abs_err / np.maximum(np.abs(gt_vals[finite]), 1e-12)
    return abs_err, rel_err


def load_ground_truth(sim_dir: Path) -> dict:
    """Ground-truth bulk properties (fort.56) plus the real phase-equilibria
    assemblage (fort.99) run through the internal HeFESTo-vec property calculator.
    """
    gt = extract_bulk_properties_from_simulation_dir(str(sim_dir))
    fort56 = gt['fort56_bulk']
    P_gpa = fort56['P(GPa)'].astype(np.float64)
    T_k = fort56['T(K)'].astype(np.float64)
    n = len(P_gpa)

    indexer = HeFESToEmulatorCPU.isothermal_emulator.ml_indexer
    gt_component_moles = load_fort99_componentMoles(str(sim_dir), indexer)[:n]

    PT = np.stack([P_gpa, T_k], axis=1)
    gt_internal_props = HeFESToEmulatorCPU.get_property_hefesto_vectorized_from_assemblage(
        torch.tensor(gt_component_moles, dtype=torch.float64),
        torch.tensor(PT, dtype=torch.float64),
        property_names=list(PROPERTY_MAP.keys()),
    )

    return {
        'sim_dir': sim_dir,
        'P_gpa': P_gpa,
        'T_k': T_k,
        'fort56': fort56,
        'gt_component_moles': gt_component_moles,
        'gt_internal': gt_internal_props,
    }


def run_isothermal_emulator(gt: dict, composition: Dict[str, float]) -> dict:
    """Predict phase equilibria from (P, T) and evaluate properties on the predicted assemblage."""
    P_gpa, T_k = gt['P_gpa'], gt['T_k']
    n = len(P_gpa)
    comp_headers = list(composition.keys())
    comp_block = np.tile(np.array([[composition[k] for k in comp_headers]], dtype=np.float32), (n, 1))
    features = np.column_stack([P_gpa, T_k, comp_block]).astype(np.float32)
    headers = ['P(GPa)(System_main)', 'T(K)(System_main)'] + comp_headers

    with torch.no_grad():
        out = HeFESToEmulatorCPU.ForwardMB(features, headers=headers, outputs=['component_moles'])
    component_moles = _to_numpy(out['component_moles']).astype(np.float64)

    PT = np.stack([P_gpa, T_k], axis=1)
    return HeFESToEmulatorCPU.get_property_hefesto_vectorized_from_assemblage(
        torch.tensor(component_moles, dtype=torch.float64),
        torch.tensor(PT, dtype=torch.float64),
        property_names=list(PROPERTY_MAP.keys()),
    )


def run_isentropic_emulator(gt: dict, composition: Dict[str, float]) -> dict:
    """Predict phase equilibria + temperature from (P, S) and evaluate properties
    on the predicted assemblage at the predicted temperature."""
    P_gpa = gt['P_gpa']
    S_jgk = gt['fort56']['S(J/g/K)'].astype(np.float64)
    n = len(P_gpa)
    comp_headers = list(composition.keys())
    comp_block = np.tile(np.array([[composition[k] for k in comp_headers]], dtype=np.float32), (n, 1))
    features = np.column_stack([P_gpa, S_jgk, comp_block]).astype(np.float32)
    headers = ['P(GPa)(System_main)', 'S(J/g/K)(System_main)'] + comp_headers

    with torch.no_grad():
        out = HeFESToEmulatorCPU.ForwardMB(
            features, headers=headers, outputs=['component_moles', 'temperature']
        )
    component_moles = _to_numpy(out['component_moles']).astype(np.float64)
    T_emulated = _to_numpy(out['temperature']).reshape(-1).astype(np.float64)

    PT = np.stack([P_gpa, T_emulated], axis=1)
    props = HeFESToEmulatorCPU.get_property_hefesto_vectorized_from_assemblage(
        torch.tensor(component_moles, dtype=torch.float64),
        torch.tensor(PT, dtype=torch.float64),
        property_names=list(PROPERTY_MAP.keys()),
    )
    props['T_emulated'] = T_emulated
    return props


def compare_simulation(sim_dir: Path) -> dict:
    composition = load_composition(sim_dir)
    gt = load_ground_truth(sim_dir)
    result = {
        'sim_dir': sim_dir,
        'composition': composition,
        'P_gpa': gt['P_gpa'],
        'fort56': gt['fort56'],
        'gt_internal': gt['gt_internal'],
        'isothermal_emulator': run_isothermal_emulator(gt, composition),
        'isentropic_emulator': run_isentropic_emulator(gt, composition),
    }
    return result


def plot_comparison(results: list, mode: str, save_path: str) -> None:
    """mode is 'isothermal' or 'isentropic'."""
    assert mode in ('isothermal', 'isentropic')
    emulator_source = f'{mode}_emulator'
    prop_keys = list(PROPERTY_MAP.keys())
    show_T_panel = mode == 'isentropic'
    n_cols = len(prop_keys) + (1 if show_T_panel else 0)
    n_sims = len(results)

    fig, axes = plt.subplots(n_sims, n_cols, figsize=(3.6 * n_cols, 3.2 * n_sims), squeeze=False)

    for row, result in enumerate(results):
        P = result['P_gpa']
        for col, prop in enumerate(prop_keys):
            ax = axes[row][col]
            _, label = PROPERTY_MAP[prop]
            gt_vals = get_prop_array(result, 'fort56', prop)
            em_vals = get_prop_array(result, emulator_source, prop)
            gi_vals = get_prop_array(result, 'gt_internal', prop)

            ax.plot(P, gt_vals, 'k-', lw=1.5, label='HeFESTo (GT)')
            ax.plot(P, em_vals, 'r--', lw=1.5, label='Emulator (pred. assemblage)')
            ax.plot(P, gi_vals, 'b:', lw=1.8, label='GT assemblage + internal calc')
            ax.set_xlabel('P (GPa)')
            ax.set_ylabel(label if col > 0 else f"{result['sim_dir'].name}\n{label}", fontsize=9)
            ax.grid(True, alpha=0.25)
            if row == 0:
                ax.set_title(label, fontsize=10)
            if row == 0 and col == len(prop_keys) - 1:
                ax.legend(fontsize=7, loc='best')

        if show_T_panel:
            ax = axes[row][-1]
            T_gt = result['fort56']['T(K)'].astype(np.float64)
            T_em = result['isentropic_emulator']['T_emulated']
            ax.plot(P, T_gt, 'k-', lw=1.5, label='HeFESTo (GT)')
            ax.plot(P, T_em, 'r--', lw=1.5, label='Emulator T(P,S)')
            ax.set_xlabel('P (GPa)')
            ax.set_ylabel('T (K)')
            ax.grid(True, alpha=0.25)
            if row == 0:
                ax.set_title('T (K)', fontsize=10)
                ax.legend(fontsize=7, loc='best')

    mode_label = 'Isothermal (P, T) pathway' if mode == 'isothermal' else 'Isentropic (P, S) pathway'
    fig.suptitle(f'{mode_label}: emulator vs HeFESTo ground truth', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(save_path, dpi=150, facecolor='white')
    plt.close(fig)
    print(f'Saved -> {save_path}')


def print_error_summary(results: list, mode: str) -> None:
    """mode is 'isothermal' or 'isentropic'."""
    emulator_source = f'{mode}_emulator'
    sources = [
        (emulator_source, f'{mode.capitalize()} emulator (predicted assemblage)'),
        ('gt_internal', 'GT assemblage + internal calc (isolates property-calculator error)'),
    ]
    prop_keys = list(PROPERTY_MAP.keys())

    print('\n' + '=' * 78)
    print(f'Relative error vs HeFESTo ground truth (fort.56) -- {mode} pathway')
    print('=' * 78)
    for result in results:
        print(f"\n{result['sim_dir'].name}  ({len(result['P_gpa'])} P-T points)")
        for source_key, display_name in sources:
            print(f'  -- {display_name} --')
            for prop in prop_keys:
                _, label = PROPERTY_MAP[prop]
                gt_vals = get_prop_array(result, 'fort56', prop)
                pred_vals = get_prop_array(result, source_key, prop)
                abs_err, rel_err = compute_error_stats(gt_vals, pred_vals)
                print(
                    f'    {prop:>4s} ({label})  mean_abs_err={abs_err.mean():.4g}  '
                    f'mean_rel_err={rel_err.mean() * 100:.3f}%  max_rel_err={rel_err.max() * 100:.3f}%'
                )
        if mode == 'isentropic':
            T_gt = result['fort56']['T(K)'].astype(np.float64)
            T_em = result['isentropic_emulator']['T_emulated']
            abs_err, rel_err = compute_error_stats(T_gt, T_em)
            print(
                f'    {"T":>4s} (T (K))  mean_abs_err={abs_err.mean():.4g}  '
                f'mean_rel_err={rel_err.mean() * 100:.3f}%  max_rel_err={rel_err.max() * 100:.3f}%'
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=str, default='data/HeFESToWorkspace/JiChingSims',
                         help='HeFESTo workspace directory to sample simulations from')
    parser.add_argument('--n', type=int, default=3, help='Number of random simulations to compare')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducible sampling')
    parser.add_argument('--sims', type=str, nargs='+', default=None,
                         help='Explicit simulation directory names (e.g. model_000167 model_000012), '
                              'overriding random sampling. Use this — not --seed — to guarantee '
                              'property_comparison.py and phase_comparison.py compare the same simulations.')
    parser.add_argument('--save-path-isothermal', type=str, default='property_comparison_isothermal.png')
    parser.add_argument('--save-path-isentropic', type=str, default='property_comparison_isentropic.png')
    args = parser.parse_args()

    workspace_dir = REPO_ROOT / args.workspace
    chosen = select_sim_dirs(workspace_dir, sims=args.sims, n=args.n, seed=args.seed)

    print(f'Selected {len(chosen)} simulations from {workspace_dir}:')
    for d in chosen:
        print(f'  {d.name}')

    # Phase in/out flags are identical across JiChingSims control files (only the
    # bulk composition lines differ), so a single load_control call suffices.
    # The parameter directory embedded in the control file is a stale macOS path
    # from the original run machine — override it with the local copy.
    HeFESToEmulatorCPU.hefesto_params = load_control(
        str(chosen[0] / 'control'), param_dir_override=str(PARAM_DIR)
    )
    HeFESToEmulatorCPU.hefesto_npz_path = None

    results = [compare_simulation(d) for d in chosen]

    print_error_summary(results, mode='isothermal')
    print_error_summary(results, mode='isentropic')

    plot_comparison(results, mode='isothermal', save_path=args.save_path_isothermal)
    plot_comparison(results, mode='isentropic', save_path=args.save_path_isentropic)
