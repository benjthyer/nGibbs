"""Static phase-abundance comparison — emulator vs HeFESTo ground truth.

Companion to scripts/property_comparison.py: for the same random, complete
simulation directories, compares stacked phase-fraction-vs-pressure diagrams
between HeFESTo ground truth (real assemblage, read from fort.99) and the
emulator's predicted assemblage, for both the isothermal (P, T) and isentropic
(P, S) pathways.

Unlike property_comparison.py there is no separate "internal calculator" line:
phase fractions are a direct aggregation of component moles, so the only two
things to compare are the real assemblage and the emulator-predicted assemblage.

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
import pandas as pd
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
from src.builder.plotting import phase_colors, build_ordered_phases, draw_phase_stack
from builder.HeFESTo.HeFESTo_functions import (
    extract_bulk_properties_from_simulation_dir,
    load_fort99_componentMoles,
    _safe_read_ws_table,
    _resolve_component_name_from_abbr,
)

# A component contributing less than this fraction of a row's total assemblage
# moles is treated as numerical noise rather than a genuinely unrepresented phase.
MISSING_PHASE_FRACTION_THRESHOLD = 1.0e-3

REQUIRED_FILES = ['control', 'fort.56', 'fort.61', 'fort.68', 'fort.99', 'params.json']
ELEMENT_KEYS = ['Si', 'Mg', 'Fe', 'Ca', 'Al', 'Na', 'Cr', 'O']


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


def normalize_phase_moles(phase_moles: np.ndarray) -> np.ndarray:
    """Normalize per-row phase moles to phase mole fractions summing to 1."""
    row_totals = phase_moles.sum(axis=1, keepdims=True)
    row_totals = np.where(row_totals == 0, 1.0, row_totals)
    return phase_moles / row_totals


def check_missing_phases(sim_dir: Path, indexer, gt_component_moles: np.ndarray, P_gpa: np.ndarray) -> dict:
    """Flag fort.99 components with nonzero abundance that the emulator's
    indexer cannot represent (i.e. not in ``indexer.label_names``).

    ``load_fort99_componentMoles`` silently drops such components (only
    ``gt_component_moles`` columns known to the indexer get populated), so any
    phase built entirely from unrepresented components is invisible in the GT
    phase-fraction stack, and the *other* GT phases end up renormalized over a
    smaller total (i.e. slightly overstated). This reads fort.99 directly to
    quantify what's missing.

    Returns
    -------
    dict with:
        'missing_fraction' : (n,) array — fraction of each row's total assemblage
                              moles attributable to unrepresented components
        'unresolved'       : {abbrev: {'resolved_name', 'n_active_rows',
                              'peak_fraction', 'P_at_peak', 'significant'}} for
                              every fort.99 component not represented by the
                              emulator's indexer. 'significant' marks whether
                              that component's own peak contribution exceeds
                              MISSING_PHASE_FRACTION_THRESHOLD; components below
                              that bar are still listed (name + peak fraction)
                              but may not individually explain a hatched span —
                              hatching is driven by the row-wise sum of *all*
                              unresolved components (see 'missing_fraction'), so
                              several sub-threshold components can jointly cross
                              the threshold in the same row.
    """
    comp_df = _safe_read_ws_table(str(sim_dir / 'fort.99'), skiprows=0)
    n = len(gt_component_moles)
    component_cols = list(comp_df.columns)[3:-2]
    label_names = set(indexer.label_names)

    total_all = np.zeros(n, dtype=np.float64)
    candidates = {}
    for comp_abbr in component_cols:
        values = pd.to_numeric(comp_df[comp_abbr], errors='coerce').fillna(0.0).to_numpy(dtype=np.float64)[:n]
        total_all += values
        comp_abbr_str = str(comp_abbr).strip()
        comp_name = _resolve_component_name_from_abbr(comp_abbr_str)
        if comp_name in label_names or not np.any(values > 0):
            continue
        candidates[comp_abbr_str] = (comp_name, values)

    total_resolved = gt_component_moles.sum(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        missing_fraction = np.where(total_all > 0, 1.0 - total_resolved / total_all, 0.0)

    unresolved = {}
    for comp_abbr_str, (comp_name, values) in candidates.items():
        with np.errstate(invalid='ignore', divide='ignore'):
            row_fraction = np.where(total_all > 0, values / total_all, 0.0)
        peak_idx = int(np.argmax(row_fraction))
        peak_fraction = float(row_fraction[peak_idx])
        unresolved[comp_abbr_str] = {
            'resolved_name': comp_name,
            'n_active_rows': int(np.sum(values > 0)),
            'peak_fraction': peak_fraction,
            'P_at_peak': float(P_gpa[peak_idx]),
            'significant': peak_fraction >= MISSING_PHASE_FRACTION_THRESHOLD,
        }

    return {'missing_fraction': missing_fraction, 'unresolved': unresolved}


def component_moles_to_phase_fractions(component_moles: np.ndarray, indexer) -> np.ndarray:
    """Aggregate per-component moles (indexer.label_names layout) into per-phase
    mole fractions (indexer.mass_phasedict layout) — the ground-truth analogue
    of the emulator's normalized 'phase_moles' output."""
    n_phases = max(indexer.mass_phasedict.values()) + 1
    phase_moles = np.zeros((component_moles.shape[0], n_phases), dtype=np.float64)
    for phase, col in indexer.mass_phasedict.items():
        idx = indexer.label_indices.get(phase)
        if idx is None or len(idx) == 0:
            continue
        phase_moles[:, col] = component_moles[:, idx].sum(axis=1)
    return normalize_phase_moles(phase_moles)


def compare_simulation_phases(sim_dir: Path) -> dict:
    composition = load_composition(sim_dir)
    comp_headers = list(composition.keys())

    gt = extract_bulk_properties_from_simulation_dir(str(sim_dir))
    fort56 = gt['fort56_bulk']
    P_gpa = fort56['P(GPa)'].astype(np.float64)
    T_k = fort56['T(K)'].astype(np.float64)
    S_jgk = fort56['S(J/g/K)'].astype(np.float64)
    n = len(P_gpa)

    indexer = HeFESToEmulatorCPU.isothermal_emulator.ml_indexer
    gt_component_moles = load_fort99_componentMoles(str(sim_dir), indexer)[:n]
    gt_phase_fractions = component_moles_to_phase_fractions(gt_component_moles, indexer)
    phase_coverage = check_missing_phases(sim_dir, indexer, gt_component_moles, P_gpa)

    comp_block = np.tile(np.array([[composition[k] for k in comp_headers]], dtype=np.float32), (n, 1))

    # Isothermal pathway: predict assemblage from (P, T).
    iso_features = np.column_stack([P_gpa, T_k, comp_block]).astype(np.float32)
    iso_headers = ['P(GPa)(System_main)', 'T(K)(System_main)'] + comp_headers
    with torch.no_grad():
        iso_out = HeFESToEmulatorCPU.ForwardMB(iso_features, headers=iso_headers, outputs=['phase_moles'])
    iso_phase_fractions = normalize_phase_moles(_to_numpy(iso_out['phase_moles']).astype(np.float64))

    # Isentropic pathway: predict assemblage from (P, S).
    isen_features = np.column_stack([P_gpa, S_jgk, comp_block]).astype(np.float32)
    isen_headers = ['P(GPa)(System_main)', 'S(J/g/K)(System_main)'] + comp_headers
    with torch.no_grad():
        isen_out = HeFESToEmulatorCPU.ForwardMB(isen_features, headers=isen_headers, outputs=['phase_moles'])
    isen_phase_fractions = normalize_phase_moles(_to_numpy(isen_out['phase_moles']).astype(np.float64))

    return {
        'sim_dir': sim_dir,
        'P_gpa': P_gpa,
        'mass_phasedict': indexer.mass_phasedict,
        'ordered_phases': build_ordered_phases(indexer.mass_phasedict),
        'colors': phase_colors(),
        'gt_phase_fractions': gt_phase_fractions,
        'isothermal_phase_fractions': iso_phase_fractions,
        'isentropic_phase_fractions': isen_phase_fractions,
        'phase_coverage': phase_coverage,
    }


def print_phase_coverage_report(results: list) -> None:
    """Print, per simulation, every HeFESTo component that the emulator's
    indexer cannot represent at all, sorted by peak contribution to the
    assemblage (largest first). Components below MISSING_PHASE_FRACTION_THRESHOLD
    are still listed but marked '(below noise threshold)' -- individually they
    don't cross the bar, though several of them together can still push a
    row's *combined* missing fraction over threshold and trigger hatching in
    the plot (see check_missing_phases)."""
    print('\n' + '=' * 78)
    print('Phase coverage check -- components in fort.99 not represented by the emulator')
    print('=' * 78)
    any_unresolved = False
    for result in results:
        unresolved = result['phase_coverage']['unresolved']
        if not unresolved:
            continue
        any_unresolved = True
        print(f"\n{result['sim_dir'].name}:")
        for abbrev, info in sorted(unresolved.items(), key=lambda kv: kv[1]['peak_fraction'], reverse=True):
            note = '' if info['significant'] else '  (below noise threshold)'
            print(
                f"  '{abbrev}' (resolved as '{info['resolved_name']}')  "
                f"active in {info['n_active_rows']}/{len(result['P_gpa'])} rows  "
                f"peak={info['peak_fraction'] * 100:.3f}% of assemblage at P={info['P_at_peak']:.2f} GPa"
                f"{note}"
            )
    if not any_unresolved:
        print('\nNone -- every component present in fort.99 is represented in the emulator.')


def _flagged_pressure_spans(pres_sorted: np.ndarray, flagged: np.ndarray) -> list:
    """Group a boolean mask (aligned to ascending pres_sorted) into contiguous (start, end) spans."""
    spans = []
    start = None
    for i, f in enumerate(flagged):
        if f and start is None:
            start = i
        elif not f and start is not None:
            spans.append((pres_sorted[start], pres_sorted[i - 1]))
            start = None
    if start is not None:
        spans.append((pres_sorted[start], pres_sorted[-1]))
    return spans


def plot_phase_comparison(results: list, mode: str, save_path: str) -> None:
    """mode is 'isothermal' or 'isentropic'."""
    assert mode in ('isothermal', 'isentropic')
    emulator_key = f'{mode}_phase_fractions'
    n_sims = len(results)

    fig, axes = plt.subplots(n_sims, 2, figsize=(11, 3.4 * n_sims), squeeze=False)

    for row, result in enumerate(results):
        pressures = result['P_gpa']
        row_idx = np.argsort(pressures)
        ordered_phases = result['ordered_phases']
        mass_phasedict = result['mass_phasedict']
        colors = result['colors']
        sim_name = result['sim_dir'].name
        coverage = result['phase_coverage']

        gt_title = f'{sim_name} -- HeFESTo (GT)'
        if coverage['unresolved']:
            gt_title += '  [UNREPRESENTED PHASE]'

        draw_phase_stack(
            axes[row][0], result['gt_phase_fractions'], row_idx, pressures,
            ordered_phases, mass_phasedict, colors, gt_title,
        )
        draw_phase_stack(
            axes[row][1], result[emulator_key], row_idx, pressures,
            ordered_phases, mass_phasedict, colors, f'{sim_name} -- Emulator ({mode})',
        )

        # Hatch the pressure range(s) where an unrepresented component makes up
        # a non-trivial share of the real assemblage, on the GT panel.
        flagged = coverage['missing_fraction'][row_idx] > MISSING_PHASE_FRACTION_THRESHOLD
        if flagged.any():
            pres_sorted = pressures[row_idx]
            pad = 0.15 * (pres_sorted[-1] - pres_sorted[0]) / max(len(pres_sorted) - 1, 1)
            for start_p, end_p in _flagged_pressure_spans(pres_sorted, flagged):
                if start_p == end_p:
                    start_p, end_p = start_p - pad, end_p + pad
                axes[row][0].axvspan(start_p, end_p, facecolor='none', edgecolor='black',
                                      hatch='//', alpha=0.6, zorder=5)

        for ax in axes[row]:
            ax.set_xlabel('P (GPa)')

    mode_label = 'Isothermal (P, T)' if mode == 'isothermal' else 'Isentropic (P, S)'
    fig.suptitle(f'Phase abundances -- {mode_label} pathway: GT vs emulator', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(save_path, dpi=150, facecolor='white')
    plt.close(fig)
    print(f'Saved -> {save_path}')


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
    parser.add_argument('--save-path-isothermal', type=str, default='phase_comparison_isothermal.png')
    parser.add_argument('--save-path-isentropic', type=str, default='phase_comparison_isentropic.png')
    args = parser.parse_args()

    workspace_dir = REPO_ROOT / args.workspace
    chosen = select_sim_dirs(workspace_dir, sims=args.sims, n=args.n, seed=args.seed)

    print(f'Selected {len(chosen)} simulations from {workspace_dir}:')
    for d in chosen:
        print(f'  {d.name}')

    results = [compare_simulation_phases(d) for d in chosen]

    print_phase_coverage_report(results)

    plot_phase_comparison(results, mode='isothermal', save_path=args.save_path_isothermal)
    plot_phase_comparison(results, mode='isentropic', save_path=args.save_path_isentropic)
