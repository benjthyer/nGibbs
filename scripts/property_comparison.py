"""Static thermodynamic property comparison — emulator vs HeFESTo ground truth.

Picks N random, complete simulation directories from a HeFESTo workspace (e.g.
data/HeFESToWorkspace/JiChingSims/), reads each simulation's composition
(control) and (P, T, S) profile, and compares three ways of computing
bulk properties (density, VP, VS, entropy, Cp, KS) at the same conditions:

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

from src.ngibbs.engine.API import HeFESToEmulatorCPU as HeFESToEmulatorCPU
from src.ngibbs.engine.EOS_arithmetic.hefesto_vec import load_control, compute_within_phase_frac
from builder.HeFESTo.HeFESTo_functions import (
    extract_bulk_properties_from_simulation_dir,
    load_fort99_componentMoles,
)
from src.ngibbs.utils.file_utils import _parse_control_file

REQUIRED_FILES = ['control', 'fort.56', 'fort.61', 'fort.68', 'fort.99']
ELEMENT_KEYS = ['Si', 'Mg', 'Fe', 'Ca', 'Al', 'Na', 'Cr', 'O']

# HeFESTo-vec EOS property key -> (fort.56 column, axis label).
#
# Cp and KS used to be excluded here, on the grounds that "fort.56's KS(GPa)
# uses a different aggregation scheme".  That is now understood: fort.56's KS is
# `bstot`, the *metamorphic* (phase-change-softened) aggregate modulus, not the
# isomorphic VRH Hill `Kh`.  Comparing Kh against it gives ~30% mean error;
# comparing KStot gives 0.016%.  Same story for cp -> cptot.
#
# Likewise VP: fort.56's VP carries the intra-phase order-disorder relaxation,
# so the comparable key is `Vp_fast`, not the isomorphic `Vp` (which is biased
# high by up to 0.26% in the lower mantle).  VS is unaffected — the shear
# modulus has no metamorphic term — so plain 'Vs' is correct.
#
# Requesting any of cptot/KStot/Vp_fast makes the API run the metamorphic pass;
# see HeFESToAPI._METAMORPHIC_KEYS.
PROPERTY_MAP = {
    'rho':     ('rho(g/cm^3)', 'rho (g/cm3)'),
    'Vp_fast': ('VP(km/s)',    'VP (km/s)'),
    'Vs':      ('VS(km/s)',    'VS (km/s)'),
    'S':       ('S(J/g/K)',    'S (J/g/K)'),
    'cptot':   ('cp(J/g/K)',   'Cp (J/g/K)'),
    'KStot':   ('KS(GPa)',     'KS (GPa)'),
}

# HeFESTo_Parameters_010123 embedded in the control files points at the
# original run machine (macOS) — override with the local copy.
PARAM_DIR = SRC_ROOT / 'ngibbs' / 'engine' / 'EOS_arithmetic' / 'HeFESTo_Parameters_010123'


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, 'detach'):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def find_complete_sim_dirs(workspace_dir: Path) -> List[Path]:
    if not workspace_dir.exists():
        raise ValueError(f'Workspace directory {workspace_dir} does not exist')
    out = sorted(
        d for d in workspace_dir.glob('model_*')
        if all((d / f).exists() for f in REQUIRED_FILES)
    )
    if len(out) == 0:
        return sorted(
            d for d in workspace_dir.glob('Simulation*')
            if all((d / f).exists() for f in REQUIRED_FILES)
        )
    return out


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
    element_moles, _ = _parse_control_file(str(sim_dir / 'control'))
    return {k: float(element_moles[k]) for k in ELEMENT_KEYS}


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


# Set by main() when --diagnose is passed: adds the per-species derivatives the
# trace-species diagnostics need.  They cost nothing extra — the metamorphic
# solve already computes them — but they are (B, nspec), so they are only
# requested when actually wanted.
DIAGNOSE = False


def property_names() -> list:
    names = list(PROPERTY_MAP.keys())
    if DIAGNOSE:
        names += ['dndt', 'sspeca']
    return names


def to_full_species(component_moles: np.ndarray) -> np.ndarray:
    """Component moles (emulator label space) -> full (B, nspec) species array.

    Mirrors what HeFESToAPI.get_property_hefesto_vectorized_from_assemblage does
    internally, so the diagnostics below see exactly the array the EOS sees.
    """
    cm = np.asarray(component_moles, dtype=np.float64)
    X = np.zeros((cm.shape[0], HeFESToEmulatorCPU.hefesto_params.nspec), dtype=np.float64)
    X[:, HeFESToEmulatorCPU.PropertyIDX] = cm
    return X


def diagnose_trace_species(result: dict, mode: str) -> None:
    """Report whether trace species are driving the latent-heat term.

    The metamorphic Cp/KS terms come from a dn/dT solve in which a species that
    is the sole occupant of an otherwise-absent phase is completely
    unregularised (its mole fraction within that phase is ~1, so the ideal
    configurational penalty never fires).  Such a species carries no mass or
    volume, so it is invisible in rho/Vp/Vs and in the phase proportions, but it
    can dominate cpmet and produce a spurious Cp spike or drag a transition to
    higher pressure.

    hefesto_vec.prune_active_set guards against this, but only if
    HeFESToAPI.metamorphic_nsmall_rel sits ABOVE the noise floor of whatever
    produced the moles.  This function measures that floor so the threshold can
    be set from evidence.  Read `min depletion` first: it is threshold-free.
    """
    from src.ngibbs.engine.EOS_arithmetic.hefesto_vec import trace_species_leverage

    nsmall = HeFESToEmulatorCPU.metamorphic_nsmall_rel
    # Leverage is reported at a FIXED 1e-3 reference, not at nsmall: measuring it
    # at the threshold that just did the pruning always returns ~0 and would hide
    # a species sitting just above the threshold.  min depletion is threshold-free.
    LEV_REF = 1.0e-3
    print(f"\n  trace-species diagnostics  (metamorphic_nsmall_rel = {nsmall:.0e})")
    print(f"    {'source':22s} {'min nonzero':>12s} {'#below thresh':>14s} "
          f"{'leverage@1e-3':>14s} {'min depletion':>14s}")
    for key, label in (('gt_internal', 'GT assemblage'),
                       (f'{mode}_emulator', f'{mode} emulator')):
        props = result.get(key) or {}
        cm = props.get('component_moles')
        if cm is None or 'dndt' not in props:
            print(f"    {label:22s} {'(not captured — rerun with --diagnose)':>50s}")
            continue
        X = to_full_species(cm)
        tot = X.sum(axis=1, keepdims=True)
        nz = X[X > 0]
        d = trace_species_leverage(props['dndt'], X, props['sspeca'],
                                   result['T_k'], nsmall_rel=LEV_REF)
        n_trace = int(((X > 0) & (X < nsmall * tot)).sum())
        print(f"    {label:22s} {nz.min() if nz.size else 0:12.2e} {n_trace:14d} "
              f"{d['leverage'].max():14.4f} {np.min(d['depletion_K']):11.2e} K")
    print("    Read 'min depletion' first — it is the threshold-free indicator.")
    print("    healthy: >~ 1e-1 K.   pathological: <~ 1e-2 K (a species depleting")
    print("    itself in microkelvins is an unregularised trace species, not physics).")
    print("    If the emulator row is bad and the GT row is not, set")
    print("    --nsmall-rel above the emulator's 'min nonzero' and rerun.")


def _expand_chem_out_to_components(chem_out: np.ndarray, indexer) -> np.ndarray:
    """Expand the emulator's 'chem_out' output (within-phase mole fraction,
    but only for the compositionally-variable subset of components) back to
    full (B, C) component space.

    chem_out is itself a slice of a full (B, C) within-phase-fraction array
    (see NN_MELTS.forwardMB/forwardNN: it computes componentMoles/phaseMoles
    summed over phases -- which, since each component belongs to exactly one
    phase, is just that phase's within-phase fraction in full C-space -- then
    slices to compositionally_variable_subset). The complement (components of
    single-member, "pure" phases) is missing because it's trivially 1.0: a
    phase's sole member IS its own whole phase.
    """
    B = chem_out.shape[0]
    full = np.ones((B, len(indexer.label_names)), dtype=np.float64)
    vc = np.asarray(indexer.compositionally_variable_subset, dtype=np.int64)
    full[:, vc] = chem_out
    return full


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

    # Within-phase mole fraction, computed directly from the true assemblage
    # (no network output to reuse here, unlike the emulator paths below) --
    # this is a quality-of-EOS-reconstruction check, not a speed-sensitive
    # path, so a plain per-phase division is fine.
    gt_within_phase_frac = compute_within_phase_frac(
        gt_component_moles, list(indexer.label_indices.values())
    )

    PT = np.stack([P_gpa, T_k], axis=1)
    gt_internal_props = HeFESToEmulatorCPU.get_property_hefesto_vectorized_from_assemblage(
        torch.tensor(gt_component_moles, dtype=torch.float64),
        torch.tensor(PT, dtype=torch.float64),
        property_names=property_names(),
        within_phase_frac=torch.tensor(gt_within_phase_frac, dtype=torch.float64),
    )
    gt_internal_props['component_moles'] = gt_component_moles

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
        out = HeFESToEmulatorCPU.ForwardMB(
            features, headers=headers, outputs=['component_moles', 'chem_out']
        )
    component_moles = _to_numpy(out['component_moles']).astype(np.float64)
    # Within-phase mole fraction is already computed as part of this same
    # forward pass -- reuse it directly rather than re-deriving it from
    # component_moles by division.
    within_phase_frac = _expand_chem_out_to_components(
        _to_numpy(out['chem_out']).astype(np.float64),
        HeFESToEmulatorCPU.isothermal_emulator.ml_indexer,
    )

    PT = np.stack([P_gpa, T_k], axis=1)
    props = HeFESToEmulatorCPU.get_property_hefesto_vectorized_from_assemblage(
        torch.tensor(component_moles, dtype=torch.float64),
        torch.tensor(PT, dtype=torch.float64),
        property_names=property_names(),
        within_phase_frac=torch.tensor(within_phase_frac, dtype=torch.float64),
    )
    props['component_moles'] = component_moles
    return props


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
            features, headers=headers, outputs=['component_moles', 'chem_out', 'temperature']
        )
    component_moles = _to_numpy(out['component_moles']).astype(np.float64)
    T_emulated = _to_numpy(out['temperature']).reshape(-1).astype(np.float64)
    within_phase_frac = _expand_chem_out_to_components(
        _to_numpy(out['chem_out']).astype(np.float64),
        HeFESToEmulatorCPU.isentropic_emulator.ml_indexer,
    )

    PT = np.stack([P_gpa, T_emulated], axis=1)
    props = HeFESToEmulatorCPU.get_property_hefesto_vectorized_from_assemblage(
        torch.tensor(component_moles, dtype=torch.float64),
        torch.tensor(PT, dtype=torch.float64),
        property_names=property_names(),
        within_phase_frac=torch.tensor(within_phase_frac, dtype=torch.float64),
    )
    props['T_emulated'] = T_emulated
    props['component_moles'] = component_moles
    return props


def compare_simulation(sim_dir: Path) -> dict:
    composition = load_composition(sim_dir)
    gt = load_ground_truth(sim_dir)
    result = {
        'sim_dir': sim_dir,
        'composition': composition,
        'P_gpa': gt['P_gpa'],
        'T_k': gt['T_k'],
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

    # Panel width shrinks as columns are added so the figure stays legible when
    # the metamorphic properties (Cp, KS) push the grid to 6-7 columns.
    col_w = 3.6 if n_cols <= 5 else 3.0
    fig, axes = plt.subplots(n_sims, n_cols, figsize=(col_w * n_cols, 3.2 * n_sims), squeeze=False)

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
                    f'    {prop:>8s} ({label})  mean_abs_err={abs_err.mean():.4g}  '
                    f'mean_rel_err={rel_err.mean() * 100:.3f}%  max_rel_err={rel_err.max() * 100:.3f}%'
                )
        if mode == 'isentropic':
            T_gt = result['fort56']['T(K)'].astype(np.float64)
            T_em = result['isentropic_emulator']['T_emulated']
            abs_err, rel_err = compute_error_stats(T_gt, T_em)
            print(
                f'    {"T":>8s} (T (K))  mean_abs_err={abs_err.mean():.4g}  '
                f'mean_rel_err={rel_err.mean() * 100:.3f}%  max_rel_err={rel_err.max() * 100:.3f}%'
            )
        if DIAGNOSE:
            diagnose_trace_species(result, mode)


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
    parser.add_argument('--diagnose', action='store_true',
                         help='report whether trace species are driving the metamorphic '
                              'Cp/KS terms. Use this to read off your emulator\'s noise '
                              'floor and set HeFESToAPI.metamorphic_nsmall_rel from it.')
    parser.add_argument('--nsmall-rel', type=float, default=None,
                         help='override HeFESToAPI.metamorphic_nsmall_rel, the active-set '
                              'smallness threshold for the metamorphic solve (fraction of '
                              'total moles). Must exceed the emulator noise floor.')
    parser.add_argument('--save-path-isothermal', type=str, default='property_comparison_isothermal.png')
    parser.add_argument('--save-path-isentropic', type=str, default='property_comparison_isentropic.png')
    args = parser.parse_args()
    DIAGNOSE = args.diagnose

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
    if args.nsmall_rel is not None:
        HeFESToEmulatorCPU.metamorphic_nsmall_rel = args.nsmall_rel
        print(f'metamorphic_nsmall_rel = {args.nsmall_rel:.0e}')

    results = [compare_simulation(d) for d in chosen]

    print_error_summary(results, mode='isothermal')
    print_error_summary(results, mode='isentropic')

    plot_comparison(results, mode='isothermal', save_path=args.save_path_isothermal)
    plot_comparison(results, mode='isentropic', save_path=args.save_path_isentropic)
