"""Animated ternary composition comparison: emulator vs ground truth.

Sweeps over pressure at a fixed entropy and animates phase mineral compositions
in two ternary spaces:
  1. Ca – Mg – Fe²⁺  (all phases; olivine/wadsleyite/ringwoodite also here)
  2. Si – Al – Fe³⁺  (shown as a trapezoid whose north limit = max Fe³⁺ in dataset)

Emulated results → red 'x' markers (no text labels)
Ground truth      → black '+' markers with phase abbreviation text labels

Ground truth is optional: pass ``ground_truth_el_comps`` as a dict
  {phase_long_name: (N_P, E) ndarray}
where rows correspond to the same pressure array used for the emulator.
Leave it as None to show emulator results only.
"""

import sys
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

base_path = str(Path(__file__).parent.parent)
if base_path not in sys.path:
    sys.path.insert(0, base_path)

from src.ngibbs.engine.API import HeFESToEmulatorCPU
from src.ngibbs.utils.math_utils import make_ternary, grid_sample, mix_compositions
from src.ngibbs.utils.file_utils import load_fort99_componentMoles, extract_bulk_properties_from_simulation_dir


# ── Phase configuration ────────────────────────────────────────────────────────
# Long names must match HeFESToEmulatorCPU.isentropic_emulator.ml_indexer.all_phases
# Print that list to verify if you get KeyErrors.
PHASES_BOTH = [
    'garnet', 'bridgmanite', 'clinopyroxene', 'orthopyroxene',
    'post-perovskite', 'ferropericlase', 'hp-clinopyroxene'
]
PHASES_SPACE1_ONLY = ['olivine', 'wadsleyite', 'ringwoodite']

PHASE_ABBREV = {
    'garnet':           'gt',
    'bridgmanite':      'pv',
    'clinopyroxene':    'cpx',
    'orthopyroxene':    'opx',
    'hp-clinopyroxene': 'c2c',
    'post-perovskite':  'ppv',
    'ferropericlase':   'mw',
    'olivine':          'ol',
    'wadsleyite':       'wa',
    'ringwoodite':      'ri',
}

PHASE_COLORS = {
    'garnet':           'red',
    'bridgmanite':      '#b8860b',
    'clinopyroxene':    'steelblue',
    'orthopyroxene':    'skyblue',
    'hp-clinopyroxene': 'royalblue',
    'post-perovskite':  'goldenrod',
    'ferropericlase':   'magenta',
    'olivine':          'forestgreen',
    'wadsleyite':       'limegreen',
    'ringwoodite':      'darkgreen',
}

# ── Default composition (DMM-PS from Workman & Hart 2005) ─────────────────────
DMM_PS = {
    'Si': 3.79222, 'Mg': 4.88475, 'Fe': 0.57874,
    'Ca': 0.28734, 'Al': 0.39685, 'Na': 0.02197,
    'Cr': 0.03813, 'O':  14.00450,
}

S_FIXED     = 2.8    # J/g/K
N_PRESSURES = 140    # frames / pressure steps (0 → 140 GPa)


# ── Element-composition helpers ───────────────────────────────────────────────
def phase_element_compositions(component_moles_np, mli, phases):
    """Map component moles to element abundances for each phase.

    Parameters
    ----------
    component_moles_np : (N, C) ndarray
    mli                : ml_indexer from emulator
    phases             : list of phase long-names

    Returns
    -------
    dict {phase: (N, E) ndarray} — phases absent from model are skipped
    """
    out = {}
    for phase in phases:
        if phase not in mli.label_indices:
            continue
        idx = np.asarray(mli.label_indices[phase])
        if idx.size == 0:
            continue
        # (N, n_comp) @ (n_comp, O) @ (O, E) → (N, E)
        out[phase] = component_moles_np[:, idx] @ mli.compToOx[idx] @ mli.OxToEl
    return out


def _ternary_coords(el, i0, i1, i2_main, i2_sub=None):
    """Build (N, 3) normalised ternary coords from element columns.

    i2_main and optional i2_sub: the third axis is max(0, el[:,i2_main] - el[:,i2_sub])
    so Fe²⁺ = max(0, Fe_total - Fe³⁺).
    """
    v0 = el[:, i0] if i0 is not None else np.zeros(el.shape[0])
    v1 = el[:, i1] if i1 is not None else np.zeros(el.shape[0])
    v2 = el[:, i2_main] if i2_main is not None else np.zeros(el.shape[0])
    if i2_sub is not None:
        v2 = np.maximum(v2 - el[:, i2_sub], 0.0)

    raw = np.clip(np.stack([v0, v1, v2], axis=1), 0.0, None)
    total = raw.sum(axis=1, keepdims=True)
    mask = total[:, 0] > 0#1e-10
    coords = raw.copy()
    coords[mask] /= total[mask]
    return coords, mask


# ── Main animation function ───────────────────────────────────────────────────
def animate_phase_comparison(
    composition: dict,
    save_path: str = 'phase_comparison.gif',
    interval: int = 80,
    ground_truth_el_comps: dict = None,
    ground_truth_PS: np.ndarray = None,
    max_pressure_gpa: float = None,
):
    """Run and save the animated ternary comparison.

    Parameters
    ----------
    composition           : element dict (e.g. DMM_PS) passed to the emulator
    save_path             : output GIF path
    interval              : ms per frame
    ground_truth_el_comps : optional dict {phase_name: (N, E) ndarray}
                            Pre-computed element abundances for the ground truth.
                            N must match the number of rows in ground_truth_PS.
                            E must match the emulator's ml_indexer.Elkeys length and order.
    ground_truth_PS       : optional (N, 2) ndarray — [P in Pa, S in J/g/K].
                            When supplied, the emulator is run at exactly these
                            (P, S) conditions (after converting Pa → GPa) instead
                            of the default fixed-entropy pressure grid.  Both the
                            emulator and GT results are then sorted by pressure for
                            the animation.
    max_pressure_gpa      : if given, frames with pressure above this value (GPa)
                            are excluded from the animation.
    """
    mli = HeFESToEmulatorCPU.isentropic_emulator.ml_indexer
    elkeys = list(mli.Elkeys)

    def _idx(key):
        return elkeys.index(key) if key in elkeys else None

    i_Ca  = _idx('Ca')
    i_Mg  = _idx('Mg')
    i_Fe  = _idx('Fe')
    i_Fe3 = _idx('Fe3')
    i_Si  = _idx('Si')
    i_Al  = _idx('Al')

    comp_headers = list(composition.keys())
    comp_array   = np.array([list(composition.values())], dtype=np.float32)

    # ── Build feature array and sort index ───────────────────────────────────
    if ground_truth_PS is not None:
        gt_P_gpa = ground_truth_PS[:, 0] / 1e9   # Pa → GPa
        gt_S     = ground_truth_PS[:, 1]           # J/g/K
        N        = len(gt_P_gpa)

        # sort_idx maps frame index → row index in the unsorted GT/emulator arrays
        sort_idx  = np.argsort(gt_P_gpa)
        pressures = gt_P_gpa[sort_idx]             # ascending GPa
        S_vals    = gt_S[sort_idx]                 # matching S for title
        n_frames  = N

        # Tile the single composition to every GT PS point
        feat_comp = np.tile(comp_array, (N, 1))    # (N, n_comp)
        features  = np.column_stack([gt_P_gpa, gt_S, feat_comp]).astype(np.float32)
        headers   = ['P(GPa)(System_main)', 'S(J/g/K)(System_main)'] + comp_headers

        print(f'GT PS grid: {N} points  '
              f'P=[{gt_P_gpa.min():.1f}, {gt_P_gpa.max():.1f}] GPa  '
              f'S=[{gt_S.min():.3f}, {gt_S.max():.3f}] J/g/K')
    else:
        features  = grid_sample([[0, 140, N_PRESSURES], [S_FIXED, S_FIXED, 1]], comp_array)
        headers   = ['P(GPa)(System_main)', 'S(J/g/K)(System_main)'] + comp_headers
        sort_idx  = np.arange(N_PRESSURES)
        pressures = features[sort_idx, 0]
        S_vals    = np.full(N_PRESSURES, S_FIXED)
        n_frames  = N_PRESSURES

    # ── Apply pressure ceiling (pressures is already sorted ascending) ────────
    if max_pressure_gpa is not None:
        keep      = pressures <= max_pressure_gpa
        sort_idx  = sort_idx[keep]
        pressures = pressures[keep]
        S_vals    = S_vals[keep]
        n_frames  = int(keep.sum())
        print(f'max_pressure_gpa={max_pressure_gpa:.1f} → {n_frames} frames retained')

    # ── Run emulator ─────────────────────────────────────────────────────────
    print(f'Running emulator on {len(features)} assemblages …')
    t0 = time.time()
    output = HeFESToEmulatorCPU.ForwardMB(
        features, headers=headers, outputs=['component_moles']
    )
    print(f'  done in {time.time()-t0:.1f}s')

    comp_moles = output['component_moles']
    if hasattr(comp_moles, 'detach'):
        comp_moles = comp_moles.detach().cpu().numpy()
    comp_moles = np.asarray(comp_moles, dtype=np.float64)

    all_phases = PHASES_BOTH + PHASES_SPACE1_ONLY

    # ── Element compositions → ternary coords (all rows, unsorted) ────────────
    em_el = phase_element_compositions(comp_moles, mli, all_phases)
    em_s1, em_s2 = {}, {}
    for phase, el in em_el.items():
        em_s1[phase] = _ternary_coords(el, i_Ca, i_Mg, i_Fe, i2_sub=i_Fe3)
        em_s2[phase] = _ternary_coords(el, i_Si, i_Al, i_Fe3)

    if ground_truth_el_comps is not None:
        gt_s1, gt_s2 = {}, {}
        for phase, el in ground_truth_el_comps.items():
            gt_s1[phase] = _ternary_coords(el, i_Ca, i_Mg, i_Fe, i2_sub=i_Fe3)
            gt_s2[phase] = _ternary_coords(el, i_Si, i_Al, i_Fe3)
    else:
        gt_s1 = gt_s2 = {}

    # ── Max Fe³⁺ across all rows for trapezoid north limit ───────────────────
    max_fe3 = 0.0
    for src in (em_s2, gt_s2):
        for phase in PHASES_BOTH:
            if phase in src:
                coords, mask = src[phase]
                if mask.any():
                    max_fe3 = max(max_fe3, float(coords[mask, 2].max()))
    max_fe3 = max(max_fe3, 0.05)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.subplots_adjust(left=0.03, right=0.97, wspace=0.15)
    tern1 = make_ternary(['Ca', 'Mg', 'Fe²⁺'], ax=ax1, max_top=1.0)
    tern2 = make_ternary(['Si', 'Al', 'Fe³⁺'], ax=ax2, max_top=max_fe3 * 1.15)

    def update(frame):
        tern1.clear_data()
        tern2.clear_data()

        # sort_idx maps animation frame → original (unsorted) row index
        i = sort_idx[frame]
        P = pressures[frame]
        S = S_vals[frame]

        em_present  = []   # phases with valid emulator coords at this frame
        gt_present  = []   # phases with valid GT coords at this frame
        em_absent   = []   # phases in model but mask[i]=False (zero/absent)
        gt_absent   = []

        for phase in all_phases:
            color  = PHASE_COLORS.get(phase, 'grey')
            abbrev = PHASE_ABBREV.get(phase, phase)

            # Emulator — space 1
            if phase in em_s1:
                coords, mask = em_s1[phase]
                if mask[i]:
                    em_present.append(abbrev)
                    tern1.add_points(coords[i].reshape(1, 3),
                                     marker='x', color=color, ls='none', ms=7, mew=1.5)
                else:
                    em_absent.append(abbrev)

            # Emulator — space 2
            if phase in PHASES_BOTH and phase in em_s2:
                coords, mask = em_s2[phase]
                if mask[i]:
                    tern2.add_points(coords[i].reshape(1, 3),
                                     marker='x', color=color, ls='none', ms=7, mew=1.5)

            # Ground truth — space 1
            if phase in gt_s1:
                coords, mask = gt_s1[phase]
                if mask[i]:
                    gt_present.append(abbrev)
                    tern1.add_points({abbrev: coords[i]},
                                     marker='+', color='k', ls='none', ms=8, mew=1.5)
                else:
                    gt_absent.append(abbrev)

            # Ground truth — space 2
            if phase in PHASES_BOTH and phase in gt_s2:
                coords, mask = gt_s2[phase]
                if mask[i]:
                    tern2.add_points({abbrev: coords[i]},
                                     marker='+', color='k', ls='none', ms=8, mew=1.5)

        print(
            f'frame {frame+1:>4d}/{n_frames}  '
            f'P={P:6.2f} GPa  S={S:.3f} J/g/K  '
            f'row={i}  '
            f'em=[{",".join(em_present) or "—"}]  '
            f'em_absent=[{",".join(em_absent) or "—"}]  '
            f'gt=[{",".join(gt_present) or "—"}]  '
            f'gt_absent=[{",".join(gt_absent) or "—"}]'
        )

        fig.suptitle(
            f'Phase compositions  |  P = {P:.1f} GPa,  S = {S:.3f} J/g/K  '
            f'(frame {frame+1}/{n_frames})',
            fontsize=13,
        )
        return []

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=interval, blit=False, repeat=True,
    )
    anim.save(save_path, writer='pillow', savefig_kwargs={'facecolor': 'white'}, dpi = 300)
    plt.close(fig)
    print(f'Saved → {save_path}')
    return anim


# ── Legend helper (static figure) ────────────────────────────────────────────
def _print_phase_legend():
    mli = HeFESToEmulatorCPU.isentropic_emulator.ml_indexer
    print('Phases in model:', mli.all_phases)
    print('Elkeys:',         list(mli.Elkeys))


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Uncomment to see model's exact phase/element names before running:
    # _print_phase_legend()

    # ── Optional: load ground truth element compositions ──────────────────────
    # Expected format: {phase_long_name: (N_PRESSURES, E) ndarray}
    # where rows correspond to pressures = np.linspace(0, 140, N_PRESSURES) GPa
    # and E matches HeFESToEmulatorCPU.isentropic_emulator.ml_indexer.Elkeys.
    #
    # Example (fill in your own loading logic):
    # import pandas as pd
    # gt_raw = pd.read_csv('path/to/hefesto_output.csv')
    # ground_truth = {
    #     'garnet':    gt_raw[['Ca','Mg','Fe','Fe3','Si','Al','Na','Cr']].to_numpy(),
    #     'bridgmanite': ...,
    # }
    ground_truth = None   # set to dict to enable GT overlay

    
    #sim_dir = 'C:\Git_Repositories\nMELTS\data\HeFESToWorkspace\DMM_PS'# 'nMELTS/data/HeFESToWorkspace/DMM_PS/'
    sim_dir =  'data/HeFESToWorkspace/DMM_PS/'

    indexer = HeFESToEmulatorCPU.isentropic_emulator.ml_indexer

    gt_componentMoles = load_fort99_componentMoles(sim_dir, indexer)

    

    gt_features = extract_bulk_properties_from_simulation_dir(sim_dir, indexer)

    # Build PT matrix from fort56
    fort56 = gt_features['fort56_bulk']
    P_gpa = fort56['P(GPa)']
    S_K = fort56['S(J/g/K)']
    PS = np.stack([P_gpa*1E9, S_K], axis=1)
    Desired_S = PS[:,1] == S_FIXED

    phase_el_comps = {}
    for phase in PHASES_BOTH + PHASES_SPACE1_ONLY:
        if phase not in indexer.label_indices:
            print(f'  skipping {phase!r} — not in indexer.label_indices')
            continue
        idx = indexer.label_indices[phase]
        phase_el_comps[phase] = (
            gt_componentMoles[np.ix_(Desired_S, idx)] @ indexer.compToOx[idx]
        ) @ indexer.OxToEl

    animate_phase_comparison(
        composition=DMM_PS,
        save_path='phase_composition_comparison.gif',
        interval=150,
        #ground_truth_el_comps=phase_el_comps,
        #ground_truth_PS=PS[Desired_S],
        max_pressure_gpa=40.0
    )
