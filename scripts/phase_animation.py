"""Animated Phase Diagrams — emulator-only phase abundance animations across pressure transects."""

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

from src.module.engine.API import HeFESToEmulatorCPU
from src.module.config.constants import HEFESTO_ABBREVIATION_TO_SHORT_NAMES

INDEX_TO_PHASE = {
     1: 'plg',  2: 'sp',  3: 'opx',  4: 'c2c',  5: 'cpx',
     6: 'wo',   7: 'pwo', 8: 'gt',   9: 'cpv', 10: 'ol',
    11: 'wa',  12: 'ri',  13: 'il',  14: 'pv',  15: 'ppv',
    16: 'cf',  17: 'nal', 18: 'mw',  19: 'qtz', 20: 'coes',
    21: 'st',  22: 'apbo', 23: 'ky', 24: 'neph', 25: 'fea',
    26: 'feg', 27: 'fee',
}

IP_PYROLITE = [19, 20, 21, 22, 10, 11, 12, 18, 14, 15, 13, 8, 16, 9,
               3, 4, 5, 1, 2, 23, 24, 17, 25, 26, 27, 6, 7]


def _phase_colors() -> dict:
    col = {}
    for p in ['qtz', 'coes', 'st', 'apbo', 'cacl']:
        col[p] = 'dimgray'
    for p in ['plg', 'pwo', 'wo', 'neph']:
        col[p] = 'lightgray'
    for p in ['ky', 'nal']:
        col[p] = 'mediumpurple'
    for p in ['opx', 'cpx', 'c2c']:
        col[p] = 'steelblue'
    col['sp']  = 'brown'
    col['gt']  = 'red'
    for p in ['ol', 'wa', 'ri', 'il']:
        col[p] = 'forestgreen'
    col['pv']  = 'khaki'
    col['ppv'] = 'goldenrod'
    col['cf']  = 'wheat'
    col['cpv'] = 'tan'
    for p in ['fea', 'feg', 'fee']:
        col[p] = 'sienna'
    col['mw'] = 'magenta'
    return col


def _build_ordered_phases(mass_phasedict: dict) -> list:
    """Return (long_name, abbrev) pairs in IP_PYROLITE order, restricted to mass_phasedict."""
    ordered = []
    for hf_idx in IP_PYROLITE:
        abbrev = INDEX_TO_PHASE[hf_idx]
        if abbrev not in HEFESTO_ABBREVIATION_TO_SHORT_NAMES:
            continue
        long_name = HEFESTO_ABBREVIATION_TO_SHORT_NAMES[abbrev]
        if long_name in mass_phasedict:
            ordered.append((long_name, abbrev))
    return ordered


def _draw_stack(ax, data: np.ndarray, row_idx: np.ndarray,
                pressures: np.ndarray, ordered_phases: list,
                mass_phasedict: dict, colors: dict, title: str) -> None:
    ax.cla()
    pres = pressures[row_idx]
    bottoms = np.zeros(len(pres), dtype=np.float32)
    for long_name, abbrev in ordered_phases:
        col = mass_phasedict[long_name]
        vals = np.clip(data[row_idx, col], 0.0, None)
        top = bottoms + vals
        ax.fill_between(pres, bottoms, top,
                        facecolor=colors.get(abbrev, '#aaaaaa'),
                        edgecolor='k', linewidth=0.4)
        peak = float(np.nanmax(vals))
        if peak > 0.03:
            band = vals > 0.35 * peak
            if band.any():
                x_c = np.average(pres[band], weights=vals[band])
                y_c = np.average(0.5 * (bottoms[band] + top[band]), weights=vals[band])
                ax.text(x_c, y_c, abbrev, ha='center', va='center',
                        fontsize=7, clip_on=True)
        bottoms = top
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(pres.min(), pres.max())
    ax.set_ylabel('Phase fraction')
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.2)


def _strip_suffix(header: str) -> str:
    return header.replace('(System_main)', '').strip()


def animate_phase_diagram(
    features: np.ndarray,
    headers: list,
    save_path: str = 'phase_animation.gif',
    interval: int = 50,
) -> animation.FuncAnimation:
    """Animate emulator phase abundances over a pressure grid.

    Parameters
    ----------
    features  : (N, F) array — full grid of input features
    headers   : length-F list of column names, e.g.
                ['P(GPa)(System_main)', 'S(J/g/K)(System_main)', 'Na2O', 'SiO2', ...]
    save_path : output GIF path
    interval  : milliseconds per animation frame

    Exactly one or two non-pressure columns may vary across rows:
      * one  → single subplot; animation sweeps that variable in ascending order
      * two  → subplots for the variable with fewer unique values (≤ 5),
                animation sweeps the variable with more unique values
    """
    features = np.asarray(features, dtype=np.float32)

    t0 = time.time()
    with torch.no_grad():
        output = HeFESToEmulatorCPU.ForwardMB(
            features, headers=headers, outputs=['phase_moles']
        )
    print(f'Emulator: {features.shape[0]} assemblages in {time.time() - t0:.2f}s')

    indexer = HeFESToEmulatorCPU.isentropic_emulator.ml_indexer
    mass_phasedict = indexer.mass_phasedict

    phase_data = output['phase_moles']
    if hasattr(phase_data, 'detach'):
        phase_data = phase_data.detach().cpu().numpy()
    phase_data = np.asarray(phase_data, dtype=np.float32)
    row_totals = phase_data.sum(axis=1, keepdims=True)
    row_totals = np.where(row_totals == 0, 1.0, row_totals)
    phase_data = phase_data / row_totals

    # --- Identify pressure column ---
    p_col = next((i for i, h in enumerate(headers) if 'P(GPa)' in h), None)
    if p_col is None:
        raise ValueError("No pressure column found — expected a header containing 'P(GPa)'")
    pressures = features[:, p_col]

    # --- Detect other varying columns (unique values > 1, excluding pressure) ---
    varying = []
    for i, h in enumerate(headers):
        if i == p_col:
            continue
        unique_vals = np.unique(np.round(features[:, i], 4))
        if len(unique_vals) > 1:
            varying.append((i, _strip_suffix(h), unique_vals))

    if len(varying) not in (1, 2):
        names = ', '.join(name for _, name, _ in varying)
        raise ValueError(
            f'Expected 1 or 2 other varying features besides pressure, '
            f'found {len(varying)}: {names}'
        )

    colors = _phase_colors()
    ordered_phases = _build_ordered_phases(mass_phasedict)

    # --- Case 1: one other varying feature ---
    if len(varying) == 1:
        var_col, var_name, var_vals = varying[0]
        var_rounded = np.round(features[:, var_col], 4)
        n_frames = len(var_vals)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_xlabel('Pressure (GPa)')

        def update_single(frame):
            v = var_vals[frame]
            idx = np.where(var_rounded == v)[0]
            idx = idx[np.argsort(pressures[idx])]
            _draw_stack(ax, phase_data, idx, pressures, ordered_phases,
                        mass_phasedict, colors, f'{var_name} = {v:.4g}')
            fig.suptitle(f'Frame {frame + 1}/{n_frames}  |  {var_name} ascending', fontsize=12)
            return []

        anim = animation.FuncAnimation(
            fig, update_single, frames=n_frames, interval=interval, blit=False, repeat=True,
        )

    # --- Case 2: two other varying features ---
    else:
        va, vb = varying
        subplot_var, anim_var = (va, vb) if len(va[2]) <= len(vb[2]) else (vb, va)

        sp_col, sp_name, sp_vals = subplot_var
        an_col, an_name, an_vals = anim_var

        assert len(sp_vals) <= 5, (
            f"Subplot variable '{sp_name}' has {len(sp_vals)} unique values (max 5). "
            "The variable with fewer unique values is assigned to subplots."
        )

        sp_rounded = np.round(features[:, sp_col], 4)
        an_rounded = np.round(features[:, an_col], 4)
        n_subplots = len(sp_vals)
        n_frames   = len(an_vals)

        n_cols = 2 if n_subplots >= 4 else 1
        n_rows = (n_subplots + n_cols - 1) // n_cols
        fig, axes_arr = plt.subplots(n_rows, n_cols,
                                     figsize=(6 * n_cols, 5 * n_rows),
                                     sharey=True, constrained_layout=True)
        axes_flat = np.asarray(axes_arr).reshape(-1).tolist()
        for ax in axes_flat[n_subplots:]:   # hide any unused grid cells
            ax.set_visible(False)
        axes = axes_flat[:n_subplots]
        for ax in axes:
            ax.set_xlabel('Pressure (GPa)')

        def update_multi(frame):
            an_v = an_vals[frame]
            for ax, sp_v in zip(axes, sp_vals):
                idx = np.where((sp_rounded == sp_v) & (an_rounded == an_v))[0]
                idx = idx[np.argsort(pressures[idx])]
                _draw_stack(ax, phase_data, idx, pressures, ordered_phases,
                            mass_phasedict, colors, f'{sp_name} = {sp_v:.4g}')
            fig.suptitle(
                f'{an_name} = {an_v:.4g}  |  Frame {frame + 1}/{n_frames}',
                fontsize=12,
            )
            return []

        anim = animation.FuncAnimation(
            fig, update_multi, frames=n_frames, interval=interval, blit=False, repeat=True,
        )

    plt.close(fig)
    anim.save(save_path, writer='pillow', savefig_kwargs={'facecolor': 'white'}, dpi = 300)
    print(f'Saved → {save_path}')
    return anim


if __name__ == '__main__':
    from src.module.utils.math_utils import grid_sample

    # Caracas et al., 2019 pyrolite (fixed composition)
    input_dict = {
        'SiO2': 45.0,
        'Al2O3': 4.45,
        'FeO': 8.05,
        'MgO': 37.8,
        'CaO': 3.55,
        'Fe2O3': 0.30,
        'Cr2O3': 0.1
    }
    comp_array = np.zeros((1, len(input_dict)), dtype=np.float32)
    comp_headers = list(input_dict.keys())
    for i, val in enumerate(input_dict.values()):
        comp_array[0, i] = val

    # Grid: P(300 steps) × S(4 steps) × Na2O(50 steps)
    # → subplot variable = S (4 unique values), animation variable = Na2O (50 frames)
    features = grid_sample([[0, 140, 140], [2.4, 2.7, 4], [0.5, 4, 50]], comp_array)
    headers = ['P(GPa)(System_main)', 'S(J/g/K)(System_main)', 'Na2O'] + comp_headers



    animate_phase_diagram(features, headers, save_path='pyrolite_phase_animation.gif')


