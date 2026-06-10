"""
Phase-abundance animation utilities.

Generic primitives
------------------
_draw_stack              — draw one stacked-area frame onto a matplotlib Axes
animate_phase_diagram    — single-panel animation: X axis, Y animation frames
animate_phase_comparison — two-panel side-by-side for benchmarking/validation

MELTS helpers
-------------
melts_colors             — default color dict for ptt phase abbreviations
prepare_melts_phase_data — run MELTS emulator → (x_coords, y_coords, phase_grid, names, x_name, y_name)
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


# ---------------------------------------------------------------------------
# Generic primitives
# ---------------------------------------------------------------------------

def _draw_stack(ax, x_vals: np.ndarray, phase_frame: np.ndarray,
                phase_names: list, colors: dict, title: str,
                x_name: str = 'X') -> None:
    """Draw one stacked-area frame onto *ax*.

    Parameters
    ----------
    x_vals      : (Nx,) x-axis coordinates
    phase_frame : (Nx, P) phase fractions for this frame — columns match phase_names
    phase_names : length-P phase label strings (also used as color-dict keys)
    colors      : mapping phase name → matplotlib color
    title       : axes title for this frame
    x_name      : x-axis label
    """
    from src.ngibbs.config.constants import ptt_to_short
    x_vals = np.asarray(x_vals, dtype=np.float32).ravel()
    phase_frame = np.asarray(phase_frame, dtype=np.float32)
    phase_names = [ptt_to_short[n].lstrip('_') if n in ptt_to_short else n for n in phase_names]
    ax.cla()
    bottoms = np.zeros(len(x_vals), dtype=np.float32)
    for i, name in enumerate(phase_names):
        vals = np.clip(phase_frame[:, i], 0.0, None)
        top = bottoms + vals
        ax.fill_between(x_vals, bottoms, top,
                        facecolor=colors.get(name, '#aaaaaa'),
                        edgecolor='k', linewidth=0.4)
        peak = float(np.nanmax(vals))
        if peak > 0.03:
            band = vals > 0.35 * peak
            if band.any():
                x_c = np.average(x_vals[band], weights=vals[band])
                y_c = np.average(0.5 * (bottoms[band] + top[band]), weights=vals[band])
                ax.text(x_c, y_c, name, ha='center', va='center',
                        fontsize=7, clip_on=True)
        bottoms = top
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(x_vals.min(), x_vals.max())
    ax.set_xlabel(x_name)
    ax.set_ylabel('Phase fraction')
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.2)


def animate_phase_diagram(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    phase_abundances: np.ndarray,
    phase_names: list,
    x_name: str = 'X',
    y_name: str = 'Y',
    colors: dict = None,
    save_path: str = 'phase_animation.gif',
    interval: int = 50,
    dpi: int = 150,
) -> animation.FuncAnimation:
    """Animate stacked phase fractions along X, sweeping over Y.

    Parameters
    ----------
    x_coords        : (Nx,) x-axis values (e.g. pressure)
    y_coords        : (Ny,) animation-frame values (e.g. temperature)
    phase_abundances: (Ny, Nx, P) phase fraction array
    phase_names     : length-P phase label strings
    x_name          : x-axis label
    y_name          : animation-variable label shown in the title
    colors          : phase name → color mapping; grey fill used for unknowns
    save_path       : output GIF path
    interval        : milliseconds per frame
    dpi             : output resolution
    """
    x_coords = np.asarray(x_coords, dtype=np.float32)
    y_coords = np.asarray(y_coords, dtype=np.float32)
    phase_abundances = np.asarray(phase_abundances, dtype=np.float32)
    if colors is None:
        colors = {}

    Ny, Nx, P = len(y_coords), len(x_coords), len(phase_names)
    if phase_abundances.shape != (Ny, Nx, P):
        raise ValueError(
            f'phase_abundances shape {phase_abundances.shape} does not match '
            f'(Ny={Ny}, Nx={Nx}, P={P})'
        )

    fig, ax = plt.subplots(figsize=(10, 5))

    def update(frame):
        _draw_stack(ax, x_coords, phase_abundances[frame], phase_names, colors,
                    f'{y_name} = {y_coords[frame]:.4g}', x_name)
        fig.suptitle(f'Frame {frame + 1}/{Ny}', fontsize=12)
        return []

    anim = animation.FuncAnimation(
        fig, update, frames=Ny, interval=interval, blit=False, repeat=True,
    )
    plt.close(fig)
    anim.save(save_path, writer='pillow', savefig_kwargs={'facecolor': 'white'}, dpi=dpi)
    print(f'Saved → {save_path}')
    return anim


def animate_phase_comparison(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    phase_abundances_a: np.ndarray,
    phase_abundances_b: np.ndarray,
    phase_names: list,
    x_name: str = 'X',
    y_name: str = 'Y',
    label_a: str = 'A',
    label_b: str = 'B',
    colors: dict = None,
    save_path: str = 'phase_comparison.gif',
    interval: int = 50,
    dpi: int = 150,
) -> animation.FuncAnimation:
    """Side-by-side comparison of two phase-abundance datasets.

    Both datasets share the same X/Y grid and phase ordering.  Useful for
    benchmarking an emulator against reference calculations.

    Parameters
    ----------
    x_coords            : (Nx,) x-axis values
    y_coords            : (Ny,) animation-frame values
    phase_abundances_a  : (Ny, Nx, P) first dataset (e.g. emulator)
    phase_abundances_b  : (Ny, Nx, P) second dataset (e.g. reference)
    phase_names         : length-P phase label strings
    x_name              : x-axis label
    y_name              : animation-variable label
    label_a             : subplot title for the left panel
    label_b             : subplot title for the right panel
    colors              : phase name → color mapping
    save_path           : output GIF path
    interval            : milliseconds per frame
    dpi                 : output resolution
    """
    x_coords = np.asarray(x_coords, dtype=np.float32)
    y_coords = np.asarray(y_coords, dtype=np.float32)
    phase_abundances_a = np.asarray(phase_abundances_a, dtype=np.float32)
    phase_abundances_b = np.asarray(phase_abundances_b, dtype=np.float32)
    if colors is None:
        colors = {}

    Ny, Nx, P = len(y_coords), len(x_coords), len(phase_names)
    for tag, arr in (('a', phase_abundances_a), ('b', phase_abundances_b)):
        if arr.shape != (Ny, Nx, P):
            raise ValueError(
                f'phase_abundances_{tag} shape {arr.shape} does not match '
                f'(Ny={Ny}, Nx={Nx}, P={P})'
            )

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(16, 5),
                                      sharey=True, constrained_layout=True)

    def update(frame):
        yv = y_coords[frame]
        frame_title = f'{y_name} = {yv:.4g}'
        _draw_stack(ax_a, x_coords, phase_abundances_a[frame], phase_names, colors,
                    f'{label_a} — {frame_title}', x_name)
        _draw_stack(ax_b, x_coords, phase_abundances_b[frame], phase_names, colors,
                    f'{label_b} — {frame_title}', x_name)
        ax_b.set_ylabel('')
        fig.suptitle(f'Frame {frame + 1}/{Ny}', fontsize=12)
        return []

    anim = animation.FuncAnimation(
        fig, update, frames=Ny, interval=interval, blit=False, repeat=True,
    )
    plt.close(fig)
    anim.save(save_path, writer='pillow', savefig_kwargs={'facecolor': 'white'}, dpi=dpi)
    print(f'Saved → {save_path}')
    return anim


# ---------------------------------------------------------------------------
# MELTS-specific helpers
# ---------------------------------------------------------------------------

def melts_colors() -> dict:
    """Default color mapping for ptt phase abbreviations."""
    return {
        'Liq':   'red',
        'Ol':    'forestgreen',
        'Opx':   'steelblue',
        'Cpx':   'deepskyblue',
        'Grt':   'orange',
        'Sp':    'brown',
        'Kspar': 'lightgray',
        'Qtz':   'dimgray',
        'Rhm':   'darkorange',
        'Apa':   'mediumpurple',
        'Plag':  'silver',
        'Fl':    'cyan',
    }


def prepare_melts_phase_data(
    features: np.ndarray,
    headers: list,
) -> tuple:
    """Run MELTS emulator and return data ready for the animation functions.

    The input grid must have exactly two varying columns: one containing
    'P(GPa)' (becomes the X axis) and one other (becomes the Y animation axis).
    All remaining columns must be constant across rows.

    Parameters
    ----------
    features : (N, F) array of emulator input features
    headers  : length-F list of column names

    Returns
    -------
    x_coords      : (Nx,) pressure values in ascending order
    y_coords      : (Ny,) animation-variable values in ascending order
    phase_grid    : (Ny, Nx, P) normalized phase fractions, columns ordered
                    by ptt_to_short and restricted to phases present in the data
    phase_names   : length-P list of phase abbreviations (e.g. 'Ol', 'Cpx')
    x_name        : stripped x-axis label (e.g. 'P(GPa)')
    y_name        : stripped y-axis label (e.g. 'T(K)')
    """
    from src.ngibbs.engine.API import MELTS102EmulatorCPU
    from src.ngibbs.config.constants import ptt_to_short, ptt_longs

    features = np.asarray(features, dtype=np.float32)

    t0 = time.time()
    with torch.no_grad():
        output = MELTS102EmulatorCPU.ForwardMB(
            features, headers=headers, outputs=['phase_moles']
        )
    print(f'Emulator: {features.shape[0]} assemblages in {time.time() - t0:.2f}s')

    sub_api = MELTS102EmulatorCPU._route(headers=headers)
    mass_phasedict = sub_api.isentropic_emulator.ml_indexer.mass_phasedict

    phase_data = output['phase_moles']
    if hasattr(phase_data, 'detach'):
        phase_data = phase_data.detach().cpu().numpy()
    phase_data = np.asarray(phase_data, dtype=np.float32)
    row_totals = phase_data.sum(axis=1, keepdims=True)
    phase_data = phase_data / np.where(row_totals == 0, 1.0, row_totals)

    # --- Identify pressure (X) column ---
    p_col = next((i for i, h in enumerate(headers) if 'P(GPa)' in h), None)
    if p_col is None:
        raise ValueError("No pressure column found — expected a header containing 'P(GPa)'")
    pressures = features[:, p_col]
    x_name = headers[p_col].replace('(System_main)', '').strip()

    # --- Find the single other varying column (Y) ---
    varying = []
    for i, h in enumerate(headers):
        if i == p_col:
            continue
        unique_vals = np.unique(np.round(features[:, i], 4))
        if len(unique_vals) > 1:
            varying.append((i, h.replace('(System_main)', '').strip(), unique_vals))

    if len(varying) != 1:
        names = ', '.join(n for _, n, _ in varying)
        raise ValueError(
            f'Expected exactly 1 other varying feature besides pressure, '
            f'found {len(varying)}: {names}. '
            f'Slice features to a single Y variable before calling this function.'
        )

    y_col, y_name, y_coords = varying[0]
    y_data = features[:, y_col]

    # --- Build (Ny, Nx, P) array via vectorised index assignment ---
    x_coords = np.unique(np.round(pressures, 4))
    x_idx = np.searchsorted(x_coords, np.round(pressures, 4))
    y_idx = np.searchsorted(y_coords, np.round(y_data, 4))

    n_phases = phase_data.shape[1]
    phase_grid = np.zeros((len(y_coords), len(x_coords), n_phases), dtype=np.float32)
    phase_grid[y_idx, x_idx, :] = phase_data

    # --- Select and order phases via ptt_to_short, drop absent ones ---
    ordered_cols, ordered_names = [], []
    for ptt_long, ptt_short in ptt_to_short.items():
        internal = ptt_longs[ptt_long] if ptt_long in ptt_longs else ptt_long[:-1]
        if internal not in mass_phasedict:
            continue
        col = mass_phasedict[internal]
        if phase_grid[:, :, col].max() > 1e-4:
            ordered_cols.append(col)
            ordered_names.append(ptt_short.lstrip('_'))

    phase_grid = phase_grid[:, :, ordered_cols]
    return x_coords, y_coords, phase_grid, ordered_names, x_name, y_name


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    from src.ngibbs.utils.math_utils import grid_sample

    # MORB-like fixed composition
    input_dict = {
        'SiO2':  50.5,
        'TiO2':   1.5,
        'Al2O3': 15.5,
        'FeO':    9.0,
        'MgO':    8.5,
        'CaO':   11.5,
        'K2O':    0.1,
        'P2O5':   0.1,
        'H2O':    0.1,
    }
    comp_array = np.zeros((1, len(input_dict)), dtype=np.float32)
    for i, val in enumerate(input_dict.values()):
        comp_array[0, i] = val
    comp_headers = list(input_dict.keys())

    # Grid: P(100 steps) × T(40 steps) — animate over T
    features = grid_sample([[0, 3, 100], [1273, 1573, 40]], comp_array)
    headers = ['P(GPa)(System_main)', 'T(K)(System_main)'] + comp_headers

    x, y, data, names, x_name, y_name = prepare_melts_phase_data(features, headers)
    colors = melts_colors()

    # Single animation
    animate_phase_diagram(x, y, data, names,
                          x_name=x_name, y_name=y_name,
                          colors=colors,
                          save_path='morb_melts_phase_animation.gif')

    # Comparison: low-Na vs high-Na end-member grids
    lo_na = input_dict.copy(); lo_na['Na2O'] = 0.5  # not in input_dict originally, demo only
    hi_na = input_dict.copy(); hi_na['Na2O'] = 4.0

    def make_grid(comp):
        arr = np.zeros((1, len(comp)), dtype=np.float32)
        for i, v in enumerate(comp.values()):
            arr[0, i] = v
        feats = grid_sample([[0, 3, 100], [1273, 1573, 40]], arr)
        hdrs = ['P(GPa)(System_main)', 'T(K)(System_main)'] + list(comp.keys())
        return prepare_melts_phase_data(feats, hdrs)

    x, y, data_lo, names, x_name, y_name = make_grid(lo_na)
    _, _,  data_hi, _,     _,      _      = make_grid(hi_na)

    animate_phase_comparison(x, y, data_lo, data_hi, names,
                              x_name=x_name, y_name=y_name,
                              label_a='Low Na₂O', label_b='High Na₂O',
                              colors=colors,
                              save_path='morb_melts_na_comparison.gif')
