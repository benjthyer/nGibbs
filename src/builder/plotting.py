"""Shared plotting utilities for HeFESTo phase-abundance diagrams.

Used by scripts/phase_animation.py (emulator-only, animated) and
scripts/phase_comparison.py (emulator vs ground truth, static) so the stack
drawing logic and phase color/ordering conventions live in one place.
"""

import sys
from pathlib import Path

import numpy as np

src_path = str(Path(__file__).parent.parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from ngibbs.config.constants import HEFESTO_ABBREVIATION_TO_SHORT_NAMES

# HeFESTo phase index (as used in control files) -> abbreviation
INDEX_TO_PHASE = {
     1: 'plg',  2: 'sp',  3: 'opx',  4: 'c2c',  5: 'cpx',
     6: 'wo',   7: 'pwo', 8: 'gt',   9: 'cpv', 10: 'ol',
    11: 'wa',  12: 'ri',  13: 'il',  14: 'pv',  15: 'ppv',
    16: 'cf',  17: 'nal', 18: 'mw',  19: 'qtz', 20: 'coes',
    21: 'st',  22: 'apbo', 23: 'ky', 24: 'neph', 25: 'fea',
    26: 'feg', 27: 'fee',
}

# Stacking order for a pyrolitic mantle assemblage, bottom to top.
IP_PYROLITE = [19, 20, 21, 22, 10, 11, 12, 18, 14, 15, 13, 8, 16, 9,
               3, 4, 5, 1, 2, 23, 24, 17, 25, 26, 27, 6, 7]


def phase_colors() -> dict:
    """Return {phase_abbreviation: matplotlib_color} for stacked phase diagrams."""
    col = {}
    for p in ['qtz', 'coes', 'st', 'apbo', 'cacl']:  # SiO2 polymorphs
        col[p] = 'dimgray'
    for p in ['plg', 'pwo', 'wo', 'neph']:  # Feldspars and feldspathoids
        col[p] = 'lightgray'
    for p in ['ky', 'nal']:
        col[p] = 'mediumpurple'
    for p in ['opx', 'cpx', 'c2c']:  # Pyroxenes
        col[p] = 'steelblue'
    col['sp'] = 'brown'
    col['gt'] = 'red'
    for p in ['ol', 'wa', 'ri', 'il']:
        col[p] = 'forestgreen'
    col['pv'] = 'khaki'
    col['ppv'] = 'goldenrod'
    col['cf'] = 'wheat'
    col['cpv'] = 'tan'
    for p in ['fea', 'feg', 'fee']:
        col[p] = 'sienna'
    col['mw'] = 'magenta'
    return col


def build_ordered_phases(mass_phasedict: dict) -> list:
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


def draw_phase_stack(ax, data: np.ndarray, row_idx: np.ndarray,
                      pressures: np.ndarray, ordered_phases: list,
                      mass_phasedict: dict, colors: dict, title: str) -> None:
    """Draw a stacked phase-fraction-vs-pressure diagram onto ``ax``.

    Parameters
    ----------
    ax             : matplotlib Axes to draw onto (cleared first)
    data           : (N, n_phases) array of phase fractions (or moles), columns
                     indexed per ``mass_phasedict``
    row_idx        : row indices of ``data``/``pressures`` to draw, in the order
                     they should be plotted (typically sorted by pressure)
    pressures      : (N,) array of pressures (GPa), aligned with ``data`` rows
    ordered_phases : list of (long_name, abbrev) pairs, bottom-to-top stacking
                     order (see ``build_ordered_phases``)
    mass_phasedict : {phase_long_name: column_index_in_data}
    colors         : {phase_abbrev: matplotlib_color} (see ``phase_colors``)
    title          : subplot title
    """
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
