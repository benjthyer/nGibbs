"""Stacked phase-abundance diagram helpers for MELTS (deployable copy).

MELTS analogue of ``_phase_plotting.py``: same stacked-area-by-mass-fraction
drawing logic, but keyed to the phase set alphaMELTS actually produces
(olivine, orthopyroxene, clinopyroxene, spinel, plagioclase, alkali-feldspar,
rhm-oxide, melts-liquid, quartz, tridymite, whitlockite, apatite, fluid, ...)
rather than HeFESTo's mantle mineralogy, and on a MASS fraction basis
throughout (matching Phase_mass_tbl.txt and the isentropic emulator's own
gram-mass normalization), not mole fraction.
"""

from __future__ import annotations

import numpy as np

# Bottom-to-top stacking order for a typical crustal/mantle-melt assemblage.
# Silica polymorphs and oxides at the bottom, mafic silicates, feldspars,
# liquid on top (usually the dominant high-T phase).
MELTS_PHASE_STACK_ORDER = [
    'quartz', 'tridymite', 'rhm-oxide', 'spinel', 'apatite', 'whitlockite',
    'olivine', 'orthopyroxene', 'clinopyroxene', 'hornblende', 'biotite',
    'garnet', 'plagioclase', 'alkali-feldspar', 'nepheline', 'leucite',
    'muscovite', 'calcite', 'graphite', 'fluid', 'melts-liquid',
]


def melts_phase_colors() -> dict:
    """{phase_name: matplotlib_color} for stacked MELTS phase diagrams."""
    return {
        'quartz': 'dimgray', 'tridymite': 'darkgray',
        'rhm-oxide': 'magenta', 'spinel': 'brown',
        'apatite': 'gold', 'whitlockite': 'khaki',
        'olivine': 'forestgreen', 'orthopyroxene': 'steelblue',
        'clinopyroxene': 'royalblue', 'hornblende': 'seagreen',
        'biotite': 'saddlebrown', 'garnet': 'red',
        'plagioclase': 'lightgray', 'alkali-feldspar': 'plum',
        'nepheline': 'mediumpurple', 'leucite': 'orchid',
        'muscovite': 'silver', 'calcite': 'wheat', 'graphite': 'black',
        'fluid': 'lightskyblue', 'melts-liquid': 'orangered',
    }


def build_ordered_melts_phases(phase_names_present) -> list:
    """MELTS_PHASE_STACK_ORDER phases restricted to those actually present in
    ``phase_names_present`` (any iterable of phase name strings); anything
    present but not in the fixed order list is appended at the end."""
    present = list(phase_names_present)
    ordered = [p for p in MELTS_PHASE_STACK_ORDER if p in present]
    ordered += [p for p in present if p not in MELTS_PHASE_STACK_ORDER]
    return ordered


def draw_melts_phase_stack(ax, mass_frac: dict, x: np.ndarray, ordered_phases: list,
                            colors: dict, title: str) -> None:
    """Draw a stacked phase-mass-fraction-vs-x diagram onto ``ax`` (cleared
    first). ``mass_frac``: {phase_name: (N,) array of mass fraction of the
    total system, aligned to ``x``}."""
    ax.cla()
    order = np.argsort(x)
    x_sorted = np.asarray(x)[order]
    bottoms = np.zeros(len(x_sorted), dtype=np.float64)
    for phase in ordered_phases:
        vals = np.clip(np.asarray(mass_frac.get(phase, np.zeros_like(x_sorted)))[order], 0.0, None)
        top = bottoms + vals
        ax.fill_between(x_sorted, bottoms, top,
                         facecolor=colors.get(phase, '#aaaaaa'),
                         edgecolor='k', linewidth=0.3, label=phase)
        peak = float(np.nanmax(vals)) if vals.size else 0.0
        if peak > 0.04:
            band = vals > 0.35 * peak
            if band.any():
                x_c = np.average(x_sorted[band], weights=vals[band])
                y_c = np.average(0.5 * (bottoms[band] + top[band]), weights=vals[band])
                ax.text(x_c, y_c, phase, ha='center', va='center', fontsize=6, clip_on=True)
        bottoms = top
    ax.set_ylim(0.0, 1.0)
    if x_sorted.size:
        ax.set_xlim(x_sorted.min(), x_sorted.max())
    ax.set_ylabel('Phase mass fraction')
    ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.2)
