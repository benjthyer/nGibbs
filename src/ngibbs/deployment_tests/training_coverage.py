"""
Training-data-coverage diagnostic plots.

Contextualizes a single loaded emulator checkpoint's evaluation (as run by
``EmulatorAPI.test()`` / ``HeFESToAPI.test()`` / ``MELTSAPI.test()``) against
the P-T-S-X conditions its own ``*_Test_subset*.tar.gz`` ML-ready bundle was
drawn from, by rendering that bundle as 2D density heatmaps in four fixed
views:

    1. P vs S
    2. P vs T
    3. a Harker grid: SiO2 (MELTS) / Si (HeFESTo) vs every other oxide/element
    4. a Harker grid: MgO (MELTS) / Mg (HeFESTo) vs every other oxide/element

and overlaying the real standard-rock conditions (the ones read straight off
HeFESTo's MELTStable CSVs or MELTS's alphaMELTS text-table output -- never
synthetic) as 'x' markers on top, so a modeler can see at a glance whether a
given evaluation point sits inside or outside where the model actually has
training density.

Two things are deliberately NOT hardcoded, because they vary bundle-to-bundle
(different checkpoints of even the same model family can carry a different
oxide set, e.g. the Cr variant adds Cr2O3):

- which of {S, T} is a native input feature vs. a derived ``free_outputs``
  column (``_bundle_ps_t``);
- the composition columns' identity and order, read straight off the
  bundle's own ``ml_indexer`` 'Oxides' (MELTS) or 'Elkeys' (HeFESTo) list
  (``_bundle_composition``) -- MELTS's composition space is oxide mole
  fraction, HeFESTo's is elemental mole fraction (with a ferric-iron
  fraction, 'Fe3', split out of total Fe); see ``emulator.py``'s
  ``reorder_input_table(composition_space=...)`` and
  ``meltstable_comparison.py``'s ``ELEMENT_KEYS`` usage, which this module's
  standard-rock loaders both defer to rather than re-deriving independently.

All heatmap/overlay pairs are built from dicts keyed by oxide/element name,
never by column position, so a bundle and a standards directory that don't
share the exact same oxide/element set (e.g. NoCr vs Cr) still line up
correctly on whichever names they DO share.
"""
from __future__ import annotations

import io
import json
import tarfile
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


# --------------------------------------------------------------------------- #
# bundle loading
# --------------------------------------------------------------------------- #
def _read_bundle_arrays(bundle_path: Path):
    """Load features.npy, free_outputs.npy (if present), feature_bounds.json
    and ml_indexer/indexer_metadata.json straight out of a
    '*_Test_subset*.tar.gz' bundle, without extracting it to disk."""
    bundle_path = Path(bundle_path)
    with tarfile.open(bundle_path, 'r:gz') as tf:
        names = set(tf.getnames())

        def _load_npy(name):
            if name not in names:
                return None
            f = tf.extractfile(name)
            return np.load(io.BytesIO(f.read()))

        def _load_json(name):
            f = tf.extractfile(name)
            return json.loads(f.read().decode('utf-8'))

        features = _load_npy('features.npy')
        free_outputs = _load_npy('free_outputs.npy')
        feature_bounds = _load_json('feature_bounds.json')
        indexer_meta = _load_json('ml_indexer/indexer_metadata.json')
    return features, free_outputs, feature_bounds, indexer_meta


def _bundle_ps_t(features, free_outputs, feature_bounds, indexer_meta):
    """(P, S, T) 1D arrays for a bundle. P and whichever of {S, T}
    parameterizes this bundle come straight off ``features``; the other
    (when available at all) comes from ``free_outputs.npy`` -- e.g. an
    isentropic bundle carries P, S as features and ground-truth T as a free
    output, and vice versa for an isothermal bundle. An open-system/logfO2
    bundle has neither S feature nor S free-output, so S comes back None.

    Temperature is named 'Temperature(System_main)' by MELTS and
    'T(K)(System_main)' by HeFESTo -- both checked for."""
    def _is_S(name):
        return name.startswith('S(')

    def _is_T(name):
        return name.startswith('T(') or name.startswith('Temperature')

    names = feature_bounds['featureNames']
    P = features[:, 0]
    second = features[:, 1]
    S = second if _is_S(names[1]) else None
    T = second if _is_T(names[1]) else None
    free_names = indexer_meta.get('free_outputs') or []
    for j, name in enumerate(free_names):
        if free_outputs is None:
            break
        col = free_outputs[:, j]
        if _is_S(name) and S is None:
            S = col
        elif _is_T(name) and T is None:
            T = col
    return P, S, T


def _bundle_composition(features, indexer_meta, composition_space: str) -> Dict[str, np.ndarray]:
    """{oxide_or_element_name: 1D array} for a bundle's composition columns,
    sliced out of ``features`` right after its native P/(T|S)/(logfO2)
    feature columns, in the exact order given by the bundle's own
    ``ml_indexer`` 'Oxides' (composition_space='oxides') or 'Elkeys'
    ('elements') list.

    An open-system/logfO2 bundle has one fewer free composition column than
    the isothermal/isentropic bundles of the same model family: with fO2
    externally imposed, the FeO/Fe2O3 split is derived from it rather than
    independently specified, so there's a combined-iron oxide slot instead
    of two separate ones (``len(Elkeys) == len(Oxides) - 1`` for such a
    bundle, vs. equal for a closed-system one). This is the exact same
    convention ``emulator.py``'s ``reorder_input_table`` already documents
    as ``Oxides[:len(Elkeys)]`` ('excludes ferric column') -- by
    construction the ferric column, when there is one, is always last in
    ``Oxides``, so truncating to ``len(Elkeys)`` columns drops exactly it.
    """
    n_native = len(indexer_meta['featureNames'])
    oxides, elkeys = indexer_meta['Oxides'], indexer_meta['Elkeys']
    if composition_space == 'oxides':
        labels = oxides if len(oxides) == len(elkeys) else oxides[:len(elkeys)]
    else:
        labels = elkeys
    comp = features[:, n_native:n_native + len(labels)]
    if comp.shape[1] != len(labels):
        raise ValueError(
            f"Bundle has {comp.shape[1]} composition columns but "
            f"{len(labels)} {composition_space} labels -- can't align them."
        )
    return {name: comp[:, i] for i, name in enumerate(labels)}


# --------------------------------------------------------------------------- #
# standard-rock (ground-truth) loaders -- MELTS
# --------------------------------------------------------------------------- #
def melts_standard_points(standards_dir: Path, variant: str = 'NoCr',
                           rocks: Optional[Sequence[str]] = None):
    """(P_bar, T_C, S_specific, comp_molefrac, rock_ids) for every row of
    every standard rock's isobaric-cooling run under
    ``standards_dir/<variant>/<rock>/``, reusing ``melts_comparison``'s own
    table readers -- so these values are guaranteed consistent with what's
    actually fed to the emulator elsewhere in this codebase.

    ``comp_molefrac`` is a dict {oxide: array}: ``read_bulk_comp`` returns
    oxide wt%, converted here to mole fraction (via each oxide's molar mass)
    to match the bundle's own oxide-mole-fraction convention. Returns None
    if no rock directory exists for this variant at all.
    """
    from ngibbs.config.constants import get_oxide_molar_mass
    from ngibbs.deployment_tests.melts_comparison import (
        ROCKS, read_system_main, read_bulk_comp,
        _EMULATOR_OXIDE_COLS_CR, _EMULATOR_OXIDE_COLS_NOCR,
    )
    rocks = list(rocks) if rocks is not None else list(ROCKS)
    oxide_cols = _EMULATOR_OXIDE_COLS_CR if variant == 'Cr' else _EMULATOR_OXIDE_COLS_NOCR

    P_all, T_all, S_all, ids_all = [], [], [], []
    wtpct_cols = {ox: [] for ox in oxide_cols}
    for rock in rocks:
        rock_dir = Path(standards_dir) / variant / rock
        if not rock_dir.exists():
            continue
        sysmain = read_system_main(rock_dir)
        bulk = read_bulk_comp(rock_dir)
        mass_g = sysmain['mass'].values
        P_all.append(sysmain['Pressure'].values)
        T_all.append(sysmain['Temperature'].values)
        S_all.append(sysmain['S'].values / mass_g)
        ids_all.extend([rock] * len(sysmain))
        for ox in oxide_cols:
            wtpct_cols[ox].append(bulk[ox].fillna(0.0).values if ox in bulk.columns
                                   else np.zeros(len(sysmain)))
    if not P_all:
        return None

    wtpct = {ox: np.concatenate(cols) for ox, cols in wtpct_cols.items()}
    moles = {ox: wtpct[ox] / get_oxide_molar_mass(ox) for ox in oxide_cols}
    total_moles = np.sum(list(moles.values()), axis=0)
    total_moles = np.where(total_moles == 0, 1.0, total_moles)
    comp_molefrac = {ox: moles[ox] / total_moles for ox in oxide_cols}

    return np.concatenate(P_all), np.concatenate(T_all), np.concatenate(S_all), comp_molefrac, ids_all


# --------------------------------------------------------------------------- #
# standard-rock (ground-truth) loaders -- HeFESTo
# --------------------------------------------------------------------------- #
def hefesto_standard_points(standards_dir: Path):
    """(P_GPa, T_K, S, comp_molefrac, rock_ids) for every row of every raw
    HeFESTo standard-rock simulation directory under ``standards_dir``,
    reusing ``meltstable_comparison``'s own readers (fort.56 for P/T/S,
    ``control`` for bulk composition -- never a pre-baked CSV).

    ``comp_molefrac`` is a dict {element: array} keyed by the SAME element
    vocabulary as a HeFESTo bundle's own Elkeys (e.g. 'Fe3' present, 'O'
    dropped), and on the SAME mole-FRACTION (sum == 1) basis a bundle's own
    composition columns are -- ``_load_composition`` renormalizes HeFESTo's
    raw control-file element moles (a fixed total-moles basis, not sum == 1)
    before returning them; without that, every standard-rock overlay point
    would sit off that same total-moles factor outside this plot's axes.
    'Fe3' is filled with 0.0, mirroring how the existing HeFESTo comparison
    pipeline already feeds these standards to the emulator itself
    (``reorder_input_table(composition_space='elements')``) -- not a new
    approximation introduced here. Returns None if no simulation directories
    are found.
    """
    from ngibbs.deployment_tests.meltstable_comparison import (
        find_tables, _load_composition, _read_conditions, ELEMENT_KEYS,
    )
    try:
        tables = find_tables(standards_dir)
    except (FileNotFoundError, NotADirectoryError):
        return None
    if not tables:
        return None

    P_all, T_all, S_all, ids_all = [], [], [], []
    comp_cols = {k: [] for k in ELEMENT_KEYS if k != 'O'}
    for sim_dir in tables:
        _, P, T, S = _read_conditions(sim_dir)
        comp = _load_composition(sim_dir)  # single dict of scalars for this rock
        n = len(P)
        P_all.append(P)
        T_all.append(T)
        S_all.append(S)
        ids_all.extend([sim_dir.name] * n)
        for k in comp_cols:
            comp_cols[k].append(np.full(n, comp[k]))
    if not P_all:
        return None

    comp_molefrac = {k: np.concatenate(v) for k, v in comp_cols.items()}
    comp_molefrac['Fe3'] = np.zeros_like(comp_molefrac['Fe'])  # see docstring
    return np.concatenate(P_all), np.concatenate(T_all), np.concatenate(S_all), comp_molefrac, ids_all


# --------------------------------------------------------------------------- #
# plotting primitives
# --------------------------------------------------------------------------- #
def _density_heatmap(ax, x, y, xlabel, ylabel, title, bins=50,
                      standard_xy: Optional[Tuple[np.ndarray, np.ndarray]] = None):
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[finite], np.asarray(y)[finite]
    if x.size == 0:
        ax.set_title(f'{title}\n(no data)', fontsize=8)
        ax.axis('off')
        return
    try:
        ax.hist2d(x, y, bins=bins, cmap='viridis', norm=LogNorm())
    except ValueError:
        # LogNorm chokes if every count is identical (e.g. a degenerate axis)
        ax.hist2d(x, y, bins=bins, cmap='viridis')
    if standard_xy is not None:
        sx, sy = standard_xy
        sfinite = np.isfinite(sx) & np.isfinite(sy)
        if sfinite.any():
            ax.scatter(np.asarray(sx)[sfinite], np.asarray(sy)[sfinite],
                       marker='x', c='red', s=45, linewidths=1.5, zorder=5,
                       label='standard rocks')
            ax.legend(loc='best', fontsize=6, framealpha=0.6)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)


def _plot_p_vs_second(P, second, second_label, out_path, title,
                       standard_P=None, standard_second=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    sxy = (standard_second, standard_P) if standard_P is not None else None
    _density_heatmap(ax, second, P, second_label, 'Pressure', title, bins=60, standard_xy=sxy)
    ax.invert_yaxis()  # pressure increases downward
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_harker_grid(comp: Dict[str, np.ndarray], pivot: str, out_path, title,
                       standard_comp: Optional[Dict[str, np.ndarray]] = None, ncols=4):
    if pivot not in comp:
        warnings.warn(f"Harker pivot '{pivot}' not in this bundle's composition columns; skipping.")
        return False
    others = [k for k in comp if k != pivot]
    n = len(others)
    if n == 0:
        return False
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 2.9 * nrows), squeeze=False)
    pivot_vals = comp[pivot]
    std_pivot = standard_comp.get(pivot) if standard_comp is not None else None
    for k, other in enumerate(others):
        ax = axes[k // ncols][k % ncols]
        sxy = None
        if standard_comp is not None and std_pivot is not None and other in standard_comp:
            sxy = (std_pivot, standard_comp[other])
        _density_heatmap(ax, pivot_vals, comp[other], pivot, other, other, bins=40, standard_xy=sxy)
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].axis('off')
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


# --------------------------------------------------------------------------- #
# top-level orchestration
# --------------------------------------------------------------------------- #
def plot_training_coverage(
    bundle_path,
    out_dir,
    *,
    composition_space: str,
    model_label: str,
    standards=None,
) -> Dict[str, Path]:
    """Build & save all 4 diagnostic plots for one loaded emulator's own
    Test_subset bundle.

    Parameters
    ----------
    bundle_path : the '*_Test_subset*.tar.gz' bundle for this checkpoint.
    out_dir : directory the 4 PNGs are written into (created if needed).
    composition_space : 'oxides' (MELTS) or 'elements' (HeFESTo).
    model_label : used only in filenames/titles, e.g. 'isothermal_NoCr'.
    standards : optional (P, T, S, comp_dict, rock_ids) tuple, as returned by
        ``melts_standard_points`` / ``hefesto_standard_points`` -- the 'x'
        overlay. None (or all-empty) simply omits the overlay.

    Returns
    -------
    dict mapping {'p_vs_s', 'p_vs_t', 'harker_<pivot1>', 'harker_<pivot2>'} to
    the PNG path actually written -- a key is omitted when that plot
    couldn't be built (e.g. no S data at all for an open-system bundle).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    features, free_outputs, feature_bounds, indexer_meta = _read_bundle_arrays(bundle_path)
    P, S, T = _bundle_ps_t(features, free_outputs, feature_bounds, indexer_meta)
    comp = _bundle_composition(features, indexer_meta, composition_space)
    pivot1 = 'SiO2' if composition_space == 'oxides' else 'Si'
    pivot2 = 'MgO' if composition_space == 'oxides' else 'Mg'

    std_P, std_T, std_S, std_comp = None, None, None, None
    if standards is not None:
        std_P, std_T, std_S, std_comp, _ids = standards

    paths: Dict[str, Path] = {}

    if S is not None:
        p = out_dir / f'{model_label}_P_vs_S.png'
        _plot_p_vs_second(P, S, 'Specific entropy S', p,
                           f'{model_label}: training coverage (P vs S)', std_P, std_S)
        paths['p_vs_s'] = p
    if T is not None:
        p = out_dir / f'{model_label}_P_vs_T.png'
        _plot_p_vs_second(P, T, 'Temperature', p,
                           f'{model_label}: training coverage (P vs T)', std_P, std_T)
        paths['p_vs_t'] = p

    p = out_dir / f'{model_label}_Harker_{pivot1}.png'
    if _plot_harker_grid(comp, pivot1, p, f'{model_label}: {pivot1} Harker grid', std_comp):
        paths[f'harker_{pivot1}'] = p
    p = out_dir / f'{model_label}_Harker_{pivot2}.png'
    if _plot_harker_grid(comp, pivot2, p, f'{model_label}: {pivot2} Harker grid', std_comp):
        paths[f'harker_{pivot2}'] = p

    return paths
