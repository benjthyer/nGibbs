"""Bulk-reconstruction diagnostics — how well an emulator's predicted assemblage
adds back up to the bulk composition it was asked about.

For every row of a Test_subset bundle the requested bulk ``b`` (closed element
mole fractions, exactly the target the mass balance corrects onto) is compared
with the reconstructed bulk

    b_hat = normalize(componentMoles @ compToEl)

once from the RAW network output (``mass_balance='none'``) and once AFTER the
emulator's own mass-balance correction (its configured mode, normally
'iterative').  Two figures per emulator, one panel per element, each with the
raw reconstruction in orange and the mass-balanced one in blue drawn on top:

    <label>_bulk_parity.<fmt>   b_hat vs b
    <label>_bulk_relerr.<fmt>   (b - b_hat) / b vs b (log x, symlog y)

and one table, ``bulk_reconstruction_<name>.csv`` (rows = statistic, columns =
elements + an ``all`` column), so the numbers are trackable without the plots.

The per-row L2 norm of ``b_hat - b`` reproduces ``recon_residual_l2_mean`` in
``emulator_quality_<name>.csv`` for the mass-balanced mode.

Nothing here imports ``builder``.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from ngibbs.utils.file_utils import load_ml_bundle
from .emulator_quality import NA_REP, _staged_forward, _to_np

# Relative error is only defined where the element is actually in the bulk.
RELERR_MIN_B = 1e-6
# A row counts as "badly reconstructed" for an element above this |relative error|.
RELERR_FLAG = 0.01

_MODE_LABEL = {'none': 'raw network (no mass balance)'}
_MODE_COLOR = {'raw': '#eb6834', 'massbalanced': '#2a78d6'}   # orange under, blue on top


def _mode_key(mode: str) -> str:
    return 'raw' if mode == 'none' else 'massbalanced'


def evaluate_bulk_reconstruction(
    emulator,
    bundle_path,
    *,
    max_samples: Optional[int] = None,
    seed: int = 1337,
    batch_size: int = 2 ** 14,
    corrected_mode: Optional[str] = None,
) -> Dict[str, object]:
    """Reconstruct the bulk of every bundle row with and without mass balance.

    ``corrected_mode`` defaults to the emulator's own ``mass_balance`` setting;
    if that is 'none' only the raw reconstruction is produced.  Row sub-sampling
    uses the same ``max_samples`` / ``seed`` rule as ``evaluate_emulator_quality``.

    Returns dict with 'elements', 'b' (n, E), 'b_hat' {mode_key: (n, E)},
    'modes' {mode_key: mass_balance mode}, 'summary' (DataFrame) and 'meta'.
    """
    bundle = load_ml_bundle(bundle_path, arrays=('features',))
    n_total = bundle.features.shape[0]
    if max_samples is not None and max_samples < n_total:
        rng = np.random.default_rng(seed)
        subset = np.sort(rng.choice(n_total, size=max_samples, replace=False))
    else:
        subset = np.arange(n_total)

    device = emulator.dev if str(emulator.dev) == 'cuda' and torch.cuda.is_available() else 'cpu'
    feats = torch.tensor(bundle.features[subset], device=device, dtype=torch.float32)
    off = emulator.feature_offset
    elements = list(emulator.ml_indexer.Elkeys)

    b = _to_np(emulator.convertOxToMol(feats, convert=False)[:, off:]).astype(np.float64)
    compToEl = _to_np(emulator.compToEl).astype(np.float64)

    corrected_mode = corrected_mode or getattr(emulator, 'mass_balance', 'iterative')
    modes = {'raw': 'none'}
    if corrected_mode != 'none':
        modes['massbalanced'] = corrected_mode

    b_hat = {}
    with warnings.catch_warnings():
        # the quality test already reported any training-range excursions
        warnings.simplefilter('ignore')
        for key, mode in modes.items():
            out = _staged_forward(
                emulator.forwardMB, feats, batch_size, Normalize=True, optimize_masses=True,
                protect_opx=False, outputs=['component_moles'], mass_balance=mode)
            bl = _to_np(out['component_moles']).astype(np.float64) @ compToEl
            b_hat[key] = bl / np.clip(bl.sum(axis=1, keepdims=True), 1e-6, None)

    summary = _summarize(b, b_hat, elements)
    meta = {'bundle': str(bundle_path), 'n_samples': int(len(subset)), 'n_total_rows': int(n_total),
            'modes': modes, 'relerr_min_b': RELERR_MIN_B, 'relerr_flag': RELERR_FLAG}
    return {'elements': elements, 'b': b, 'b_hat': b_hat, 'modes': modes,
            'summary': summary, 'meta': meta}


_PCTS = (95, 98, 99)


def _summarize(b, b_hat, elements) -> pd.DataFrame:
    rows = {}
    for key, bh in b_hat.items():
        err = bh - b
        with np.errstate(divide='ignore', invalid='ignore'):
            rel = np.where(b > RELERR_MIN_B, (b - bh) / b, np.nan)
        absrel = np.abs(rel)
        rows[f'{key}_mae'] = list(np.mean(np.abs(err), axis=0)) + [float(np.mean(np.abs(err)))]
        rows[f'{key}_relerr_median_pct'] = (list(100 * np.nanmedian(absrel, axis=0))
                                           + [float(100 * np.nanmedian(absrel))])
        for q in _PCTS:
            rows[f'{key}_relerr_p{q}_pct'] = (list(100 * np.nanpercentile(absrel, q, axis=0))
                                             + [float(100 * np.nanpercentile(absrel, q))])
        valid = np.isfinite(absrel)
        frac = np.where(valid.sum(0) > 0, (absrel > RELERR_FLAG).sum(0) / np.maximum(valid.sum(0), 1),
                        np.nan)
        rows[f'{key}_frac_relerr_gt_{int(RELERR_FLAG * 100)}pct'] = (
            list(frac) + [float((absrel > RELERR_FLAG).sum() / max(valid.sum(), 1))])
        l2 = np.linalg.norm(err, axis=1)
        rows[f'{key}_l2_mean'] = [np.nan] * len(elements) + [float(np.mean(l2))]
        rows[f'{key}_l2_median'] = [np.nan] * len(elements) + [float(np.median(l2))]
        for q in _PCTS:
            rows[f'{key}_l2_p{q}'] = [np.nan] * len(elements) + [float(np.percentile(l2, q))]
    rows['frac_rows_element_present'] = list((b > RELERR_MIN_B).mean(axis=0)) + [np.nan]
    table = pd.DataFrame(rows, index=elements + ['all']).T
    table.index.name = 'statistic'
    return table


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #
_INK = '#0b0b0b'
_INK2 = '#52514e'
_REF = '#898781'


def _pct(frac: float) -> str:
    v = 100.0 * float(frac)
    return f'{v:.0f}%' if v >= 10 else (f'{v:.1f}%' if v >= 0.1 else f'{v:.1g}%')


def _grid(n):
    ncols = min(4, n)
    return (n + ncols - 1) // ncols, ncols


def _style_axis(ax):
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#b5b4ae')
    ax.tick_params(colors=_INK2, labelsize=7, length=2)
    ax.grid(True, color='#e6e5e0', linewidth=0.5)
    ax.set_axisbelow(True)


def _series(result):
    """(key, label, color) for each mode, raw first so the corrected points draw on top."""
    out = []
    for key in ('raw', 'massbalanced'):
        if key in result['b_hat']:
            mode = result['modes'][key]
            out.append((key, _MODE_LABEL.get(mode, f'mass-balanced ({mode})'), _MODE_COLOR[key]))
    return out


def _legend(fig, series, n):
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker='o', linestyle='', markersize=5, color=c, label=lab)
               for _, lab, c in series]
    fig.legend(handles=handles, loc='upper right', ncol=len(handles), frameon=False,
               fontsize=8, bbox_to_anchor=(0.995, 0.995), title=f'n = {n:,} rows',
               title_fontsize=7)


def plot_bulk_reconstruction(result: Dict[str, object], out_dir, model_label: str,
                             *, elements: Optional[Sequence[str]] = None, fmt: str = 'png',
                             dpi: int = 150) -> Dict[str, Path]:
    """Write the two figures for ``result`` (as returned by
    ``evaluate_bulk_reconstruction``), one panel per element, each with the raw
    network reconstruction in orange and the mass-balanced one in blue on top:

        <model_label>_bulk_parity.<fmt>   b_hat vs b
        <model_label>_bulk_relerr.<fmt>   (b - b_hat) / b vs b

    Elements absent from every row are skipped.  ``fmt`` is any matplotlib
    format ('png', 'pdf', 'svg'); scatter layers are rasterized (at ``dpi``) so
    vector PDFs stay small.  Returns {'parity': Path, 'relerr': Path}."""
    import matplotlib
    # matplotlib.use('Agg')  # disabled: a global backend switch breaks interactive plotting (e.g. PetThermoTools)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    b = result['b']
    all_el = list(result['elements'])
    elements = [e for e in (elements or all_el) if (b[:, all_el.index(e)] > RELERR_MIN_B).any()]
    nrows, ncols = _grid(len(elements))
    n = b.shape[0]
    series = _series(result)
    figsize = (2.8 * ncols, 2.55 * nrows + 0.35)
    paths = {}

    # ---- parity: b_hat vs b ----
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    for ax in axes.flat[len(elements):]:
        ax.set_visible(False)
    for ax, e in zip(axes.flat, elements):
        j = all_el.index(e)
        vals = np.concatenate([b[:, j]] + [result['b_hat'][k][:, j] for k, _, _ in series])
        hi = float(np.nanpercentile(vals, 99.9))
        lo, hi = 0.0, (hi * 1.05 if hi > 0 else 1.0)
        ax.plot([lo, hi], [lo, hi], color=_REF, linewidth=0.8, linestyle='--', zorder=1)
        notes = []
        for z, (key, _, color) in enumerate(series):
            bh = result['b_hat'][key]
            ax.scatter(b[:, j], bh[:, j], s=2, alpha=0.35, color=color, linewidths=0,
                       rasterized=True, zorder=2 + z)
            notes.append(f"{'raw' if key == 'raw' else 'MB'} MAE {np.mean(np.abs(bh[:, j] - b[:, j])):.1e}")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(e, fontsize=9, loc='left', color=_INK)
        ax.set_title('\n'.join(notes), fontsize=6, loc='right', color=_INK2)
        _style_axis(ax)
    for ax in axes[-1]:
        ax.set_xlabel('b  (requested, mole frac.)', fontsize=7, color=_INK2)
    for ax in axes[:, 0]:
        ax.set_ylabel(r'$\hat{b}$  (reconstructed)', fontsize=7, color=_INK2)
    fig.suptitle(f'{model_label} - bulk reconstruction', fontsize=10, x=0.01, ha='left',
                 y=0.995, va='top')
    _legend(fig, series, n)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.45 / figsize[1]))
    p = out_dir / f'{model_label}_bulk_parity.{fmt}'
    fig.savefig(p, dpi=dpi)
    plt.close(fig)
    paths['parity'] = p

    # ---- relative error: (b - b_hat) / b vs b ----
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    for ax in axes.flat[len(elements):]:
        ax.set_visible(False)
    for ax, e in zip(axes.flat, elements):
        j = all_el.index(e)
        ok = b[:, j] > RELERR_MIN_B
        x = b[ok, j]
        ys = {k: (b[ok, j] - result['b_hat'][k][ok, j]) / b[ok, j] for k, _, _ in series}
        allabs = np.concatenate([np.abs(y[np.isfinite(y)]) for y in ys.values()])
        # symlog: linear inside the smallest typical error (median |delta| of the best
        # series, rounded down to a decade), logarithmic outside, so the corrected cloud,
        # the raw cloud and both tails are all visible and nothing is clipped
        meds = [float(np.median(np.abs(y[np.isfinite(y)]))) for y in ys.values() if np.isfinite(y).any()]
        linthresh = 10.0 ** np.floor(np.log10(max(min(meds) if meds else 1e-6, 1e-12)))
        span = max(float(allabs.max()) if len(allabs) else 1.0, 10 * linthresh) * 1.3
        ax.axhspan(-linthresh, linthresh, color='#f3f2ee', zorder=0, linewidth=0)
        ax.axhline(0.0, color=_REF, linewidth=0.8, linestyle='--', zorder=1)
        notes = []
        for z, (key, _, color) in enumerate(series):
            y = ys[key]
            ax.scatter(x, y, s=2, alpha=0.35, color=color, linewidths=0, rasterized=True,
                       zorder=2 + z)
            a = np.abs(y[np.isfinite(y)])
            if len(a):
                notes.append(f"{'raw' if key == 'raw' else 'MB'}  |δ| med {_pct(np.median(a))}, "
                             f"p98 {_pct(np.percentile(a, 98))}")
        if len(x) and x.max() / max(x.min(), 1e-30) > 20:
            ax.set_xscale('log')
            ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_yscale('symlog', linthresh=linthresh, linscale=0.6)
        ax.set_ylim(-span, span)
        lo_d = int(np.floor(np.log10(linthresh)))
        hi_d = int(np.floor(np.log10(span)))
        step = max(1, int(np.ceil((hi_d - lo_d + 1) / 3)))
        decs = list(range(hi_d, lo_d - 1, -step))
        ax.yaxis.set_major_locator(FixedLocator(
            sorted([-(10.0 ** d) for d in decs] + [0.0] + [10.0 ** d for d in decs])))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_title(e, fontsize=9, loc='left', color=_INK)
        ax.set_title('\n'.join(notes), fontsize=6, loc='right', color=_INK2)
        _style_axis(ax)
    for ax in axes[-1]:
        ax.set_xlabel('b  (requested, mole frac.)', fontsize=7, color=_INK2)
    for ax in axes[:, 0]:
        ax.set_ylabel(r'$(b-\hat{b})\,/\,b$', fontsize=7, color=_INK2)
    fig.suptitle(f'{model_label} - relative bulk error  (rows with b > {RELERR_MIN_B:g}; '
                 'symlog y, shaded band = linear region)', fontsize=10, x=0.01, ha='left',
                 y=0.995, va='top')
    _legend(fig, series, n)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.45 / figsize[1]))
    p = out_dir / f'{model_label}_bulk_relerr.{fmt}'
    fig.savefig(p, dpi=dpi)
    plt.close(fig)
    paths['relerr'] = p
    return paths


def write_summary(result: Dict[str, object], path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result['summary'].to_csv(path, na_rep=NA_REP)
    return path
