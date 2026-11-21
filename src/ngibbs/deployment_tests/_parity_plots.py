"""Parity plots shared by every dataset the deployment tests score.

* ``plot_abundance_parity``: per-phase predicted vs ground-truth abundance as a
  log-density hexbin (one panel per phase), used for the validation dataset,
  the standard workspace and any custom workspace alike.
* ``plot_property_parity``: per-property built-in EOS value vs the simulator's
  own reported value, in two colours -- red for the EOS on the emulated
  assemblage, blue for the EOS on the ground-truth assemblage -- each with its
  R^2 against the 1:1 line. Only possible where the simulator's own property
  output exists (raw workspaces, not the preprocessed validation dataset).

Both use matplotlib's object API (no global backend switch).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

EMULATED_COLOR = 'tab:red'
GT_ASSEMBLAGE_COLOR = 'tab:blue'


def r2_score(truth: np.ndarray, pred: np.ndarray) -> float:
    """Coefficient of determination of ``pred`` as a prediction of ``truth``
    (R^2 against the 1:1 line, not the squared correlation): 1 - SS_res/SS_tot
    over rows where both are finite. NaN with fewer than two such rows or no
    spread in ``truth``."""
    truth, pred = np.asarray(truth, dtype=np.float64), np.asarray(pred, dtype=np.float64)
    ok = np.isfinite(truth) & np.isfinite(pred)
    if ok.sum() < 2:
        return float('nan')
    t, p = truth[ok], pred[ok]
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    if ss_tot <= 0:
        return float('nan')
    return 1.0 - float(np.sum((t - p) ** 2)) / ss_tot


def plot_abundance_parity(gt_mass_wtpct, pred_mass_wtpct, ml_indexer, phase_order, save_path,
                          title: str = '', max_phases: int = 12) -> Path:
    """Grid of predicted vs ground-truth phase abundance (wt% of system), one
    panel per phase (most abundant first), over rows where the phase is present
    in either. Colour is the number of assemblages per hexagonal bin, on a log
    scale (shared colorbar)."""
    from matplotlib.colors import LogNorm
    from matplotlib.figure import Figure

    gt_mass_wtpct = np.asarray(gt_mass_wtpct, dtype=np.float64)
    pred_mass_wtpct = np.asarray(pred_mass_wtpct, dtype=np.float64)
    phases = [p for p in phase_order
              if np.any(gt_mass_wtpct[:, ml_indexer.mass_phasedict[p]] > 0)][:max_phases]
    ncol = 4
    nrow = max(1, int(np.ceil(len(phases) / ncol)))
    fig = Figure(figsize=(3.2 * ncol + 0.8, 3.0 * nrow), constrained_layout=True)
    axes = fig.subplots(nrow, ncol, squeeze=False).ravel()

    selections = []
    for phase in phases:
        j = ml_indexer.mass_phasedict[phase]
        gt, pr = gt_mass_wtpct[:, j], pred_mass_wtpct[:, j]
        sel = (gt > 0) | (pr > 0.01)
        selections.append((phase, gt, pr, sel))
    # One colour scale for every panel so densities are comparable across phases.
    vmax = 1
    for _, gt, pr, sel in selections:
        if sel.any():
            counts, _, _ = np.histogram2d(gt[sel], pr[sel], bins=60)
            vmax = max(vmax, int(counts.max()))
    norm = LogNorm(vmin=1, vmax=max(vmax, 2))

    mappable = None
    for ax, (phase, gt, pr, sel) in zip(axes, selections):
        if sel.any():
            mappable = ax.hexbin(gt[sel], pr[sel], gridsize=60, norm=norm, mincnt=1, cmap='viridis')
            hi = float(max(gt[sel].max(), pr[sel].max(), 1e-3))
        else:
            hi = 1.0
        ax.plot([0, hi], [0, hi], color='0.4', lw=0.8, ls='--')
        ax.set_title(f'{phase} (n={int((gt > 0).sum())})', fontsize=9)
        ax.set_xlabel('ground truth wt%', fontsize=8)
        ax.set_ylabel('emulator wt%', fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes[len(phases):]:
        ax.set_visible(False)
    if mappable is not None:
        cb = fig.colorbar(mappable, ax=list(axes[:len(phases)]), shrink=0.9, pad=0.01)
        cb.set_label('assemblages per bin (log scale)', fontsize=8)
        cb.ax.tick_params(labelsize=7)
    if title:
        fig.suptitle(title, fontsize=10)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=130)
    return save_path


def plot_property_parity(truth: Mapping[str, np.ndarray], emulated: Mapping[str, np.ndarray],
                         gt_assemblage: Mapping[str, np.ndarray], save_path, *,
                         units: Optional[Mapping[str, str]] = None, title: str = '',
                         properties: Optional[Sequence[str]] = None) -> Dict[str, Dict[str, float]]:
    """One panel per property: built-in EOS value (y) vs the simulator's own
    value (x). Red: EOS on the emulated assemblage. Blue: EOS on the
    ground-truth assemblage. Each legend entry carries that source's R^2
    against the 1:1 line. Returns ``{property: {'emulated': r2, 'gt_assemblage': r2}}``.
    """
    from matplotlib.figure import Figure

    units = units or {}
    props = [p for p in (properties or truth.keys()) if p in truth]
    ncol = min(4, max(1, len(props)))
    nrow = max(1, int(np.ceil(len(props) / ncol)))
    fig = Figure(figsize=(3.4 * ncol, 3.2 * nrow), constrained_layout=True)
    axes = fig.subplots(nrow, ncol, squeeze=False).ravel()
    scores: Dict[str, Dict[str, float]] = {}
    for ax, prop in zip(axes, props):
        x = np.asarray(truth[prop], dtype=np.float64)
        r2s = {}
        finite_vals = [x[np.isfinite(x)]]
        for key, source, color, label in (
                ('gt_assemblage', gt_assemblage, GT_ASSEMBLAGE_COLOR, 'EOS on GT assemblage'),
                ('emulated', emulated, EMULATED_COLOR, 'EOS on emulated assemblage')):
            if prop not in source:
                continue
            y = np.asarray(source[prop], dtype=np.float64)
            r2s[key] = r2_score(x, y)
            ok = np.isfinite(x) & np.isfinite(y)
            finite_vals.append(y[ok])
            ax.scatter(x[ok], y[ok], s=4, alpha=0.5, color=color, linewidths=0,
                       label=f'{label} (R²={r2s[key]:.4f})', rasterized=True)
        allv = np.concatenate(finite_vals) if finite_vals else np.array([])
        if allv.size:
            lo, hi = float(allv.min()), float(allv.max())
            pad = 0.02 * (hi - lo or abs(hi) or 1.0)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color='0.3', lw=0.8, ls='--')
        unit = f' ({units[prop]})' if prop in units else ''
        ax.set_title(f'{prop}{unit}', fontsize=9)
        ax.set_xlabel('simulator', fontsize=8)
        ax.set_ylabel('built-in EOS', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, loc='upper left', markerscale=3)
        scores[prop] = r2s
    for ax in axes[len(props):]:
        ax.set_visible(False)
    if title:
        fig.suptitle(title, fontsize=10)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=130)
    return scores
