#!/usr/bin/env python
"""Mass-balance iteration sweep: how much does each extra iteration of
``MassBalanceProjector`` improve the bulk reconstruction and the phase
predictions, and what does it cost?

For every emulator / Test_subset bundle pair:

1. one network forward pass over the bundle (cached),
2. ``apply_mass_balance`` with ``mass_balance='none'`` and with the iterative
   projector at ``iters = 0, 1, ..., K`` (iters=0 = phase seeding only, which the
   continuous models do before projecting),
3. per setting: bulk-residual statistics (per-row L2 and per-element relative
   error percentiles), how many phases the clamp removed, timing (warm-up +
   median of repeats, projector-only and full correction), and the phase-quality
   metrics of ``evaluate_emulator_quality`` aggregated over phases.

Outputs (``--out``, default ``generated/mass_balance_sweep`` next to this file):
``sweep.csv`` (one row per emulator x setting), ``sweep_<label>.png`` and a
``sweep_summary.txt``.  Only the ``ngibbs`` package is used.

    python mass_balance_iteration_sweep.py --families 102 120 --kinds isothermal --iters 10
"""

from __future__ import annotations

import argparse
import time
import types
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ngibbs.deployment_tests.discovery import shipped_emulators
from ngibbs.deployment_tests.emulator_quality import evaluate_emulator_quality
from ngibbs.engine.NN import rebuild_MELTS_model
from ngibbs.engine.emulator import NN_MELTS
from ngibbs.utils.file_utils import load_ml_bundle

RELERR_MIN_B = 1e-6


def _timed(fn, repeats):
    fn()                                    # warm-up
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t0)
    return out, float(np.median(ts))


def _bulk_stats(cm, compToEl, b):
    bl = (cm.double() @ compToEl.double()).numpy()
    bh = bl / np.clip(bl.sum(1, keepdims=True), 1e-6, None)
    l2 = np.linalg.norm(bh - b, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel = np.abs(np.where(b > RELERR_MIN_B, (b - bh) / b, np.nan))
    rel_row = np.nanmax(rel, axis=1)        # worst element per row
    flat = rel[np.isfinite(rel)]
    return {
        'l2_mean': l2.mean(), 'l2_median': np.median(l2), 'l2_p95': np.percentile(l2, 95),
        'l2_p98': np.percentile(l2, 98), 'l2_p99': np.percentile(l2, 99), 'l2_max': l2.max(),
        'relerr_median_pct': 100 * np.median(flat), 'relerr_p95_pct': 100 * np.percentile(flat, 95),
        'relerr_p98_pct': 100 * np.percentile(flat, 98), 'relerr_p99_pct': 100 * np.percentile(flat, 99),
        'frac_rows_any_relerr_gt_1pct': float(np.mean(rel_row > 0.01)),
        'frac_rows_any_relerr_gt_0p1pct': float(np.mean(rel_row > 0.001)),
    }


def _quality_aggregate(emu, bundle, max_samples, seed):
    api = types.SimpleNamespace(isothermal_emulator=emu)
    t = evaluate_emulator_quality(api, bundle, emulator_name='isothermal', max_samples=max_samples,
                                  seed=seed)['phase_quality_metrics']
    w = t.loc['n_rows_present'].clip(lower=0)
    comp = [m for m in t.index if m.endswith('_comp_mae_wtpct')]
    relc = [m for m in t.index if m.endswith('_comp_relerr_pct')]

    def wmean(row):
        v = t.loc[row]
        ok = np.isfinite(v) & (w > 0)
        return float(np.average(v[ok], weights=w[ok])) if ok.any() else np.nan

    return {
        'q_precision_wmean': wmean('precision'),
        'q_recall_wmean': wmean('recall'),
        'q_abundance_mae_wtpct_wmean': wmean('abundance_mae_wtpct'),
        'q_abundance_relerr_pct_wmean': wmean('abundance_relerr_pct'),
        'q_comp_mae_wtpct_mean': float(np.nanmean(t.loc[comp].to_numpy())) if comp else np.nan,
        'q_comp_relerr_pct_mean': float(np.nanmean(t.loc[relc].to_numpy())) if relc else np.nan,
    }


def sweep_one(label, ckpt, bundle, *, iters, repeats, max_samples, seed, quality, device):
    emu = NN_MELTS(rebuild_MELTS_model(str(ckpt)), cuda=(device == 'cuda'), mass_balance='iterative')
    data = load_ml_bundle(bundle, arrays=('features',))
    n_total = data.features.shape[0]
    if max_samples is not None and max_samples < n_total:
        idx = np.sort(np.random.default_rng(seed).choice(n_total, size=max_samples, replace=False))
    else:
        idx = np.arange(n_total)
    feats = torch.tensor(data.features[idx], dtype=torch.float32, device=emu.dev)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mol = emu.convertOxToMol(feats, convert=False)
        nf = emu.norm_features.norm(mol)
        with torch.no_grad():
            _lik, _chem, logM, reconB, cm0, pp, pm = emu.model.forward(nf, detailed=True)
    off = emu.feature_offset
    b = mol[:, off:].detach().cpu().double().numpy()
    compToEl = emu.compToEl.to(emu.dev)
    bulk_dir = nf[:, off:]
    P2C = emu.phaseToCompMap.to(emu.dev)

    def present(cm):
        return ((cm @ P2C.T) > 0).cpu().numpy()

    raw_present = present(cm0)
    seeded = emu._seed_phases
    if seeded:
        cm_seed, _, _, _ = emu.seed_unexplained_phases(cm0, pm, pp, logM, nf, reconBulk=reconB)
    else:
        cm_seed = cm0
    seeded_present = present(cm_seed)

    rows = []
    settings = [('none', None)] + [('iterative', k) for k in range(iters + 1)]
    for mode, k in settings:
        if k is not None:
            emu._projector.iters = k

        def full():
            with torch.no_grad():
                return emu.apply_mass_balance(pm, reconB, cm0, pp, nf, mode=mode,
                                              outputs=['component_moles'], phase_affinity=logM)

        def proj_only():
            with torch.no_grad():
                return emu._projector(cm_seed.clone(), compToEl, bulk_dir,
                                      polish=emu.polish_component_moles)

        out, t_full = _timed(full, repeats)
        cm = out['component_moles'].detach().cpu()
        t_proj = _timed(proj_only, repeats)[1] if mode == 'iterative' else 0.0
        pres = present(cm.to(emu.dev))
        base = seeded_present if mode == 'iterative' else raw_present
        row = {'emulator': label, 'model_class': type(emu.model).__name__, 'n_rows': len(idx),
               'setting': 'none' if mode == 'none' else f'iters={k}',
               'iters': -1 if mode == 'none' else k,
               'time_full_s': t_full, 'time_projector_s': t_proj,
               'time_projector_us_per_row': 1e6 * t_proj / len(idx),
               'phases_removed_per_row': float((base & ~pres).sum(1).mean()),
               'frac_rows_phase_removed': float((base & ~pres).any(1).mean())}
        row.update(_bulk_stats(cm, compToEl.cpu(), b))
        if quality:
            emu.mass_balance = mode
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                row.update(_quality_aggregate(emu, bundle, max_samples, seed))
            emu.mass_balance = 'iterative'
        rows.append(row)
        print(f"  {label:32s} {row['setting']:9s} L2 mean {row['l2_mean']:.2e} p99 {row['l2_p99']:.2e} "
              f"| rel p98 {row['relerr_p98_pct']:.2e}% | rows>1% {100 * row['frac_rows_any_relerr_gt_1pct']:.2f}% "
              f"| proj {1e3 * t_proj:7.1f} ms  full {1e3 * t_full:7.1f} ms", flush=True)
    emu._projector.iters = 3
    return rows


def plot_sweep(df, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
    panels = [('l2_mean', 'L2 residual, mean', True), ('l2_p99', 'L2 residual, p99', True),
              ('relerr_p98_pct', '|rel. error| p98 (%)', True),
              ('frac_rows_any_relerr_gt_1pct', 'rows with any element |rel| > 1%', True),
              ('q_abundance_mae_wtpct_wmean', 'phase abundance MAE (wt%, weighted)', False),
              ('time_projector_us_per_row', 'projector cost (µs / row)', False)]
    panels = [p for p in panels if p[0] in df.columns]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), squeeze=False)
    for ax, (col, title, logy) in zip(axes.flat, panels):
        for i, (lab, g) in enumerate(df[df.iters >= 0].groupby('emulator', sort=False)):
            c = colors[i % len(colors)]
            ax.plot(g.iters, g[col], marker='o', markersize=3, linewidth=1.5, color=c, label=lab)
            none = df[(df.emulator == lab) & (df.iters < 0)][col]
            if len(none) and col != 'time_projector_us_per_row':
                ax.axhline(float(none.iloc[0]), color=c, linewidth=0.8, linestyle=':')
        if logy:
            ax.set_yscale('log')
        ax.set_title(title, fontsize=9, loc='left')
        ax.set_xlabel('projector iterations', fontsize=8)
        ax.tick_params(labelsize=7)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        ax.grid(True, color='#e6e5e0', linewidth=0.5)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=min(4, len(l)), fontsize=8, frameon=False)
    fig.suptitle('Mass-balance iteration sweep  (dotted = no mass balance)', x=0.01, ha='left',
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    p = out_dir / 'sweep.png'
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--families', nargs='+')
    ap.add_argument('--kinds', nargs='+', default=['isothermal'],
                    choices=['isothermal', 'isentropic', 'openox'])
    ap.add_argument('--pair', nargs=2, action='append', metavar=('CHECKPOINT', 'BUNDLE'))
    ap.add_argument('--iters', type=int, default=10)
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--max-samples', type=int, default=None)
    ap.add_argument('--seed', type=int, default=1337)
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    ap.add_argument('--no-quality', action='store_true', help='skip the phase-quality metrics')
    ap.add_argument('--out', default=None)
    a = ap.parse_args(argv)
    out = Path(a.out) if a.out else Path(__file__).resolve().parent / 'generated' / 'mass_balance_sweep'
    out.mkdir(parents=True, exist_ok=True)

    if a.pair:
        jobs = [(Path(c).stem, Path(c), Path(b)) for c, b in a.pair]
    else:
        jobs = [(e.label, e.checkpoint, e.bundle)
                for e in shipped_emulators(families=a.families, kinds=a.kinds) if e.bundle]
    print(f'torch {torch.__version__}, device {a.device}, threads {torch.get_num_threads()}')
    rows = []
    for label, ckpt, bundle in jobs:
        rows += sweep_one(label, ckpt, bundle, iters=a.iters, repeats=a.repeats,
                          max_samples=a.max_samples, seed=a.seed, quality=not a.no_quality,
                          device=a.device)
    df = pd.DataFrame(rows)
    df.to_csv(out / 'sweep.csv', index=False)
    plot_sweep(df, out)
    print(f'-> {out}')


if __name__ == '__main__':
    main()
