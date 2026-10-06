"""
Diagnostic: phase moles, or activated labels y, vs T for one run of the FxTal/batch reference bundle.

Usage
  python scripts/plot_batch_run_moles.py --run 2                 # n vs T
  python scripts/plot_batch_run_moles.py --run 2 --labels        # y vs T, c from the whole bundle
      [--dT 25] [--eps 0.02] [--fx-recipe nstep] [--batch-recipe brief] [--bundle PATH] [--out DIR]

Runs are numbered as in test_fxtal_dndT_projection.py (split wherever T increases):
0-19 MORB batch, 20-39 Lherzolite batch, 40 MORB FxTal, 41 Lherzolite FxTal.

c_phi = median over all runs (of the per-run mean slope s at that run's crossings) * dT,
with the same crossing / slope code as test_fxtal_dndT_projection.py.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
for p_ in (REPO_ROOT / 'src', REPO_ROOT / 'scripts'):
    if str(p_) not in sys.path:
        sys.path.insert(0, str(p_))
from ngibbs.utils.file_utils import load_ml_bundle  # noqa: E402
from test_fxtal_dndT_projection import (split_runs, find_crossings, run_labels, activate,  # noqa: E402
                                        PRESENT_TOL)

DEFAULT_BUNDLE = REPO_ROOT / 'data' / 'TestFxTal_dndT' / 'MELTS120_RefStandardsFxTalBatchOpenNoCr.tar.gz'
DEFAULT_OUT = REPO_ROOT / 'data' / 'TestFxTal_dndT' / 'projection_test'
INK, MUTED, GRID, LINE = '#222222', '#6b6b6b', '#e4e4e4', '#2a78d6'


def style(ax):
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=int, default=2)
    ap.add_argument('--labels', action='store_true', help='plot activated labels y instead of n')
    ap.add_argument('--dT', type=float, default=25.0)
    ap.add_argument('--eps', type=float, default=0.02)
    ap.add_argument('--fx-recipe', default='nstep', choices=['nstep', 'brief', 'last'])
    ap.add_argument('--batch-recipe', default='brief', choices=['nstep', 'brief', 'last'])
    ap.add_argument('--bundle', default=str(DEFAULT_BUNDLE))
    ap.add_argument('--out', default=str(DEFAULT_OUT))
    args = ap.parse_args()

    b = load_ml_bundle(args.bundle, arrays=('features', 'molar_labels'))
    F, M = np.asarray(b.features, float), np.asarray(b.molar_labels, float)
    phases = list(b.ml_indexer.all_phases)
    fn = list(b.ml_indexer.featureNames)
    nc = len(fn)
    T, P = F[:, fn.index('Temperature(System_main)')], F[:, fn.index('Pressure(System_main)')]
    runs = split_runs(T)
    is_fx = [bool(np.abs(F[a:e, nc:] - F[a, nc:]).max() > 1e-3) for a, e in runs]

    a, e = runs[args.run]
    t, m = T[a:e], M[a:e]
    kind = 'FxTal' if is_fx[args.run] else 'batch'
    rock = {500.0: 'MORB', 9000.0: 'Lherzolite'}.get(float(P[a]), f'P={P[a]:.0f}')
    act = [j for j in range(len(phases)) if (m[:, j] > PRESENT_TOL).any()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if not args.labels:
        print(f'run {args.run}: {rock} {kind}, T {t[0]:.0f} -> {t[-1]:.0f}, {len(t)} rows')
        ncol = 3
        nrow = int(np.ceil(len(act) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 3.2 * nrow), squeeze=False)
        for k, j in enumerate(act):
            ax = axes.flat[k]
            style(ax)
            ax.axhline(0, color='black', lw=1.0, zorder=1)
            ax.plot(t, m[:, j], color=LINE, lw=1.2, zorder=2)
            ax.scatter(t, m[:, j], s=5, color=LINE, zorder=3)
            ax.set_title(phases[j], color=INK, fontsize=10)
            ax.set_xlabel('T (°C)', color=INK, fontsize=8)
            ax.set_ylabel('n (cation-normalised moles)', color=INK, fontsize=8)
        for k in range(len(act), nrow * ncol):
            axes.flat[k].set_visible(False)
        fig.suptitle(f'Run {args.run}: {rock} {kind}, P = {P[a]:.0f} bar  (dots = rows in the bundle)',
                     color=INK, fontsize=11)
        fig.tight_layout()
        f = out / f'diag_run{args.run}_moles_vs_T.png'
        fig.savefig(f, dpi=140)
        print(f'wrote {f}')
        return

    # ---- c from the whole bundle ------------------------------------------------------
    slopes = {p: [] for p in phases}
    for (ra, re_), fx in zip(runs, is_fx):
        recipe = args.fx_recipe if fx else args.batch_recipe
        for j, p in enumerate(phases):
            n = M[ra:re_, j]
            if (n > PRESENT_TOL).any():
                cr = find_crossings(T[ra:re_], n, recipe)
                if cr:
                    slopes[p].append(np.mean([c['s'] for c in cr]))
    s_med = {p: float(np.median(v)) for p, v in slopes.items() if v}
    c = {p: s * args.dT for p, s in s_med.items()}
    tab = pd.DataFrame([dict(phase=p, runs_with_crossing=len(slopes[p]), median_s_per_C=s_med[p], c=c[p],
                             n_max_in_bundle=float(M[:, phases.index(p)].max()),
                             n98_present=float(np.percentile(M[M[:, phases.index(p)] > 0, phases.index(p)], 98)))
                        for p in s_med])
    tab['c_over_n98'] = tab.c / tab.n98_present
    tag = f'fx{args.fx_recipe}_b{args.batch_recipe}_dT{int(args.dT)}'
    tab.to_csv(out / f'c_{tag}.csv', index=False)
    print(f'c per phase (dT = {args.dT:g} C, FxTal recipe {args.fx_recipe}, batch recipe {args.batch_recipe}):')
    print(tab.to_string(index=False, float_format=lambda x: f'{x:.4g}'))

    # ---- y vs T for the chosen run ----------------------------------------------------
    recipe = args.fx_recipe if is_fx[args.run] else args.batch_recipe
    act = [j for j in act if phases[j] in c]
    ncol = 3
    nrow = int(np.ceil(len(act) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 3.2 * nrow), squeeze=False)
    print(f'\nrun {args.run}: {rock} {kind}; crossings ({recipe} recipe):')
    for k, j in enumerate(act):
        p = phases[j]
        ax = axes.flat[k]
        style(ax)
        cr = find_crossings(t, m[:, j], recipe)
        g = run_labels(t, m[:, j], cr)[0]
        y = activate(g, c[p], args.eps)
        ax.axhline(0, color='black', lw=1.0, zorder=1)
        ax.axhline(-c[p], color=MUTED, lw=0.9, ls='--', zorder=1)
        for cc in cr:
            ax.axvline(cc['T_star'], color='#eb6834', lw=0.8, ls=':', zorder=1)
        ax.plot(t, y, color=LINE, lw=1.2, zorder=2)
        ax.scatter(t, y, s=5, color=LINE, zorder=3)
        ax.set_title(f'{p}   c = {c[p]:.3g}', color=INK, fontsize=10)
        ax.set_xlabel('T (°C)', color=INK, fontsize=8)
        ax.set_ylabel('y', color=INK, fontsize=8)
        print(f'  {p:<14} ' + '; '.join(f"T*={cc['T_star']:.2f} s={cc['s']:.3g}/C stable {cc['side']}" for cc in cr))
    for k in range(len(act), nrow * ncol):
        axes.flat[k].set_visible(False)
    fig.suptitle(f'Run {args.run}: {rock} {kind}, labels y vs T  (dT = {args.dT:g} °C, eps = {args.eps:g}, '
                 f'{recipe} slope; dashed = −c, dotted = T*)', color=INK, fontsize=11)
    fig.tight_layout()
    f = out / f'diag_run{args.run}_y_vs_T_{tag}.png'
    fig.savefig(f, dpi=140)
    print(f'wrote {f}')


if __name__ == '__main__':
    main()
