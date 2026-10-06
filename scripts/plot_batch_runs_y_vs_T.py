"""
y vs T for every batch run of the FxTal/batch reference bundle, with the matched FxTal label,
plus histograms of the crossing slope s per phase.

For each batch run (40), one figure with a panel per phase:
    blue line      batch-run label y(T) (batch recipe for s; constant composition)
    orange X       the FxTal label y at the single FxTal row with the same bulk, P and T.
                   It is computed from the FxTal run alone (FxTal recipe for s), with no use of the
                   batch runs, as it would be when labelling an FxTal dataset.
    dashed line    -c ;   dotted orange lines: T* of the batch run's crossings
Labels and c use the same code as test_fxtal_dndT_projection.py:
    g = n (present), -s*|T - T*| (absent, nearest crossing in the run)
    y = h(max(g, -3c)), h(g) = g (g >= 0), c*tanh(g/c) + eps*g (g < 0)
    c_phi = median over all 42 runs of the per-run mean s, times dT

Histogram figure: s at every crossing, per phase, log10 axis, split into entries on cooling
(phase stable below T*) and exits on cooling (stable above T*). FxTal crossings are marked as a rug.

Usage
  python scripts/plot_batch_runs_y_vs_T.py [--dT 25] [--eps 0.02] [--fx-recipe nstep] [--batch-recipe brief]
                                          [--bundle PATH] [--out DIR]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parents[1]
for p_ in (REPO_ROOT / 'src', REPO_ROOT / 'scripts'):
    if str(p_) not in sys.path:
        sys.path.insert(0, str(p_))
from ngibbs.utils.file_utils import load_ml_bundle  # noqa: E402
from test_fxtal_dndT_projection import (split_runs, find_crossings, run_labels, activate,  # noqa: E402
                                        PRESENT_TOL, MATCH_TOL)

DEFAULT_BUNDLE = REPO_ROOT / 'data' / 'TestFxTal_dndT' / 'MELTS120_RefStandardsFxTalBatchOpenNoCr.tar.gz'
DEFAULT_OUT = REPO_ROOT / 'data' / 'TestFxTal_dndT' / 'projection_test'
INK, MUTED, GRID = '#222222', '#6b6b6b', '#e4e4e4'
BLUE, ORANGE, FAIL_COL = '#2a78d6', '#eb6834', '#b3123a'
ENTRY_COL, EXIT_COL = '#2a78d6', '#1baf7a'


def style(ax):
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dT', type=float, default=25.0)
    ap.add_argument('--eps', type=float, default=0.02)
    ap.add_argument('--fx-recipe', default='nstep', choices=['nstep', 'brief', 'last'])
    ap.add_argument('--batch-recipe', default='brief', choices=['nstep', 'brief', 'last'])
    ap.add_argument('--filter', default='ratio', choices=['ratio', 'lognormal'],
                    help='ratio: s outside [s_med/R, R*s_med] fails; lognormal: s outside mu +/- k sigma '
                         'of log10 s (fit after dropping the outermost ceil(trim*N) crossings) fails')
    ap.add_argument('--flag-ratio', type=float, default=5.0, help='R for --filter ratio (default 5)')
    ap.add_argument('--sigma', type=float, default=2.0, help='k for --filter lognormal (default 2)')
    ap.add_argument('--trim', type=float, default=0.05,
                    help='fraction of crossings dropped (rounded up) before the lognormal fit (default 0.05)')
    ap.add_argument('--bundle', default=str(DEFAULT_BUNDLE))
    ap.add_argument('--out', default=str(DEFAULT_OUT))
    args = ap.parse_args()
    eps = args.eps

    b = load_ml_bundle(args.bundle, arrays=('features', 'molar_labels'))
    F, M = np.asarray(b.features, float), np.asarray(b.molar_labels, float)
    phases = list(b.ml_indexer.all_phases)
    fn = list(b.ml_indexer.featureNames)
    nc = len(fn)
    T, P = F[:, fn.index('Temperature(System_main)')], F[:, fn.index('Pressure(System_main)')]
    rock_of = {500.0: 'MORB', 9000.0: 'Lherzolite'}

    runs = []
    for k, (a, e) in enumerate(split_runs(T)):
        runs.append(dict(k=k, a=a, e=e, fx=bool(np.abs(F[a:e, nc:] - F[a, nc:]).max() > 1e-3),
                         rock=rock_of.get(float(P[a]), f'P{P[a]:.0f}')))
    fx_by_rock = {r['rock']: r for r in runs if r['fx']}

    # --- crossings, labels g for every run (each run labelled from its own rows only) -------
    G = np.full(M.shape, np.nan)
    SU = np.full(M.shape, np.nan)              # slope s of the crossing each row is projected from
    TS = {}                                    # (run, phase) -> list of crossings
    xrows = []
    for r in runs:
        a, e = r['a'], r['e']
        recipe = args.fx_recipe if r['fx'] else args.batch_recipe
        for j, p in enumerate(phases):
            n = M[a:e, j]
            if not (n > PRESENT_TOL).any():
                continue
            cr = find_crossings(T[a:e], n, recipe)
            TS[(r['k'], p)] = cr
            lab = run_labels(T[a:e], n, cr)
            G[a:e, j], SU[a:e, j] = lab[0], lab[2]
            for c_ in cr:
                xrows.append(dict(run=r['k'], rock=r['rock'], fxtal=r['fx'], phase=p, recipe=recipe, **c_))
    X = pd.DataFrame(xrows)
    per_run = X.groupby(['run', 'phase']).s.mean().reset_index()
    s_med = per_run.groupby('phase').s.median().to_dict()
    c = {p: s * args.dT for p, s in s_med.items()}
    X['s_over_median'] = X.s / X.phase.map(s_med)
    X['entry_on_cooling'] = X.side == 'below'
    R = args.flag_ratio

    # Per-phase acceptance band [lo, hi] for the crossing slope s.
    #   ratio     : [s_med/R, R*s_med]
    #   lognormal : drop the ceil(trim*N) crossings farthest from the median of log10 s,
    #               fit mean mu and std sigma (ddof=1) of log10 s to the rest,
    #               accept mu +/- k*sigma.
    bounds, fit_info = {}, {}
    for p, d in X.groupby('phase'):
        if args.filter == 'ratio':
            bounds[p] = (s_med[p] / R, s_med[p] * R)
            continue
        ls = np.log10(d.s.values)
        n_drop = int(np.ceil(args.trim * len(ls)))
        order = np.argsort(np.abs(ls - np.median(ls)))
        keep = order[:len(ls) - n_drop]
        dropped = order[len(ls) - n_drop:]
        if len(keep) < 3:                      # too few to fit a spread: accept everything
            bounds[p] = (0.0, np.inf)
            fit_info[p] = dict(mu=np.nan, sigma=np.nan, n=len(ls), n_drop=n_drop,
                               dropped=d.s.values[dropped])
            continue
        mu, sig = float(ls[keep].mean()), float(ls[keep].std(ddof=1))
        bounds[p] = (10 ** (mu - args.sigma * sig), 10 ** (mu + args.sigma * sig))
        fit_info[p] = dict(mu=mu, sigma=sig, n=len(ls), n_drop=n_drop, dropped=d.s.values[dropped])
    X['lo'] = X.phase.map(lambda p: bounds[p][0])
    X['hi'] = X.phase.map(lambda p: bounds[p][1])
    X['fails_filter'] = (X.s < X.lo) | (X.s > X.hi)

    # Row-level filter: an absent row fails when the crossing it is projected from has
    # s outside the band. Present rows (g = n) are never flagged.
    lo_vec = np.array([bounds.get(p, (np.nan, np.nan))[0] for p in phases])
    hi_vec = np.array([bounds.get(p, (np.nan, np.nan))[1] for p in phases])
    FAIL = np.isfinite(G) & (G < 0) & ((SU < lo_vec[None, :]) | (SU > hi_vec[None, :]))
    FAIL_LOW = FAIL & (SU < lo_vec[None, :])

    def fails(s, p):
        return s < bounds[p][0] or s > bounds[p][1]

    if args.filter == 'ratio':
        ftag, fdesc = f'filter{R:g}', f'[s̃/{R:g}, {R:g}s̃]'
    else:
        ftag = f'lognorm{args.sigma:g}sig_trim{args.trim:g}'
        fdesc = f'μ ± {args.sigma:g}σ of log s'
    tag = f'fx{args.fx_recipe}_b{args.batch_recipe}_dT{int(args.dT)}_{ftag}'
    out = Path(args.out)
    X.to_csv(out / f'crossings_s_{tag}.csv', index=False)

    # --- one figure per batch run ---------------------------------------------------------
    fig_dir = out / f'batch_y_vs_T_{tag}'
    fig_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for r in runs:
        if r['fx']:
            continue
        a, e = r['a'], r['e']
        fx = fx_by_rock[r['rock']]
        dX = np.abs(F[fx['a']:fx['e'], nc:] - F[a, nc:]).max(axis=1)
        i_fx = fx['a'] + int(dX.argmin())
        matched = dX.min() <= MATCH_TOL and (T[a:e] == T[i_fx]).any()
        t = T[a:e]
        show = [j for j, p in enumerate(phases) if p in c and
                (np.isfinite(G[a:e, j]).any() or (matched and np.isfinite(G[i_fx, j])))]
        ncol = 3
        nrow = int(np.ceil(len(show) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 3.2 * nrow), squeeze=False)
        for kk, j in enumerate(show):
            p = phases[j]
            ax = axes.flat[kk]
            style(ax)
            ax.axhline(0, color='black', lw=1.0, zorder=1)
            ax.axhline(-c[p], color=MUTED, lw=0.9, ls='--', zorder=1)
            cr = TS.get((r['k'], p), [])
            for c_ in cr:
                bad = fails(c_['s'], p)
                ax.axvline(c_['T_star'], color=FAIL_COL if bad else ORANGE, lw=1.1 if bad else 0.8,
                           ls=':', zorder=1)
            gb = G[a:e, j]
            if np.isfinite(gb).any():
                ok = np.isfinite(gb)
                yy = np.full(len(t), np.nan)
                yy[ok] = activate(gb[ok], c[p], eps)
                ax.plot(t, yy, color=BLUE, lw=1.3, zorder=2)
                fb = FAIL[a:e, j]
                if fb.any():
                    ax.plot(t, np.where(fb, yy, np.nan), color=FAIL_COL, lw=3.2, alpha=0.75,
                            solid_capstyle='butt', zorder=3)
            else:
                ax.text(0.5, 0.5, 'never present in this batch run', transform=ax.transAxes,
                        ha='center', va='center', color=MUTED, fontsize=8)
            yfx = yb = np.nan
            fx_fail = b_fail = False
            if matched and np.isfinite(G[i_fx, j]):
                yfx = float(activate(G[i_fx, j], c[p], eps))
                fx_fail = bool(FAIL[i_fx, j])
                if fx_fail:
                    ax.scatter([T[i_fx]], [yfx], marker='X', s=90, facecolor='white', edgecolor=FAIL_COL,
                               linewidth=1.6, zorder=5)
                else:
                    ax.scatter([T[i_fx]], [yfx], marker='X', s=70, color=ORANGE, edgecolor='white',
                               linewidth=0.6, zorder=5)
            if matched:
                ib = a + int(np.where(t == T[i_fx])[0][0])
                if np.isfinite(G[ib, j]):
                    yb = float(activate(G[ib, j], c[p], eps))
                    b_fail = bool(FAIL[ib, j])
                summary.append(dict(run=r['k'], rock=r['rock'], T=T[i_fx], phase=p,
                                    y_batch=yb, y_fxtal=yfx, c=c[p], batch_fails=b_fail, fxtal_fails=fx_fail,
                                    res_over_c=(yfx - yb) / c[p] if np.isfinite(yfx) and np.isfinite(yb) else np.nan))
            s_txt = ', '.join(f"{c_['s']:.2g}" + ('✗' if fails(c_['s'], p) else '') for c_ in cr[:3])
            ax.set_title(f'{p}   c = {c[p]:.3g}' + (f'   s = {s_txt}' if s_txt else ''), color=INK, fontsize=9)
            ax.set_xlabel('T (°C)', color=INK, fontsize=8)
            ax.set_ylabel('y', color=INK, fontsize=8)
        for kk in range(len(show), nrow * ncol):
            axes.flat[kk].set_visible(False)
        handles = [Line2D([], [], color=BLUE, lw=1.3, label='batch y'),
                   Line2D([], [], color=FAIL_COL, lw=3.2, alpha=0.75,
                          label=f'batch: projected from s outside {fdesc}'),
                   Line2D([], [], ls='', marker='X', ms=8, color=ORANGE, label='FxTal y at same bulk, P, T'),
                   Line2D([], [], ls='', marker='X', ms=8, mfc='white', mec=FAIL_COL, mew=1.6,
                          label='FxTal y, fails the s filter'),
                   Line2D([], [], color=MUTED, ls='--', label='−c'),
                   Line2D([], [], color=ORANGE, ls=':', label='T* (batch); red = s fails (✗)')]
        axes.flat[0].legend(handles=handles, fontsize=7, frameon=False, loc='best')
        mtxt = f'matched FxTal row at T = {T[i_fx]:.0f} °C' if matched else 'NO FxTal match'
        fig.suptitle(f'Run {r["k"]}: {r["rock"]} batch, P = {P[a]:.0f} bar; {mtxt}  '
                     f'(dT = {args.dT:g} °C, eps = {eps:g}; s: FxTal {args.fx_recipe}, batch {args.batch_recipe})',
                     color=INK, fontsize=10.5)
        fig.tight_layout()
        fig.savefig(fig_dir / f'run{r["k"]:02d}_{r["rock"]}_T{T[i_fx]:.0f}.png', dpi=120)
        plt.close(fig)
    S = pd.DataFrame(summary)
    S.to_csv(fig_dir / 'matched_points.csv', index=False)
    print(f'wrote {len(S.run.unique())} figures to {fig_dir}')

    # --- histograms of s per phase --------------------------------------------------------
    hp = [p for p in phases if p in s_med]
    ncol = 4
    nrow = int(np.ceil(len(hp) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.0 * nrow), squeeze=False)
    for kk, p in enumerate(hp):
        ax = axes.flat[kk]
        style(ax)
        d = X[X.phase == p]
        ls = np.log10(d.s)
        blo, bhi = bounds[p]
        lo, hi = np.floor(ls.min() * 4) / 4, np.ceil(ls.max() * 4) / 4
        bins = np.arange(lo, hi + 0.25, 0.25) if hi > lo else np.array([lo - 0.125, lo + 0.125])
        ax.hist([ls[d.entry_on_cooling], ls[~d.entry_on_cooling]], bins=bins, stacked=True,
                color=[ENTRY_COL, EXIT_COL], edgecolor='white', linewidth=0.8, zorder=2)
        fxs = ls[d.fxtal]
        if len(fxs):
            ax.plot(fxs, np.zeros(len(fxs)), ls='', marker='|', ms=14, mew=2, color=ORANGE, zorder=4,
                    clip_on=False)
        ax.axvline(np.log10(s_med[p]), color=INK, lw=1.0, ls='--', zorder=3)
        # acceptance band and its bounds
        xl = ax.get_xlim()
        bl = np.log10(blo) if blo > 0 else -np.inf
        bh = np.log10(bhi) if np.isfinite(bhi) else np.inf
        ax.axvspan(max(bl, -50), min(bh, 50), color='#f1efe8', zorder=0)
        for xb in (bl, bh):
            if np.isfinite(xb):
                ax.axvline(xb, color=FAIL_COL, lw=1.4, zorder=3)
        if args.filter == 'lognormal' and p in fit_info:
            fi = fit_info[p]
            if np.isfinite(fi['mu']):
                ax.axvline(fi['mu'], color=FAIL_COL, lw=1.0, ls=':', zorder=3)
            if len(fi['dropped']):
                ax.plot(np.log10(fi['dropped']), np.zeros(len(fi['dropped'])), ls='', marker='v', ms=7,
                        color=MUTED, zorder=5, clip_on=False)
        # keep the band lines visible even when they fall outside the data range
        finite_b = [x for x in (bl, bh) if np.isfinite(x)]
        ax.set_xlim(min([xl[0]] + [x - 0.2 for x in finite_b]), max([xl[1]] + [x + 0.2 for x in finite_b]))
        nf = int(d.fails_filter.sum())
        extra = ''
        if args.filter == 'lognormal' and p in fit_info:
            fi = fit_info[p]
            extra = (f'\nfit: dropped {fi["n_drop"]}, μ = {fi["mu"]:.2f}, σ = {fi["sigma"]:.2f}'
                     if np.isfinite(fi['mu']) else f'\n<3 left after trim: no filter')
        ax.set_title(f'{p}  (n = {len(d)}, fail {nf}){extra}', color=INK, fontsize=8.5)
        ax.set_xlabel('log10 s  (mol/°C, cation-normalised)', color=INK, fontsize=8)
        ax.set_ylabel('crossings', color=INK, fontsize=8)
    for kk in range(len(hp), nrow * ncol):
        axes.flat[kk].set_visible(False)
    hh = [Patch(color=ENTRY_COL, label='entry on cooling (stable below T*)'),
          Patch(color=EXIT_COL, label='exit on cooling (stable above T*)'),
          Line2D([], [], ls='', marker='|', ms=10, mew=2, color=ORANGE, label='FxTal crossing'),
          Line2D([], [], color=INK, ls='--', label='median of per-run means (sets c)'),
          Line2D([], [], color=FAIL_COL, lw=1.4, label=f'filter bounds, {fdesc}')]
    if args.filter == 'lognormal':
        hh += [Line2D([], [], color=FAIL_COL, lw=1.0, ls=':', label='μ (lognormal fit)'),
               Line2D([], [], ls='', marker='v', ms=7, color=MUTED, label=f'dropped before fit (outermost {args.trim:.0%})')]
    fig.legend(handles=hh, loc='lower right', fontsize=8, frameon=False, ncol=1,
               bbox_to_anchor=(0.98, 0.02))
    fig.suptitle(f'Crossing slope s per phase, all 42 runs  (FxTal: {args.fx_recipe}, batch: {args.batch_recipe})',
                 color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(out / f'hist_s_per_phase_{tag}.png', dpi=140)
    plt.close(fig)

    # --- text summary -------------------------------------------------------------------
    print(f'\nfilter {fdesc}; per phase: band, crossings failing low / high:')
    for p in hp:
        d = X[X.phase == p]
        lo_, hi_ = bounds[p]
        print(f'  {p:<14} n={len(d):>3}  band [{lo_:.2g}, {hi_:.2g}] (x{lo_ / s_med[p]:.2g} .. x{hi_ / s_med[p]:.2g} of s_med)  '
              f'low={int((d.s < lo_).sum())}  high={int((d.s > hi_).sum())}')
    yc = activate(G, np.array([c.get(p, np.nan) for p in phases])[None, :], eps) / \
        np.array([c.get(p, np.nan) for p in phases])[None, :]
    print(f'\nrows failing the filter (absent rows projected from s outside {fdesc}):')
    for rk in ('batch', 'FxTal'):
        rows = np.zeros(len(T), bool)
        for r in runs:
            if r['fx'] == (rk == 'FxTal'):
                rows[r['a']:r['e']] = True
        absent = np.isfinite(G[rows]) & (G[rows] < 0)
        fl, fh = FAIL_LOW[rows], FAIL[rows] & ~FAIL_LOW[rows]
        unsat = yc[rows] > -0.9
        print(f'  {rk}: {int(FAIL[rows].sum())} of {int(absent.sum())} absent labelled cells '
              f'({FAIL[rows].sum() / max(absent.sum(), 1):.1%}); low-s {int(fl.sum())} '
              f'({int((fl & unsat).sum())} with y/c > -0.9), high-s {int(fh.sum())} ({int((fh & unsat).sum())} with y/c > -0.9)')
    v = S.dropna(subset=['res_over_c'])
    keep = ~(v.batch_fails | v.fxtal_fails)
    for name, vv in (('all', v), ('neither side fails', v[keep]), ('a side fails', v[~keep])):
        if len(vv):
            print(f'matched cells, {name}: n={len(vv)}; |res|/c median {vv.res_over_c.abs().median():.3f}, '
                  f'p90 {vv.res_over_c.abs().quantile(0.9):.3f}, max {vv.res_over_c.abs().max():.3f}')
    print(v.reindex(v.res_over_c.abs().sort_values(ascending=False).index).head(12).to_string(index=False,
          float_format=lambda x: f'{x:.4g}'))


if __name__ == '__main__':
    main()
