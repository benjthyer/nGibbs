"""
Test of the affinity-free dn/dT projection on fractional-crystallisation (FxTal) runs.

Bundle: data/TestFxTal_dndT/MELTS120_RefStandardsFxTalBatchOpenNoCr.tar.gz
  42 isobaric cooling runs, concatenated row-wise (rows of one run are in descending T):
    - 2 FxTal runs ('fractionate solids'): MORB (500 bar, FMQ+0) and Lherzolite (9 kbar, FMQ-2)
    - 40 batch runs (constant bulk), 20 per rock, each started from a bulk composition
      sampled along the FxTal liquid line of descent.
  Each batch run contains one row (same bulk, same T) that is the same equilibrium
  as a row of its parent FxTal run. That batch run is the constant-composition ground truth
  for the FxTal row's signed abundance / undersaturation label.

Unactivated label g (see the 'signed phase-abundance' brief):
    present:  g = n
    absent:   g = -s * |T - T*| using the nearest crossing in the run
    phase never present in the run: undefined (NaN)

Slope recipes at a crossing (p = present row next to the crossing, a = absent row next to it,
p2 = next present row further into the field):
    'brief' : s = |n_p2 - n_p| / |T_p2 - T_p|; T* = T_p -/+ n_p/s, clamped inside [T_p, T_a].
    'nstep' : s = n_p2 / |T_p2 - T_p|  (FxTal: n is the per-step increment, so n/step = dn/dT);
              T* = T_p -/+ n_p/s, clamped inside [T_p, T_a].
    'last'  : s = n_p / |T_p - T_a|  (last saturated step's n / dT); continuing that line through
              zero puts T* on the absent row, T* = T_a.

Label schemes (FxTal recipe, batch recipe):
    nstep : ('nstep', 'brief')   previous default
    brief : ('brief', 'brief')   brief recipe everywhere
    last  : ('last',  'last')    same recipe in both run types

c_phi = median over runs (of the per-run mean s at its crossings) * dT_window, from each
scheme's own slopes. No n98 floor/cap. Labels y = h(max(g, -3c)),
h(g) = g (g >= 0), c*tanh(g/c) + eps*g (g < 0).

Outputs (in --out):
    fig1_abundance_1to1.png                 equilibrium sanity check at the 40 matched rows
    fig2_residuals_vs_dsat.png              (y_FxTal - y_batch)/c vs FxTal distance from saturation
    fig3_gspace_decomposition.png           g, |T - T*| and s ratios (batch / FxTal), absent in both
    fig4_<scheme>_fxtal_y_vs_TmT_{symlog,linear}.png
                                            y vs T - T* along both FxTal runs, per phase, with the
                                            matched batch (ground-truth) values
    fig5_<scheme>_batch_y_vs_T_<rock>.png   y vs T for 5 batch runs along the LLD, per phase,
                                            with each run's matched FxTal value joined to it
    matched_cells.csv, crossings.csv, c_per_phase.csv, summary.txt

Usage
  python scripts/test_fxtal_dndT_projection.py [--bundle PATH] [--out DIR] [--eps 0.02] [--dT-plot 25]
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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / 'src'))
from ngibbs.utils.file_utils import load_ml_bundle  # noqa: E402

DEFAULT_BUNDLE = REPO_ROOT / 'data' / 'TestFxTal_dndT' / 'MELTS120_RefStandardsFxTalBatchOpenNoCr.tar.gz'
DEFAULT_OUT = REPO_ROOT / 'data' / 'TestFxTal_dndT' / 'projection_test'

PRESENT_TOL = 0.0        # molar_labels carry exact zeros for absent phases
CLIP_DEPTH = 3.0         # labels clipped at -3c before activation
DT_WINDOWS = (10.0, 25.0, 50.0)
MATCH_TOL = 1e-4         # max |d(element fraction)| for a bulk match (true matches ~1e-6, next best ~2e-4)
N_BATCH_SHOWN = 5        # batch runs per rock in fig 5
SCHEMES = {'nstep': ('nstep', 'brief'), 'brief': ('brief', 'brief'), 'last': ('last', 'last')}
SCHEME_TITLE = {'nstep': 'FxTal s = n(full step)/dT; batch: brief recipe',
                'brief': 'brief recipe (2-row difference) in both',
                'last': 'last saturated step n/dT in both (T* on absent row)'}
ROCK_BY_P = {500.0: 'MORB', 9000.0: 'Lherzolite'}
ROCK_COLOR = {'MORB': '#2a78d6', 'Lherzolite': '#eb6834'}
# one-hue ordinal ramps (early -> late along the LLD) for fig 5
ROCK_RAMP = {'MORB': ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#0d366b'],
             'Lherzolite': ['#f6b394', '#f08a5d', '#eb6834', '#c04a1b', '#7f2c0b']}
INK, MUTED, GRID = '#222222', '#6b6b6b', '#e4e4e4'
MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '<', '>', 'h', '*', 'p']


# ----------------------------------------------------------------------------------------
# Runs, crossings, labels
# ----------------------------------------------------------------------------------------
def split_runs(T):
    """Rows of a run descend in T; a new run starts wherever T goes up."""
    starts = np.r_[0, np.where(np.diff(T) > 0)[0] + 1]
    ends = np.r_[starts[1:], len(T)]
    return list(zip(starts, ends))


def find_crossings(t, n, recipe):
    """Saturation crossings of one phase along one monotonic run (see module docstring)."""
    present = n > PRESENT_TOL
    out = []
    for i in range(len(t) - 1):
        if present[i] == present[i + 1]:
            continue
        p, a = (i, i + 1) if present[i] else (i + 1, i)
        p2 = p + (p - a)
        step_pa = abs(t[a] - t[p])
        has_p2 = 0 <= p2 < len(t) and present[p2]
        direction = np.sign(t[a] - t[p])         # from present row towards absent row
        if recipe == 'last':
            s, T_star = n[p] / step_pa, float(t[a])
        else:
            s = np.nan
            if has_p2:
                dt2 = abs(t[p2] - t[p])
                s = n[p2] / dt2 if recipe == 'nstep' else abs(n[p2] - n[p]) / dt2
            if not np.isfinite(s) or s <= 0:
                s = n[p] / step_pa
            T_star = float(np.clip(t[p] + direction * n[p] / s, min(t[p], t[a]), max(t[p], t[a])))
        out.append(dict(T_star=T_star, s=float(s), single=not has_p2,
                        side='below' if direction > 0 else 'above'))
    return out


def run_labels(t, n, crossings):
    """g, dsat (+ undersaturated distance, - inside the field), and the slope / T* / side used."""
    m = len(t)
    g = np.full(m, np.nan)
    dsat = np.full(m, np.nan)
    s_used = np.full(m, np.nan)
    Tstar_used = np.full(m, np.nan)
    side_used = np.array([''] * m, dtype=object)
    present = n > PRESENT_TOL
    g[present] = n[present]
    if crossings:
        Ts = np.array([c['T_star'] for c in crossings])
        S = np.array([c['s'] for c in crossings])
        D = np.abs(t[:, None] - Ts[None, :])
        k = D.argmin(axis=1)
        d = D[np.arange(m), k]
        absent = ~present
        g[absent] = -S[k[absent]] * d[absent]
        dsat = np.where(present, -d, d)
        s_used = S[k]
        Tstar_used = Ts[k]
        side_used = np.array([crossings[j]['side'] for j in k], dtype=object)
    return g, dsat, s_used, Tstar_used, side_used


def activate(g, c, eps):
    """y = h(max(g, -3c)); h(g) = g for g >= 0, c*tanh(g/c) + eps*g for g < 0."""
    g = np.asarray(g, dtype=float)
    gc = np.maximum(g, -CLIP_DEPTH * c)
    with np.errstate(invalid='ignore'):
        return np.where(gc >= 0, gc, c * np.tanh(gc / c) + eps * gc)


def style(ax):
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)


# ----------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bundle', default=str(DEFAULT_BUNDLE))
    ap.add_argument('--out', default=str(DEFAULT_OUT))
    ap.add_argument('--eps', type=float, default=0.02, help='leak of the tanh-ReLU (default 0.02)')
    ap.add_argument('--dT-plot', type=float, default=25.0, help='dT window used for figs 4-5 (default 25)')
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    eps = args.eps

    bundle = load_ml_bundle(args.bundle, arrays=('features', 'molar_labels'))
    F = np.asarray(bundle.features, dtype=float)
    M = np.asarray(bundle.molar_labels, dtype=float)
    idx = bundle.ml_indexer
    phases = list(idx.all_phases)
    fnames = list(idx.featureNames)
    iP = fnames.index('Pressure(System_main)')
    iT = fnames.index('Temperature(System_main)')
    n_cond = len(fnames)
    T = F[:, iT]
    N = len(T)

    # --- runs ---------------------------------------------------------------------------
    runs = []
    for k, (a, b) in enumerate(split_runs(T)):
        X = F[a:b, n_cond:]
        runs.append(dict(k=k, a=a, b=b, fx=bool(np.abs(X - X[0]).max() > 1e-3),
                         rock=ROCK_BY_P.get(float(F[a, iP]), f'P{F[a, iP]:.0f}')))
    n_fx_runs = sum(r['fx'] for r in runs)
    print(f'{len(runs)} runs: {n_fx_runs} FxTal, {len(runs) - n_fx_runs} batch')
    fx_by_rock = {r['rock']: r for r in runs if r['fx']}
    rocks = list(fx_by_rock)

    # --- labels per scheme --------------------------------------------------------------
    store, run_slopes, crossing_rows = {}, {}, []
    for sc, (rec_fx, rec_b) in SCHEMES.items():
        store[sc] = {p: dict(g=np.full(N, np.nan), dsat=np.full(N, np.nan), s=np.full(N, np.nan),
                             Tstar=np.full(N, np.nan), side=np.array([''] * N, dtype=object))
                     for p in phases}
        run_slopes[sc] = {p: [] for p in phases}
        for r in runs:
            a, b = r['a'], r['b']
            t = T[a:b]
            recipe = rec_fx if r['fx'] else rec_b
            for j, p in enumerate(phases):
                n = M[a:b, j]
                if not (n > PRESENT_TOL).any():
                    continue
                cr = find_crossings(t, n, recipe)
                g, dsat, s_u, Ts_u, side_u = run_labels(t, n, cr)
                st = store[sc][p]
                st['g'][a:b], st['dsat'][a:b], st['s'][a:b] = g, dsat, s_u
                st['Tstar'][a:b], st['side'][a:b] = Ts_u, side_u
                if cr:
                    run_slopes[sc][p].append(np.mean([c['s'] for c in cr]))
                for c in cr:
                    crossing_rows.append(dict(scheme=sc, run=r['k'], rock=r['rock'], fxtal=r['fx'], phase=p, **c))
    pd.DataFrame(crossing_rows).to_csv(out / 'crossings.csv', index=False)

    s_med = {sc: {p: (float(np.median(v)) if v else np.nan) for p, v in run_slopes[sc].items()} for sc in SCHEMES}
    c_rows = []
    for sc in SCHEMES:
        for p in phases:
            if run_slopes[sc][p]:
                c_rows.append(dict(scheme=sc, phase=p, n_runs_with_crossing=len(run_slopes[sc][p]),
                                   median_s_per_C=s_med[sc][p],
                                   **{f'c_dT{int(w)}': s_med[sc][p] * w for w in DT_WINDOWS}))
    c_tab = pd.DataFrame(c_rows)
    c_tab.to_csv(out / 'c_per_phase.csv', index=False)
    print('\nmedian slope per phase and scheme (cation-normalised moles / degC):')
    print(c_tab.pivot(index='phase', columns='scheme', values='median_s_per_C').to_string(float_format=lambda x: f'{x:.3g}'))

    def cval(sc, p, w):
        return s_med[sc][p] * w

    # --- matching -----------------------------------------------------------------------
    matches = []
    for r in runs:
        if r['fx']:
            continue
        fx = fx_by_rock[r['rock']]
        dX = np.abs(F[fx['a']:fx['b'], n_cond:] - F[r['a'], n_cond:]).max(axis=1)
        i_fx = fx['a'] + int(dX.argmin())
        if dX.min() > MATCH_TOL:
            print(f'run {r["k"]}: no FxTal match (best {dX.min():.2e}) - skipped')
            continue
        same_T = np.where(T[r['a']:r['b']] == T[i_fx])[0]
        if len(same_T) == 0:
            print(f'run {r["k"]}: matched bulk but T={T[i_fx]} not in batch run - skipped')
            continue
        matches.append(dict(run=r['k'], rock=r['rock'], i_fx=i_fx, i_b=r['a'] + int(same_T[0]),
                            T=T[i_fx], bulk_mismatch=dX.min()))
    print(f'\n{len(matches)} batch runs matched to an FxTal row')

    # --- matched cells ------------------------------------------------------------------
    rows = []
    for mt in matches:
        for j, p in enumerate(phases):
            rec = dict(run=mt['run'], rock=mt['rock'], T=mt['T'], phase=p,
                       n_fx=M[mt['i_fx'], j], n_b=M[mt['i_b'], j])
            for sc in SCHEMES:
                st = store[sc][p]
                for side_tag, i in (('fx', mt['i_fx']), ('b', mt['i_b'])):
                    for key in ('g', 'dsat', 's', 'Tstar', 'side'):
                        rec[f'{key}_{side_tag}_{sc}'] = st[key][i]
                for w in DT_WINDOWS:
                    c, tag = cval(sc, p, w), f'{sc}_dT{int(w)}'
                    rec[f'c_{tag}'] = c
                    if np.isfinite(c):
                        rec[f'yc_fx_{tag}'] = float(activate(rec[f'g_fx_{sc}'], c, eps) / c)
                        rec[f'yc_b_{tag}'] = float(activate(rec[f'g_b_{sc}'], c, eps) / c)
                        rec[f'res_{tag}'] = rec[f'yc_fx_{tag}'] - rec[f'yc_b_{tag}']
            rows.append(rec)
    cells = pd.DataFrame(rows)
    cells.to_csv(out / 'matched_cells.csv', index=False)

    phase_marker = {}
    for p in phases:
        if ((cells.phase == p) & ((cells.n_fx > 0) | (cells.n_b > 0))).any():
            phase_marker[p] = MARKERS[len(phase_marker) % len(MARKERS)]
    handles = [Line2D([], [], ls='', marker='o', color=ROCK_COLOR[r], label=r) for r in ROCK_COLOR]
    handles += [Line2D([], [], ls='', marker=phase_marker[p], color=MUTED, label=p) for p in phase_marker]

    # =====================================================================================
    # Figure 1: 1:1 abundances at the matched rows
    # =====================================================================================
    floor = 1e-7
    pres = cells[(cells.n_fx > 0) | (cells.n_b > 0)]
    fig, ax = plt.subplots(figsize=(6.2, 5.8))
    style(ax)
    for (rock, p), d in pres.groupby(['rock', 'phase']):
        both = (d.n_fx > 0) & (d.n_b > 0)
        ax.scatter(d.n_b[both], d.n_fx[both], s=34, marker=phase_marker[p], color=ROCK_COLOR[rock],
                   edgecolor='white', linewidth=0.6, alpha=0.9, zorder=3)
        if (~both).any():
            ax.scatter(np.maximum(d.n_b[~both], floor), np.maximum(d.n_fx[~both], floor), s=40,
                       marker=phase_marker[p], facecolor='none', edgecolor=ROCK_COLOR[rock], linewidth=1.2, zorder=4)
    lims = [floor * 0.5, 1.5]
    ax.plot(lims, lims, color=MUTED, lw=1, ls='--', zorder=2)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('n, batch run at the matched T (cation-normalised moles)', color=INK)
    ax.set_ylabel('n, FxTal run (cation-normalised moles)', color=INK)
    one_only = (pres.n_fx > 0) != (pres.n_b > 0)
    rel = (pres.n_fx - pres.n_b).abs() / np.maximum(pres.n_fx, pres.n_b)
    ax.set_title(f'Equilibrium reproducibility at {len(matches)} matched rows\n'
                 f'{len(pres)} present-phase cells; {int(one_only.sum())} present in one run only (open, at {floor:g})',
                 color=INK, fontsize=10)
    ax.legend(handles=handles, fontsize=7.5, frameon=False, loc='upper left', ncol=2)
    fig.tight_layout(); fig.savefig(out / 'fig1_abundance_1to1.png', dpi=170); plt.close(fig)

    # =====================================================================================
    # Figure 2: residuals in label space vs distance from saturation in FxTal space
    # =====================================================================================
    sch = list(SCHEMES)
    fig, axes = plt.subplots(len(sch), len(DT_WINDOWS), figsize=(4.6 * len(DT_WINDOWS), 3.9 * len(sch)),
                             sharey=True, sharex=True)
    rms_tab = {}
    for i, sc in enumerate(sch):
        for jw, w in enumerate(DT_WINDOWS):
            ax = axes[i, jw]
            style(ax)
            col, xcol = f'res_{sc}_dT{int(w)}', f'dsat_fx_{sc}'
            d = cells[np.isfinite(cells[col]) & np.isfinite(cells[xcol])]
            for (rock, p), dd in d.groupby(['rock', 'phase']):
                ax.scatter(dd[xcol], dd[col], s=26, marker=phase_marker.get(p, 'o'), color=ROCK_COLOR[rock],
                           edgecolor='white', linewidth=0.5, alpha=0.85, zorder=3)
            ax.axhline(0, color=MUTED, lw=0.8, zorder=2)
            ax.axvspan(-1, 0, color='#f4f4f2', zorder=0)
            ax.set_xscale('symlog', linthresh=1.0)
            ab = d[d[xcol] > 0]
            rms = float(np.sqrt(np.mean(ab[col] ** 2))) if len(ab) else np.nan
            rms_tab[(sc, w)] = rms
            ax.set_title(f'dT window = {int(w)} °C   (absent-cell RMS {rms:.2f})', color=INK, fontsize=9.5)
            if jw == 0:
                ax.set_ylabel(f'{SCHEME_TITLE[sc]}\n\n(y_FxTal − y_batch) / c', color=INK, fontsize=8.5)
            if i == len(sch) - 1:
                ax.set_xlabel('distance from saturation in FxTal run (°C)\n< 0: phase present in FxTal row',
                              color=INK, fontsize=9)
    axes[0, 0].legend(handles=handles, fontsize=7, frameon=False, loc='lower left', ncol=2)
    fig.suptitle(f'Label-space residuals at matched rows (batch run = truth), eps = {eps:g}', color=INK, fontsize=11)
    fig.tight_layout(); fig.savefig(out / 'fig2_residuals_vs_dsat.png', dpi=150); plt.close(fig)

    # =====================================================================================
    # Figure 3: g-space decomposition (absent in both runs)
    # =====================================================================================
    fig, axes = plt.subplots(len(sch), 3, figsize=(14, 3.9 * len(sch)), sharex=True)
    for i, sc in enumerate(sch):
        d = cells[(cells.n_fx <= 0) & (cells.n_b <= 0)
                  & (cells[f'g_fx_{sc}'] < 0) & (cells[f'g_b_{sc}'] < 0)].copy()
        d['d_fx'] = d[f'dsat_fx_{sc}']
        ratios = {'g': d[f'g_b_{sc}'] / d[f'g_fx_{sc}'], 'd': d[f'dsat_b_{sc}'] / d.d_fx,
                  's': d[f's_b_{sc}'] / d[f's_fx_{sc}']}
        same = (d[f'side_b_{sc}'] == d[f'side_fx_{sc}']).values
        for jx, (key, lab) in enumerate([('g', 'g_batch / g_FxTal'), ('d', 'd_batch / d_FxTal'),
                                          ('s', 's_batch / s_FxTal')]):
            ax = axes[i, jx]
            style(ax)
            for (rock, p), dd in d.groupby(['rock', 'phase']):
                ii = d.index.get_indexer(dd.index)
                ss = same[ii]
                yv = ratios[key].loc[dd.index]
                ax.scatter(dd.d_fx[ss], yv[ss], s=24, marker=phase_marker.get(p, 'o'), color=ROCK_COLOR[rock],
                           edgecolor='white', linewidth=0.5, alpha=0.85, zorder=3)
                ax.scatter(dd.d_fx[~ss], yv[~ss], s=28, marker=phase_marker.get(p, 'o'), facecolor='none',
                           edgecolor=ROCK_COLOR[rock], linewidth=1.1, zorder=3)
            ax.axhline(1, color=MUTED, lw=0.8, ls='--')
            ax.set_xscale('log'); ax.set_yscale('log')
            if jx == 0:
                ax.set_ylabel(f'{SCHEME_TITLE[sc]}\n\n{lab}', color=INK, fontsize=8.5)
            else:
                ax.set_ylabel(lab, color=INK, fontsize=9)
            if i == len(sch) - 1:
                ax.set_xlabel('d_FxTal = |T − T*| in FxTal run (°C)', color=INK, fontsize=9)
    axes[0, 0].legend(handles=handles, fontsize=7, frameon=False, loc='best', ncol=2)
    fig.suptitle('Absent in both runs: batch / FxTal ratios of g, |T − T*| and s  '
                 '(open = nearest crossings on opposite sides)', color=INK, fontsize=10.5)
    fig.tight_layout(); fig.savefig(out / 'fig3_gspace_decomposition.png', dpi=150); plt.close(fig)

    # =====================================================================================
    # Figure 4: y vs T - T* along the FxTal runs, per phase, with matched batch truth
    # =====================================================================================
    w = args.dT_plot
    # phases with at least one saturation crossing in an FxTal run (T - T* undefined otherwise)
    fx_phases = [p for p in phases
                 if any(np.isfinite(store['nstep'][p]['Tstar'][fx_by_rock[r]['a']:fx_by_rock[r]['b']]).any()
                        for r in rocks)]
    fx_phases = [p for p in fx_phases if p != 'melts-liquid'] + (['melts-liquid'] if 'melts-liquid' in fx_phases else [])
    for sc in SCHEMES:
        for xmode in ('symlog', 'linear'):
            ncol = 3
            nrow = int(np.ceil(len(fx_phases) / ncol))
            fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 3.4 * nrow), squeeze=False)
            for k_, p in enumerate(fx_phases):
                ax = axes.flat[k_]
                style(ax)
                c = cval(sc, p, w)
                for rock in rocks:
                    fx = fx_by_rock[rock]
                    sl = slice(fx['a'], fx['b'])
                    g = store[sc][p]['g'][sl]
                    ok = np.isfinite(g) & np.isfinite(store[sc][p]['Tstar'][sl])
                    if not ok.any():
                        continue
                    x = (T[sl] - store[sc][p]['Tstar'][sl])[ok]
                    y = activate(g[ok], c, eps)
                    ax.scatter(x, y, s=4, color=ROCK_COLOR[rock], alpha=0.55, lw=0, zorder=2)
                    mm = cells[(cells.rock == rock) & (cells.phase == p)
                               & np.isfinite(cells[f'g_b_{sc}']) & np.isfinite(cells[f'g_fx_{sc}'])]
                    if len(mm):
                        xm = mm['T'] - mm[f'Tstar_fx_{sc}']
                        yb = activate(mm[f'g_b_{sc}'].values, c, eps)
                        yf = activate(mm[f'g_fx_{sc}'].values, c, eps)
                        for xx, a_, b_ in zip(xm, yf, yb):
                            ax.plot([xx, xx], [a_, b_], color=INK, lw=0.8, zorder=3)
                        ax.scatter(xm, yb, s=34, color=ROCK_COLOR[rock], edgecolor=INK, linewidth=0.9, zorder=4)
                ax.axhline(0, color=MUTED, lw=0.7, zorder=1)
                ax.axhline(-c, color=MUTED, lw=0.7, ls=':', zorder=1)
                ax.axvline(0, color=GRID, lw=1.0, zorder=1)
                if xmode == 'symlog':
                    ax.set_xscale('symlog', linthresh=2.0)
                ax.set_title(f'{p}   (c = {c:.3g})', color=INK, fontsize=9.5)
                ax.set_xlabel('T − T* (°C), nearest FxTal crossing', color=INK, fontsize=8)
                ax.set_ylabel('y', color=INK, fontsize=8)
            for k_ in range(len(fx_phases), nrow * ncol):
                axes.flat[k_].set_visible(False)
            h4 = [Line2D([], [], ls='', marker='o', ms=3, color=ROCK_COLOR[r], label=f'{r} FxTal') for r in rocks]
            h4 += [Line2D([], [], ls='', marker='o', ms=6, color=ROCK_COLOR[r], mec=INK, label=f'{r} batch truth')
                   for r in rocks]
            axes.flat[0].legend(handles=h4, fontsize=7, frameon=False, loc='best')
            fig.suptitle(f'FxTal labels y vs T − T*  [{SCHEME_TITLE[sc]}]  dT window {int(w)} °C, eps {eps:g}; '
                         f'black bars join FxTal label to batch truth at matched rows; dotted line = −c',
                         color=INK, fontsize=10)
            fig.tight_layout(); fig.savefig(out / f'fig4_{sc}_fxtal_y_vs_TmT_{xmode}.png', dpi=140); plt.close(fig)

    # =====================================================================================
    # Figure 5: y vs T along N_BATCH_SHOWN batch runs per rock, with matched FxTal values
    # =====================================================================================
    for sc in SCHEMES:
        for rock in rocks:
            b_runs = [mt for mt in matches if mt['rock'] == rock]
            pick = np.unique(np.round(np.linspace(0, len(b_runs) - 1, N_BATCH_SHOWN)).astype(int))
            shown = [b_runs[i] for i in pick]
            rk = {r['k']: r for r in runs}
            fx = fx_by_rock[rock]
            ph = [p for p in phases if p != 'melts-liquid' and np.isfinite(s_med[sc][p]) and (
                any(np.isfinite(store[sc][p]['g'][rk[mt['run']]['a']:rk[mt['run']]['b']]).any() for mt in shown)
                or np.isfinite(store[sc][p]['g'][fx['a']:fx['b']]).any())]
            if np.isfinite(s_med[sc]['melts-liquid']):
                ph.append('melts-liquid')
            ncol = 3
            nrow = int(np.ceil(len(ph) / ncol))
            fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 3.3 * nrow), squeeze=False)
            for k_, p in enumerate(ph):
                ax = axes.flat[k_]
                style(ax)
                c = cval(sc, p, w)
                gfx = store[sc][p]['g'][fx['a']:fx['b']]
                if np.isfinite(gfx).any():
                    ok = np.isfinite(gfx)
                    ax.plot(T[fx['a']:fx['b']][ok], activate(gfx[ok], c, eps), color='#b9b9b4', lw=1.0, zorder=1)
                for col_, mt in zip(ROCK_RAMP[rock], shown):
                    r = rk[mt['run']]
                    sl = slice(r['a'], r['b'])
                    g = store[sc][p]['g'][sl]
                    ok = np.isfinite(g)
                    if ok.any():
                        ax.plot(T[sl][ok], activate(g[ok], c, eps), color=col_, lw=1.4, zorder=2)
                    gb, gf = store[sc][p]['g'][mt['i_b']], store[sc][p]['g'][mt['i_fx']]
                    if np.isfinite(gb) and np.isfinite(gf):
                        yb, yf = float(activate(gb, c, eps)), float(activate(gf, c, eps))
                        ax.plot([mt['T'], mt['T']], [yb, yf], color=INK, lw=1.0, zorder=4)
                        ax.scatter([mt['T']], [yb], s=30, color=col_, edgecolor=INK, linewidth=0.8, zorder=5)
                        ax.scatter([mt['T']], [yf], s=30, marker='D', facecolor='white', edgecolor=INK,
                                   linewidth=0.9, zorder=5)
                    elif np.isfinite(gf):
                        ax.scatter([mt['T']], [float(activate(gf, c, eps))], s=30, marker='D', facecolor='white',
                                   edgecolor=MUTED, linewidth=0.9, zorder=5)
                ax.axhline(0, color=MUTED, lw=0.7, zorder=0)
                ax.axhline(-c, color=MUTED, lw=0.7, ls=':', zorder=0)
                ax.set_title(f'{p}   (c = {c:.3g})', color=INK, fontsize=9.5)
                ax.set_xlabel('T (°C)', color=INK, fontsize=8)
                ax.set_ylabel('y', color=INK, fontsize=8)
            for k_ in range(len(ph), nrow * ncol):
                axes.flat[k_].set_visible(False)
            fig.suptitle(f'{rock}: batch labels y vs T for {len(shown)} bulks along the LLD (light = early, dark = late); '
                         f'circle = batch at matched T, white diamond = FxTal label there; grey = full FxTal run\n'
                         f'[{SCHEME_TITLE[sc]}]  dT window {int(w)} °C, eps {eps:g}', color=INK, fontsize=9.5)
            fig.tight_layout(); fig.savefig(out / f'fig5_{sc}_batch_y_vs_T_{rock}.png', dpi=140); plt.close(fig)

    # =====================================================================================
    # Text summary
    # =====================================================================================
    lines = []
    P = lines.append
    P(f'bundle: {args.bundle}')
    P(f'runs: {len(runs)} ({n_fx_runs} FxTal); matched batch runs: {len(matches)}; eps = {eps}')
    P(f'max bulk mismatch of a match: {max(mt["bulk_mismatch"] for mt in matches):.2e}')
    P('')
    P('1:1 sanity check')
    both = (pres.n_fx > 0) & (pres.n_b > 0)
    P(f'  present-phase cells: {len(pres)}; present in one run only: {int(one_only.sum())}')
    P(f'  both present: median |rel diff| {np.median(rel[both]):.2e}, max {rel[both].max():.2e}')
    for _, rr in pres[one_only].iterrows():
        P(f'    run {rr.run:>2} {rr.rock:<10} T={rr["T"]:.0f} {rr.phase:<14} n_fx={rr.n_fx:.3e} n_b={rr.n_b:.3e} '
          f'batch |T-T*|={rr.dsat_b_nstep:.0f}')
    P('')
    P('Absent-cell RMS of (y_fx - y_b)/c')
    for sc in SCHEMES:
        P(f'  {sc:<6} ' + '  '.join(f'dT{int(w)}={rms_tab[(sc, w)]:.3f}' for w in DT_WINDOWS))
    P('')
    for sc in SCHEMES:
        gfx, gb = np.isfinite(cells[f'g_fx_{sc}']), np.isfinite(cells[f'g_b_{sc}'])
        P(f'[{sc}] label coverage: both {int((gfx & gb).sum())}, FxTal only {int((gfx & ~gb).sum())}, '
          f'batch only {int((~gfx & gb).sum())}')
        for w in DT_WINDOWS:
            sel = ~gfx & gb & np.isfinite(cells[f'yc_b_{sc}_dT{int(w)}'])
            if sel.any():
                v = cells.loc[sel, f'yc_b_{sc}_dT{int(w)}']
                P(f'    dT={int(w)}: batch-truth y/c at one-sided cells: fraction <= -0.5: {(v <= -0.5).mean():.2f}')
    P('')
    bins = [-np.inf, 0, 2, 5, 10, 25, 50, 100, np.inf]
    P('Residual (y_fx - y_b)/c by FxTal distance from saturation (deg C). cols: n, median, mean|.|, p90|.|')
    for sc in SCHEMES:
        for w in DT_WINDOWS:
            col, xcol = f'res_{sc}_dT{int(w)}', f'dsat_fx_{sc}'
            d = cells[np.isfinite(cells[col]) & np.isfinite(cells[xcol])]
            P(f'  scheme={sc}  dT={int(w)}')
            for b_, dd in d.groupby(pd.cut(d[xcol], bins), observed=True):
                r_ = dd[col]
                P(f'    {str(b_):<16} {len(r_):>4} {r_.median():+.3f} {r_.abs().mean():.3f} {r_.abs().quantile(0.9):.3f}')
    (out / 'summary.txt').write_text('\n'.join(lines))
    print('\n'.join(lines))
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
