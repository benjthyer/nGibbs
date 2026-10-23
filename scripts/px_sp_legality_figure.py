#!/usr/bin/env python3
"""Pyroxene / spinel legality check: emulator vs ML-ready bundle, with and without polish.

Figure-generation version of nGibbs' scripts/px_sp_legality_check.py. Instead of taking
checkpoint/bundle paths on the command line, it imports the shipped MELTS 1.0.2 emulator
from the installed `ngibbs` package and uses the Cr isothermal (NPT) emulator,
102Closed_Cr_NPT, together with its auto-discovered Test_subset bundle
(102Closed_Cr_NPT_Test_subset15000.tar.gz). All settings are the constants below.

Runs the emulator over the bundle twice:

    uncorrected   the network's compositions as they come out of the chemistry heads,
                  with NO pyroxene/spinel legality polish anywhere in the pipeline
    corrected     the deployment path: `polish_negative_px` / `polish_negative_cpx` /
                  `polish_negative_sp` applied (inside the iterative mass balance, or once on the raw
                  component moles when MASS_BALANCE = 'none')

Both passes use the same mass-balance mode (MASS_BALANCE), so the only difference
between them is the polish. For each pass and each phase in PHASES it writes

    <OUT_DIR>/<pass>/<phase>_1to1.png   predicted vs true oxide wt% within the phase,
                                        one panel per oxide the phase can contain,
                                        rows with any negative oxide highlighted
    <OUT_DIR>/<pass>/log.txt            ground-truth phase counts, presence precision /
                                        recall, negative-oxide counts per oxide, rows with
                                        any negative oxide, per-oxide MAE

and, across both passes, <OUT_DIR>/summary.csv (long format) and <OUT_DIR>/summary.txt, plus
a publication figure (<OUT_DIR>/<FIG_PHASE>_figure.pdf/.png): a 2 x k grid of 1:1 panels for
one phase and k oxides (FIG_OXIDES), uncorrected on top, corrected below.

Where the polish can run (and how this script switches it off)
-------------------------------------------------------------
- Training never polishes: ContinuousModel / AffinityModel train through
  `upper_forward`, and the gated model's training branch (binaries given) skips it.
- MidLevelNetwork.forward polishes `chem_out` at inference unless NN_only=True.
  The uncorrected pass calls it with NN_only=True. (ContinuousModel.forward never
  polishes; it ignores the flag.)
- NN_MELTS.apply_mass_balance(mode='iterative') polishes after every projection step
  via `polish_component_moles`. The uncorrected pass turns the emulator's
  `_polish_px` / `_polish_cpx` / `_polish_sp` flags off for the duration of the call, which makes
  `polish_component_moles` a no-op. mode='none' never polishes, so the corrected pass
  applies `polish_component_moles` once itself.

Oxides
------
Ground truth: bundle `labels` (raw MELTS endmember fractions) -> PxSp-transformed
basis -> oxide moles with `compToOx` (transformed basis), as in
ngibbs.deployment_tests.emulator_quality. Prediction: each phase's component moles ->
fractions -> oxide moles with the same `compToOx`. Both are converted to wt% with the
oxide molar masses. A predicted oxide is counted negative when its wt% is below
-NEG_TOL. Negative counts are taken over rows where the phase is PREDICTED present
(those are the compositions the emulator outputs); 1:1 plots and MAE use rows where
the phase is present in both ground truth and prediction.

Usage
-----
    python px_sp_legality_check.py

Note: importing `ngibbs.engine.models` builds every shipped emulator API (HeFESTo and
MELTS), so start-up takes a while.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
# matplotlib.use('Agg')  # disabled: a global backend switch breaks interactive plotting (e.g. PetThermoTools)
import matplotlib.pyplot as plt

from ngibbs.engine.models import MELTS102EmulatorCPU
from ngibbs.utils.file_utils import load_ml_bundle

# Emulator and its ground-truth bundle: MELTS 1.0.2, Cr variant, isothermal (102Closed_Cr_NPT)
API = MELTS102EmulatorCPU.cr
EMU = API.isothermal_emulator                     # NN_MELTS
BUNDLE_PATH = Path(API._test_bundles['isothermal'])
MODEL_NAME = Path(API._iso_path).name

# Settings
OUT_DIR = Path(__file__).resolve().parent / 'plots' / 'px_sp_legality' / Path(MODEL_NAME).stem
PHASES = ('orthopyroxene', 'clinopyroxene', 'spinel')
MASS_BALANCE = 'iterative'      # for BOTH passes ('iterative', the deployment default, or 'none')
MAX_SAMPLES = None              # random subset of bundle rows (None: all)
BATCH_SIZE = 2 ** 14
NEG_TOL = 1e-4                  # an oxide is negative when its wt% < -NEG_TOL (above float32 round-off)
MAX_PLOT_POINTS = 30000
FIG_PHASE = 'spinel'            # phase of the publication figure ('' to skip)
FIG_OXIDES = ('FeO', 'Al2O3', 'MgO')
SEED = 1337

PASSES = ('uncorrected', 'corrected')

# Chart styling (validated reference palette, light mode; first two categorical slots).
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'
GRID = '#e4e3df'
C_LEGAL = '#2a78d6'      # rows with no negative oxide
C_NEG = '#eb6834'        # rows with >= 1 negative oxide


# --------------------------------------------------------------------------------- #
#  Forward passes
# --------------------------------------------------------------------------------- #
def _forward_batch(emu, feats_raw, corrected, mode):
    """One batch -> (component_moles (B, C), phase_present (B, P)), both on CPU."""
    norm = emu.norm_features.norm(feats_raw)
    flags = ('_polish_px', '_polish_cpx', '_polish_sp')
    saved = {f: getattr(emu, f, False) for f in flags}
    try:
        if not corrected:
            for f in flags:
                setattr(emu, f, False)
        with torch.no_grad():
            out = emu.model.forward(norm, detailed=True, NN_only=not corrected)
            _lik, _chem, _lm, recon_bulk, comp_moles, phase_prop, phase_moles = out
            if corrected and mode == 'none':
                comp_moles = emu.polish_component_moles(comp_moles)
            res = emu.apply_mass_balance(
                phase_moles, recon_bulk, comp_moles, phase_prop, norm,
                mode=mode, outputs=['component_moles', 'phase_present'],
                phase_affinity=_lm)   # phase seeding before the correction, as forwardMB does
    finally:
        for f, v in saved.items():
            setattr(emu, f, v)
    return (res['component_moles'].detach().cpu().numpy().astype(np.float64),
            res['phase_present'].detach().cpu().numpy())


def run_pass(emu, features, corrected, mode, batch_size):
    cms, pps = [], []
    dev = emu.dev
    for a in range(0, len(features), batch_size):
        f = torch.as_tensor(features[a:a + batch_size], dtype=torch.float32, device=dev)
        cm, pp = _forward_batch(emu, f, corrected, mode)
        cms.append(cm)
        pps.append(pp)
    return np.concatenate(cms), np.concatenate(pps)


# --------------------------------------------------------------------------------- #
#  Oxides
# --------------------------------------------------------------------------------- #
def oxide_wt(fractions, comp_to_ox_phase, mw):
    """(N, k) transformed fractions -> (N, O) oxide wt%, signed (negatives kept)."""
    mol = fractions @ comp_to_ox_phase
    mass = mol * mw[None, :]
    tot = mass.sum(axis=1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(np.abs(tot) > 0, 100.0 * mass / tot, np.nan)


def predicted_phase_fractions(comp_moles, comp_idx):
    block = comp_moles[:, comp_idx]
    n = block.sum(axis=1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(n > 0, block / n, np.nan)


# --------------------------------------------------------------------------------- #
#  Metrics
# --------------------------------------------------------------------------------- #
def phase_metrics(gt_pos, pred_pos, wt_true, wt_pred, ox_cols, oxides, neg_tol):
    tp = int(np.sum(gt_pos & pred_pos))
    fp = int(np.sum(~gt_pos & pred_pos))
    fn = int(np.sum(gt_pos & ~pred_pos))
    m = {
        'n_rows': int(len(gt_pos)),
        'n_gt_present': int(gt_pos.sum()),
        'n_pred_present': int(pred_pos.sum()),
        'n_both_present': tp,
        'true_pos': tp, 'false_pos': fp, 'false_neg': fn,
        'precision': tp / (tp + fp) if tp + fp else np.nan,
        'recall': tp / (tp + fn) if tp + fn else np.nan,
    }
    neg_pred = wt_pred[pred_pos][:, ox_cols] < -neg_tol          # (n_pred, n_ox)
    neg_gt = wt_true[gt_pos][:, ox_cols] < -neg_tol
    m['n_rows_any_negative'] = int(neg_pred.any(axis=1).sum())
    m['frac_rows_any_negative'] = (m['n_rows_any_negative'] / m['n_pred_present']
                                   if m['n_pred_present'] else np.nan)
    m['n_gt_rows_any_negative'] = int(neg_gt.any(axis=1).sum())
    both = gt_pos & pred_pos
    for j, c in enumerate(ox_cols):
        ox = oxides[c]
        m[f'n_negative_{ox}'] = int(neg_pred[:, j].sum())
        m[f'min_wt_{ox}'] = float(np.nanmin(wt_pred[pred_pos, c])) if pred_pos.any() else np.nan
        m[f'mae_wt_{ox}'] = (float(np.nanmean(np.abs(wt_pred[both, c] - wt_true[both, c])))
                             if both.any() else np.nan)
    return m


def format_log(header, per_phase, ox_cols_by_phase, oxides, overall):
    lines = list(header) + ['']
    for phase, m in per_phase.items():
        lines += [f'=== {phase} ===',
                  f'  rows evaluated                 {m["n_rows"]}',
                  f'  ground truth present           {m["n_gt_present"]}',
                  f'  predicted present              {m["n_pred_present"]}',
                  f'  present in both                {m["n_both_present"]}',
                  f'  precision                      {m["precision"]:.4f}',
                  f'  recall                         {m["recall"]:.4f}',
                  f'  predicted rows with any negative oxide   {m["n_rows_any_negative"]}'
                  f'  ({100 * m["frac_rows_any_negative"]:.2f}% of predicted-present)',
                  f'  ground-truth rows with any negative oxide {m["n_gt_rows_any_negative"]}'
                  '  (sanity check, should be 0)',
                  f'  {"oxide":<8}{"n_negative":>12}{"min wt%":>12}{"MAE wt%":>12}']
        for c in ox_cols_by_phase[phase]:
            ox = oxides[c]
            lines.append(f'  {ox:<8}{m[f"n_negative_{ox}"]:>12d}{m[f"min_wt_{ox}"]:>12.3f}'
                         f'{m[f"mae_wt_{ox}"]:>12.3f}')
        lines.append('')
    lines.append(f'Rows with a negative oxide in any checked phase: {overall}')
    return '\n'.join(lines) + '\n'


# --------------------------------------------------------------------------------- #
#  Plots
# --------------------------------------------------------------------------------- #
def plot_phase(path, phase, pass_name, mode, wt_true, wt_pred, both, any_neg, ox_cols,
               oxides, limits, plot_rows, maes):
    n_ox = len(ox_cols)
    ncols = min(4, n_ox)
    nrows = int(np.ceil(n_ox / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.5 * nrows + 0.7),
                             squeeze=False, facecolor=SURFACE, layout='constrained')
    rows = plot_rows[both[plot_rows]]
    legal = rows[~any_neg[rows]]
    neg = rows[any_neg[rows]]
    n_both = int(both.sum())
    n_neg_both = int((both & any_neg).sum())
    for k, c in enumerate(ox_cols):
        ax = axes[k // ncols, k % ncols]
        ax.set_facecolor(SURFACE)
        lo, hi = limits[c]
        ax.plot([lo, hi], [lo, hi], color=INK_2, lw=1, ls='--', zorder=1)
        ax.axhline(0, color=GRID, lw=1, zorder=0)
        ax.scatter(wt_true[legal, c], wt_pred[legal, c], s=6, c=C_LEGAL, alpha=0.1,
                   linewidths=0, zorder=2, rasterized=True)
        ax.scatter(wt_true[neg, c], wt_pred[neg, c], s=9, c=C_NEG, alpha=0.35,
                   linewidths=0, zorder=3, rasterized=True)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(oxides[c], color=INK, fontsize=11, loc='left')
        ax.text(0.97, 0.04, f'MAE {maes[c]:.2f}', transform=ax.transAxes, ha='right',
                va='bottom', fontsize=8, color=INK_2)
        ax.tick_params(colors=INK_2, labelsize=8)
        for s in ax.spines.values():
            s.set_color(GRID)
        if k % ncols == 0:
            ax.set_ylabel('Predicted (wt%)', color=INK_2, fontsize=9)
        if k // ncols == nrows - 1 or k + ncols >= n_ox:
            ax.set_xlabel('True (wt%)', color=INK_2, fontsize=9)
    for k in range(n_ox, nrows * ncols):
        axes[k // ncols, k % ncols].axis('off')
    handles = [
        plt.Line2D([], [], marker='o', ls='', color=C_LEGAL, ms=5,
                   label=f'no negative oxide (n={n_both - n_neg_both})'),
        plt.Line2D([], [], marker='o', ls='', color=C_NEG, ms=5,
                   label=f'≥1 negative oxide (n={n_neg_both})'),
    ]
    fig.legend(handles=handles, loc='outside upper right', frameon=False, fontsize=9,
               labelcolor=INK, ncol=2)
    shown = '' if len(rows) == n_both else f'; {len(rows)} of {n_both} rows plotted'
    fig.suptitle(f'{phase}: {pass_name}, mass balance "{mode}" (present in both{shown})',
                 x=0.01, ha='left', color=INK, fontsize=12)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _ox_tex(ox):
    """'Al2O3' -> 'Al$_2$O$_3$' for axis labels."""
    import re
    return re.sub(r'(\d+)', r'$_{\1}$', ox)


def plot_pub_figure(path_base, phase, ox_sel, oxides, wt_true, wt_pred_by_pass,
                    neg_by_pass, rows, limits, maes_by_pass, neg_counts_by_pass, neg_before):
    """2 x k grid: rows = uncorrected / corrected, columns = oxides. Points are coloured
    by whether the row had ANY negative oxide in this phase BEFORE correction, in both
    rows, so the bottom row shows where the correction moved those compositions."""
    rc = {'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 9, 'xtick.labelsize': 7,
          'ytick.labelsize': 7, 'legend.fontsize': 7.5, 'axes.linewidth': 0.6,
          'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'pdf.fonttype': 42,
          'font.family': 'sans-serif'}
    k = len(ox_sel)
    letters = 'abcdefghijklmnop'
    with plt.rc_context(rc):
        fig, axes = plt.subplots(2, k, figsize=(2.35 * k, 4.9), squeeze=False,
                                 layout='constrained', facecolor='white')
        legal = rows[~neg_before[rows]]
        bad = rows[neg_before[rows]]
        for r, pname in enumerate(PASSES):
            wt_pred = wt_pred_by_pass[pname]
            for j, c in enumerate(ox_sel):
                ax = axes[r, j]
                lo, hi = limits[c]
                ax.axhline(0, color=GRID, lw=0.6, zorder=0)
                ax.plot([lo, hi], [lo, hi], color=INK_2, lw=0.7, ls='--', zorder=1)
                ax.scatter(wt_true[legal, c], wt_pred[legal, c], s=3, c=C_LEGAL, alpha=0.25,
                           linewidths=0, zorder=2, rasterized=True)
                ax.scatter(wt_true[bad, c], wt_pred[bad, c], s=6, c=C_NEG, alpha=0.8,
                           linewidths=0, zorder=3, rasterized=True)
                ax.set_xlim(lo, hi)
                ax.set_ylim(lo, hi)
                ax.set_aspect('equal', adjustable='box')
                ax.text(0.04, 0.96, f'({letters[r * k + j]})', transform=ax.transAxes,
                        ha='left', va='top', fontsize=8.5, fontweight='bold', color=INK)
                ax.text(0.96, 0.04,
                        f'MAE {maes_by_pass[pname][c]:.2f} wt%\n'
                        f'negative: {neg_counts_by_pass[pname][c]}',
                        transform=ax.transAxes, ha='right', va='bottom', fontsize=7,
                        color=INK_2)
                for sp_ in ax.spines.values():
                    sp_.set_color('#9a9993')
                ax.tick_params(colors=INK_2)
                if r == 0:
                    ax.set_title(_ox_tex(oxides[c]), color=INK)
                if r == 1:
                    ax.set_xlabel(f'True {_ox_tex(oxides[c])} (wt%)', color=INK)
                if j == 0:
                    ax.set_ylabel(f'{pname.capitalize()}\npredicted (wt%)', color=INK)
        handles = [
            plt.Line2D([], [], marker='o', ls='', color=C_LEGAL, ms=4,
                       label=f'No negative oxide before correction (n={len(legal)})'),
            plt.Line2D([], [], marker='o', ls='', color=C_NEG, ms=4,
                       label=f'Negative oxide before correction (n={len(bad)})'),
        ]
        fig.legend(handles=handles, loc='outside lower center', ncol=2, frameon=False)
        for ext, dpi in (('pdf', 300), ('png', 300)):
            fig.savefig(f'{path_base}.{ext}', dpi=dpi, facecolor='white')
        plt.close(fig)


# --------------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------------- #
def main():
    t0 = time.time()
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    emu = EMU
    model = emu.model
    mass_balance = MASS_BALANCE
    if mass_balance not in ('iterative', 'none'):
        raise SystemExit(f"MASS_BALANCE must be 'iterative' or 'none', got {mass_balance!r}")

    mi = emu.ml_indexer
    if 'melts-liquid' not in mi.label_indices:
        raise SystemExit('This check is for MELTS emulators (the polish only exists there).')
    if not any(getattr(emu, f, False) for f in ('_polish_px', '_polish_cpx', '_polish_sp')):
        print('[warn] emulator has no orthopyroxene, clinopyroxene or spinel: the corrected '
              'pass will equal the uncorrected one')

    bundle = load_ml_bundle(BUNDLE_PATH, arrays=['features', 'binary_labels', 'labels'])
    bi = bundle.ml_indexer
    if list(bi.featureNames) != list(mi.featureNames):
        raise SystemExit(f'Feature mismatch: bundle {list(bi.featureNames)} vs model '
                         f'{list(mi.featureNames)}')
    if list(bi.all_phases) != list(mi.all_phases):
        raise SystemExit('Phase order differs between bundle and model:\n'
                         f'  bundle {list(bi.all_phases)}\n  model  {list(mi.all_phases)}')

    n_total = bundle.features.shape[0]
    rng = np.random.default_rng(SEED)
    if MAX_SAMPLES is not None and MAX_SAMPLES < n_total:
        subset = np.sort(rng.choice(n_total, size=MAX_SAMPLES, replace=False))
    else:
        subset = np.arange(n_total)
    features = np.asarray(bundle.features[subset], dtype=np.float32)
    binaries = np.asarray(bundle.binary_labels[subset]) > 0.5
    comp_sub = mi.compositional_component_subset
    px_sp_sub = np.asarray(mi.PxSpTransform)[np.ix_(comp_sub, comp_sub)]
    labels_t = np.asarray(bundle.labels[subset], dtype=np.float64) @ px_sp_sub
    n = len(subset)
    print(f'[data] {n} of {n_total} rows; model {MODEL_NAME}; bundle {BUNDLE_PATH.name}')

    comp_to_ox = np.asarray(mi.compToOx, dtype=np.float64)          # transformed basis
    mw = np.diag(np.asarray(mi.MM, dtype=np.float64))
    oxides = list(mi.Oxides)

    phases = [p for p in PHASES if p in mi.label_indices_comp]
    missing = sorted(set(PHASES) - set(phases))
    if missing:
        print(f'[warn] not compositionally variable phases of this model, skipped: {missing}')

    # ground truth per phase (pass-independent)
    gt = {}
    ox_cols = {}
    for ph in phases:
        idx = np.asarray(mi.label_indices[ph])
        ox_cols[ph] = np.flatnonzero(np.any(comp_to_ox[idx] != 0, axis=0))
        frac = labels_t[:, np.asarray(mi.label_indices_comp[ph])]
        gt[ph] = (binaries[:, mi.mass_phasedict[ph]], oxide_wt(frac, comp_to_ox[idx], mw))

    # forward passes
    results = {}
    for pname in PASSES:
        t = time.time()
        cm, pp = run_pass(emu, features, pname == 'corrected', mass_balance, BATCH_SIZE)
        results[pname] = {}
        for ph in phases:
            idx = np.asarray(mi.label_indices[ph])
            pred_pos = pp[:, mi.mass_phasedict[ph]] > 0.5
            wt = oxide_wt(predicted_phase_fractions(cm, idx), comp_to_ox[idx], mw)
            results[pname][ph] = (pred_pos, wt)
        print(f'[{pname}] forward + mass balance: {time.time() - t:.1f} s')

    # shared plot rows and axis limits, so both passes are drawn on identical axes
    plot_rows = rng.permutation(n)
    limits = {}
    for ph in phases:
        gt_pos, wt_true = gt[ph]
        for c in ox_cols[ph]:
            vals = [wt_true[gt_pos, c]]
            for pname in PASSES:
                pred_pos, wt = results[pname][ph]
                vals.append(wt[gt_pos & pred_pos, c])
            v = np.concatenate(vals)
            v = v[np.isfinite(v)]
            lo = min(0.0, float(v.min())) if v.size else 0.0
            hi = float(v.max()) if v.size else 1.0
            pad = 0.03 * (hi - lo if hi > lo else 1.0)
            limits.setdefault(ph, {})[c] = (lo - pad, hi + pad)

    summary_rows = []
    summary_lines = []
    for pname in PASSES:
        pdir = out_dir / pname
        pdir.mkdir(exist_ok=True)
        per_phase = {}
        any_neg_overall = np.zeros(n, dtype=bool)
        for ph in phases:
            gt_pos, wt_true = gt[ph]
            pred_pos, wt = results[pname][ph]
            m = phase_metrics(gt_pos, pred_pos, wt_true, wt, ox_cols[ph], oxides, NEG_TOL)
            per_phase[ph] = m
            any_neg = pred_pos & (wt[:, ox_cols[ph]] < -NEG_TOL).any(axis=1)
            any_neg_overall |= any_neg
            both = gt_pos & pred_pos
            maes = {c: m[f'mae_wt_{oxides[c]}'] for c in ox_cols[ph]}
            sel = plot_rows[both[plot_rows]][:MAX_PLOT_POINTS]
            sel_mask = np.zeros(n, dtype=bool)
            sel_mask[sel] = True
            plot_phase(pdir / f'{ph}_1to1.png', ph, pname, mass_balance, wt_true, wt,
                       both, any_neg, ox_cols[ph], oxides, limits[ph],
                       plot_rows[sel_mask[plot_rows]], maes)
            for k, v in m.items():
                summary_rows.append({'pass': pname, 'phase': ph, 'metric': k, 'value': v})
        overall = int(any_neg_overall.sum())
        summary_rows.append({'pass': pname, 'phase': 'ALL', 'metric': 'n_rows_any_negative',
                             'value': overall})
        header = [f'pass            {pname}',
                  f'model           {API._iso_path}',
                  f'model class     {type(model).__name__}',
                  f'bundle          {BUNDLE_PATH}',
                  f'rows            {n} of {n_total}',
                  f'mass balance    {mass_balance}',
                  f'negative oxide  wt% < -{NEG_TOL:g}']
        (pdir / 'log.txt').write_text(format_log(header, per_phase, ox_cols, oxides, overall),
                                      encoding='utf-8')
        summary_lines.append((pname, per_phase, overall))

    if FIG_PHASE:
        if FIG_PHASE not in phases:
            print(f'[warn] FIG_PHASE {FIG_PHASE!r} not evaluated; figure skipped')
        else:
            ph = FIG_PHASE
            missing_ox = [o for o in FIG_OXIDES if o not in oxides]
            if missing_ox:
                raise SystemExit(f'unknown oxides {missing_ox}; choose from {oxides}')
            ox_sel = [oxides.index(o) for o in FIG_OXIDES]
            gt_pos, wt_true = gt[ph]
            both_all = gt_pos.copy()
            wt_pred_by_pass, neg_by_pass, maes_by_pass, negc_by_pass = {}, {}, {}, {}
            for pname in PASSES:
                pred_pos, wt = results[pname][ph]
                both_all &= pred_pos
                wt_pred_by_pass[pname] = wt
                neg = pred_pos[:, None] & (wt < -NEG_TOL)
                neg_by_pass[pname] = neg
                negc_by_pass[pname] = {c: int(neg[:, c].sum()) for c in ox_sel}
                m = summary_lines[PASSES.index(pname)][1][ph]
                maes_by_pass[pname] = {c: m[f'mae_wt_{oxides[c]}'] for c in ox_sel}
            neg_before = neg_by_pass['uncorrected'][:, ox_cols[ph]].any(axis=1)
            rows = plot_rows[both_all[plot_rows]][:MAX_PLOT_POINTS]
            base = out_dir / f'{ph}_figure'
            plot_pub_figure(base, ph, ox_sel, oxides, wt_true, wt_pred_by_pass, neg_by_pass,
                            rows, limits[ph], maes_by_pass, negc_by_pass, neg_before)
            print(f'[figure] {base}.pdf / .png')

    df = pd.DataFrame(summary_rows)
    df.to_csv(out_dir / 'summary.csv', index=False)

    # side-by-side comparison of the headline numbers
    lines = [f'model {MODEL_NAME} | bundle {BUNDLE_PATH.name} | {n} rows | '
             f'mass balance {mass_balance} | negative: wt% < -{NEG_TOL:g}', '']
    keys = ['n_gt_present', 'n_pred_present', 'precision', 'recall', 'n_rows_any_negative']
    for ph in phases:
        lines.append(f'=== {ph} ===')
        lines.append(f'  {"metric":<24}{"uncorrected":>14}{"corrected":>14}')
        u, c = summary_lines[0][1][ph], summary_lines[1][1][ph]
        for k in keys + [f'n_negative_{oxides[j]}' for j in ox_cols[ph]]:
            fmt = (lambda v: f'{v:>14.4f}') if isinstance(u[k], float) else (lambda v: f'{v:>14d}')
            lines.append(f'  {k:<24}{fmt(u[k])}{fmt(c[k])}')
        lines.append('')
    lines.append(f'rows with a negative oxide in any checked phase: '
                 f'uncorrected {summary_lines[0][2]}, corrected {summary_lines[1][2]}')
    text = '\n'.join(lines) + '\n'
    (out_dir / 'summary.txt').write_text(text, encoding='utf-8')
    print(text)
    print(f'[done] {time.time() - t0:.1f} s -> {out_dir}')


if __name__ == '__main__':
    main()
