#!/usr/bin/env python
"""Grid deployment test: every model against every validation dataset.

Runs N = M x D deployment-quality tests for M emulator checkpoints (``*.tar``)
against D ML-ready validation bundles (``*.tar.gz``), writes one stats table per
pair in the same ``metrics x phases`` format as ``EmulatorAPI.test()``
(``emulator_quality_<name>.csv``), and then builds ranking tables that make the
M models directly comparable on each dataset:

    rankings/<dataset>/order.csv              one M-digit integer per cell, the
                                              model numbers ordered best -> worst
                                              (e.g. 2431: model 2 best, model 1 worst)
    rankings/<dataset>/placement_<model>.csv  that model's placement (1 = best)
                                              in every cell

plus a colour-coded table plot of each (``plots/``).  That is D order tables and
N placement tables, all with identical rows and columns per dataset.

Geometry mismatches (the reason this script exists)
---------------------------------------------------
A model and a dataset need not share the same phase set, element set or
per-phase component set.  Everything is matched by NAME (phases, elements,
oxides, condition features), never by column index, and each pair is scored only
on the rows both sides can represent:

* dataset-only phase   - a phase in the dataset the model cannot predict. Rows in
                         which it is present (ground truth) are excluded.
* unsupported element  - an element in the dataset bulk the model has no input
                         for. Rows with a non-zero amount (> --element-tol) are
                         excluded. A model element missing from the dataset is
                         fed as 0.
* model-only phase     - a phase the model can predict but the dataset never
                         contains. Validation assemblages are complete, so any
                         row where the model predicts it present is WRONG, but
                         precision/recall cannot be computed for it. Those rows
                         are excluded from the bulk stats (``--keep-spurious-rows``
                         keeps them) and their proportion is reported per phase.
* condition features   - every condition feature the model needs (P, T, S, fO2..)
                         must exist in the dataset, else the pair is skipped.

All of this is reported loudly (stdout and ``grid_log.txt``) with the phases
involved and the fraction of the dataset each rule removed; ``pair_summary.csv``
holds the same numbers in tabular form.

Row policy: by default each dataset is scored on its COMMON rows - the
intersection of the rows every (compatible) model keeps - so every model on a
dataset is ranked on exactly the same rows.  ``--per-pair-rows`` scores each pair
on its own kept rows instead.

Ground truth is reconstructed with the DATASET's own ml_indexer (never the
model's), so it is independent of which model is being scored.  For a matched
pair with --per-pair-rows the per-pair table reproduces
``ngibbs.deployment_tests.evaluate_emulator_quality`` exactly.

Example
-------
    python scripts/grid_deployment_test.py \\
        --models  src/ngibbs/engine/TrainedModels/102/102Closed_NoCr_NPT.tar \\
                  src/ngibbs/engine/TrainedModels/102/102Closed_Cr_NPT.tar \\
                  runs/new_candidate_NPT.tar \\
        --datasets src/ngibbs/engine/TrainedModels/102/102Closed_NoCr_NPT_Test_subset15000.tar.gz \\
                   src/ngibbs/engine/TrainedModels/102/102Closed_Cr_NPT_Test_subset15000.tar.gz \\
        --out grid_test_out/cr_swap

Only ``ngibbs`` is imported (no ``builder``), so it runs anywhere the shipped
package does.  matplotlib is needed only for the plots (``--no-plots`` skips).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import gc
import json
import re
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / 'src') not in sys.path:
    sys.path.insert(0, str(_REPO / 'src'))

from ngibbs.engine.NN import rebuild_MELTS_model  # noqa: E402
from ngibbs.engine.emulator import NN_MELTS  # noqa: E402
from ngibbs.utils.file_utils import load_ml_bundle  # noqa: E402
from ngibbs.utils.math_utils import Normalizer  # noqa: E402
from ngibbs.deployment_tests.emulator_quality import (  # noqa: E402
    NA_REP, _FIXED_METRICS, _compute_active_oxides, _safe_mean_abs_rel, _staged_forward, _to_np,
    legend_text,
)

# --------------------------------------------------------------------------- #
# metric ranking directions
# --------------------------------------------------------------------------- #
# 'high'  : larger is better            'low' : smaller is better
# 'match' : |proportion_predicted - proportion_in_dataset|, smaller is better
# None    : descriptive (a property of the ground truth / row count) - not ranked
_DIRECTION_FIXED = {
    'proportion_in_dataset': None,
    'proportion_predicted': 'match',
    'n_rows_present': None,
    'precision': 'high',
    'recall': 'high',
    'abundance_gt_mean_wtpct': None,
    'abundance_mae_wtpct': 'low',
    'abundance_relerr_pct': 'low',
    'n_samples': None,
    'recon_residual_l2_mean': 'low',
    'recon_residual_l2_p95': 'low',
}


def metric_direction(metric: str) -> Optional[str]:
    if metric in _DIRECTION_FIXED:
        return _DIRECTION_FIXED[metric]
    if metric.endswith('_comp_mae_wtpct') or metric.endswith('_comp_relerr_pct'):
        return 'low'
    return None


# --------------------------------------------------------------------------- #
# logging: everything printed is also kept for grid_log.txt
# --------------------------------------------------------------------------- #
class GridLog:
    def __init__(self):
        self.lines: List[str] = []

    def __call__(self, msg: str = ''):
        print(msg, flush=True)
        self.lines.extend(str(msg).split('\n'))

    def loud(self, title: str, body: Sequence[str]):
        bar = '!' * 88
        self(bar)
        self(f'!!  WARNING: {title}')
        for line in body:
            self(f'!!    {line}')
        self(bar)

    def write(self, path: Path):
        path.write_text('\n'.join(self.lines) + '\n', encoding='utf-8')


# --------------------------------------------------------------------------- #
# containers
# --------------------------------------------------------------------------- #
@dataclass
class Dataset:
    number: int
    label: str
    path: Path
    indexer: object
    feature_cols: List[str]            # condition featureNames + Elkeys
    features: np.ndarray               # (n, F) raw, subset rows
    binaries: np.ndarray               # (n, P_d) bool
    subset: np.ndarray                 # row ids into the bundle
    n_total: int
    gt_mass: np.ndarray                # (n, P_d) wt% of system
    gt_ox: Dict[str, np.ndarray]       # phase -> (n, O_d) wt% within phase
    gt_active_ox: Dict[str, List[str]]
    phases: List[str]
    phase_col: Dict[str, int]
    abundance_order: List[str]         # phases, most -> least abundant (all subset rows)


@dataclass
class Model:
    number: int
    label: str
    path: Path
    emulator: NN_MELTS


@dataclass
class Pair:
    model: Model
    dataset: Dataset
    compatible: bool = True
    reason: str = ''
    shared_phases: List[str] = field(default_factory=list)
    dataset_only: List[str] = field(default_factory=list)
    model_only: List[str] = field(default_factory=list)
    unsupported_elements: List[str] = field(default_factory=list)
    missing_model_elements: List[str] = field(default_factory=list)
    component_mismatch: Dict[str, tuple] = field(default_factory=dict)
    excl_element: Optional[np.ndarray] = None
    excl_dsonly: Optional[np.ndarray] = None
    excl_spurious: Optional[np.ndarray] = None
    spurious_by_phase: Dict[str, float] = field(default_factory=dict)
    dsonly_by_phase: Dict[str, float] = field(default_factory=dict)
    pred_present: Optional[np.ndarray] = None   # (n, P_m) bool, False on unrun rows
    pred_mass: Optional[np.ndarray] = None      # (n, P_m)
    pred_comp: Optional[np.ndarray] = None      # (n, Pc_m, O_m)
    recon: Optional[np.ndarray] = None          # (n,)
    range_warnings: List[str] = field(default_factory=list)
    keep: Optional[np.ndarray] = None           # final scored rows
    table: Optional[pd.DataFrame] = None
    spurious_kept: bool = False

    @property
    def structural_keep(self):
        return ~(self.excl_element | self.excl_dsonly)

    @property
    def pair_keep(self):
        return self.structural_keep & ~self.excl_spurious


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def _slug(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9._+-]+', '_', s).strip('_')


def _default_labels(paths: Sequence[Path], strip: Sequence[str] = ()) -> List[str]:
    def stem(p):
        s = p.name
        for suf in ('.tar.gz', '.tar', '.pt'):
            if s.endswith(suf):
                s = s[:-len(suf)]
                break
        for tok in strip:
            s = s.replace(tok, '')
        return s
    labels = [stem(p) for p in paths]
    if len(set(labels)) < len(labels):           # disambiguate with the parent dir
        labels = [f'{p.parent.name}-{l}' for p, l in zip(paths, labels)]
    if len(set(labels)) < len(labels):
        labels = [f'{l}-{i + 1}' for i, l in enumerate(labels)]
    return [_slug(l) for l in labels]


def _expand(paths: Sequence[str], pattern: str) -> List[Path]:
    out = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            out += sorted(p.glob(pattern))
        else:
            out.append(p)
    missing = [str(p) for p in out if not p.exists()]
    if missing:
        raise FileNotFoundError(f'not found: {missing}')
    return out


def _gt_engine(indexer) -> NN_MELTS:
    """An NN_MELTS shell carrying only the DATASET's indexer matrices, used to
    rebuild ground-truth phase masses / compositions in the dataset's own
    geometry. Features are passed un-normalised (identity normaliser), which is
    exactly what the iron speciator ends up seeing after its own denorm."""
    eng = NN_MELTS.__new__(NN_MELTS)
    eng.ml_indexer = indexer
    eng.dev = 'cpu'
    eng._setup_indexer_tensors()
    n_f = len(indexer.featureNames) + len(indexer.Elkeys)
    eng.norm_features = Normalizer(torch.zeros(n_f), torch.ones(n_f))
    return eng


def load_dataset(number, label, path, *, max_samples, seed, batch_size, log) -> Dataset:
    bundle = load_ml_bundle(path)
    idx = bundle.ml_indexer
    n_total = bundle.features.shape[0]
    if max_samples is not None and max_samples < n_total:
        rng = np.random.default_rng(seed)
        subset = np.sort(rng.choice(n_total, size=max_samples, replace=False))
    else:
        subset = np.arange(n_total)

    eng = _gt_engine(idx)
    comp_subset = idx.compositional_component_subset
    px_sp_sub = idx.PxSpTransform[np.ix_(comp_subset, comp_subset)]
    labels_trans = bundle.labels @ px_sp_sub

    labels_t = torch.tensor(labels_trans[subset], dtype=torch.float32)
    moles_t = torch.tensor(bundle.molar_labels[subset], dtype=torch.float32)
    feats_t = torch.tensor(bundle.features[subset], dtype=torch.float32)
    n_vc, n_ph = labels_t.shape[1], moles_t.shape[1]

    def gt_batch_fn(chunk):
        comps = eng.getExtensiveComps(intensiveLabels=chunk[:, :n_vc],
                                      molarLabels=chunk[:, n_vc:n_vc + n_ph])
        return eng.make_phase_tables(comps, eng.compToOx, eng.MM, eng.phaseToCompMap.T,
                                     chunk[:, n_vc + n_ph:], out=None)

    gt_mass = _to_np(_staged_forward(gt_batch_fn, torch.cat([labels_t, moles_t, feats_t], 1),
                                     batch_size))

    oxides = list(idx.Oxides)
    gt_ox, gt_active = {}, {}
    for phase in idx.label_indices_comp:
        indices = idx.label_indices[phase]
        comp_indices = idx.label_indices_comp[phase]
        ox_gt = labels_trans[np.ix_(subset, comp_indices)] @ idx.compToOx[indices]
        if phase == 'melts-liquid' and 'Fe3' not in idx.Elkeys:
            ox_gt = eng.Iron_Speciator(torch.tensor(ox_gt, dtype=torch.float32), feats_t).numpy()
        ox_gt = ox_gt @ idx.MM
        gt_ox[phase] = ox_gt * (100.0 / (1e-6 + ox_gt.sum(axis=1, keepdims=True)))
        gt_active[phase] = [oxides[i] for i in _compute_active_oxides(idx.compToOx, indices)]

    binaries = bundle.binary_labels[subset] > 0.5
    phases = list(idx.label_indices.keys())
    pcol = {p: idx.mass_phasedict[p] for p in phases}
    abundance = {p: float(np.mean(np.where(binaries[:, pcol[p]], gt_mass[:, pcol[p]], 0.0)))
                 for p in phases}
    order = sorted(phases, key=lambda p: abundance[p], reverse=True)

    ds = Dataset(number=number, label=label, path=Path(path), indexer=idx,
                 feature_cols=list(idx.featureNames) + list(idx.Elkeys),
                 features=bundle.features[subset].astype(np.float32), binaries=binaries,
                 subset=subset, n_total=n_total, gt_mass=gt_mass, gt_ox=gt_ox,
                 gt_active_ox=gt_active, phases=phases, phase_col=pcol, abundance_order=order)
    log(f'  [D{number}] {label}: {len(subset):,}/{n_total:,} rows, {len(phases)} phases, '
        f'elements {list(idx.Elkeys)}, conditions {list(idx.featureNames)}')
    del bundle
    gc.collect()
    return ds


def load_model(number, label, path, *, device, mass_balance, log) -> Model:
    emu = NN_MELTS(rebuild_MELTS_model(str(path)), cuda=(device == 'cuda'),
                   mass_balance=mass_balance)
    mi = emu.ml_indexer
    log(f'  [M{number}] {label}: {type(emu.model).__name__}, {len(mi.label_indices)} phases, '
        f'elements {list(mi.Elkeys)}, conditions {list(mi.featureNames)}')
    return Model(number=number, label=label, path=Path(path), emulator=emu)


# --------------------------------------------------------------------------- #
# per-pair: reconcile geometry, run inference
# --------------------------------------------------------------------------- #
def reconcile(pair: Pair, *, element_tol: float):
    ds, mi = pair.dataset, pair.model.emulator.ml_indexer
    n = len(ds.subset)
    pair.excl_element = np.zeros(n, bool)
    pair.excl_dsonly = np.zeros(n, bool)
    pair.excl_spurious = np.zeros(n, bool)

    missing_cond = [f for f in mi.featureNames if f not in ds.indexer.featureNames]
    if missing_cond:
        pair.compatible = False
        pair.reason = (f'model condition feature(s) {missing_cond} not in dataset '
                       f'(dataset has {list(ds.indexer.featureNames)})')
        return

    m_phases = list(mi.label_indices.keys())
    pair.shared_phases = [p for p in ds.phases if p in mi.label_indices]
    pair.dataset_only = [p for p in ds.phases if p not in mi.label_indices]
    pair.model_only = [p for p in m_phases if p not in ds.indexer.label_indices]
    if not pair.shared_phases:
        pair.compatible = False
        pair.reason = 'no phases in common'
        return

    m_el, d_el = list(mi.Elkeys), list(ds.indexer.Elkeys)
    pair.unsupported_elements = [e for e in d_el if e not in m_el]
    pair.missing_model_elements = [e for e in m_el if e not in d_el]
    off = len(ds.indexer.featureNames)
    for e in pair.unsupported_elements:
        pair.excl_element |= ds.features[:, off + d_el.index(e)] > element_tol

    for p in pair.dataset_only:
        present = ds.binaries[:, ds.phase_col[p]]
        pair.dsonly_by_phase[p] = float(present.mean())
        pair.excl_dsonly |= present

    for p in pair.shared_phases:
        if p == 'melts-liquid':          # liquid components are the element list, reported above
            continue
        cm = list(mi.components_in_phases.get(p, []))
        cd = list(ds.indexer.components_in_phases.get(p, []))
        if cm != cd:
            pair.component_mismatch[p] = (cm, cd)


def model_features(pair: Pair) -> np.ndarray:
    """Dataset rows re-expressed in the model's feature order (by name)."""
    ds, mi = pair.dataset, pair.model.emulator.ml_indexer
    col = {c: i for i, c in enumerate(ds.feature_cols)}
    n = ds.features.shape[0]
    out = np.zeros((n, len(mi.featureNames) + len(mi.Elkeys)), dtype=np.float32)
    for j, name in enumerate(list(mi.featureNames) + list(mi.Elkeys)):
        if name in col:
            out[:, j] = ds.features[:, col[name]]
    return out


def run_inference(pair: Pair, *, batch_size: int):
    emu = pair.model.emulator
    mi = emu.ml_indexer
    ds = pair.dataset
    n = len(ds.subset)
    rows = np.where(pair.structural_keep)[0]

    n_ph = len(mi.label_indices)
    n_cph = len(mi.comp_phasedict)
    n_ox = len(mi.Oxides)
    pair.pred_present = np.zeros((n, n_ph), bool)
    pair.pred_mass = np.full((n, n_ph), np.nan, np.float32)
    pair.pred_comp = np.full((n, n_cph, n_ox), np.nan, np.float32)
    pair.recon = np.full(n, np.nan, np.float32)
    if len(rows) == 0:
        return

    dev = emu.dev if str(emu.dev) == 'cuda' and torch.cuda.is_available() else 'cpu'
    feats = torch.tensor(model_features(pair)[rows], device=dev, dtype=torch.float32)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        out = _staged_forward(
            emu.forwardMB, feats, batch_size, Normalize=True, optimize_masses=True,
            protect_opx=False,
            outputs=['phase_present', 'phase_tables', 'reconstruction_residual'])
    msgs = []
    for w in caught:
        m = str(w.message).strip()
        if m and m not in msgs:
            msgs.append(m)
    pair.range_warnings = msgs

    comp_t, mass_t = out['phase_tables']
    pair.pred_present[rows] = _to_np(out['phase_present']) > 0.5
    pair.pred_mass[rows] = _to_np(mass_t)
    pair.pred_comp[rows] = _to_np(comp_t)
    pair.recon[rows] = _to_np(out['reconstruction_residual']).reshape(-1)

    for p in pair.model_only:
        hit = pair.pred_present[:, mi.mass_phasedict[p]] & pair.structural_keep
        pair.spurious_by_phase[p] = float(hit.mean())
        pair.excl_spurious |= hit


# --------------------------------------------------------------------------- #
# metrics (mirrors ngibbs.deployment_tests.emulator_quality exactly)
# --------------------------------------------------------------------------- #
def score_pair(pair: Pair) -> pd.DataFrame:
    ds, mi = pair.dataset, pair.model.emulator.ml_indexer
    keep = pair.keep
    m_ox = {o: i for i, o in enumerate(mi.Oxides)}
    d_oxides = list(ds.indexer.Oxides)

    records, abundance = {}, {}
    for phase in pair.shared_phases:
        dcol, mcol = ds.phase_col[phase], mi.mass_phasedict[phase]
        real = ds.binaries[keep, dcol]
        pred = pair.pred_present[keep, mcol]
        tp = int(np.sum(real & pred)); fp = int(np.sum(~real & pred)); fn = int(np.sum(real & ~pred))
        gt_mass_all = ds.gt_mass[keep, dcol]
        abundance[phase] = float(np.mean(np.where(real, gt_mass_all, 0.0))) if keep.any() else 0.0
        if real.any():
            g = gt_mass_all[real]
            pm = pair.pred_mass[keep, mcol][real]
            mae = float(np.mean(np.abs(g - pm)))
            rel = _safe_mean_abs_rel(g - pm, g, threshold=0.01)
        else:
            mae = rel = np.nan
        rec = {
            'proportion_in_dataset': float(real.mean()) if keep.any() else np.nan,
            'proportion_predicted': float(pred.mean()) if keep.any() else np.nan,
            'n_rows_present': int(real.sum()),
            'precision': tp / (tp + fp) if (tp + fp) > 0 else np.nan,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else np.nan,
            'abundance_gt_mean_wtpct': abundance[phase],
            'abundance_mae_wtpct': mae,
            'abundance_relerr_pct': rel * 100 if np.isfinite(rel) else np.nan,
        }
        if phase in ds.gt_ox and phase in mi.comp_phasedict:
            both = real & pred
            if both.any():
                gt_rows = ds.gt_ox[phase][keep][both]
                pr_rows = pair.pred_comp[keep][both][:, mi.comp_phasedict[phase]]
                for ox in ds.gt_active_ox[phase]:
                    if ox not in m_ox:
                        continue
                    g = gt_rows[:, d_oxides.index(ox)]
                    pr = pr_rows[:, m_ox[ox]]
                    rec[f'{ox}_comp_mae_wtpct'] = float(np.nanmean(np.abs(g - pr)))
                    r = _safe_mean_abs_rel(g - pr, g, threshold=0.1)
                    rec[f'{ox}_comp_relerr_pct'] = r * 100 if np.isfinite(r) else np.nan
        records[phase] = rec

    phase_order = sorted(records, key=lambda p: abundance.get(p, 0.0), reverse=True)
    metric_order = list(_FIXED_METRICS)
    ox_metrics = []
    for ox in d_oxides:
        ox_metrics += [f'{ox}_comp_mae_wtpct', f'{ox}_comp_relerr_pct']
    metric_order += [m for m in ox_metrics if any(m in records[p] for p in phase_order)]
    table = pd.DataFrame({p: {m: records[p].get(m, np.nan) for m in metric_order}
                          for p in phase_order}, index=metric_order)
    table.index.name = 'metric'
    rr = pair.recon[keep]
    rr = rr[np.isfinite(rr)]
    table.loc['n_samples'] = float(keep.sum())
    table.loc['recon_residual_l2_mean'] = float(np.mean(rr)) if len(rr) else np.nan
    table.loc['recon_residual_l2_p95'] = float(np.percentile(rr, 95)) if len(rr) else np.nan
    return table


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def _pct(x):
    return f'{100.0 * x:.2f}%'


def report_pair(pair: Pair, log: GridLog):
    tag = f'[M{pair.model.number} x D{pair.dataset.number}] {pair.model.label}  x  {pair.dataset.label}'
    if not pair.compatible:
        log.loud(f'{tag}: PAIR SKIPPED', [pair.reason])
        return
    body = []
    if pair.dataset_only:
        body.append(f'dataset-only phases (model cannot predict): {pair.dataset_only}')
        for p, f in pair.dsonly_by_phase.items():
            body.append(f'    {p:22s} present in {_pct(f)} of rows')
        body.append(f'  -> {_pct(pair.excl_dsonly.mean())} of the dataset ignored '
                    '(rows containing any of these phases)')
    if pair.unsupported_elements:
        body.append(f'dataset elements the model has no input for: {pair.unsupported_elements}')
        body.append(f'  -> {_pct(pair.excl_element.mean())} of the dataset ignored '
                    '(rows with a non-zero amount)')
    if pair.model_only:
        body.append(f'model-only phases (absent from every validation assemblage): {pair.model_only}')
        for p, f in pair.spurious_by_phase.items():
            body.append(f'    {p:22s} predicted present in {_pct(f)} of rows  '
                        '(always incorrect; precision/recall undefined)')
        body.append(f'  -> {_pct(pair.excl_spurious.mean())} of rows predict at least one of these '
                    + ('(KEPT in the bulk stats: --keep-spurious-rows)' if pair.spurious_kept
                       else '(excluded from the bulk stats)'))
    if body:
        excl = ~pair.pair_keep if not pair.spurious_kept else ~pair.structural_keep
        body.append(f'TOTAL for this pair: {_pct(excl.mean())} ignored, '
                    f'{int((~excl).sum()):,} of {len(excl):,} rows scored')
        log.loud(f'{tag}: phase/element geometry mismatch', body)
    else:
        log(f'  {tag}: geometries match ({len(pair.shared_phases)} phases)')
    if pair.missing_model_elements:
        log(f'    note: model element(s) {pair.missing_model_elements} absent from the dataset; fed as 0')
    for p, (cm, cd) in pair.component_mismatch.items():
        log(f'    note: {p} components differ (compared by oxide name): model {cm} | dataset {cd}')
    if pair.range_warnings:
        log(f'    note: {len(pair.range_warnings)} training-range warning(s) from the model:')
        for m in pair.range_warnings:
            for line in m.split('\n')[:12]:
                log(f'      {line}')


def pair_summary_row(pair: Pair) -> dict:
    n = len(pair.dataset.subset)
    row = {'model_no': pair.model.number, 'model': pair.model.label,
           'dataset_no': pair.dataset.number, 'dataset': pair.dataset.label,
           'compatible': pair.compatible, 'skip_reason': pair.reason, 'n_rows': n}
    if pair.compatible:
        row.update({
            'dataset_only_phases': ';'.join(pair.dataset_only),
            'model_only_phases': ';'.join(pair.model_only),
            'unsupported_elements': ';'.join(pair.unsupported_elements),
            'frac_excl_dataset_only_phase': float(pair.excl_dsonly.mean()),
            'frac_excl_unsupported_element': float(pair.excl_element.mean()),
            'frac_rows_predicting_model_only_phase': float(pair.excl_spurious.mean()),
            'frac_scored': float(pair.keep.mean()),
            'n_scored': int(pair.keep.sum()),
            'n_training_range_warnings': len(pair.range_warnings),
        })
        for p, f in pair.spurious_by_phase.items():
            row[f'spurious_{p}'] = f
    return row


# --------------------------------------------------------------------------- #
# rankings
# --------------------------------------------------------------------------- #
def _eligible(pair: 'Pair', metric: str, phase: str) -> bool:
    """Can this model structurally produce a value for (metric, phase)?  A model is
    left out of a cell (not ranked last) when the quantity does not exist in its
    geometry - e.g. Cr2O3 in liquid for a NoCr model, or composition metrics for a
    phase the model treats as fixed-composition."""
    if pair is None or not pair.compatible or pair.table is None:
        return False
    if metric in _GLOBAL_ROWS:
        return True
    if phase not in pair.shared_phases:
        return False
    for suf in ('_comp_mae_wtpct', '_comp_relerr_pct'):
        if metric.endswith(suf):
            ox = metric[:-len(suf)]
            mi = pair.model.emulator.ml_indexer
            return phase in mi.comp_phasedict and ox in list(mi.Oxides)
    return True


_GLOBAL_ROWS = ('n_samples', 'recon_residual_l2_mean', 'recon_residual_l2_p95')


def build_rankings(ds_pairs: Dict[int, 'Pair'], models: List[Model], ds: Dataset):
    """ds_pairs: model number -> Pair on this dataset.
    Returns (order_df, {model number: placement_df}) with identical index/columns,
    for the models actually scored on this dataset.

    Per cell, eligible models (see ``_eligible``) are ordered best -> worst; ties
    keep model-number order in ``order`` and share a placement (1, 1, 3); an
    eligible model with no value (NaN, e.g. it never predicts the phase where the
    ground truth has it) is placed last.  Ineligible models are left out of the
    cell, so a cell can carry fewer than M digits."""
    scored = [mo for mo in models if ds_pairs[mo.number].table is not None]
    tables = {mo.number: ds_pairs[mo.number].table for mo in scored}
    present = list(tables.values())
    phases = [p for p in ds.abundance_order if any(p in t.columns for t in present)]
    metrics = []
    for t in present:
        metrics += [m for m in t.index if m not in metrics]
    fixed = [m for m in list(_FIXED_METRICS) if m in metrics]
    ox = [m for m in metrics if m.endswith('_comp_mae_wtpct') or m.endswith('_comp_relerr_pct')]
    ox_sorted = []
    for o in ds.indexer.Oxides:
        ox_sorted += [m for m in (f'{o}_comp_mae_wtpct', f'{o}_comp_relerr_pct') if m in ox]
    tail = [m for m in metrics if m not in fixed and m not in ox_sorted]
    metrics = fixed + ox_sorted + tail

    aligned = {k: t.reindex(index=metrics, columns=phases) for k, t in tables.items()}
    many = len(scored) > 9
    order = pd.DataFrame(index=metrics, columns=phases, dtype=object if many else 'Int64')
    place = {mo.number: pd.DataFrame(index=metrics, columns=phases, dtype='Int64') for mo in scored}
    order.index.name = 'metric'
    for p in place.values():
        p.index.name = 'metric'

    for metric in metrics:
        direction = metric_direction(metric)
        if direction is None:
            continue
        for phase in phases:
            elig, scores = [], []
            for mo in scored:
                if not _eligible(ds_pairs[mo.number], metric, phase):
                    continue
                t = aligned[mo.number]
                if metric in _GLOBAL_ROWS:       # model-global scalar, broadcast across phases
                    v = tables[mo.number].at[metric, tables[mo.number].columns[0]]
                else:
                    v = t.at[metric, phase]
                if direction == 'match':
                    v = abs(v - t.at['proportion_in_dataset', phase])
                elif direction == 'high':
                    v = -v
                elig.append(mo.number)
                scores.append(float(v) if v is not None else np.nan)
            s = np.array(scores, float)
            finite = np.isfinite(s)
            if not finite.any():
                continue
            idx = sorted(range(len(elig)), key=lambda i: (not finite[i], s[i] if finite[i] else 0.0,
                                                          elig[i]))
            nums = [elig[i] for i in idx]
            order.at[metric, phase] = ('-'.join(str(k) for k in nums) if many
                                       else int(''.join(str(k) for k in nums)))
            for i, k in enumerate(elig):
                place[k].at[metric, phase] = (int(1 + np.sum(s[finite] < s[i])) if finite[i]
                                              else int(1 + finite.sum()))
    return order, place, scored


# --------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------- #
# Categorical slots (validated reference palette, fixed order) - identity of the
# best model in the order tables. Text in every cell carries the identity too.
_CATEGORICAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300',
                '#6250d6', '#e34948']
_EMPTY = '#ececea'


def _placement_colors(M: int):
    import matplotlib
    cmap = matplotlib.colormaps['RdYlGn_r']
    if M == 1:
        return {1: cmap(0.0)}
    return {k: cmap(0.08 + 0.84 * (k - 1) / (M - 1)) for k in range(1, M + 1)}


def plot_table(df: pd.DataFrame, path: Path, title: str, *, kind: str,
               models: List[Model], subtitle: str = ''):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    nr, nc = df.shape
    M = len(models)
    cw, rh = 0.62 if kind == 'placement' else max(0.62, 0.11 * M + 0.25), 0.24
    fig_w = 2.6 + cw * nc
    fig_h = 2.1 + rh * nr
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    rgb = np.zeros((nr, nc, 4))
    place_cols = _placement_colors(M)
    for i in range(nr):
        for j in range(nc):
            v = df.iat[i, j]
            if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v):
                rgb[i, j] = matplotlib.colors.to_rgba(_EMPTY)
                continue
            if kind == 'placement':
                rgb[i, j] = place_cols[int(v)]
                txt = str(int(v))
            else:
                best = int(str(v).split('-')[0]) if M > 9 else int(str(v)[0])
                rgb[i, j] = matplotlib.colors.to_rgba(_CATEGORICAL[(best - 1) % len(_CATEGORICAL)])
                txt = str(v)
            ax.text(j, i, txt, ha='center', va='center', fontsize=6.5, color='#0b0b0b')
    ax.imshow(rgb, aspect='auto', interpolation='nearest')
    # 2px-ish surface gaps between cells
    ax.set_xticks(np.arange(-0.5, nc, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nr, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.5)
    ax.tick_params(which='minor', length=0)
    ax.set_xticks(range(nc))
    ax.set_xticklabels(df.columns, rotation=55, ha='right', fontsize=7)
    ax.set_yticks(range(nr))
    ax.set_yticklabels(df.index, fontsize=7)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title + (f'\n{subtitle}' if subtitle else ''), fontsize=9, loc='left')
    if kind == 'placement':
        handles = [Patch(color=place_cols[k], label=f'{k}' + (' (best)' if k == 1 else
                                                              ' (worst)' if k == M else ''))
                   for k in range(1, M + 1)]
        handles.append(Patch(color=_EMPTY, label='not ranked'))
        ax.legend(handles=handles, title='placement', fontsize=7, title_fontsize=7,
                  loc='upper left', bbox_to_anchor=(1.01, 1.0), frameon=False)
    else:
        handles = [Patch(color=_CATEGORICAL[(mo.number - 1) % len(_CATEGORICAL)],
                         label=f'{mo.number}: {mo.label}') for mo in models]
        handles.append(Patch(color=_EMPTY, label='not ranked'))
        ax.legend(handles=handles, title='cell colour = best model\ndigits = model numbers, best -> worst',
                  fontsize=7, title_fontsize=7, loc='upper left',
                  bbox_to_anchor=(1.01, 1.0), frameon=False, alignment='left')
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def run_grid(model_paths, dataset_paths, *, out_dir: Path, model_labels=None,
             dataset_labels=None, max_samples=None, seed=1337, batch_size=2 ** 14,
             device='cpu', mass_balance='iterative', common_rows=True,
             keep_spurious_rows=False, element_tol=1e-6, plots=True) -> dict:
    log = GridLog()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_paths = [Path(p) for p in model_paths]
    dataset_paths = [Path(p) for p in dataset_paths]
    model_labels = [_slug(l) for l in model_labels] if model_labels else _default_labels(model_paths)
    dataset_labels = ([_slug(l) for l in dataset_labels] if dataset_labels
                      else _default_labels(dataset_paths, strip=('_Test_subset15000',)))
    if len(model_labels) != len(model_paths) or len(dataset_labels) != len(dataset_paths):
        raise ValueError('number of labels must match number of models/datasets')

    M, D = len(model_paths), len(dataset_paths)
    log(f'nGibbs grid deployment test  {_dt.datetime.now().isoformat(timespec="seconds")}')
    log(f'{M} model(s) x {D} dataset(s) = {M * D} tests;  row policy: '
        f'{"COMMON rows per dataset" if common_rows else "per-pair rows"};  '
        f'spurious rows {"kept" if keep_spurious_rows else "excluded"};  '
        f'max_samples={max_samples}  seed={seed}  mass_balance={mass_balance}')
    if M > 9:
        log('note: more than 9 models - order cells are written as dash-separated strings')
    log('Models:')
    models = [load_model(i + 1, l, p, device=device, mass_balance=mass_balance, log=log)
              for i, (l, p) in enumerate(zip(model_labels, model_paths))]
    log('Datasets:')
    datasets = [load_dataset(j + 1, l, p, max_samples=max_samples, seed=seed,
                             batch_size=batch_size, log=log)
                for j, (l, p) in enumerate(zip(dataset_labels, dataset_paths))]

    log('')
    log('=' * 88)
    log('Running pairs')
    log('=' * 88)
    pairs: Dict[tuple, Pair] = {}
    for ds in datasets:
        for mo in models:
            pair = Pair(model=mo, dataset=ds)
            pair.spurious_kept = keep_spurious_rows
            reconcile(pair, element_tol=element_tol)
            if pair.compatible:
                run_inference(pair, batch_size=batch_size)
            pairs[(mo.number, ds.number)] = pair
            report_pair(pair, log)
            gc.collect()

    # ---- decide scored rows ----
    log('')
    for ds in datasets:
        ok = [pairs[(mo.number, ds.number)] for mo in models if pairs[(mo.number, ds.number)].compatible]
        own = {p.model.number: (p.structural_keep if keep_spurious_rows else p.pair_keep) for p in ok}
        if common_rows and ok:
            common = np.logical_and.reduce(list(own.values()))
            for p in ok:
                p.keep = common
            lost = {p.model.number: float((own[p.model.number] & ~common).mean()) for p in ok}
            body = [f'{int(common.sum()):,} of {len(common):,} rows ({_pct(common.mean())}) '
                    f'are scored for every model']
            body += [f'M{k} {models[k - 1].label}: own rows {_pct(own[k].mean())}, '
                     f'further {_pct(v)} dropped by other models\' exclusions' for k, v in lost.items()]
            if common.mean() < 1.0:
                log.loud(f'[D{ds.number}] {ds.label}: common-row intersection', body)
            else:
                log(f'[D{ds.number}] {ds.label}: all rows scored for every model')
            if common.sum() == 0:
                log.loud(f'[D{ds.number}] {ds.label}: NO common rows',
                         ['every model is unscored on this dataset; try --per-pair-rows'])
        else:
            for p in ok:
                p.keep = own[p.model.number]

    # ---- per-pair tables ----
    summary = []
    for ds in datasets:
        sdir = out_dir / 'stats' / f'D{ds.number}_{ds.label}'
        sdir.mkdir(parents=True, exist_ok=True)
        for mo in models:
            pair = pairs[(mo.number, ds.number)]
            if pair.compatible:
                pair.table = score_pair(pair)
                pair.table.to_csv(sdir / f'M{mo.number}_{mo.label}.csv', na_rep=NA_REP)
            summary.append(pair_summary_row(pair))
    (out_dir / 'stats' / 'emulator_quality_legend.txt').write_text(legend_text())
    pd.DataFrame(summary).to_csv(out_dir / 'pair_summary.csv', index=False)

    # ---- rankings + plots ----
    log('')
    log('Rankings')
    for ds in datasets:
        ds_pairs = {mo.number: pairs[(mo.number, ds.number)] for mo in models}
        if all(p.table is None for p in ds_pairs.values()):
            log(f'  [D{ds.number}] {ds.label}: no compatible model - no rankings')
            continue
        order, place, scored = build_rankings(ds_pairs, models, ds)
        skipped = [mo for mo in models if mo not in scored]
        if skipped:
            log(f'  [D{ds.number}] {ds.label}: ranked among {len(scored)} scored model(s); '
                f'not scored (pair skipped): ' + ', '.join(f'M{m.number}' for m in skipped))
        rdir = out_dir / 'rankings' / f'D{ds.number}_{ds.label}'
        rdir.mkdir(parents=True, exist_ok=True)
        order.to_csv(rdir / 'order.csv', na_rep='')
        for mo in scored:
            place[mo.number].to_csv(rdir / f'placement_M{mo.number}_{mo.label}.csv', na_rep='')
        # tally of first places, a quick headline
        wins = {mo.number: int((place[mo.number] == 1).sum().sum()) for mo in scored}
        n_cells = int(order.notna().sum().sum())
        log(f'  [D{ds.number}] {ds.label}: {n_cells} ranked cells; first places (ties shared): '
            + ', '.join(f'M{k}={v}' for k, v in wins.items()))
        if plots:
            pdir = out_dir / 'plots' / f'D{ds.number}_{ds.label}'
            first_ok = next(pairs[(m.number, ds.number)] for m in models
                            if pairs[(m.number, ds.number)].compatible)
            rows_note = (f'{int(first_ok.keep.sum()):,} common rows' if common_rows
                         else 'per-pair rows')
            plot_table(order, pdir / 'order.png', f'D{ds.number} {ds.label} - model order per cell',
                       kind='order', models=scored, subtitle=rows_note)
            for mo in scored:
                plot_table(place[mo.number], pdir / f'placement_M{mo.number}_{mo.label}.png',
                           f'D{ds.number} {ds.label} - placement of M{mo.number} {mo.label}',
                           kind='placement', models=scored,
                           subtitle=f'1 = best of {len(scored)};  {rows_note}')

    manifest = {
        'created': _dt.datetime.now().isoformat(timespec='seconds'),
        'models': [{'number': m.number, 'label': m.label, 'path': str(m.path)} for m in models],
        'datasets': [{'number': d.number, 'label': d.label, 'path': str(d.path),
                      'n_rows': len(d.subset), 'n_total': d.n_total} for d in datasets],
        'settings': {'max_samples': max_samples, 'seed': seed, 'mass_balance': mass_balance,
                     'common_rows': common_rows, 'keep_spurious_rows': keep_spurious_rows,
                     'element_tol': element_tol, 'device': device},
        'metric_directions': {m: d for m, d in _DIRECTION_FIXED.items()},
    }
    (out_dir / 'grid_manifest.json').write_text(json.dumps(manifest, indent=2))
    log('')
    log(f'Outputs written to {out_dir}')
    log.write(out_dir / 'grid_log.txt')
    return {'pairs': pairs, 'models': models, 'datasets': datasets, 'out_dir': out_dir}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models', nargs='+', required=True,
                    help='checkpoint .tar files (a directory expands to its *.tar). '
                         'Numbered 1..M in the order given.')
    ap.add_argument('--datasets', nargs='+', required=True,
                    help='ML-ready .tar.gz bundles (a directory expands to its *Test_subset*.tar.gz)')
    ap.add_argument('--model-labels', nargs='+', help='short names, one per model')
    ap.add_argument('--dataset-labels', nargs='+', help='short names, one per dataset')
    ap.add_argument('--out', default=None, help='output directory (default grid_test_out/<timestamp>)')
    ap.add_argument('--max-samples', type=int, default=None, help='random row cap per dataset')
    ap.add_argument('--seed', type=int, default=1337)
    ap.add_argument('--batch-size', type=int, default=2 ** 14)
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    ap.add_argument('--mass-balance', default='iterative', choices=['iterative', 'pinv', 'none'])
    ap.add_argument('--per-pair-rows', action='store_true',
                    help='score each pair on its own kept rows instead of the per-dataset '
                         'intersection across models')
    ap.add_argument('--keep-spurious-rows', action='store_true',
                    help='keep rows where the model predicts a phase the dataset never contains')
    ap.add_argument('--element-tol', type=float, default=1e-6,
                    help='element mole fraction above which an unsupported dataset element '
                         'excludes the row')
    ap.add_argument('--no-plots', action='store_true')
    a = ap.parse_args(argv)

    out = Path(a.out) if a.out else (_REPO / 'grid_test_out' /
                                     _dt.datetime.now().strftime('%Y%m%d_%H%M%S'))
    run_grid(_expand(a.models, '*.tar'), _expand(a.datasets, '*Test_subset*.tar.gz'),
             out_dir=out, model_labels=a.model_labels, dataset_labels=a.dataset_labels,
             max_samples=a.max_samples, seed=a.seed, batch_size=a.batch_size, device=a.device,
             mass_balance=a.mass_balance, common_rows=not a.per_pair_rows,
             keep_spurious_rows=a.keep_spurious_rows, element_tol=a.element_tol,
             plots=not a.no_plots)


if __name__ == '__main__':
    main()
