"""Base emulator quality metrics — one emulator against one ML-ready bundle.

The single-model analogue of ``builder/training/validation/ModelComparison.py``.
Because only one model is evaluated, every metric ModelComparison spreads across
its ``phase_metrics.csv`` / ``oxide_metrics.csv`` collapses into a single table:

    rows    = metrics / quality stats / errors (see ``METRIC_DESCRIPTIONS``)
    columns = phases, ordered most- to least-abundant in the bundle

NaN cells (e.g. an oxide a phase does not contain, or a phase absent from every
evaluated row) are written as ``--``.  To read the CSV straight back into
numbers::

    pd.read_csv(path, index_col='metric', na_values='--')

Nothing here imports ``builder`` — the forward pass and ground-truth mass
reconstruction run through the already-wrapped ``NN_MELTS`` emulator the API
holds.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch

from ngibbs.utils.file_utils import load_ml_bundle

NA_REP = '--'

# Fixed metric rows (per phase), in table order.  Oxide-composition rows
# (``<ox>_comp_mae_wtpct`` / ``<ox>_comp_relerr_pct``) are appended per bundle
# for whichever oxides are active.  The three ``__`` rows at the end are
# model-global scalars broadcast across every phase column so the file stays a
# single table.
METRIC_DESCRIPTIONS = {
    'proportion_in_dataset':
        'fraction of evaluated rows in which the phase is present (ground truth)',
    'proportion_predicted':
        'fraction of evaluated rows in which the model predicts the phase present',
    'n_rows_present':
        'count of evaluated rows in which the phase is present (ground truth)',
    'precision':
        'phase-presence precision  TP / (TP + FP)',
    'recall':
        'phase-presence recall  TP / (TP + FN)',
    'abundance_gt_mean_wtpct':
        'mean ground-truth phase abundance, wt% of the whole system '
        '(rows where the phase is absent count as 0)',
    'abundance_mae_wtpct':
        'mean |predicted - ground-truth| phase abundance, wt% of system, '
        'over rows where the phase is present in the ground truth',
    'abundance_relerr_pct':
        'mean |predicted - ground-truth| / ground-truth of phase abundance, in percent, '
        'over ground-truth-present rows with ground-truth abundance > 0.01 wt%',
    '<ox>_comp_mae_wtpct':
        'mean |predicted - ground-truth| of that oxide, wt% WITHIN the phase, '
        'over rows where the phase is present in both ground truth and prediction',
    '<ox>_comp_relerr_pct':
        'mean |predicted - ground-truth| / ground-truth of that within-phase oxide wt%, '
        'in percent, over both-present rows with ground-truth wt% > 0.1',
    'n_samples':
        'number of bundle rows evaluated',
    'recon_residual_l2_mean':
        'mean L2 residual of the mass-balanced bulk reconstruction (normalized units)',
    'recon_residual_l2_p95':
        '95th percentile of the mass-balanced bulk reconstruction L2 residual',
}

_FIXED_METRICS = [
    'proportion_in_dataset', 'proportion_predicted', 'n_rows_present',
    'precision', 'recall',
    'abundance_gt_mean_wtpct', 'abundance_mae_wtpct', 'abundance_relerr_pct',
]
_GLOBAL_METRICS = ['n_samples', 'recon_residual_l2_mean', 'recon_residual_l2_p95']


def legend_text() -> str:
    """Human-readable description of every metric row, for a sidecar file."""
    lines = ['Emulator quality metrics — one row per metric, one column per phase.',
             f'NaN is written as {NA_REP!r}.  Phase columns are ordered most- to '
             'least-abundant in the bundle.', '']
    for name, desc in METRIC_DESCRIPTIONS.items():
        lines.append(f'{name:24s} {desc}')
    return '\n'.join(lines) + '\n'


# --------------------------------------------------------------------------- #
# small helpers (ported from ModelComparison)
# --------------------------------------------------------------------------- #
def _compute_active_oxides(comp_to_ox: np.ndarray, phase_indices: np.ndarray) -> np.ndarray:
    return np.where(np.any(comp_to_ox[phase_indices] != 0, axis=0))[0]


def _safe_mean_abs_rel(residuals: np.ndarray, reference: np.ndarray, threshold: float = 0.1) -> float:
    """Mean |residual / reference|, ignoring reference values below threshold."""
    mask = reference > threshold
    if not np.any(mask):
        return float('nan')
    with np.errstate(invalid='ignore', divide='ignore'):
        return float(np.nanmean(np.abs(residuals[mask] / reference[mask])))


def _staged_forward(func, input_tensor: torch.Tensor, batch_size: int, **kwargs):
    """Batch-execute func(input_tensor[start:end], **kwargs), merging results."""
    def _merge(a, b):
        if isinstance(a, dict) and isinstance(b, dict):
            return {k: _merge(a[k], b[k]) for k in a}
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            return type(a)([_merge(a[i], b[i]) for i in range(len(a))])
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            return torch.cat([a.detach().cpu(), b.detach().cpu()], dim=0)
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            return np.concatenate([a, b], axis=0)
        return b

    n = input_tensor.size(0)
    if n <= batch_size:
        return func(input_tensor, **kwargs)

    n_batches = (n + batch_size - 1) // batch_size
    with torch.no_grad():
        out = func(input_tensor[:batch_size], **kwargs)
        for i in range(1, n_batches):
            start = i * batch_size
            end = min(start + batch_size, n)
            out = _merge(out, func(input_tensor[start:end], **kwargs))
            gc.collect()
    return out


def _to_np(t):
    return t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)


# --------------------------------------------------------------------------- #
# core evaluation
# --------------------------------------------------------------------------- #
_EMULATOR_ATTR = {'isothermal': 'isothermal_emulator',
                  'isentropic': 'isentropic_emulator',
                  'openox': 'open_emulator'}


def _select_emulator(api, bundle, emulator_name: Optional[str] = None):
    """Return (name, NN_MELTS). Honour ``emulator_name`` if given, else pick the
    wrapped emulator whose training feature space matches the bundle."""
    if emulator_name is not None:
        emu = getattr(api, _EMULATOR_ATTR.get(emulator_name, f'{emulator_name}_emulator'), None)
        if emu is None:
            raise RuntimeError(f"API has no '{emulator_name}' emulator to run the quality test.")
        want = list(getattr(bundle.ml_indexer, 'featureNames', []))
        have = list(emu.ml_indexer.featureNames)
        if want and have and want != have:
            print(f"[quality] warning: bundle feature space {want} != "
                  f"'{emulator_name}' emulator feature space {have}")
        return emulator_name, emu

    want = list(getattr(bundle.ml_indexer, 'featureNames', []))
    candidates = [(n, getattr(api, a, None)) for n, a in _EMULATOR_ATTR.items()]
    for name, emu in candidates:
        if emu is not None and list(emu.ml_indexer.featureNames) == want:
            return name, emu
    joined = ' '.join(want).lower()
    is_isentropic = 's(j/g/k)' in joined or 'entropy' in joined
    for name, emu in candidates:
        if emu is not None and (name == 'isentropic') == is_isentropic:
            return name, emu
    for name, emu in candidates:
        if emu is not None:
            return name, emu
    raise RuntimeError('No emulator available on this API to run the quality test.')


def evaluate_emulator_quality(
    api,
    bundle_path,
    *,
    emulator_name: Optional[str] = None,
    max_samples: Optional[int] = None,
    seed: int = 1337,
    batch_size: int = 2 ** 14,
    normalize_features: bool = True,
) -> Dict[str, object]:
    """Evaluate one emulator against one ML-ready bundle.

    Parameters
    ----------
    api : EmulatorAPI
        Provides the wrapped ``NN_MELTS`` emulators.
    bundle_path : str or Path
        ML-ready ``.tar.gz`` bundle (features / binary_labels / molar_labels /
        labels / ml_indexer).
    emulator_name : {'isothermal', 'isentropic', 'openox'}, optional
        Which wrapped emulator to score. Default: auto-detect from the bundle's
        feature space.
    max_samples : int, optional
        Cap on evaluated rows (random subset, ``seed``). None -> every row.

    Returns
    -------
    dict with 'phase_quality_metrics' (DataFrame, metric rows x phase columns,
    columns abundance-sorted) and 'meta' (dict).
    """
    bundle_path = Path(bundle_path)
    bundle = load_ml_bundle(bundle_path)
    emu_name, emulator = _select_emulator(api, bundle, emulator_name)
    ml_indexer = emulator.ml_indexer
    device = emulator.dev if hasattr(emulator, 'dev') else 'cpu'
    device = 'cuda' if str(device) == 'cuda' and torch.cuda.is_available() else 'cpu'

    n_total = bundle.features.shape[0]
    if max_samples is not None and max_samples < n_total:
        rng = np.random.default_rng(seed)
        subset = np.sort(rng.choice(n_total, size=max_samples, replace=False))
    else:
        subset = np.arange(n_total)
    n = len(subset)

    px_sp_transform = ml_indexer.PxSpTransform
    comp_subset = ml_indexer.compositional_component_subset
    px_sp_sub = px_sp_transform[np.ix_(comp_subset, comp_subset)]

    validation_features = bundle.features
    validation_binaries = bundle.binary_labels
    validation_moles = bundle.molar_labels
    validation_labels = bundle.labels
    validation_labels_trans = validation_labels @ px_sp_sub

    labels_sub_tensor = torch.tensor(
        validation_labels_trans[subset], device=device, dtype=torch.float32
    )
    moles_sub_tensor = torch.tensor(validation_moles[subset], device=device, dtype=torch.float32)
    features_sub_raw = torch.tensor(validation_features[subset], device=device, dtype=torch.float32)
    features_sub_normed = (
        emulator.norm_features.norm(features_sub_raw) if normalize_features else features_sub_raw
    )

    # --- Ground-truth masses: batched getExtensiveComps + make_phase_tables ---
    n_vc = labels_sub_tensor.shape[1]
    n_ph = moles_sub_tensor.shape[1]
    combined_gt = torch.cat([labels_sub_tensor, moles_sub_tensor, features_sub_raw], dim=1)

    def gt_batch_fn(chunk: torch.Tensor) -> torch.Tensor:
        labels_c = chunk[:, :n_vc]
        moles_c = chunk[:, n_vc:n_vc + n_ph]
        feats_c = chunk[:, n_vc + n_ph:]
        if normalize_features:
            feats_c = emulator.norm_features.norm(feats_c)
        newcomps = emulator.getExtensiveComps(intensiveLabels=labels_c, molarLabels=moles_c)
        return emulator.make_phase_tables(
            newcomps, emulator.compToOx, emulator.MM, emulator.phaseToCompMap.T, feats_c, out=None
        )

    gt_masses = _to_np(_staged_forward(gt_batch_fn, combined_gt, batch_size))

    # --- Model inference: batched ---
    infer_out = _staged_forward(
        emulator.forwardMB,
        features_sub_raw,
        batch_size,
        Normalize=normalize_features,
        optimize_masses=True,
        protect_opx=False,
        outputs=["phase_present", "phase_tables", "reconstruction_residual"],
    )
    binary_hat = _to_np(infer_out["phase_present"])
    comp_tens, mass_tens = infer_out["phase_tables"]
    comp_tens = _to_np(comp_tens)     # (n, n_comp_phases, n_oxides) wt% within phase
    mass_tens = _to_np(mass_tens)     # (n, n_phases) wt% of system
    recon_resid = _to_np(infer_out["reconstruction_residual"]).reshape(-1)

    label_indices = ml_indexer.label_indices
    label_indices_comp = ml_indexer.label_indices_comp
    comp_phasedict = ml_indexer.comp_phasedict
    mass_phasedict = ml_indexer.mass_phasedict
    Oxides = list(ml_indexer.Oxides)
    comp_to_ox = ml_indexer.compToOx
    mm = ml_indexer.MM

    # ---- per-phase metric records ----
    records: Dict[str, dict] = {}
    phase_abundance: Dict[str, float] = {}

    for phase in label_indices:
        phase_idx = mass_phasedict[phase]
        real_pos = validation_binaries[subset, phase_idx] > 0.5
        pred_pos = binary_hat[:, phase_idx] > 0.5

        tp = int(np.sum(real_pos & pred_pos))
        fp = int(np.sum(~real_pos & pred_pos))
        fn = int(np.sum(real_pos & ~pred_pos))
        precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan

        gt_mass_all = gt_masses[:, phase_idx]
        phase_abundance[phase] = float(np.mean(np.where(real_pos, gt_mass_all, 0.0)))

        if real_pos.any():
            gt_mass = gt_masses[real_pos, phase_idx]
            pred_mass = mass_tens[real_pos, phase_idx]
            abundance_mae = float(np.mean(np.abs(gt_mass - pred_mass)))
            abundance_relerr = _safe_mean_abs_rel(gt_mass - pred_mass, gt_mass, threshold=0.01)
        else:
            abundance_mae = np.nan
            abundance_relerr = np.nan

        rec = {
            'proportion_in_dataset': float(real_pos.mean()),
            'proportion_predicted': float(pred_pos.mean()),
            'n_rows_present': int(real_pos.sum()),
            'precision': precision,
            'recall': recall,
            'abundance_gt_mean_wtpct': phase_abundance[phase],
            'abundance_mae_wtpct': abundance_mae,
            'abundance_relerr_pct': abundance_relerr * 100 if np.isfinite(abundance_relerr) else np.nan,
        }

        # ---- per-oxide within-phase composition errors ----
        if phase in label_indices_comp:
            indices = label_indices[phase]
            comp_indices = label_indices_comp[phase]
            active_oxides = _compute_active_oxides(comp_to_ox, indices)

            oxides_gt = validation_labels_trans[np.ix_(subset, comp_indices)] @ comp_to_ox[indices]
            if phase == "melts-liquid" and "Fe3" not in ml_indexer.Elkeys:
                oxides_gt = emulator.Iron_Speciator(
                    torch.tensor(oxides_gt, device=device, dtype=torch.float32),
                    features_sub_normed,
                ).detach().cpu().numpy()
            oxides_gt = oxides_gt @ mm
            oxides_gt_wt = oxides_gt * (100.0 / (1e-6 + np.sum(oxides_gt, axis=1, keepdims=True)))

            both = real_pos & pred_pos
            if both.any():
                both_idx = np.where(both)[0]
                for ox_i in active_oxides:
                    ox = Oxides[ox_i]
                    gt_ox = oxides_gt_wt[both_idx, ox_i]
                    pred_ox = comp_tens[both_idx, comp_phasedict[phase], ox_i]
                    rec[f'{ox}_comp_mae_wtpct'] = float(np.nanmean(np.abs(gt_ox - pred_ox)))
                    r = _safe_mean_abs_rel(gt_ox - pred_ox, gt_ox, threshold=0.1)
                    rec[f'{ox}_comp_relerr_pct'] = r * 100 if np.isfinite(r) else np.nan

        records[phase] = rec
        gc.collect()

    # ---- assemble metrics x phases table, columns abundance-sorted ----
    phase_order = sorted(records, key=lambda p: phase_abundance.get(p, 0.0), reverse=True)

    metric_order = list(_FIXED_METRICS)
    oxide_metrics = []
    for ox in Oxides:
        oxide_metrics += [f'{ox}_comp_mae_wtpct', f'{ox}_comp_relerr_pct']
    metric_order += [m for m in oxide_metrics if any(m in records[p] for p in phase_order)]

    table = pd.DataFrame(
        {phase: {m: records[phase].get(m, np.nan) for m in metric_order} for phase in phase_order},
        index=metric_order,
    )
    table.index.name = 'metric'

    # global scalars, broadcast across phase columns so the file stays one table
    for name, val in (
        ('n_samples', float(n)),
        ('recon_residual_l2_mean', float(np.mean(recon_resid))),
        ('recon_residual_l2_p95', float(np.percentile(recon_resid, 95))),
    ):
        table.loc[name] = val

    meta = {
        'bundle': str(bundle_path),
        'emulator': emu_name,
        'n_samples': int(n),
        'n_total_rows': int(n_total),
        'mass_balance': getattr(emulator, 'mass_balance', None),
        'recon_residual_l2_mean': float(np.mean(recon_resid)),
        'recon_residual_l2_p95': float(np.percentile(recon_resid, 95)),
        'phase_order': phase_order,
        'metric_descriptions': METRIC_DESCRIPTIONS,
    }
    return {'phase_quality_metrics': table, 'meta': meta}
