"""
Check predicted phase-assemblage coverage of bulk oxide space.

This script loads a trained model and an ML data bundle, predicts phase
assemblages, and verifies whether each predicted assemblage can span the oxide
support of the corresponding bulk composition.

It also reports statistics for zero-valued bulk oxides in feature compositions.

Usage example:
    python scripts/check_phase_oxide_coverage.py \
        --model .\src\builder\training\temp_models\model.tar \
        --bundle .\data\MLready\p\some_bundle.tar.gz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


repo_root = str(Path(__file__).resolve().parents[1])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

repo_src = str(Path(__file__).resolve().parents[1] / 'src')
if repo_src not in sys.path:
    sys.path.insert(0, repo_src)

from nMELTS.engine.NN import rebuild_MELTS_model
from nMELTS.engine.emulator import NN_MELTS
from nMELTS.utils.file_utils import load_ml_bundle


def _resolve_bundle_path(bundle_path: str) -> Path:
    path = Path(bundle_path)
    if path.suffixes[-2:] == ['.tar', '.gz']:
        return path
    if path.suffix == '.tar':
        return path.with_suffix('.tar.gz')
    if path.suffix == '.gz' and path.name.endswith('.tar.gz'):
        return path
    return path.with_suffix(path.suffix + '.tar.gz') if path.suffix else path.with_suffix('.tar.gz')


def _check_indexer_compat(bundle_indexer, model_indexer) -> None:
    if model_indexer is None:
        return
    mismatch = []
    if len(bundle_indexer.all_phases) != len(model_indexer.all_phases):
        mismatch.append('nphases')
    if bundle_indexer.ncompsVaried != model_indexer.ncompsVaried:
        mismatch.append('ncompsVaried')
    if bundle_indexer.ncomps != model_indexer.ncomps:
        mismatch.append('ncomps')
    if mismatch:
        joined = ', '.join(mismatch)
        raise ValueError(f'Model ml_indexer mismatch on: {joined}')


def _get_bulk_oxide_array(features: np.ndarray, ml_indexer) -> tuple[np.ndarray, list[str]]:
    feature_offset = len(ml_indexer.featureNames)
    n_elements = len(ml_indexer.Elkeys)
    oxide_names = list(ml_indexer.Oxides[:n_elements])

    element_comp = np.asarray(features[:, feature_offset:feature_offset + n_elements], dtype=np.float64)
    el_to_ox = np.asarray(ml_indexer.ElToOx, dtype=np.float64)
    bulk_oxides = element_comp @ el_to_ox
    return bulk_oxides, oxide_names


def _format_pct(count: int, total: int) -> str:
    if total <= 0:
        return '0.00%'
    return f'{100.0 * float(count) / float(total):.2f}%'


def run_coverage_check(
    model_path: str,
    bundle_path: str,
    threshold: float,
    bulk_zero_tol: float,
    support_tol: float,
    use_cuda: bool,
    normalize_features: bool,
    max_samples: int | None,
    seed: int,
) -> None:
    bundle_path_resolved = _resolve_bundle_path(bundle_path)
    bundle = load_ml_bundle(bundle_path_resolved)

    model = rebuild_MELTS_model(model_path)
    _check_indexer_compat(bundle.ml_indexer, getattr(model, 'ml_indexer', None))
    emulator = NN_MELTS(model, cuda=use_cuda)

    features = np.asarray(bundle.features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError(f'Expected 2D bundle.features, got shape {features.shape}')

    n_rows = features.shape[0]
    indices = np.arange(n_rows)
    if max_samples is not None and max_samples > 0 and max_samples < n_rows:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(indices, size=max_samples, replace=False))

    features_eval = features[indices]

    device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
    features_tensor = torch.tensor(features_eval, device=device, dtype=torch.float32)
    if normalize_features:
        features_tensor = emulator.norm_features.norm(features_tensor)

    with torch.no_grad():
        likelihoods, _, _, _ = emulator.model.forward(features_tensor, detailed=False)

    predicted_binary = (likelihoods.detach().cpu().numpy() > threshold).astype(np.int8)

    ml_indexer = model.ml_indexer
    phase_to_comp = np.asarray(ml_indexer.phaseToCompMap, dtype=np.float64)
    comp_to_ox = np.asarray(ml_indexer.compToOx, dtype=np.float64)

    bulk_oxides, oxide_names = _get_bulk_oxide_array(features_eval, ml_indexer)
    n_oxides = len(oxide_names)

    required_mask = bulk_oxides[:, :n_oxides] > bulk_zero_tol
    component_support = np.abs(comp_to_ox[:, :n_oxides]) > support_tol

    active_components = (predicted_binary @ phase_to_comp) > 0
    covered_mask = (active_components.astype(np.int8) @ component_support.astype(np.int8)) > 0
    missing_mask = required_mask & (~covered_mask)
    coverage_ok = ~np.any(missing_mask, axis=1)

    zero_mask = np.isclose(bulk_oxides[:, :n_oxides], 0.0, atol=bulk_zero_tol)

    total = features_eval.shape[0]
    n_ok = int(np.sum(coverage_ok))
    n_fail = total - n_ok

    print('--- Predicted Assemblage Oxide-Coverage Check ---')
    print(f'Model: {model_path}')
    print(f'Bundle: {bundle_path_resolved}')
    print(f'Samples evaluated: {total}')
    print(f'Coverage pass: {n_ok} / {total} ({_format_pct(n_ok, total)})')
    print(f'Coverage fail: {n_fail} / {total} ({_format_pct(n_fail, total)})')
    print('')

    fail_counts_by_oxide = np.sum(missing_mask, axis=0)
    if np.any(fail_counts_by_oxide > 0):
        print('Missing-support frequency by oxide among failed samples:')
        for oxide, count in zip(oxide_names, fail_counts_by_oxide):
            if count > 0:
                print(f'  {oxide}: {int(count)}')
    else:
        print('No missing-support oxides found; all required oxides were covered.')

    print('')
    print('--- Feature Bulk Zero-Oxide Statistics ---')
    rows_with_any_zero = int(np.sum(np.any(zero_mask, axis=1)))
    print(
        f'Rows with >=1 zero oxide: {rows_with_any_zero} / {total} '
        f'({_format_pct(rows_with_any_zero, total)})'
    )

    zero_counts_by_oxide = np.sum(zero_mask, axis=0)
    for oxide, count in zip(oxide_names, zero_counts_by_oxide):
        print(f'  {oxide}: {int(count)} zero rows ({_format_pct(int(count), total)})')

    if n_fail > 0:
        failed_global_ids = indices[~coverage_ok]
        preview = failed_global_ids[:25]
        print('')
        print('First failing sample indices (bundle row indices):')
        print('  ' + ', '.join(str(int(x)) for x in preview))
        if failed_global_ids.size > preview.size:
            print(f'  ... and {failed_global_ids.size - preview.size} more')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Check if predicted phase assemblages cover bulk oxide support.'
    )
    parser.add_argument(
        '--model',
        required=True,
        help='Path to trained model (.tar/.pt/.zip depending on your setup).',
    )
    parser.add_argument(
        '--bundle',
        required=True,
        help='Path to ML bundle (.tar.gz; extension inferred when omitted).',
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Phase-likelihood threshold for predicted assemblage (default: 0.5).',
    )
    parser.add_argument(
        '--bulk-zero-tol',
        type=float,
        default=1e-12,
        help='Tolerance below which bulk oxide is treated as zero (default: 1e-12).',
    )
    parser.add_argument(
        '--support-tol',
        type=float,
        default=1e-12,
        help='Tolerance for component oxide support in compToOx (default: 1e-12).',
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Optional random subset size for quick checks.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=1337,
        help='Random seed used with --max-samples.',
    )
    parser.add_argument(
        '--cuda',
        action='store_true',
        help='Use CUDA if available.',
    )
    parser.add_argument(
        '--no-normalize-features',
        action='store_true',
        help='Disable feature normalization before model forward pass.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_coverage_check(
        model_path=args.model,
        bundle_path=args.bundle,
        threshold=float(args.threshold),
        bulk_zero_tol=float(args.bulk_zero_tol),
        support_tol=float(args.support_tol),
        use_cuda=bool(args.cuda),
        normalize_features=not bool(args.no_normalize_features),
        max_samples=args.max_samples,
        seed=int(args.seed),
    )


if __name__ == '__main__':
    main()
