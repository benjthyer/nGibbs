"""
remap_bundle_basis.py

Convert an ML-ready bundle between thermodynamic state-variable bases
(e.g. P-S → P-T, V-T → V-S) by swapping one feature column with one
free-output column.

Bundle layout produced by resampling_to_datasets():
  features.npy       shape (N, n_features + n_elements)
                     columns 0..n_features-1  → state variable inputs (P, T, S, ...)
                     columns n_features..     → bulk chemistry (element fractions)
  free_outputs.npy   shape (N, n_free_outputs)
  ml_indexer/        directory with featureNames / free_outputs stored in metadata

The swap:
  - column ``demote`` in features  → replaces matching column in free_outputs
  - column ``promote`` in free_outputs → replaces matching column in features
  - ml_indexer.featureNames / free_outputs are updated to match

Usage
-----
  # Single bundle
  python scripts/remap_bundle_basis.py \\
      --input  data/MLready/102/102Isentropic_NoCr_Train.tar.gz \\
      --demote "S(System_main)" \\
      --promote "Temperature(System_main)" \\
      --output data/MLready/102/102PT_NoCr_Train.tar.gz

  # All three splits in one call (supply the shared stem, omit _Train/_Test/_Valid suffix)
  python scripts/remap_bundle_basis.py \\
      --input  data/MLready/102/102Isentropic_NoCr \\
      --demote "S(System_main)" \\
      --promote "Temperature(System_main)" \\
      --output data/MLready/102/102PT_NoCr
"""

import argparse
import json
import shutil
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import sys

# Ensure repo root and src are on Python path
_repo_root = Path(__file__).resolve().parents[1]
for _p in [str(_repo_root), str(_repo_root / 'src')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ngibbs.config.ml_indexer import load_ml_indexer_from_state


def _norm(name: str) -> str:
    return name.strip()


def _compute_feature_bounds(features: np.ndarray, feature_names: list) -> dict:
    """Min/max over the physical feature columns (0..len(feature_names)-1),
    matching the layout MLexporter.generate_dataset_stats() writes."""
    n = len(feature_names)
    physical = features[:, :n]
    return {
        "featureNames": list(feature_names),
        "min": [float(v) for v in np.min(physical, axis=0)],
        "max": [float(v) for v in np.max(physical, axis=0)],
    }


def _patch_stats_condition_bounds(stats_text: str, bounds: dict) -> str:
    """Replace just the CONDITION BOUNDS table in stats.txt with `bounds`, in
    the exact "{name:>20} {min:15.2f} {max:15.2f}" layout
    MLexporter.generate_dataset_stats() writes. Every other section (phase
    abundances, oxide bounds, liquid fraction) is untouched - the swap doesn't
    change those, so there's no need to rescan the dataset to refresh them."""
    lines = stats_text.splitlines(keepends=True)
    header_idx = next(i for i, l in enumerate(lines) if l.strip() == "CONDITION BOUNDS")
    rows_start = header_idx + 3  # skip the section's own dash / column-header / dash lines
    rows_end = rows_start
    while rows_end < len(lines) and lines[rows_end].strip():
        rows_end += 1
    new_rows = [
        f"{name:>20} {lo:15.2f} {hi:15.2f}\n"
        for name, lo, hi in zip(bounds['featureNames'], bounds['min'], bounds['max'])
    ]
    return ''.join(lines[:rows_start] + new_rows + lines[rows_end:])


def remap_bundle(input_path: Path, demote: str, promote: str, output_path: Path,
                  bounds_override: dict = None) -> dict:
    """
    Swap one feature and one free output in a single bundle.

    Parameters
    ----------
    input_path : Path
        Source .tar.gz bundle.
    demote : str
        Name currently in featureNames to move into free_outputs.
        Format: 'Component(Phase)'  e.g. 'S(System_main)'
    promote : str
        Name currently in free_outputs to move into features.
        Format: 'Component(Phase)'  e.g. 'Temperature(System_main)'
    output_path : Path
        Destination .tar.gz path (created or overwritten).
    bounds_override : dict, optional
        A feature_bounds.json payload (featureNames/min/max) to write into this
        bundle verbatim instead of recomputing bounds from its own data. Used so
        Test/Valid splits inherit the Train split's bounds rather than fitting
        their own - the normalizer built from feature_bounds.json must match
        across splits, the same way Test/Valid already inherit Train's fitted
        feature_normalizer at training time (see dataset_workspace.py).

    Returns
    -------
    dict
        The feature_bounds.json payload written into this bundle (either
        ``bounds_override``, or freshly computed from this bundle's own data
        when no override is given).
    """
    tmp_base = _repo_root / "data" / "tmp"
    tmp_base.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="remap_bundle_", dir=tmp_base))
    try:
        # Extract
        with tarfile.open(input_path, 'r:gz') as tar:
            tar.extractall(path=tmp)

        # Load arrays
        features = np.load(tmp / 'features.npy')
        free_outputs_path = tmp / 'free_outputs.npy'
        if not free_outputs_path.exists():
            raise FileNotFoundError(
                f"Bundle '{input_path.name}' has no free_outputs.npy.\n"
                "Rebuild the bundle with the variable you want to promote listed in free_outputs."
            )
        free_outputs = np.load(free_outputs_path)

        # Load indexer to get current name lists
        indexer = load_ml_indexer_from_state(tmp / 'ml_indexer')
        feature_names = list(indexer.featureNames or [])
        free_output_names = list(indexer.free_outputs or [])

        demote_n = _norm(demote)
        promote_n = _norm(promote)

        norm_feat = [_norm(n) for n in feature_names]
        norm_free = [_norm(n) for n in free_output_names]

        if demote_n not in norm_feat:
            raise ValueError(
                f"'{demote}' not in featureNames: {feature_names}\n"
                "Use the exact 'Component(Phase)' string stored in the bundle."
            )
        if promote_n not in norm_free:
            raise ValueError(
                f"'{promote}' not in free_outputs: {free_output_names}\n"
                "Use the exact 'Component(Phase)' string stored in the bundle."
            )

        fi = norm_feat.index(demote_n)   # column index inside features (state-var section)
        oi = norm_free.index(promote_n)  # column index inside free_outputs

        # Swap columns (copy to avoid aliasing)
        new_features = features.copy()
        new_free = free_outputs.copy()
        new_features[:, fi] = free_outputs[:, oi]
        new_free[:, oi] = features[:, fi]

        # Update name lists (preserve original objects/casing from indexer)
        new_feature_names = list(feature_names)
        new_free_names = list(free_output_names)
        new_feature_names[fi] = free_output_names[oi]
        new_free_names[oi] = feature_names[fi]

        indexer.featureNames = new_feature_names
        indexer.free_outputs = new_free_names

        # Write updated arrays back to temp dir
        np.save(tmp / 'features.npy', new_features)
        np.save(tmp / 'free_outputs.npy', new_free)

        # Overwrite ml_indexer directory with updated metadata
        indexer_dir = tmp / 'ml_indexer'
        shutil.rmtree(indexer_dir, ignore_errors=True)
        indexer.save(str(indexer_dir))

        # feature_bounds.json is not just carried through unchanged: the swap put
        # a differently-scaled quantity (e.g. entropy) into the column that used
        # to hold the demoted quantity's own bounds (e.g. temperature), which
        # would badly mis-normalize training inputs downstream. Recompute it from
        # this bundle's own (post-swap) data, unless the caller supplied another
        # split's bounds to inherit instead.
        bounds = bounds_override if bounds_override is not None else _compute_feature_bounds(
            new_features, new_feature_names
        )
        with open(tmp / 'feature_bounds.json', 'w') as f:
            json.dump(bounds, f, indent=2)

        # stats.txt's CONDITION BOUNDS table carries the same stale
        # name/min/max otherwise - patch it in place from the same `bounds`
        # rather than rescanning the dataset for a section that's cheap to
        # derive from what we already computed.
        stats_path = tmp / 'stats.txt'
        if stats_path.exists():
            stats_path.write_text(_patch_stats_condition_bounds(stats_path.read_text(), bounds))

        # Repack everything into the output bundle
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(output_path, 'w:gz') as tar:
            for item in sorted(tmp.iterdir()):
                tar.add(item, arcname=item.name)

        print(f"  -> {output_path.name}")
        print(f"     features    : {new_feature_names}")
        print(f"     free_outputs : {new_free_names}")
        print(f"     feature_bounds: "
              f"{'inherited from Train' if bounds_override is not None else 'recomputed'} "
              f"-> {dict(zip(bounds['featureNames'], zip(bounds['min'], bounds['max'])))}")

        return bounds

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _resolve_split_paths(stem: str, suffix: str = '.tar.gz'):
    """Given a stem (with or without _Train/_Test/_Valid), return all split paths found on disk."""
    p = Path(stem)
    # If the stem itself is an existing .tar.gz, treat as single file
    if p.suffix == '.gz' and p.exists():
        return [p]
    # Strip any accidental split suffix the user included
    for tag in ('_Train', '_Test', '_Valid'):
        if str(p).endswith(tag):
            p = Path(str(p)[:-len(tag)])
            break
    found = []
    for tag in ('_Train', '_Test', '_Valid'):
        candidate = p.parent / (p.name + tag + suffix)
        if candidate.exists():
            found.append(candidate)
    return found


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Swap a feature and a free output in an ML-ready bundle to change the "
            "thermodynamic state-variable basis (e.g. P-S ↔ P-T)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--input', required=True,
        help=(
            "Source bundle path (.tar.gz) OR a stem shared by Train/Test/Valid bundles "
            "(e.g. data/MLready/102/102Isentropic_NoCr). "
            "When a stem is given all matching *_Train/Test/Valid.tar.gz files are processed."
        ),
    )
    parser.add_argument(
        '--demote', required=True,
        help="Feature name (in featureNames) to move into free_outputs. "
             "Format: 'Component(Phase)', e.g. 'S(System_main)'",
    )
    parser.add_argument(
        '--promote', required=True,
        help="Free-output name (in free_outputs) to move into features. "
             "Format: 'Component(Phase)', e.g. 'Temperature(System_main)'",
    )
    parser.add_argument(
        '--output', default=None,
        help=(
            "Destination path. For a single bundle: the output .tar.gz path. "
            "For a stem: the output stem (split suffix is appended automatically). "
            "If omitted, appends '_remapped' to the input name."
        ),
    )
    args = parser.parse_args()

    input_path = Path(args.input)

    # Determine whether input is a single bundle or a stem
    if input_path.exists() and str(input_path).endswith('.tar.gz'):
        sources = [input_path]
        is_stem = False
    else:
        sources = _resolve_split_paths(str(input_path))
        is_stem = True
        if not sources:
            raise FileNotFoundError(
                f"No bundles found for input '{args.input}'.\n"
                "Checked for <input>_Train.tar.gz, <input>_Test.tar.gz, <input>_Valid.tar.gz."
            )

    print(f"Remapping basis:  '{args.demote}'  →→  features")
    print(f"                  '{args.promote}'  →→  free_outputs\n")

    def _split_tag(path):
        for tag in ('_Train', '_Test', '_Valid'):
            if path.name.endswith(tag + '.tar.gz'):
                return tag
        return None

    def _dst_for(src):
        if args.output:
            if is_stem:
                tag = _split_tag(src)
                if tag:
                    return Path(args.output + tag + '.tar.gz')
                return Path(args.output + '.tar.gz')
            dst = Path(args.output)
            if not str(dst).endswith('.tar.gz'):
                dst = Path(str(dst) + '.tar.gz')
            return dst
        stem = src.name.replace('.tar.gz', '')
        return src.parent / f"{stem}_remapped.tar.gz"

    dst_by_src = {src: _dst_for(src) for src in sources}

    # Recompute feature_bounds.json once (from Train's own post-swap data) and
    # reuse it verbatim for Test/Valid, so every split is normalized against the
    # same bounds - mirrors how Test/Valid already inherit Train's fitted
    # feature_normalizer at training time rather than fitting their own.
    if is_stem:
        train_srcs = [s for s in sources if _split_tag(s) == '_Train']
        other_srcs = [s for s in sources if _split_tag(s) != '_Train']
        if not train_srcs:
            print(
                "Warning: no *_Train bundle among inputs; each bundle's "
                "feature_bounds.json will be recomputed independently from its "
                "own data, so splits may not share identical normalization "
                "bounds.\n"
            )
        ordered_sources = train_srcs + other_srcs
    else:
        ordered_sources = sources

    train_bounds = None
    for src in ordered_sources:
        dst = dst_by_src[src]
        override = train_bounds if (is_stem and _split_tag(src) != '_Train') else None

        print(f"Processing: {src.name}")
        bounds = remap_bundle(src, args.demote, args.promote, dst, bounds_override=override)

        if is_stem and _split_tag(src) == '_Train':
            train_bounds = bounds

    print("\nDone.")


if __name__ == '__main__':
    main()
