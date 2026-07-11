"""
Retroactively shuffle row order and add a feature_bounds.json to an existing
ML-ready bundle, without rerunning the BigMetaTable/resampling_to_datasets
pipeline from raw MELTS output.

This exists for bundles built before two processing-pipeline changes:
  1. BigMetaTable.shuffle_rows() - a one-time out-of-core shuffle applied to
     the Train table before resampling_to_datasets(), so a later chunked/
     async training data loader can read fixed-size contiguous chunks
     without worrying about row-order structure (e.g. upsampled rows
     resample_rare_phase appends as a block at the end of the table).
  2. generate_dataset_stats() now also writes a "feature_bounds.json"
     companion (featureNames' min/max) into every bundle - an immutable
     property of the dataset that builder.training.dataset_workspace needs
     to normalize features without a separate prepass scan.

Run this once against an existing large bundle to bring it up to date with
both changes, so you can exercise the new chunked training-data workflow
without waiting on a full reprocessing run.

Usage:
    python scripts/retrofit_bundle.py --bundle <path/to/bundle.tar.gz>

By default the bundle is retrofitted in place. Pass --output to retrofit a
copy instead and leave the original untouched:

    python scripts/retrofit_bundle.py --bundle in.tar.gz --output out.tar.gz
"""

import argparse
import gc
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

repo_src = str(Path(__file__).resolve().parents[1] / "src")
if repo_src not in sys.path:
    sys.path.insert(0, repo_src)

import numpy as np

from ngibbs.utils.file_utils import chunked_permutation_copy
from ngibbs.config.ml_indexer import load_ml_indexer_from_state

# Row-aligned arrays that may be present in a bundle. Every one that actually
# exists gets shuffled with the *same* permutation, to keep rows aligned
# across arrays. Not all bundles have mass_labels/free_outputs (the latter is
# optional; the former is always written by resampling_to_datasets today, but
# guarded here in case of older/hand-built bundles).
_ROW_ALIGNED_ARRAYS = (
    "features.npy",
    "labels.npy",
    "binary_labels.npy",
    "molar_labels.npy",
    "mass_labels.npy",
    "free_outputs.npy",
)


def main():
    parser = argparse.ArgumentParser(
        description="Retroactively shuffle row order and add feature_bounds.json "
                     "to an existing ML-ready bundle."
    )
    parser.add_argument("--bundle", required=True, type=str,
                        help="Path to the .tar.gz ML-ready bundle to retrofit")
    parser.add_argument("--output", type=str, default=None,
                        help="Write the retrofitted result here instead of "
                             "overwriting --bundle in place (the bundle is "
                             "copied first, then the copy is retrofitted)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for the row permutation (reproducibility). "
                             "Default: fresh entropy.")
    parser.add_argument("--chunk-size", type=int, default=1_000_000,
                        help="Row-chunk size for the shuffle and stats scan.")
    args = parser.parse_args()

    bundle_path = Path(args.bundle)
    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle not found: {bundle_path}")

    if args.output:
        target_path = Path(args.output)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[retrofit_bundle] Copying {bundle_path} -> {target_path}")
        shutil.copy(bundle_path, target_path)
    else:
        target_path = bundle_path
        print(f"[retrofit_bundle] Retrofitting {target_path} in place")

    repo_root = Path(__file__).resolve().parents[1]
    tmp_base = repo_root / "data" / "tmp"
    tmp_base.mkdir(parents=True, exist_ok=True)
    extract_dir = Path(tempfile.mkdtemp(dir=tmp_base))

    try:
        print(f"[retrofit_bundle] Extracting {target_path} ...")
        with tarfile.open(target_path, "r:gz") as tar:
            tar.extractall(path=extract_dir)

        present_arrays = [name for name in _ROW_ALIGNED_ARRAYS if (extract_dir / name).exists()]
        print(f"[retrofit_bundle] Row-aligned arrays found: {present_arrays}")

        n_rows = np.load(extract_dir / "features.npy", mmap_mode="r").shape[0]
        rng = np.random.default_rng(args.seed)
        permutation = rng.permutation(n_rows)
        print(f"[retrofit_bundle] Shuffling {n_rows:,} rows (seed={args.seed})...")

        for name in present_arrays:
            src_path = extract_dir / name
            src = np.load(src_path, mmap_mode="r")
            tmp_path = extract_dir / f"_shuffled_{name}"
            chunked_permutation_copy(src, tmp_path, permutation, chunk_size=args.chunk_size)
            del src
            gc.collect()
            tmp_path.replace(src_path)
            print(f"[retrofit_bundle]   shuffled {name}")

        print("[retrofit_bundle] Regenerating stats.txt + feature_bounds.json...")
        ml_indexer = load_ml_indexer_from_state(str(extract_dir / "ml_indexer"))
        from builder.processing.MLexporter import generate_dataset_stats

        dataset_name = str(extract_dir) + "/"
        stats_path = generate_dataset_stats(
            dataset_name=dataset_name,
            ml_indexer=ml_indexer,
            output_dir=extract_dir,
            chunk_size=args.chunk_size,
        )
        feature_bounds_path = extract_dir / f"{Path(dataset_name).stem}_feature_bounds.json"

        print(f"[retrofit_bundle] Repacking {target_path} ...")
        file_mappings = {name: name for name in present_arrays}
        if stats_path and stats_path.exists():
            file_mappings[str(stats_path)] = "stats.txt"
        if feature_bounds_path.exists():
            file_mappings[str(feature_bounds_path)] = "feature_bounds.json"

        with tarfile.open(target_path, "w:gz") as tar:
            for name in present_arrays:
                tar.add(extract_dir / name, arcname=name)
            if stats_path and stats_path.exists():
                tar.add(stats_path, arcname="stats.txt")
            if feature_bounds_path.exists():
                tar.add(feature_bounds_path, arcname="feature_bounds.json")
            indexer_dir = extract_dir / "ml_indexer"
            if indexer_dir.is_dir():
                tar.add(indexer_dir, arcname="ml_indexer")

    finally:
        shutil.rmtree(extract_dir, ignore_errors=True)

    print(f"[retrofit_bundle] Done. Retrofitted bundle: {target_path}")


if __name__ == "__main__":
    main()
