"""
Retroactively shuffle row order and add a feature_bounds.json to an existing
ML-ready bundle, without rerunning the BigMetaTable/resampling_to_datasets
pipeline from raw MELTS output.

This exists for bundles built before two processing-pipeline changes:
  1. Shuffling row order - a one-time out-of-core shuffle applied post-
     packaging, to the finished bundle (see MLexporter.shuffle_bundle_rows,
     also called by prepareML.py/prepareML_fullvalid.py after
     resampling_to_datasets()), so a later chunked/async training data loader
     can read fixed-size contiguous chunks without worrying about row-order
     structure (e.g. upsampled rows resample_rare_phase appends as a block at
     the end of the table).
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
import shutil
import sys
from pathlib import Path

repo_src = str(Path(__file__).resolve().parents[1] / "src")
if repo_src not in sys.path:
    sys.path.insert(0, repo_src)

from builder.processing.MLexporter import shuffle_bundle_rows


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

    shuffle_bundle_rows(target_path, seed=args.seed, chunk_size=args.chunk_size)

    print(f"[retrofit_bundle] Done. Retrofitted bundle: {target_path}")


if __name__ == "__main__":
    main()
