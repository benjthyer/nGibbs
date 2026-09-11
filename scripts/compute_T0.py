#!/usr/bin/env python3
"""Retroactively compute and attach T0 to an existing ML bundle.

`T0` (shape (P,), one value per phase) is the 5th percentile of a phase's *nonzero*
training-label molar abundance -- the per-phase "how small does this actually get before
the data calls it absent" scale that anchors `ContinuousModel`'s annealed
complementarity-smoothing temperature `T = a*T0` (see NN_continuous.py's
`upper_forward`). Bundles exported by MLexporter.py after this feature was added carry
T0 already; this script is for bundles exported before that.

The whole bundle is extracted and repacked rather than round-tripped through
load_ml_bundle/save_ml_bundle, because save_ml_bundle only knows about the core
row-aligned arrays -- a round trip through it would silently drop derivative sidecars
(dndp_labels.npy etc.), stats.txt, and anything else already in the tarball.

Usage:
    python scripts/compute_T0.py --bundle path/to/HeFESTo_..._Train.tar.gz
    python scripts/compute_T0.py --bundle path/to/bundle.tar.gz --in-place
    python scripts/compute_T0.py --bundle path/to/bundle.tar.gz --output path/to/new.tar.gz
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / 'src'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ngibbs.utils.file_utils import compute_T0, load_ml_bundle  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute per-phase T0 for an existing ML bundle and attach it (as "
                     "T0.npy) without disturbing anything else already in the tarball."
    )
    parser.add_argument("--bundle", required=True, help="Path to the .tar.gz bundle")
    parser.add_argument(
        "--output", default=None,
        help="Where to write the patched bundle. Default: <bundle>_with_T0.tar.gz "
             "(a new file -- pass --in-place to overwrite the original instead)."
    )
    parser.add_argument(
        "--in-place", action="store_true",
        help="Overwrite --bundle itself instead of writing a new file."
    )
    parser.add_argument(
        "--percentile", type=float, default=5.0,
        help="Percentile of nonzero molar abundance to use as T0 (default: 5.0)."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Recompute and overwrite T0.npy even if the bundle already has one."
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    bundle_path = Path(args.bundle)
    if not bundle_path.exists():
        print(f"ERROR: bundle not found: {bundle_path}")
        sys.exit(1)

    if args.in_place and args.output:
        print("ERROR: pass --in-place OR --output, not both.")
        sys.exit(1)
    if args.in_place:
        output_path = bundle_path
    elif args.output:
        output_path = Path(args.output)
    else:
        stem = bundle_path.name
        if stem.endswith('.tar.gz'):
            stem = stem[:-len('.tar.gz')]
        output_path = bundle_path.with_name(f"{stem}_with_T0.tar.gz")

    print(f"Bundle: {bundle_path}")
    print(f"Output: {output_path}{' (in place)' if output_path == bundle_path else ''}")

    # Lightweight load: only molar_labels + ml_indexer, not features/labels/derivatives.
    print("\nLoading molar_labels + ml_indexer...")
    bundle = load_ml_bundle(bundle_path, arrays=('molar_labels',))
    ml_indexer = bundle.ml_indexer

    if bundle.T0 is not None and not args.force:
        print(f"Bundle already carries a T0 (shape {bundle.T0.shape}). "
              f"Pass --force to recompute it anyway. Nothing to do.")
        sys.exit(0)

    print(f"Computing T0 (percentile={args.percentile}) over "
          f"{bundle.molar_labels.shape[0]:,} rows, {ml_indexer.nphases} phases...")
    T0 = compute_T0(bundle.molar_labels, ml_indexer.mass_phasedict, ml_indexer.all_phases,
                    percentile=args.percentile)

    print("\nPer-phase T0:")
    zero_phases = []
    for phase in ml_indexer.all_phases:
        idx = ml_indexer.mass_phasedict[phase]
        val = T0[idx]
        print(f"  {phase:24s} {val:.6e}")
        if val == 0.0:
            zero_phases.append(phase)
    if zero_phases:
        print(f"\nWARNING: never-present phases in this dataset (T0=0, cannot anneal): "
              f"{zero_phases}")

    # Extract the WHOLE original tarball, add T0.npy, repack - preserves derivative
    # sidecars, stats.txt, ml_indexer/, everything, rather than reconstructing the
    # bundle from the handful of arrays save_ml_bundle knows about.
    print(f"\nRepacking bundle with T0.npy...")
    tmp_base = REPO_ROOT / "data" / "tmp"
    tmp_base.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(dir=tmp_base))
    try:
        with tarfile.open(bundle_path, 'r:gz') as tar:
            tar.extractall(path=temp_dir)

        np.save(temp_dir / 'T0.npy', T0)

        tmp_output = output_path.with_suffix(output_path.suffix + '.tmp')
        with tarfile.open(tmp_output, 'w:gz') as tar:
            tar.add(temp_dir, arcname='.')
        shutil.move(str(tmp_output), str(output_path))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"\nDone. T0 written to {output_path}")


if __name__ == "__main__":
    main()
