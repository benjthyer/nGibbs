"""
CLI: extract a random (or sequential) row subset from a large ML-ready bundle.

The bundle analogue of scripts/subset_csv.py. Where subset_csv.py picks rows
from paired .csv/.txt files, this picks rows from a packaged .tar.gz bundle
(as produced by prepareML.py / prepareML_fullvalid.py / resampling_to_datasets):
every row-aligned array inside it (features.npy, labels.npy, binary_labels.npy,
molar_labels.npy, mass_labels.npy, plus free_outputs.npy and the derivative
sidecars when present) is trimmed with the *same* row indices so the arrays
stay row-aligned. stats.txt, feature_bounds.json and T0.npy are regenerated
for the smaller dataset; ml_indexer/ and everything else in the tarball are
carried through unchanged.

The row selection and out-of-core repacking live in
builder.processing.MLexporter.subset_bundle_rows, next to shuffle_bundle_rows
(same extract / trim / regenerate-stats / repack machinery).

Usage:
    python scripts/subset_bundle.py --bundle path/to/bundle.tar.gz --nrows 100000
    python scripts/subset_bundle.py --bundle in.tar.gz --output out.tar.gz --nrows 50000 --seed 7
    python scripts/subset_bundle.py --bundle in.tar.gz --nrows 50000 --sequential --in-place

By default the subset is written to <bundle>_subsetN.tar.gz (a new file, the
original untouched). Pass --output to name it, or --in-place to overwrite the
source bundle.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from builder.processing.MLexporter import subset_bundle_rows  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Extract a random (or sequential) row subset from a large ML-ready "
            ".tar.gz bundle. Every row-aligned array in the bundle is trimmed "
            "with the same row indices; stats.txt / feature_bounds.json / T0.npy "
            "are regenerated for the subset. Out-of-core (chunked), so bundle "
            "size is not bounded by RAM."
        )
    )
    p.add_argument("--bundle", required=True, help="Path to the source .tar.gz ML-ready bundle")
    p.add_argument("--nrows", type=int, required=True, help="Number of output rows")
    p.add_argument(
        "--output", default=None,
        help="Where to write the subset bundle. Default: <bundle>_subset<nrows>.tar.gz "
             "(a new file -- pass --in-place to overwrite the source instead).",
    )
    p.add_argument(
        "--in-place", action="store_true",
        help="Overwrite --bundle itself instead of writing a new file.",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed for random sampling (default: 42)")
    p.add_argument(
        "--sequential", action="store_true",
        help="Take the first --nrows rows instead of a random sample",
    )
    p.add_argument(
        "--chunk-size", type=int, default=1_000_000,
        help="Row-chunk size for the trim passes and the stats scan (default: 1,000,000)",
    )
    return p


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
        name = bundle_path.name
        stem = name[:-len(".tar.gz")] if name.endswith(".tar.gz") else bundle_path.stem
        output_path = bundle_path.with_name(f"{stem}_subset{args.nrows}.tar.gz")

    print(f"Bundle: {bundle_path}")
    print(f"Output: {output_path}{' (in place)' if output_path == bundle_path else ''}")

    subset_bundle_rows(
        bundle_path,
        output_path=output_path,
        nrows=args.nrows,
        seed=args.seed,
        sequential=args.sequential,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
