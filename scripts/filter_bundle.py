"""
Apply deep_filter and/or bundle_insanity_filter to an already-built ML-ready
bundle, without rerunning the full BigMetaTable pipeline (CSV parsing,
resampling, component-moles computation, etc.).

Both filters only need the .npy arrays already exported inside the bundle
(labels.npy, features.npy, binary_labels.npy, molar_labels.npy, mass_labels.npy
[, free_outputs.npy] + the ml_indexer directory), so this re-applies bounds
directly to a .tar.gz produced by prepareML.py / prepareML_fullvalid.py.

Edit the BOUNDS section below to match the dataset being filtered - same
format as the `deep_filter` section of config/processing.yaml (see
recipes/processing/*.yaml for real examples).

Usage:
    python scripts/filter_bundle.py --bundle <path/to/bundle.tar.gz>

By default the bundle is filtered in place (matching how deep_filter/
bundle_insanity_filter already behave when called from the main pipeline).
Pass --output to filter a copy instead and leave the original untouched:

    python scripts/filter_bundle.py --bundle in.tar.gz --output out.tar.gz
"""

import argparse
import shutil
import sys
from pathlib import Path

repo_src = str(Path(__file__).resolve().parents[1] / "src")
if repo_src not in sys.path:
    sys.path.insert(0, repo_src)

from builder.processing.filters import deep_filter, bundle_insanity_filter


# --------------------------------------------------------------------------- #
# Bounds -- edit these to match the dataset being filtered. Same format as the
# `deep_filter` section of config/processing.yaml.
# --------------------------------------------------------------------------- #

# Oxide lower bounds: [phase, oxide, minimum_wt%]
OXIDE_LOWER_BOUNDS = [
     ['k-feldspar', 'K2O', 6],
    # ['clinopyroxene', 'CaO', 6],
    # ['melts-liquid', 'SiO2', 30],
]

# Oxide upper bounds: [phase, oxide, maximum_wt%]
OXIDE_UPPER_BOUNDS = [
    # ['k-feldspar', 'CaO', 4],
    # ['clinopyroxene', 'TiO2', 9],
]

# Component lower bounds: [phase, component, minimum fraction]
COMPONENT_LOWER_BOUNDS = [
    # ['plagioclase', 'sanidine', 0.0],
]

# Component upper bounds: [phase, component, maximum fraction]
COMPONENT_UPPER_BOUNDS = [
    # ['plagioclase', 'sanidine', 0.2],
]

# Whole-assemblage bulk oxide bounds: {oxide: [min_wt%, max_wt%]}
BULK_OXIDE_BOUNDS = {
    # 'SiO2': [35, 80],
}

# Row-chunk size for both deep_filter's batched scan and bundle_insanity_filter's
# chunked sanity-check scan (see recipes/processing/*.yaml performance.chunk_size
# / deep_filter.batch_size for the values used elsewhere in this dataset).
BATCH_SIZE = 200_000

# bundle_insanity_filter tolerances (see filters.py docstring for what each
# check does). bulk_tol_frac=5e-3 matches prepareML_fullvalid.py's call.
INSANITY_TOLERANCE = 1e-3
INSANITY_BULK_TOL_FRAC = 5e-3

# Toggle either stage off independently (e.g. to only run bundle_insanity_filter
# on a bundle that's already been through deep_filter).
RUN_DEEP_FILTER = True
RUN_INSANITY_FILTER = True


def main():
    parser = argparse.ArgumentParser(
        description="Apply deep_filter and/or bundle_insanity_filter to an "
                     "existing ML-ready bundle, without rerunning the "
                     "BigMetaTable pipeline. Edit the BOUNDS section at the "
                     "top of this script before running."
    )
    parser.add_argument("--bundle", required=True, type=str,
                        help="Path to the .tar.gz ML-ready bundle to filter")
    parser.add_argument("--output", type=str, default=None,
                        help="Write the filtered result here instead of "
                             "overwriting --bundle in place (the bundle is "
                             "copied first, then the copy is filtered)")
    args = parser.parse_args()

    bundle_path = Path(args.bundle)
    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle not found: {bundle_path}")

    if args.output:
        target_path = Path(args.output)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[filter_bundle] Copying {bundle_path} -> {target_path}")
        shutil.copy(bundle_path, target_path)
    else:
        target_path = bundle_path
        print(f"[filter_bundle] Filtering {target_path} in place")

    if RUN_DEEP_FILTER:
        print(f"\n[filter_bundle] === deep_filter ===")
        print(f"  oxide_lower_bounds:     {OXIDE_LOWER_BOUNDS}")
        print(f"  oxide_upper_bounds:     {OXIDE_UPPER_BOUNDS}")
        print(f"  component_lower_bounds: {COMPONENT_LOWER_BOUNDS}")
        print(f"  component_upper_bounds: {COMPONENT_UPPER_BOUNDS}")
        print(f"  bulk_oxide_bounds:      {BULK_OXIDE_BOUNDS}")
        deep_filter(
            str(target_path),
            Oxide_Lower_Bounds=OXIDE_LOWER_BOUNDS or None,
            Oxide_Upper_Bounds=OXIDE_UPPER_BOUNDS or None,
            Component_Lower_Bounds=COMPONENT_LOWER_BOUNDS or None,
            Component_Upper_Bounds=COMPONENT_UPPER_BOUNDS or None,
            Bulk_Oxide_Bounds=BULK_OXIDE_BOUNDS or None,
            batch_size=BATCH_SIZE,
        )

    if RUN_INSANITY_FILTER:
        print(f"\n[filter_bundle] === bundle_insanity_filter ===")
        bundle_insanity_filter(
            target_path,
            tolerance=INSANITY_TOLERANCE,
            bulk_tol_frac=INSANITY_BULK_TOL_FRAC,
            batch_size=BATCH_SIZE,
        )

    print(f"\n[filter_bundle] Done. Filtered bundle: {target_path}")


if __name__ == "__main__":
    main()
