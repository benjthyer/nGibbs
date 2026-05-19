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
  ml_indexer/        directory with featureNames / freeOutputs stored in metadata

The swap:
  - column ``demote`` in features  → replaces matching column in free_outputs
  - column ``promote`` in free_outputs → replaces matching column in features
  - ml_indexer.featureNames / freeOutputs are updated to match

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

from nMELTS.config.ml_indexer import load_ml_indexer_from_state


def _norm(name: str) -> str:
    return name.strip()


def remap_bundle(input_path: Path, demote: str, promote: str, output_path: Path) -> None:
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
        Name currently in freeOutputs to move into features.
        Format: 'Component(Phase)'  e.g. 'Temperature(System_main)'
    output_path : Path
        Destination .tar.gz path (created or overwritten).
    """
    tmp = Path(tempfile.mkdtemp(prefix="remap_bundle_"))
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
                "Rebuild the bundle with the variable you want to promote listed in freeOutputs."
            )
        free_outputs = np.load(free_outputs_path)

        # Load indexer to get current name lists
        indexer = load_ml_indexer_from_state(tmp / 'ml_indexer')
        feature_names = list(indexer.featureNames or [])
        free_output_names = list(indexer.freeOutputs or [])

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
                f"'{promote}' not in freeOutputs: {free_output_names}\n"
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
        indexer.freeOutputs = new_free_names

        # Write updated arrays back to temp dir
        np.save(tmp / 'features.npy', new_features)
        np.save(tmp / 'free_outputs.npy', new_free)

        # Overwrite ml_indexer directory with updated metadata
        indexer_dir = tmp / 'ml_indexer'
        shutil.rmtree(indexer_dir, ignore_errors=True)
        indexer.save(str(indexer_dir))

        # Repack everything into the output bundle
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(output_path, 'w:gz') as tar:
            for item in sorted(tmp.iterdir()):
                tar.add(item, arcname=item.name)

        print(f"  -> {output_path.name}")
        print(f"     features    : {new_feature_names}")
        print(f"     freeOutputs : {new_free_names}")

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
        help="Free-output name (in freeOutputs) to move into features. "
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

    for src in sources:
        # Derive output path
        if args.output:
            if is_stem:
                # Preserve the split suffix (_Train / _Test / _Valid)
                for tag in ('_Train', '_Test', '_Valid'):
                    if src.name.endswith(tag + '.tar.gz'):
                        dst = Path(args.output + tag + '.tar.gz')
                        break
                else:
                    dst = Path(args.output + '.tar.gz')
            else:
                dst = Path(args.output)
                if not str(dst).endswith('.tar.gz'):
                    dst = Path(str(dst) + '.tar.gz')
        else:
            stem = src.name.replace('.tar.gz', '')
            dst = src.parent / f"{stem}_remapped.tar.gz"

        print(f"Processing: {src.name}")
        remap_bundle(src, args.demote, args.promote, dst)

    print("\nDone.")


if __name__ == '__main__':
    main()
