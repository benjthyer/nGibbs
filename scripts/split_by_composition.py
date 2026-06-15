"""
Split a MELTStable CSV into train/validation datasets by unique composition.

Unique compositions are identified from columns matching the ``(Bulk_comp)``
pattern.  All rows sharing a composition are kept together — 80 % of
compositions go to the training set, 20 % to validation — so no composition
appears in both splits (preventing data leakage across PT paths).

Usage
-----
    python split_by_composition.py input.csv
    python split_by_composition.py input.csv --out-dir splits/ --seed 7 --val-frac 0.2
    python split_by_composition.py input.csv --train out_train.csv --val out_val.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _bulk_comp_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if "(Bulk_comp)" in c]
    if not cols:
        raise ValueError(
            "No columns matching '(Bulk_comp)' found. "
            "Check that the input file is a MELTStable CSV."
        )
    return cols


def run(
    input_path: Path,
    train_path: Path,
    val_path: Path,
    val_frac: float,
    seed: int,
) -> None:
    print(f"Reading {input_path} ...")
    df = pd.read_csv(input_path)
    print(f"  {len(df):,} rows, {df.shape[1]} columns.")

    comp_cols = _bulk_comp_cols(df)
    print(f"  Composition defined by {len(comp_cols)} column(s): {comp_cols}")

    # Assign each row a composition key (tuple of rounded values avoids float noise)
    comp_key = df[comp_cols].round(8).apply(tuple, axis=1)
    unique_comps = comp_key.unique()
    n_unique = len(unique_comps)
    print(f"  {n_unique} unique composition(s) found.")

    # Shuffle compositions and split
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_comps)
    n_val = max(1, round(n_unique * val_frac))
    n_train = n_unique - n_val

    val_comps = set(map(tuple, shuffled[:n_val]))
    train_comps = set(map(tuple, shuffled[n_val:]))

    is_val = comp_key.apply(tuple).isin(val_comps)
    df_train = df[~is_val].reset_index(drop=True)
    df_val = df[is_val].reset_index(drop=True)

    print(
        f"  Split: {n_train} train composition(s) ({len(df_train):,} rows)"
        f" / {n_val} val composition(s) ({len(df_val):,} rows)"
        f"  ({100*(1-val_frac):.0f}/{100*val_frac:.0f})"
    )

    train_path.parent.mkdir(parents=True, exist_ok=True)
    val_path.parent.mkdir(parents=True, exist_ok=True)

    df_train.to_csv(train_path, index=False)
    print(f"  Train -> {train_path}")

    df_val.to_csv(val_path, index=False)
    print(f"  Val   -> {val_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Split a MELTStable CSV 80/20 by unique bulk composition. "
            "All rows for a given composition stay in the same split."
        )
    )
    p.add_argument("input", help="Path to source MELTStable CSV")
    p.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Output directory. Train/val files are named after the input "
            "stem with _train/_val suffixes. Ignored if --train/--val are given."
        ),
    )
    p.add_argument("--train", default=None, help="Explicit output path for training CSV")
    p.add_argument("--val", default=None, help="Explicit output path for validation CSV")
    p.add_argument(
        "--val-frac",
        type=float,
        default=0.2,
        help="Fraction of compositions to put in validation (default: 0.2)",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    return p


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input)

    if args.train and args.val:
        train_path = Path(args.train)
        val_path = Path(args.val)
    else:
        out_dir = Path(args.out_dir) if args.out_dir else input_path.parent
        stem = input_path.stem
        train_path = out_dir / f"{stem}_train.csv"
        val_path = out_dir / f"{stem}_val.csv"

    if not (0 < args.val_frac < 1):
        print("Error: --val-frac must be between 0 and 1 (exclusive).", file=sys.stderr)
        sys.exit(1)

    run(
        input_path=input_path,
        train_path=train_path,
        val_path=val_path,
        val_frac=args.val_frac,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
