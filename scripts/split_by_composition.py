"""
Split a MELTStable table into train/validation datasets by unique composition.

Unique compositions are identified from columns matching the ``(Bulk_comp)``
pattern.  All rows sharing a composition are kept together — 80 % of
compositions go to the training set, 20 % to validation — so no composition
appears in both splits (preventing data leakage across PT paths).

The main table may be a `.csv` or a `.npy` (memmap-backed, matching how
BigMetaTable stores large tables). A `.npy` carries no column names, so its
header is read from a sibling `<stem>.csv` (header row only) or an explicit
`--header-csv`. By default, splits are written as `.npy` plus a header-only
`.csv` companion for each - regardless of the input's format - matching
merge_bigmetatables.py's 'header' csv-output convention. Pass an explicit
`--train`/`--val` path ending in `.csv` to write a full CSV split instead.

Derivative sidecars (`<stem>_dndP.npy`/`.csv`, `<stem>_dndT.npy`/`.csv`) next to
the input file are picked up automatically, mirroring merge_bigmetatables.py's
handling of BigMetaTable's dn/dP, dn/dT sidecars. Each found sidecar is split
with the exact same row mask as the main table (row-parallel, same order), and
written alongside the train/val splits with the sidecar's own suffix and format.

Usage
-----
    python split_by_composition.py input.csv
    python split_by_composition.py input.csv --out-dir splits/ --seed 7 --val-frac 0.2
    python split_by_composition.py input.csv --train out_train.csv --val out_val.csv
    python split_by_composition.py input.npy
    python split_by_composition.py input.npy --header-csv input_header.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

repo_src = str(Path(__file__).resolve().parents[1] / "src")
if repo_src not in sys.path:
    sys.path.insert(0, repo_src)

from builder.processing import sidecar


def _bulk_comp_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if "(Bulk_comp)" in c]
    if not cols:
        raise ValueError(
            "No columns matching '(Bulk_comp)' found. "
            "Check that the input file is a MELTStable CSV."
        )
    return cols


def _resolve_main_path(path: Path) -> Path:
    """Resolve `path` to an existing `.csv` or `.npy` main table.

    An extensionless `path` prefers `.npy`, mirroring `_find_sidecar`'s lookup order.
    """
    if path.suffix in (".csv", ".npy"):
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist.")
        return path
    npy_path = path.with_suffix(".npy")
    if npy_path.exists():
        return npy_path
    csv_path = path.with_suffix(".csv")
    if csv_path.exists():
        return csv_path
    raise FileNotFoundError(f"Neither {npy_path} nor {csv_path} exists.")


def _resolve_header(npy_path: Path, header_csv: Path | None) -> list[str]:
    """Column names for a `.npy` main table: from `--header-csv`, else a sibling
    `<stem>.csv` (header row only), matching BigMetaTable's own header lookup."""
    sibling_csv = npy_path.with_suffix(".csv")
    for candidate in (header_csv, sibling_csv):
        if candidate is not None and candidate.exists():
            return list(pd.read_csv(candidate, nrows=0).columns)
    raise ValueError(
        f"Cannot determine column headers for {npy_path}: pass --header-csv, or "
        f"place a sibling {sibling_csv} (a header row is enough)."
    )


def _load_main_table(input_path: Path, header_csv: Path | None) -> pd.DataFrame:
    if input_path.suffix == ".npy":
        arr = np.load(input_path, mmap_mode="r")
        header = _resolve_header(input_path, header_csv)
        if len(header) != arr.shape[1]:
            raise ValueError(
                f"Header ({len(header)} columns) does not match {input_path} "
                f"({arr.shape[1]} columns)."
            )
        return pd.DataFrame(np.asarray(arr), columns=header)
    return pd.read_csv(input_path)


def _write_table(df: pd.DataFrame, path: Path) -> None:
    """Write a main-table split as `.csv`, or as `.npy` plus a header-only `.csv`
    companion (matching merge_bigmetatables.py's 'header' csv-output convention)."""
    if path.suffix == ".npy":
        np.save(path, df.to_numpy(dtype=np.float32))
        with open(path.with_suffix(".csv"), "w", newline="") as f:
            csv.writer(f).writerow(df.columns.tolist())
    else:
        df.to_csv(path, index=False)


def _find_sidecar(input_path: Path, suffix: str) -> Path | None:
    """Locate a `<stem><suffix>.npy`/`.csv` sidecar next to the input CSV.

    Prefers the .npy, mirroring `sidecar.attach`'s preference order.
    """
    base = input_path.parent / input_path.stem
    npy_path = Path(f"{base}{suffix}.npy")
    if npy_path.exists():
        return npy_path
    csv_path = Path(f"{base}{suffix}.csv")
    if csv_path.exists():
        return csv_path
    return None


def _split_sidecar(
    path: Path,
    is_val: pd.Series,
    n_main: int,
    attr: str,
    suffix: str,
    train_path: Path,
    val_path: Path,
) -> None:
    """Split one dn/dP or dn/dT sidecar with the same row mask as the main table."""
    is_val_arr = is_val.to_numpy()
    train_out = train_path.with_name(f"{train_path.stem}{suffix}{path.suffix}")
    val_out = val_path.with_name(f"{val_path.stem}{suffix}{path.suffix}")

    if path.suffix == ".npy":
        arr = np.load(path, mmap_mode="r")
        if arr.shape[0] != n_main:
            raise ValueError(
                f"Sidecar '{attr}' ({path}) has {arr.shape[0]} rows, main table has "
                f"{n_main}. Sidecars must be row-parallel with the input CSV."
            )
        np.save(train_out, arr[~is_val_arr])
        np.save(val_out, arr[is_val_arr])
    else:
        sdf = pd.read_csv(path)
        if len(sdf) != n_main:
            raise ValueError(
                f"Sidecar '{attr}' ({path}) has {len(sdf)} rows, main table has "
                f"{n_main}. Sidecars must be row-parallel with the input CSV."
            )
        sdf[~is_val_arr].reset_index(drop=True).to_csv(train_out, index=False)
        sdf[is_val_arr].reset_index(drop=True).to_csv(val_out, index=False)

    print(f"  Sidecar '{attr}' -> {train_out}, {val_out}")


def run(
    input_path: Path,
    train_path: Path,
    val_path: Path,
    val_frac: float,
    seed: int,
    header_csv: Path | None = None,
) -> None:
    print(f"Reading {input_path} ...")
    df = _load_main_table(input_path, header_csv)
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

    _write_table(df_train, train_path)
    print(f"  Train -> {train_path}")

    _write_table(df_val, val_path)
    print(f"  Val   -> {val_path}")

    for attr, suffix in sidecar.SIDECAR_SPECS:
        sc_path = _find_sidecar(input_path, suffix)
        if sc_path is None:
            print(f"  No sidecar '{attr}' ({suffix}) found next to {input_path.name}; skipping.")
            continue
        print(f"  Found sidecar '{attr}' at {sc_path}")
        _split_sidecar(sc_path, is_val, len(df), attr, suffix, train_path, val_path)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Split a MELTStable CSV 80/20 by unique bulk composition. "
            "All rows for a given composition stay in the same split."
        )
    )
    p.add_argument(
        "input",
        help=(
            "Path to the source MELTStable table: a .csv, a .npy, or an "
            "extensionless base name (resolved preferring .npy, then .csv)."
        ),
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Output directory. Train/val files are named after the input stem with "
            "_train/_val suffixes and written as .npy (plus a header-only .csv "
            "companion). Ignored if --train/--val are given."
        ),
    )
    p.add_argument(
        "--train",
        default=None,
        help=(
            "Explicit output path for the training split. A .npy path writes a "
            "header-only .csv companion alongside it; a .csv path writes a full CSV "
            "split instead."
        ),
    )
    p.add_argument(
        "--val",
        default=None,
        help=(
            "Explicit output path for the validation split. A .npy path writes a "
            "header-only .csv companion alongside it; a .csv path writes a full CSV "
            "split instead."
        ),
    )
    p.add_argument(
        "--header-csv",
        default=None,
        help=(
            "CSV whose header row supplies column names for a .npy main table "
            "(a .npy carries no column names itself). Defaults to a sibling "
            "<stem>.csv next to the .npy if not given."
        ),
    )
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
    input_path = _resolve_main_path(Path(args.input))
    header_csv = Path(args.header_csv) if args.header_csv else None

    if args.train and args.val:
        train_path = Path(args.train)
        val_path = Path(args.val)
    else:
        out_dir = Path(args.out_dir) if args.out_dir else input_path.parent
        stem = input_path.stem
        train_path = out_dir / f"{stem}_train.npy"
        val_path = out_dir / f"{stem}_val.npy"

    if not (0 < args.val_frac < 1):
        print("Error: --val-frac must be between 0 and 1 (exclusive).", file=sys.stderr)
        sys.exit(1)

    run(
        input_path=input_path,
        train_path=train_path,
        val_path=val_path,
        val_frac=args.val_frac,
        seed=args.seed,
        header_csv=header_csv,
    )


if __name__ == "__main__":
    main()
