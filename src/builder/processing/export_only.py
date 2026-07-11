"""Export a single MELTS table to an ML bundle without preprocessing."""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

# Ensure repo root and src are on path
repo_root = str(Path(__file__).resolve().parents[3])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

src_path = str(Path(__file__).parent.parent.parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from builder.processing.BigMetaTable import BigMetaTable
from builder.processing.MLexporter import resampling_to_datasets
from ngibbs.utils.file_utils import get_baseline_files, clear_new_files


def _normalize_base_path(input_path: str) -> str:
    path_obj = Path(input_path)
    if path_obj.suffix.lower() in {".csv", ".txt", ".npy"}:
        path_obj = path_obj.with_suffix("")
    return str(path_obj)


def _close_memmaps(obj) -> int:
    memmap_attrs = [
        "table",
        "features",
        "labels",
        "binarylabels",
        "masslabels",
        "molarlabels",
        "free_outputs",
        "molar",
        "table1",
        "blurredbinaries",
    ]
    closed_count = 0
    for attr in memmap_attrs:
        if hasattr(obj, attr):
            try:
                delattr(obj, attr)
                closed_count += 1
            except Exception:
                pass
    return closed_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load BigMetaTable from csv/txt and export one ML bundle."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to table base name or to .csv/.txt/.npy file.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Output bundle base name (without .tar.gz). Defaults to input base name.",
    )
    parser.add_argument(
        "--read-dir",
        type=str,
        default=None,
        help="Optional source directory used by BigMetaTable(read_dir=...).",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Dry-run cleanup: print temp files but do not delete them.",
    )

    args = parser.parse_args()

    bmt = None
    baseline_files = None
    work_dir = None
    try:
        base_path = _normalize_base_path(args.input_path)
        name = args.name or Path(base_path).name
        bundle_name = f"{name}.tar.gz"
        work_dir = Path(base_path).resolve().parent
        baseline_files = get_baseline_files(work_dir)

        print(f"Loading BigMetaTable from: {base_path}")
        bmt = BigMetaTable(base_path, read_dir=args.read_dir)
        bmt.indexer.table_update(bmt.table) 

        print(f"Exporting bundle: {bundle_name}")
        bundle_path = resampling_to_datasets(
            bmt,
            [[1, 1]],
            bundle_name=bundle_name,
        )
        print(f"Export complete: {bundle_path}")
    finally:
        if bmt is not None:
            closed = _close_memmaps(bmt)
            del bmt
            gc.collect()
            print(f"Closed memory-map attrs: {closed}")
        if work_dir is not None and baseline_files is not None:
            deleted_count = clear_new_files(
                work_dir,
                baseline_files,
                protected_extensions=['.tar.gz'],
                dry_run=args.keep_temp,
            )
            if args.keep_temp:
                print(f"Temporary files that would be removed: {deleted_count}")
            else:
                print(f"Removed temporary files: {deleted_count}")


if __name__ == "__main__":
    main()
