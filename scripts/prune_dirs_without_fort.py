"""Prune directories that do not contain any file with "fort" in the name.

Default behavior is a dry run that prints what would be deleted.
Pass --apply to perform irreversible deletion.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def _contains_fort_file_name(file_name: str) -> bool:
    return "fort" in file_name.lower()


def find_prune_roots(root: Path) -> list[Path]:
    """Return top-level directories under root that can be removed safely.

    A directory is removable if no file in its subtree has "fort" in its name.
    """
    has_fort_in_subtree: dict[Path, bool] = {}

    for dirpath, dirnames, filenames in __import__("os").walk(root, topdown=False):
        current = Path(dirpath)
        local_has_fort = any(_contains_fort_file_name(name) for name in filenames)
        child_has_fort = any(has_fort_in_subtree.get(current / child, False) for child in dirnames)
        has_fort_in_subtree[current] = local_has_fort or child_has_fort

    all_candidates = {
        path
        for path, has_fort in has_fort_in_subtree.items()
        if path != root and not has_fort
    }

    # Keep only top-level candidates to avoid redundant nested deletes.
    prune_roots = [path for path in all_candidates if path.parent not in all_candidates]
    prune_roots.sort()
    return prune_roots


def collect_paths_to_delete(prune_roots: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for root in prune_roots:
        paths.append(root)
        for nested in root.rglob("*"):
            paths.append(nested)
    return sorted(paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Delete directories below a root path when their subtree has no file "
            "with 'fort' in its name."
        )
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Root directory to scan (default: current working directory).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete directories. Default behavior is dry run.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()

    if not root.exists() or not root.is_dir():
        print(f"Error: root path is not a directory: {root}", file=sys.stderr)
        return 2

    prune_roots = find_prune_roots(root)
    paths_to_delete = collect_paths_to_delete(prune_roots)

    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"Mode: {mode}")
    print(f"Scan root: {root}")
    print(f"Directories to delete: {len(prune_roots)}")
    print(f"Filesystem entries to delete: {len(paths_to_delete)}")

    if paths_to_delete:
        print("\nEntries slated for deletion:")
        for path in paths_to_delete:
            print(path)
    else:
        print("\nNo directories matched deletion criteria.")

    if not args.apply:
        print("\nDry run complete. Re-run with --apply to delete these directories.")
        return 0

    failures = 0
    for directory in prune_roots:
        try:
            shutil.rmtree(directory)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"Failed to delete {directory}: {exc}", file=sys.stderr)

    deleted = len(prune_roots) - failures
    print(f"\nDeleted directories: {deleted}")
    if failures:
        print(f"Failed deletions: {failures}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
