"""
Derivative sidecar tables for BigMetaTable.

`dn/dP` and `dn/dT` ride alongside the main table as parallel memmaps, following the
`blurredbinaries` pattern already in `BigMetaTable`: same row count, same row order,
carried through every operation that changes either.

Sidecars rather than extra columns, for three reasons. The shadows carry a different
column set (76 vs 185), so folding them in would mean ~150 new columns plus a parallel
index scheme anyway; injecting columns breaks `database_headers` compatibility with
every existing CSV and consumer; and `blurredbinaries` already proves the pattern
works here.

Row parity is an importer guarantee (`import_HeFESTo_derivatives` emits one row per
main-table row, NaN-filled where derivatives were unavailable), so the only invariant
these helpers have to maintain is that every row-changing operation touches all three
tables identically. Each helper asserts the shapes match before doing any work -- the
same guard `delete` already applies to `blurredbinaries`.

Scale note: sidecars hold RAW mol/GPa and mol/K, matching the raw fort.99 moles in the
main table. Because `n` and `dn/dP` are both homogeneous of degree 1 in the bulk
amount, any per-row factor applied to the moles applies unchanged to the derivatives,
so the pair stays consistent without special handling. Do not normalise one side.
"""
from __future__ import annotations

import gc
import os
from typing import Iterator, List, Optional, Tuple

import numpy as np

# attribute name on BigMetaTable -> filename suffix written by the importer
SIDECAR_SPECS: Tuple[Tuple[str, str], ...] = (('dndp', '_dndP'), ('dndt', '_dndT'))


def _npy_path(filename: str, suffix: str) -> str:
    return f'{filename}{suffix}.npy'


def _csv_path(filename: str, suffix: str) -> str:
    return f'{filename}{suffix}.csv'


def build_memmap_from_csv(csv_file: str, npy_file: str, chunk_size: int = 200_000):
    """Stream a shadow CSV into a float32 memmap. Returns (path, n_rows, header)."""
    import pandas as pd

    header = list(pd.read_csv(csv_file, nrows=0).columns)
    n_rows = sum(1 for _ in open(csv_file)) - 1
    if n_rows <= 0:
        raise ValueError(f'{csv_file}: no data rows')
    mm = np.lib.format.open_memmap(npy_file, mode='w+', dtype=np.float32,
                                   shape=(n_rows, len(header)))
    at = 0
    for chunk in pd.read_csv(csv_file, chunksize=chunk_size):
        block = chunk.to_numpy(dtype=np.float32)
        mm[at:at + len(block)] = block
        at += len(block)
    mm.flush()
    del mm
    gc.collect()
    return npy_file, at, header


def attach(table, filename: str, memmap_mode: str = 'r+', chunk_size: int = 200_000,
           verbose: bool = True) -> List[str]:
    """Attach `<filename>_dndP` / `_dndT` to `table` as `.dndp` / `.dndt`.

    Prefers an existing `.npy`; otherwise builds one from the CSV the importer wrote.
    Absent sidecars leave the attribute at None, so everything downstream degrades to
    the previous behaviour. Column headers land on `table.dndp_header` / `dndt_header`.

    Called with the original (pre-`_working`) stem, since that is where the importer
    wrote the shadows; `table.filename` is already rewritten to `<name>_working` by
    the time this runs, and every write path below (`apply_mask`, `grow_with_repeats`)
    targets `table.filename` rather than this original stem, so later filtering never
    touches the raw file this attaches from.
    """
    found: List[str] = []
    for attr, suffix in SIDECAR_SPECS:
        setattr(table, attr, None)
        setattr(table, f'{attr}_header', None)
        npy, csv = _npy_path(filename, suffix), _csv_path(filename, suffix)
        if not os.path.exists(npy) and os.path.exists(csv):
            if verbose:
                print(f'[sidecar] building {npy} from {csv}')
            _, _, header = build_memmap_from_csv(csv, npy, chunk_size)
            setattr(table, f'{attr}_header', header)
        if os.path.exists(npy):
            arr = np.load(npy, mmap_mode=memmap_mode)
            if getattr(table, 'table', None) is not None and arr.shape[0] != table.table.shape[0]:
                raise ValueError(
                    f'{npy}: {arr.shape[0]} rows against a main table of '
                    f'{table.table.shape[0]}. Sidecars must be row-parallel; re-run '
                    'the import so both are written from the same pass.')
            setattr(table, attr, arr)
            found.append(attr)
            if verbose:
                print(f'[sidecar] attached {attr}: {arr.shape} from {npy}')
    return found


def items(table) -> Iterator[Tuple[str, str, np.ndarray]]:
    """Yield (attr, suffix, array) for attached sidecars only."""
    for attr, suffix in SIDECAR_SPECS:
        arr = getattr(table, attr, None)
        if arr is not None:
            yield attr, suffix, arr


def assert_aligned(table, where: str) -> None:
    n = table.table.shape[0]
    for attr, _, arr in items(table):
        if arr.shape[0] != n:
            raise AssertionError(
                f'{where}: sidecar {attr} has {arr.shape[0]} rows, main table has {n}.')


def apply_mask(table, keep_mask: np.ndarray, chunk_size: int) -> None:
    """Row-filter every sidecar with the same mask, mirroring `delete`.

    Writes to `table.filename` (already `<name>_working` by the time this runs, same
    as `delete()`'s own `self.memmap_file`), never to the original stem `attach()` read
    from -- so filtering rewrites the working copy, not the raw import output.
    """
    assert_aligned(table, 'apply_mask')
    for attr, suffix, arr in list(items(table)):
        target = _npy_path(table.filename, suffix)
        tmp = target if not os.path.exists(target) else f'{table.filename}{suffix}_temp.npy'
        table._chunked_mask_copy(arr, tmp, keep_mask, chunk_size)
        setattr(table, attr, None)
        del arr
        gc.collect()
        if tmp != target:
            os.replace(tmp, target)
        setattr(table, attr, np.load(target, mmap_mode='r+'))


def split_into(table, new_table, move_mask: np.ndarray, keep_mask: np.ndarray,
               new_filename: str, chunk_size: int) -> None:
    """Write the moved rows onto `new_table` and keep the rest, mirroring `split`."""
    assert_aligned(table, 'split_into')
    for attr, suffix, arr in list(items(table)):
        moved = _npy_path(new_filename, suffix)
        table._chunked_mask_copy(arr, moved, move_mask, chunk_size)
        setattr(new_table, attr, np.load(moved, mmap_mode='r'))
        if new_table.table.shape[0] != getattr(new_table, attr).shape[0]:
            raise AssertionError(f'split: {attr} and main table disagree in the new table')
    apply_mask(table, keep_mask, chunk_size)


def grow_with_repeats(table, new_total: int, old_rows: int, repeat_plan, chunk_size: int):
    """Expand sidecars to `new_total` rows, duplicating the same rows `resample_rare_phase`
    duplicates.

    `repeat_plan` yields `(start, end, chunk_mask, n_resamples)` per chunk, exactly as the
    main loop walks them. No multiplier is applied: abundance resampling is guarded to
    [1, 1] (see `guardrails`), so a duplicated row's derivatives are the originals. If
    that guard is ever lifted, the same per-row factor must be applied here too, because
    `n` and `dn/dP` are both homogeneous of degree 1 in the bulk amount.
    """
    # Checked against the known row counts, not against `table.table`: the caller
    # grows the main table *after* this returns, so mid-operation the main table is
    # still at its old height and a shape comparison would fire spuriously.
    for attr, _, arr in items(table):
        if arr.shape[0] != old_rows:
            raise AssertionError(
                f'grow_with_repeats: sidecar {attr} has {arr.shape[0]} rows, expected '
                f'{old_rows} before growth.')
    handles = []
    for attr, suffix, arr in list(items(table)):
        path = f'{table.filename}{suffix}_resampled.npy'
        new = np.lib.format.open_memmap(path, mode='w+', dtype=arr.dtype,
                                        shape=(new_total, arr.shape[1]))
        table._chunked_copy_into(arr, new, chunk_size=chunk_size)
        handles.append((attr, suffix, path, new))

    write_ptr = old_rows
    for start, end, chunk_mask, n_resamples in repeat_plan:
        if not chunk_mask.any():
            continue
        n_written = None
        for attr, _, _, new in handles:
            src = getattr(table, attr)[start:end][chunk_mask]
            rep = np.repeat(src, n_resamples, axis=0)
            new[write_ptr:write_ptr + len(rep)] = rep
            n_written = len(rep)
        write_ptr += n_written or 0

    for attr, suffix, path, new in handles:
        new.flush()
        del new
        setattr(table, attr, None)
        gc.collect()
        os.replace(path, _npy_path(table.filename, suffix))
        grown = np.load(_npy_path(table.filename, suffix), mmap_mode='r+')
        if grown.shape[0] != new_total:
            raise AssertionError(
                f'grow_with_repeats: sidecar {attr} grew to {grown.shape[0]} rows, '
                f'expected {new_total}.')
        setattr(table, attr, grown)


def save(table, name: str) -> None:
    for attr, suffix, arr in items(table):
        np.save(f'{name}{suffix}.npy', arr)


def component_columns(table, attr: str, indexer) -> dict:
    """(phase, component_name) -> column index into the `attr` sidecar array.

    Mirrors the column layout `HeFESTo_derivative_import.derivative_schema` writes:
    the shadow table's first two columns are the P(GPa)/T(K) keys, followed by every
    (phase, component) placement in ascending order of its column index in the main
    table. Recomputed here rather than imported, so sidecar.py does not pull in the
    importer's heavier dependency chain (HeFESTo_functions, hefesto_vec, ...) just to
    look up column positions.
    """
    skip = {'System_main', 'Bulk_comp', 'Bulk_comp_elements'}
    component_names = {str(x) for x in getattr(indexer, 'label_names', [])}

    placements = []
    for phase_name, phase_map in indexer.MELTS_indices.items():
        if phase_name in skip:
            continue
        for comp_name, idx in phase_map.items():
            if comp_name in component_names:
                placements.append((phase_name, comp_name, int(idx)))
    placements.sort(key=lambda t: t[2])

    return {(phase, comp): 2 + i for i, (phase, comp, _) in enumerate(placements)}


def derivative_molar(table, chunk_slice, phase_label_inds, component_inds):
    """Per-chunk molar-space derivative rows, the twin of
    `molar_chunk[:, phase_label_inds] = component_moles * phase_multipliers`.

    No multiplier: it is guarded to 1. Returns a dict of attr -> array, or {} when no
    sidecars are attached.
    """
    out = {}
    for attr, _, arr in items(table):
        out[attr] = arr[chunk_slice][:, component_inds]
    return out
