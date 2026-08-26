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
import json
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
    """Stream a shadow CSV into a float32 memmap. Returns (path, n_rows, header).

    The array is sized from a LINE count but filled from pandas RECORDS, and those two
    disagree whenever the file carries a blank line (pandas skips them by default), a
    stray fragment, or an embedded newline. The gap used to be left as the zeros
    `open_memmap` initialises -- which read downstream as "this derivative is exactly
    zero", a measured value, rather than as missing data. Verified: one blank line
    produced an array one row too tall with a fabricated zero row at the end.

    So the tail is NaN-filled (the trainer masks on `isfinite`) and any shortfall raises,
    since a shadow table that silently gains rows also silently loses its alignment with
    the main table.
    """
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
    short = n_rows - at
    if short > 0:
        mm[at:] = np.nan          # never zeros: unwritten must not look measured
    mm.flush()
    del mm
    gc.collect()
    if short > 0:
        os.remove(npy_file)
        raise ValueError(
            f'{csv_file}: {n_rows:,} lines but only {at:,} parseable records '
            f'({short:,} short). Blank lines and stray fragments are the usual cause. '
            f'The sidecar would have been {short:,} rows taller than its own data and '
            f'no longer row-parallel with the main table.')
    return npy_file, at, header


def _header_path(filename: str, suffix: str) -> str:
    return f'{filename}{suffix}_header.json'


def _write_header(path: str, header) -> None:
    try:
        with open(path, 'w') as fh:
            json.dump(list(header), fh)
    except OSError:
        pass          # best effort: a missing header is recoverable from the CSV


def _read_header(path: str):
    try:
        with open(path) as fh:
            return list(json.load(fh))
    except (OSError, ValueError):
        return None


def component_columns(table, attr: str, indexer):
    """(phase, component) -> column index in the sidecar `attr`.

    The shadow tables carry a trimmed schema (P, T, then one column per component) whose
    header strings are the main table's, e.g. `forsterite(olivine)`. Resolving by header
    rather than by position is what keeps this correct when the component set changes --
    and, more sharply, it is keyed on BOTH phase and component because component names are
    NOT unique: `magnetite` appears under both `spinel` and `ferropericlase`. Keying on the
    name alone silently maps two components onto one column.
    """
    header = getattr(table, f'{attr}_header', None)
    if header is None:
        raise ValueError(
            f'sidecar {attr} has no column header; rebuild it from the CSV so '
            f'{_header_path(base_of(table), dict(SIDECAR_SPECS)[attr])} exists.')
    pos = {}
    for j, name in enumerate(header):
        pos.setdefault(str(name), j)
    out = {}
    for phase, comp_map in indexer.MELTS_indices.items():
        if phase in ('System_main', 'Bulk_comp', 'Bulk_comp_elements'):
            continue
        for comp, _main_idx in comp_map.items():
            key = f'{comp}({phase})'
            if key in pos:
                out[(phase, comp)] = pos[key]
    return out


def base_of(table) -> str:
    """Filename stem the sidecars live under.

    `BigMetaTable.__init__` rewrites `self.filename` to `<name>_working`, but the
    importer wrote the shadows under the original stem, so the base is recorded at
    attach time rather than re-derived later.
    """
    return getattr(table, '_sidecar_base', None) or table.filename


def attach(table, filename: str, memmap_mode: str = 'r+', chunk_size: int = 200_000,
           verbose: bool = True) -> List[str]:
    """Attach `<filename>_dndP` / `_dndT` to `table` as `.dndp` / `.dndt`.

    Prefers an existing `.npy`; otherwise builds one from the CSV the importer wrote.
    Absent sidecars leave the attribute at None, so everything downstream degrades to
    the previous behaviour. Column headers land on `table.dndp_header` / `dndt_header`.
    """
    table._sidecar_base = filename
    found: List[str] = []
    for attr, suffix in SIDECAR_SPECS:
        setattr(table, attr, None)
        setattr(table, f'{attr}_header', None)
        npy, csv = _npy_path(filename, suffix), _csv_path(filename, suffix)
        hdr_path = _header_path(filename, suffix)
        if not os.path.exists(npy) and os.path.exists(csv):
            if verbose:
                print(f'[sidecar] building {npy} from {csv}')
            _, _, header = build_memmap_from_csv(csv, npy, chunk_size)
            setattr(table, f'{attr}_header', header)
            _write_header(hdr_path, header)
        elif os.path.exists(npy):
            # A previously-built .npy carries no column names, and the CSV it came from is
            # often deleted. Without the header the shadow is an anonymous float block and
            # nothing downstream can say which column is which component -- so the header
            # is persisted next to it at build time and read back here.
            header = _read_header(hdr_path)
            if header is None and os.path.exists(csv):
                import pandas as pd
                header = list(pd.read_csv(csv, nrows=0).columns)
                _write_header(hdr_path, header)
            setattr(table, f'{attr}_header', header)
        if os.path.exists(npy):
            arr = np.load(npy, mmap_mode=memmap_mode)
            if getattr(table, 'table', None) is not None and arr.shape[0] != table.table.shape[0]:
                raise ValueError(
                    f'{npy}: {arr.shape[0]} rows against a main table of '
                    f'{table.table.shape[0]}. The two imports admitted different '
                    f'simulations. Realign without re-importing:\n'
                    f'    python scripts/repair_derivative_alignment.py '
                    f'{base_of(table)}.csv')
            _assert_pt_agrees(table, arr, getattr(table, f'{attr}_header', None), npy)
            setattr(table, attr, arr)
            found.append(attr)
            if verbose:
                print(f'[sidecar] attached {attr}: {arr.shape} from {npy}')
    return found


def _assert_pt_agrees(table, arr, header, npy, samples: int = 20000, tol: float = 1e-6):
    """Verify the sidecar describes the SAME rows, not merely the same number of them.

    A matching row count proves nothing. If the main import dropped two simulations and
    the derivative import dropped two *different* ones, the lengths agree, `attach`
    is satisfied, and every row's derivative belongs to some other row -- which trains,
    converges, and is wrong. That is exactly why the importer carries `P(GPa)` and
    `T(K)` into the shadow tables; this is the check they exist for.

    Sampled rather than exhaustive: any whole-simulation offset shifts a contiguous block
    of hundreds to thousands of rows, so an evenly spaced sample of 20k catches it with
    certainty while costing nothing on a 10M-row table.
    """
    if header is None or getattr(table, 'indexer', None) is None:
        return                       # nothing to compare against; row count is all we have
    sysmap = getattr(table.indexer, 'MELTS_indices', {}).get('System_main', {})
    pairs = []
    for attr_name in ('P(GPa)', 'T(K)'):
        main_col = sysmap.get(attr_name)
        full = f'{attr_name}(System_main)'
        if main_col is None or full not in header:
            return
        pairs.append((int(main_col), header.index(full)))

    n = arr.shape[0]
    idx = np.unique(np.linspace(0, n - 1, min(samples, n)).astype(np.int64))
    for main_col, side_col in pairs:
        a = np.asarray(table.table[idx, main_col], dtype=np.float64)
        b = np.asarray(arr[idx, side_col], dtype=np.float64)
        bad = ~(np.isclose(a, b, atol=tol, rtol=0) | (~np.isfinite(b)))
        if bad.any():
            first = int(idx[np.argmax(bad)])
            raise ValueError(
                f'{npy}: row counts match but the rows do not. At main row {first}, the '
                f'table has {a[np.argmax(bad)]:.6g} and the sidecar has '
                f'{b[np.argmax(bad)]:.6g} for the same column '
                f'({100 * bad.mean():.1f}% of sampled rows disagree).\n'
                f'The two imports admitted different simulations. Realign without '
                f're-importing:\n'
                f'    python scripts/repair_derivative_alignment.py {base_of(table)}.csv')


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
    """Row-filter every sidecar with the same mask, mirroring `delete`."""
    assert_aligned(table, 'apply_mask')
    for attr, suffix, arr in list(items(table)):
        target = _npy_path(base_of(table), suffix)
        tmp = target if not os.path.exists(target) else f'{base_of(table)}{suffix}_temp.npy'
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
        path = f'{base_of(table)}{suffix}_resampled.npy'
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
        os.replace(path, _npy_path(base_of(table), suffix))
        grown = np.load(_npy_path(base_of(table), suffix), mmap_mode='r+')
        if grown.shape[0] != new_total:
            raise AssertionError(
                f'grow_with_repeats: sidecar {attr} grew to {grown.shape[0]} rows, '
                f'expected {new_total}.')
        setattr(table, attr, grown)


def save(table, name: str) -> None:
    for attr, suffix, arr in items(table):
        np.save(f'{name}{suffix}.npy', arr)


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
