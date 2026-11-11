#!/usr/bin/env python3
"""
Merge MLready bundles (.tar.gz) that share a schema into one plain training bundle.

What must match across bundles (checked; any mismatch is an error):
  * features: featureNames and Elkeys (same columns, same order)
  * labels: the compositional label columns (component name, owning phase, order)
  * compositionally variable phases, free_outputs, and per-array dtypes

What may differ:
  * The phase set. The merged bundle uses the reference bundle's ml_indexer, whose
    phases must be a superset of every other bundle's. The phase-indexed arrays
    (binary_labels, molar_labels, mass_labels) are re-mapped by phase name, and a phase
    a bundle lacks is zero-filled (absent) for that bundle's rows. Because that asserts
    "absent" for data that never modelled the phase, it is refused unless
    --zero-fill-missing-phases is given.
  * Optional row-aligned arrays. Only arrays present in EVERY bundle are merged; the
    rest are dropped. This is the "downgrade" of an affinity bundle: source_rows,
    g_labels, s_labels, affinity_labels.json and lambda.json are not carried over.

The merged rows are concatenated in --bundles order, then shuffled (one permutation for
every array) unless --no-shuffle. run_ids are offset per bundle so ids from different
bundles never collide. stats.txt, feature_bounds.json and T0.npy are regenerated from
the merged data. A merge_manifest.json records sources, row counts, id offsets, dropped
arrays and zero-filled phases; each source's recipe .yaml is kept under merge_sources/.

Usage:
    python scripts/merge_ml_bundles.py \\
        --bundles data/MLready/HeFESTo/HeFESTo_EarthAffinity_NPT_Train.tar.gz \\
                  data/MLready/HeFESTo/HeFESTo_EarthReSamp_NPT_Train.tar.gz \\
        --output  data/MLready/HeFESTo/HeFESTo_EarthCombined_NPT_Train.tar.gz \\
        --zero-fill-missing-phases
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / 'src'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ngibbs.config.ml_indexer import load_ml_indexer_from_state  # noqa: E402
from ngibbs.utils.file_utils import (  # noqa: E402
    ROW_ALIGNED_BUNDLE_ARRAYS, chunked_permutation_copy, compute_T0,
)

# Columns indexed by ml_indexer.mass_phasedict (one per phase).
PHASE_ARRAYS = ('binary_labels.npy', 'molar_labels.npy', 'mass_labels.npy')
# Files regenerated or written by this script rather than copied from a source.
REGENERATED = {'stats.txt', 'feature_bounds.json', 'T0.npy', 'ml_indexer', 'merge_manifest.json'}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--bundles', nargs='+', required=True, help='Bundles (.tar.gz) to merge, in order.')
    p.add_argument('--output', required=True, help='Output bundle path (.tar.gz).')
    p.add_argument('--reference', type=int, default=None,
                   help='Index into --bundles of the bundle whose ml_indexer the output uses. '
                        'Default: the bundle with the most phases (ties: the first).')
    p.add_argument('--zero-fill-missing-phases', action='store_true',
                   help='Allow bundles lacking some reference phases; those phases are written '
                        'as absent (0) in binary/molar/mass labels for that bundle\'s rows.')
    p.add_argument('--no-shuffle', action='store_true', help='Keep rows in concatenation order.')
    p.add_argument('--seed', type=int, default=0, help='Shuffle seed (default 0).')
    p.add_argument('--no-T0', action='store_true', help='Do not compute T0.npy for the output.')
    p.add_argument('--chunk-size', type=int, default=1_000_000)
    p.add_argument('--compresslevel', type=int, default=1, help='gzip level (default 1, as the exporter).')
    p.add_argument('--workdir', default=None,
                   help='Scratch directory (default: a temp dir under data/tmp). Needs roughly the '
                        'uncompressed size of all inputs plus the output.')
    p.add_argument('--keep-workdir', action='store_true')
    p.add_argument('--overwrite', action='store_true', help='Replace --output if it exists.')
    return p.parse_args()


def _extract(bundle: Path, dest: Path) -> list[str]:
    print(f'[merge] extracting {bundle.name} -> {dest}')
    dest.mkdir(parents=True)
    with tarfile.open(bundle, 'r:gz') as tar:
        names = tar.getnames()
        tar.extractall(dest, filter='data')
    return names


def _label_columns(ix) -> list[tuple]:
    owner = {c: p for p, cs in ix.label_indices.items() for c in cs}
    return [(ix.label_names[i], owner.get(i)) for i in ix.compositional_component_subset]


def _check_schema(ref, ix, label: str, allow_fill: bool) -> list[str]:
    """Raise on any schema mismatch; return the reference phases this bundle lacks."""
    errs = []
    for att in ('featureNames', 'Elkeys', 'free_outputs'):
        if list(getattr(ref, att) or []) != list(getattr(ix, att) or []):
            errs.append(f'{att}: {getattr(ix, att)} != reference {getattr(ref, att)}')
    if list(ref.compositionally_variable_phases) != list(ix.compositionally_variable_phases):
        errs.append('compositionally_variable_phases differ')
    if _label_columns(ref) != _label_columns(ix):
        errs.append('label columns (component, phase, order) differ')
    extra = [p for p in ix.all_phases if p not in ref.all_phases]
    if extra:
        errs.append(f'phases not in the reference indexer: {extra} (pick a superset --reference)')
    missing = [p for p in ref.all_phases if p not in ix.all_phases]
    if missing and not allow_fill:
        errs.append(f'lacks reference phases {missing}; pass --zero-fill-missing-phases to write '
                    f'them as absent for this bundle')
    if errs:
        raise ValueError(f'{label} does not match the reference schema:\n  ' + '\n  '.join(errs))
    return missing


def main() -> None:
    args = parse_args()
    bundles = [Path(b) for b in args.bundles]
    out = Path(args.output)
    if len(bundles) < 2:
        sys.exit('ERROR: --bundles needs at least two bundles.')
    for b in bundles:
        if not b.is_file():
            sys.exit(f'ERROR: bundle not found: {b}')
    if not out.name.endswith('.tar.gz'):
        sys.exit('ERROR: --output must end in .tar.gz')
    if out.exists() and not args.overwrite:
        sys.exit(f'ERROR: {out} exists (pass --overwrite to replace it).')
    if out.resolve() in {b.resolve() for b in bundles}:
        sys.exit('ERROR: --output must not be one of the input bundles.')

    if args.workdir:
        work = Path(args.workdir)
        work.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(dir=work, prefix='merge_ml_bundles_'))
    else:
        tmp_base = REPO_ROOT / 'data' / 'tmp'
        tmp_base.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(dir=tmp_base, prefix='merge_ml_bundles_'))
    t_start = time.time()
    try:
        # ---- extract and load indexers --------------------------------------
        src_dirs, members = [], []
        for i, b in enumerate(bundles):
            d = work / f'src{i}'
            members.append(_extract(b, d))
            src_dirs.append(d)
        indexers = [load_ml_indexer_from_state(str(d / 'ml_indexer')) for d in src_dirs]
        labels = [f'[{i}] {b.name}' for i, b in enumerate(bundles)]

        ref_i = args.reference
        if ref_i is None:
            ref_i = max(range(len(indexers)), key=lambda i: (len(indexers[i].all_phases), -i))
        if not 0 <= ref_i < len(bundles):
            sys.exit(f'ERROR: --reference {ref_i} out of range.')
        ref = indexers[ref_i]
        print(f'[merge] reference indexer: {labels[ref_i]} ({len(ref.all_phases)} phases)')

        missing_by = {}
        try:
            for ix, lab in zip(indexers, labels):
                missing_by[lab] = _check_schema(ref, ix, lab, args.zero_fill_missing_phases)
        except ValueError as exc:
            sys.exit(f'ERROR: {exc}')
        for lab, miss in missing_by.items():
            if miss:
                print(f'[merge] WARNING: {lab} lacks {miss}; written as absent (0) for its rows.')

        # ---- which arrays to merge ------------------------------------------
        present = [{nm for nm in ROW_ALIGNED_BUNDLE_ARRAYS if (d / nm).exists()} for d in src_dirs]
        merged_arrays = [nm for nm in ROW_ALIGNED_BUNDLE_ARRAYS if all(nm in p for p in present)]
        for nm in ('features.npy', 'labels.npy') + PHASE_ARRAYS:
            if nm not in merged_arrays:
                sys.exit(f'ERROR: {nm} is missing from at least one bundle.')
        dropped = {lab: sorted(p - set(merged_arrays)) for lab, p in zip(labels, present)}
        top_files = [{Path(n).parts[0] for n in m} for m in members]
        dropped_files = {lab: sorted(t - set(merged_arrays) - REGENERATED
                                     - {n for n in t if n.endswith('.yaml')})
                         for lab, t in zip(labels, top_files)}
        for lab in labels:
            gone = dropped[lab] + dropped_files[lab]
            if gone:
                print(f'[merge] {lab}: not carried over: {sorted(set(gone))}')

        rows = []
        for d, lab in zip(src_dirs, labels):
            ns = {nm: np.load(d / nm, mmap_mode='r').shape[0] for nm in merged_arrays}
            if len(set(ns.values())) != 1:
                sys.exit(f'ERROR: {lab} arrays are not row-aligned: {ns}')
            rows.append(next(iter(ns.values())))
        total = sum(rows)
        print(f'[merge] rows: ' + ' + '.join(f'{n:,}' for n in rows) + f' = {total:,}')

        # ---- run_id offsets ---------------------------------------------------
        run_offsets = [0] * len(bundles)
        if 'run_ids.npy' in merged_arrays:
            off = 0
            for i, d in enumerate(src_dirs):
                r = np.load(d / 'run_ids.npy', mmap_mode='r')
                run_offsets[i] = off
                mx = max((int(np.max(r[a:a + args.chunk_size])) for a in range(0, len(r), args.chunk_size)),
                         default=-1)
                off += mx + 1
            dt = np.load(src_dirs[0] / 'run_ids.npy', mmap_mode='r').dtype
            if np.issubdtype(dt, np.integer) and off > np.iinfo(dt).max:
                sys.exit(f'ERROR: offset run_ids would overflow {dt}.')

        # ---- concatenate --------------------------------------------------------
        out_dir = work / 'merged'
        out_dir.mkdir()
        ref_cols = ref.mass_phasedict
        for nm in merged_arrays:
            srcs = [np.load(d / nm, mmap_mode='r') for d in src_dirs]
            dtypes = {s.dtype for s in srcs}
            if len(dtypes) != 1:
                sys.exit(f'ERROR: {nm} dtype differs between bundles: {dtypes}')
            if nm in PHASE_ARRAYS:
                width = len(ref.all_phases)
                for s, ix, lab in zip(srcs, indexers, labels):
                    if s.ndim != 2 or s.shape[1] != len(ix.all_phases):
                        sys.exit(f'ERROR: {lab} {nm} has shape {s.shape}, expected (n, {len(ix.all_phases)}).')
            else:
                widths = {s.shape[1:] for s in srcs}
                if len(widths) != 1:
                    sys.exit(f'ERROR: {nm} column shape differs between bundles: {widths}')
            shape = (total,) + ((width,) if nm in PHASE_ARRAYS else srcs[0].shape[1:])
            dst = np.lib.format.open_memmap(out_dir / nm, mode='w+', dtype=srcs[0].dtype, shape=shape)
            at = 0
            for i, (s, ix, lab) in enumerate(zip(srcs, indexers, labels)):
                n = s.shape[0]
                if nm in PHASE_ARRAYS:
                    src_idx = [ix.mass_phasedict[p] for p in ix.all_phases]
                    dst_idx = [ref_cols[p] for p in ix.all_phases]
                    fill = [ref_cols[p] for p in ref.all_phases if p not in ix.mass_phasedict]
                for a in tqdm(range(0, n, args.chunk_size), desc=f'{nm} {lab}', leave=False):
                    b = min(a + args.chunk_size, n)
                    block = np.asarray(s[a:b])
                    if nm in PHASE_ARRAYS:
                        o = np.zeros((b - a, width), dtype=block.dtype)
                        o[:, dst_idx] = block[:, src_idx]
                        if fill:
                            o[:, fill] = 0
                        block = o
                    elif nm == 'run_ids.npy' and run_offsets[i]:
                        block = block + np.asarray(run_offsets[i], dtype=block.dtype)
                    dst[at + a:at + b] = block
                at += n
            dst.flush()
            del dst, srcs
            gc.collect()
            print(f'[merge] wrote {nm} {shape}')
        # Sources no longer needed: free their disk before shuffling.
        for d in src_dirs:
            for nm in merged_arrays:
                (d / nm).unlink(missing_ok=True)

        # ---- shuffle ---------------------------------------------------------------
        if not args.no_shuffle:
            perm = np.random.default_rng(args.seed).permutation(total)
            for k, nm in enumerate(merged_arrays, 1):
                src = np.load(out_dir / nm, mmap_mode='r')
                tmp = out_dir / f'_shuffled_{nm}'
                chunked_permutation_copy(src, tmp, perm, chunk_size=args.chunk_size,
                                         desc=f'[{k}/{len(merged_arrays)}] shuffle {nm}')
                del src
                gc.collect()
                tmp.replace(out_dir / nm)
            del perm

        # ---- indexer, T0, stats -------------------------------------------------------
        shutil.copytree(src_dirs[ref_i] / 'ml_indexer', out_dir / 'ml_indexer')
        if not args.no_T0:
            molar = np.load(out_dir / 'molar_labels.npy', mmap_mode='r')
            T0 = compute_T0(molar, ref.mass_phasedict, ref.all_phases)
            np.save(out_dir / 'T0.npy', np.asarray(T0, dtype=np.float32))
            del molar
            print(f'[merge] T0 computed from {total:,} merged rows.')
        from builder.processing.MLexporter import generate_dataset_stats
        stats_path = Path(generate_dataset_stats(dataset_name=str(out_dir) + '/', ml_indexer=ref,
                                                 output_dir=out_dir, chunk_size=args.chunk_size))
        stats_path.replace(out_dir / 'stats.txt')
        (out_dir / f'{out_dir.name}_feature_bounds.json').replace(out_dir / 'feature_bounds.json')

        # ---- provenance ------------------------------------------------------------------
        srcs_dir = out_dir / 'merge_sources'
        srcs_dir.mkdir()
        for i, d in enumerate(src_dirs):
            for y in d.glob('*.yaml'):
                shutil.copy2(y, srcs_dir / f'{i}_{Path(bundles[i].name).name.removesuffix(".tar.gz")}_{y.name}')
        manifest = dict(
            sources=[dict(bundle=str(b), rows=n, row_range_before_shuffle=[s, s + n],
                          run_id_offset=o, phases_zero_filled=missing_by[lab],
                          not_carried_over=sorted(set(dropped[lab] + dropped_files[lab])))
                     for b, n, s, o, lab in zip(bundles, rows, np.cumsum([0] + rows[:-1]).tolist(),
                                                run_offsets, labels)],
            reference_indexer=str(bundles[ref_i]), total_rows=total, arrays=merged_arrays,
            shuffled=not args.no_shuffle, seed=None if args.no_shuffle else args.seed,
            T0_computed=not args.no_T0,
        )
        (out_dir / 'merge_manifest.json').write_text(json.dumps(manifest, indent=1))

        # ---- pack ---------------------------------------------------------------------------
        tmp_out = out.with_name(out.name + '.partial')
        print(f'[merge] packing {out} ...')
        with tarfile.open(tmp_out, 'w:gz', compresslevel=args.compresslevel) as tar:
            for nm in merged_arrays + sorted(n for n in REGENERATED if (out_dir / n).exists()) + ['merge_sources']:
                if nm != 'ml_indexer':
                    tar.add(out_dir / nm, arcname=nm)
            tar.add(out_dir / 'ml_indexer', arcname='ml_indexer')
        tmp_out.replace(out)
        print(f'[merge] done: {out} ({total:,} rows, {time.time() - t_start:.0f} s)')
    finally:
        if args.keep_workdir:
            print(f'[merge] workdir kept: {work}')
        else:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == '__main__':
    main()
