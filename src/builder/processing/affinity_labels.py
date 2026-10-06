"""
Signed phase-abundance labels (g) and crossing slopes (s) for MELTS ML-ready bundles.

Background: project docs `Affinity_Labels_Pipeline_Plan.md` (v2) and
`FxTal_dndT_Projection_Test.md`.

For every row and phase of an exported bundle (rows in simulation order, i.e. BEFORE the
training shuffle):

    g = n                      phase present (n > present_tol), n = molar_labels
    g = -s * |x - x*|          phase absent, projected from the nearest saturation crossing
                               in the same path segment (x = path coordinate, default T)
    g = NaN                    phase never present in that segment (no crossing to project from)
    s = slope |dn/dx| of the crossing a row is projected from (NaN where there is none)

Path segments: a new segment starts where `run_ids` changes, or where the path coordinate
stops moving in the run's overall direction (or stalls). Crossing slopes:

    batch (constant bulk)   s = |n_p2 - n_p| / |x_p2 - x_p|     (two nearest present rows)
    FxTal (bulk varies)     s = n_p2 / |x_p2 - x_p|             (per-step crystallisation)
    fallback                s = n_p / |x_a - x_p|               (single present row, or s <= 0)
    x* = x_p -/+ n_p / s, clamped into the step that brackets the crossing.

n is cation-normalised per row (molar_labels), so slopes measured at the crossing are already
in the right units for every row of the segment; g is NOT divided by anything at export.

cStats (validation bundle only): a (4, P) float64 array in ml_indexer.all_phases order,
computed from the crossings (not rows):
    row 0  c = dT_window * 10**median(log10 s)          (= dT_window * median(s))
    row 1  median(log10 s)
    row 2  mean(log10 s)   after dropping the ceil(trim*N) crossings farthest from the median
    row 3  std(log10 s)    same trimmed set, ddof=1 (raw; a sigma floor is applied at use)
Phases with no crossing are NaN; phases with < 3 crossings left after trimming have NaN in
rows 2-3 (no filter for that phase). Parameters go in cStats.json.

Entry points
    add_affinity_labels(npy_dir, ...)   operate on an unpacked bundle directory
    finalize_bundle(bundle_path, ...)   extract -> add labels (-> cStats) (-> shuffle) -> repack
"""
from __future__ import annotations

import gc
import json
import os
import shutil
import tarfile
import tempfile
import time
from pathlib import Path

import numpy as np

G_NAME = 'g_labels.npy'
S_NAME = 's_labels.npy'
RUN_NAME = 'run_ids.npy'
CSTATS_NAME = 'cStats.npy'
CSTATS_JSON = 'cStats.json'
REPORT_JSON = 'affinity_labels.json'
CSTATS_ROWS = ('c', 'median_log10_s', 'mean_log10_s_trimmed', 'std_log10_s_trimmed')

DEFAULT_PARAMS = dict(
    path_feature='Temperature(System_main)',  # path coordinate x (isobaric cooling runs)
    present_tol=0.0,        # a phase is present when n > present_tol
    fxtal_tol=1e-3,         # segment is FxTal when any bulk element fraction ranges more than this
    dT_window=25.0,         # c = dT_window * median(s)
    trim=0.05,              # fraction (rounded up) of crossings dropped before the lognormal fit
    gap_factor=1.5,         # a crossing step > gap_factor * median step is reported as a gap
)


# --------------------------------------------------------------------------------------
# Segmentation
# --------------------------------------------------------------------------------------
def _runs_from_path(x):
    """Fallback run ids for bundles exported without run_ids.npy: a new run wherever the
    path coordinate increases (isobaric cooling runs only). Unreliable in general."""
    new = np.r_[True, np.diff(x) > 0]
    return np.cumsum(new).astype(np.int64) - 1


def segment_starts(run_ids, x):
    """Row indices where a path segment starts, plus the number of extra in-run breaks.

    A segment is a maximal block of rows of one run along which x moves strictly in the
    run's overall direction (sign of x_last - x_first)."""
    n = len(x)
    new = np.zeros(n, dtype=bool)
    new[0] = True
    new[1:] = run_ids[1:] != run_ids[:-1]
    r_start = np.flatnonzero(new)
    r_end = np.r_[r_start[1:], n]
    run_dir = np.sign(x[r_end - 1].astype(np.float64) - x[r_start].astype(np.float64))
    dir_row = np.repeat(run_dir, r_end - r_start)
    step_sign = np.sign(np.diff(x.astype(np.float64)))
    extra = (step_sign != dir_row[1:]) & ~new[1:]
    new[1:] |= extra
    return np.flatnonzero(new), int(extra.sum())


# --------------------------------------------------------------------------------------
# Kernel: one chunk of whole segments
# --------------------------------------------------------------------------------------
def label_chunk(x, n, seg, seg_fx, present_tol=0.0):
    """Vectorised g/s for a block of whole segments.

    x      (m,)   path coordinate (float64)
    n      (m,P)  normalised phase abundances
    seg    (m,)   segment index, non-decreasing, 0-based within the chunk
    seg_fx (S,)   True for FxTal segments
    Returns g (m,P) f32, s (m,P) f32 and a per-crossing record dict of arrays."""
    m, P = n.shape
    g = np.full((m, P), np.nan, dtype=np.float32)
    s_out = np.full((m, P), np.nan, dtype=np.float32)
    pres = n > present_tol
    g[pres] = n[pres]
    same_next = seg[1:] == seg[:-1]
    rec = {'phase': [], 's': [], 'step': [], 'fallback': [], 'fx': []}

    for j in range(P):
        pj = pres[:, j]
        if not pj.any():
            continue
        flip = np.flatnonzero((pj[1:] != pj[:-1]) & same_next)   # boundary between flip, flip+1
        if flip.size == 0:
            continue
        nj = n[:, j].astype(np.float64)
        p_at_left = pj[flip]
        pidx = np.where(p_at_left, flip, flip + 1)
        aidx = np.where(p_at_left, flip + 1, flip)
        p2 = 2 * pidx - aidx
        p2c = np.clip(p2, 0, m - 1)
        ok2 = (p2 >= 0) & (p2 < m) & pj[p2c] & (seg[p2c] == seg[pidx])
        dx_pa = np.abs(x[aidx] - x[pidx])
        dx_2 = np.abs(x[p2c] - x[pidx])
        fx = seg_fx[seg[pidx]]
        with np.errstate(divide='ignore', invalid='ignore'):
            s = np.where(fx, nj[p2c] / dx_2, np.abs(nj[p2c] - nj[pidx]) / dx_2)
            fb = ~ok2 | ~np.isfinite(s) | (s <= 0)
            s = np.where(fb, nj[pidx] / dx_pa, s)
            x_star = x[pidx] + np.sign(x[aidx] - x[pidx]) * nj[pidx] / s
        x_star = np.clip(x_star, np.minimum(x[pidx], x[aidx]), np.maximum(x[pidx], x[aidx]))

        # nearest crossing in the same segment, for every row
        K = flip.size
        kk = np.arange(K)
        last = np.full(m, -1, dtype=np.int64)
        last[flip + 1] = kk
        last = np.maximum.accumulate(last)
        nxt = np.full(m, K, dtype=np.int64)
        nxt[flip] = kk
        nxt = np.minimum.accumulate(nxt[::-1])[::-1]
        cseg = seg[flip]
        Lc = np.clip(last, 0, K - 1)
        Nc = np.clip(nxt, 0, K - 1)
        okL = (last >= 0) & (cseg[Lc] == seg)
        okN = (nxt < K) & (cseg[Nc] == seg)
        dL = np.where(okL, np.abs(x - x_star[Lc]), np.inf)
        dN = np.where(okN, np.abs(x - x_star[Nc]), np.inf)
        k = np.where(dN < dL, Nc, Lc)
        d = np.minimum(dL, dN)
        has = np.isfinite(d)
        sk = s[k]
        absent = has & ~pj
        g[absent, j] = (-sk[absent] * d[absent]).astype(np.float32)
        s_out[has, j] = sk[has].astype(np.float32)

        rec['phase'].append(np.full(K, j, dtype=np.int32))
        rec['s'].append(s)
        rec['step'].append(dx_pa)
        rec['fallback'].append(fb)
        rec['fx'].append(fx)

    rec = {k_: (np.concatenate(v) if v else np.array([])) for k_, v in rec.items()}
    return g, s_out, rec


# --------------------------------------------------------------------------------------
# cStats
# --------------------------------------------------------------------------------------
def compute_cstats(log_s_by_phase, n_phases, dT_window, trim):
    """(4, P) cStats from per-phase lists of log10 s (see module docstring)."""
    cs = np.full((4, n_phases), np.nan)
    counts = np.zeros(n_phases, dtype=int)
    for j in range(n_phases):
        ls = np.asarray(log_s_by_phase.get(j, []), dtype=np.float64)
        ls = ls[np.isfinite(ls)]
        counts[j] = ls.size
        if ls.size == 0:
            continue
        med = float(np.median(ls))
        cs[0, j] = dT_window * 10 ** med
        cs[1, j] = med
        n_drop = int(np.ceil(trim * ls.size))
        keep = np.argsort(np.abs(ls - med))[:ls.size - n_drop]
        if keep.size >= 3:
            cs[2, j] = float(ls[keep].mean())
            cs[3, j] = float(ls[keep].std(ddof=1))
    return cs, counts


# --------------------------------------------------------------------------------------
# Driver on an unpacked bundle directory
# --------------------------------------------------------------------------------------
def add_affinity_labels(npy_dir, ml_indexer=None, compute_cstats_flag=False, params=None,
                        chunk_size=1_000_000, verbose=True):
    """Write g_labels.npy and s_labels.npy next to molar_labels.npy in `npy_dir`
    (an unpacked bundle, rows still in simulation order). Optionally write cStats.npy/json.
    Returns a report dict (also written as affinity_labels.json)."""
    npy_dir = Path(npy_dir)
    prm = dict(DEFAULT_PARAMS)
    prm.update(params or {})
    t0 = time.time()
    if ml_indexer is None:
        from ngibbs.config.ml_indexer import load_ml_indexer_from_state
        ml_indexer = load_ml_indexer_from_state(str(npy_dir / 'ml_indexer'))
    phases = list(ml_indexer.all_phases)
    feature_names = list(ml_indexer.featureNames)
    xi = feature_names.index(prm['path_feature'])
    off = len(feature_names)

    F = np.load(npy_dir / 'features.npy', mmap_mode='r')
    M = np.load(npy_dir / 'molar_labels.npy', mmap_mode='r')
    N, P = M.shape
    assert F.shape[0] == N and P == len(phases), (F.shape, M.shape, len(phases))
    x_all = np.empty(N, dtype=np.float64)
    for a in range(0, N, chunk_size):
        x_all[a:a + chunk_size] = F[a:a + chunk_size, xi]
    if (npy_dir / RUN_NAME).exists():
        run_ids = np.load(npy_dir / RUN_NAME, mmap_mode='r')
        run_ids = np.asarray(run_ids)
        run_src = RUN_NAME
    else:
        print(f'[affinity] WARNING: {RUN_NAME} not in bundle - inferring runs from path reversals '
              f'(isobaric cooling only; unreliable for general data)')
        run_ids = _runs_from_path(x_all)
        run_src = 'inferred from path reversals'
    starts, extra_breaks = segment_starts(run_ids, x_all)
    n_seg = starts.size
    seg_id = np.zeros(N, dtype=np.int64)
    seg_id[starts[1:]] = 1
    seg_id = np.cumsum(seg_id)
    seg_len = np.diff(np.r_[starts, N])
    t_seg = time.time() - t0

    G = np.lib.format.open_memmap(npy_dir / G_NAME, mode='w+', dtype=np.float32, shape=(N, P))
    S = np.lib.format.open_memmap(npy_dir / S_NAME, mode='w+', dtype=np.float32, shape=(N, P))

    log_s = {}
    n_cross = np.zeros(P, dtype=int)
    n_fallback = np.zeros(P, dtype=int)
    steps_all = []
    n_fx_seg = 0
    zero_extent = 0
    a = 0
    t_k = time.time()
    while a < N:
        b_target = min(a + chunk_size, N)
        if b_target >= N:
            b = N
        else:   # extend to the next segment start
            nxt = np.searchsorted(starts, b_target, side='left')
            b = int(starts[nxt]) if nxt < n_seg else N
        x = x_all[a:b]
        n = np.asarray(M[a:b], dtype=np.float32)
        seg = (seg_id[a:b] - seg_id[a]).astype(np.int64)
        loc_starts = np.flatnonzero(np.r_[True, seg[1:] != seg[:-1]])
        bulk = np.asarray(F[a:b, off:], dtype=np.float64)
        rng = (np.maximum.reduceat(bulk, loc_starts, axis=0)
               - np.minimum.reduceat(bulk, loc_starts, axis=0)).max(axis=1)
        seg_fx = rng > prm['fxtal_tol']
        n_fx_seg += int(seg_fx.sum())
        ext = np.maximum.reduceat(x, loc_starts) - np.minimum.reduceat(x, loc_starts)
        zero_extent += int(((ext == 0) & (np.diff(np.r_[loc_starts, len(x)]) > 1)).sum())

        g, s, rec = label_chunk(x, n, seg, seg_fx, prm['present_tol'])
        G[a:b] = g
        S[a:b] = s
        if rec['phase'].size:
            ph = rec['phase']
            np.add.at(n_cross, ph, 1)
            np.add.at(n_fallback, ph, rec['fallback'].astype(int))
            steps_all.append(rec['step'])
            if compute_cstats_flag:
                ls = np.log10(rec['s'])
                for j in np.unique(ph):
                    log_s.setdefault(int(j), []).append(ls[ph == j])
        a = b
    G.flush(); S.flush()
    del G, S
    gc.collect()
    t_kernel = time.time() - t_k

    steps = np.concatenate(steps_all) if steps_all else np.array([])
    med_step = float(np.median(steps)) if steps.size else float('nan')
    n_gap = int((steps > prm['gap_factor'] * med_step).sum()) if steps.size else 0

    report = dict(
        rows=int(N), phases=phases, params=prm, run_ids_source=run_src,
        segments=int(n_seg), extra_in_run_breaks=int(extra_breaks),
        fxtal_segments=int(n_fx_seg), zero_extent_segments=int(zero_extent),
        median_segment_rows=float(np.median(seg_len)),
        crossings_per_phase={p: int(c) for p, c in zip(phases, n_cross)},
        fallback_crossings_per_phase={p: int(c) for p, c in zip(phases, n_fallback)},
        median_crossing_step=med_step,
        crossings_across_gaps=n_gap,
        seconds=dict(segmentation=round(t_seg, 2), kernel=round(t_kernel, 2)),
    )

    if compute_cstats_flag:
        flat = {j: np.concatenate(v) for j, v in log_s.items()}
        cs, counts = compute_cstats(flat, P, prm['dT_window'], prm['trim'])
        np.save(npy_dir / CSTATS_NAME, cs)
        (npy_dir / CSTATS_JSON).write_text(json.dumps(dict(
            rows=list(CSTATS_ROWS), phases=phases, crossings_per_phase=counts.tolist(),
            path_feature=prm['path_feature'], dT_window=prm['dT_window'], trim=prm['trim'],
            note='c = dT_window * 10**median(log10 s); rows 2-3 from the trimmed set; '
                 'apply the sigma floor and k at use'), indent=1))
        report['cstats'] = {p: dict(zip(CSTATS_ROWS, [None if not np.isfinite(v) else float(v)
                                                     for v in cs[:, j]]))
                            for j, p in enumerate(phases) if counts[j]}
    (npy_dir / REPORT_JSON).write_text(json.dumps(report, indent=1))
    if verbose:
        print(f'[affinity] {N:,} rows, {n_seg:,} segments ({n_fx_seg:,} FxTal, '
              f'{extra_breaks:,} in-run breaks), {int(n_cross.sum()):,} crossings '
              f'({int(n_fallback.sum()):,} fallback, {n_gap:,} across gaps); '
              f'segmentation {t_seg:.1f} s, kernel {t_kernel:.1f} s')
    return report


# --------------------------------------------------------------------------------------
# Bundle wrapper: extract once, label, optionally shuffle, repack once
# --------------------------------------------------------------------------------------
def finalize_bundle(bundle_path, shuffle=False, compute_cstats_flag=False, params=None,
                    seed=None, chunk_size=1_000_000):
    """Add g/s (and cStats) to a packaged bundle; optionally shuffle all row-aligned arrays.
    Replaces the shuffle_bundle_rows() call for bundles produced by prepareML_affinity."""
    from ngibbs.utils.file_utils import chunked_permutation_copy, ROW_ALIGNED_BUNDLE_ARRAYS
    from ngibbs.config.ml_indexer import load_ml_indexer_from_state

    bundle_path = Path(bundle_path)
    extract_dir = Path(tempfile.mkdtemp(dir=bundle_path.parent))
    timings = {}
    try:
        t = time.time()
        with tarfile.open(bundle_path, 'r:gz') as tar:
            original_names = [m_.name for m_ in tar.getmembers() if m_.isfile()]
            tar.extractall(path=extract_dir)
        timings['extract'] = time.time() - t

        ml_indexer = load_ml_indexer_from_state(str(extract_dir / 'ml_indexer'))
        t = time.time()
        report = add_affinity_labels(extract_dir, ml_indexer, compute_cstats_flag, params, chunk_size)
        timings['affinity_labels'] = time.time() - t

        present = [nm for nm in ROW_ALIGNED_BUNDLE_ARRAYS if (extract_dir / nm).exists()]
        stats_path = None
        if shuffle:
            t = time.time()
            from builder.processing.MLexporter import generate_dataset_stats
            n_rows = np.load(extract_dir / 'features.npy', mmap_mode='r').shape[0]
            perm = np.random.default_rng(seed).permutation(n_rows)
            for nm in present:
                src = np.load(extract_dir / nm, mmap_mode='r')
                tmp = extract_dir / f'_shuffled_{nm}'
                chunked_permutation_copy(src, tmp, perm, chunk_size=chunk_size)
                del src
                gc.collect()
                tmp.replace(extract_dir / nm)
                print(f'[finalize_bundle]   shuffled {nm}')
            stats_path = generate_dataset_stats(dataset_name=str(extract_dir) + '/', ml_indexer=ml_indexer,
                                                output_dir=extract_dir, chunk_size=chunk_size)
            timings['shuffle'] = time.time() - t

        t = time.time()
        extra = [REPORT_JSON] + ([CSTATS_NAME, CSTATS_JSON] if compute_cstats_flag else [])
        handled = set(present) | set(extra)
        with tarfile.open(bundle_path, 'w:gz') as tar:
            for nm in present + extra:
                if (extract_dir / nm).exists():
                    tar.add(extract_dir / nm, arcname=nm)
            if stats_path and Path(stats_path).exists():
                tar.add(stats_path, arcname='stats.txt')
                handled.add('stats.txt')
                fb = extract_dir / f'{Path(str(extract_dir) + "/").stem}_feature_bounds.json'
                if fb.exists():
                    tar.add(fb, arcname='feature_bounds.json')
                    handled.add('feature_bounds.json')
            if (extract_dir / 'ml_indexer').is_dir():
                tar.add(extract_dir / 'ml_indexer', arcname='ml_indexer')
                handled.add('ml_indexer')
            for nm in original_names:
                top = Path(nm).parts[0] if nm else nm
                if top in handled:
                    continue
                if (extract_dir / nm).exists():
                    tar.add(extract_dir / nm, arcname=nm)
        timings['repack'] = time.time() - t
    finally:
        shutil.rmtree(extract_dir, ignore_errors=True)

    report['seconds'].update({k: round(v, 2) for k, v in timings.items()})
    print(f'[finalize_bundle] {bundle_path.name}: ' +
          ', '.join(f'{k} {v:.1f} s' for k, v in report['seconds'].items()))
    return report
