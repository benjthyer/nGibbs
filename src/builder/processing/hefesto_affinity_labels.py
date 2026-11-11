"""
Signed phase-abundance labels (g) for HeFESTo ML bundles, from HeFESTo's own affinities.

The HeFESTo counterpart of `affinity_labels.py`, and it writes the same bundle contract
(g_labels.npy, s_labels.npy, cStats.npy/json), so the training side
(`builder.training.affinity_targets`) is unchanged. The difference is where an absent
phase's g comes from. MELTS projects it from the nearest crossing along a path; HeFESTo
reports the affinity A_j (kJ/mol-atom, > 0 = undersaturated) of every absent phase at
every row. A_j is converted to "virtual abundance" with a per-phase factor lambda_j:

    g = n                      phase present (n > present_tol), n = molar_labels
    g = -lambda_j * A_j        phase absent, A_j minimised (flag 1) or an upper-bound
                               lower limit (flag 2: A_j > alarge, already far beyond the
                               training-side clip at -clip*c)
    g = 0                      phase absent and slightly supersaturated (flag 8,
                               -drop_threshold <= A_j < 0); rows with A_j < -drop_threshold
                               are disequilibrium assemblages and are dropped from the
                               table before export (`filter_disequilibrium_rows`)
    g = NaN                    no usable affinity (not evaluated / spinodal / failed /
                               conflict / not in the control file / no qout / phase not
                               in the affinity table)
    s = lambda_j wherever g is finite, NaN elsewhere

lambda_j calibration (`calibrate_lambda`). n_j and A_j vanish on the same boundary
surface, so their gradients are parallel there and lambda_j = |dn_j/dx|_present /
|dA_j/dx|_absent at a crossing is independent of the direction x. Crossings are found on
the isobar (x = T) and isotherm (x = P) scans of each simulation segment
(HeFESTo_deep_sampling.split_scans). At a present->absent step (p | a), with second
rows p2 (present) and a2 (absent) beyond them when they exist:

    s  = |n_p2 - n_p| / |x_p2 - x_p|            x*_n = x_p + dir * n_p / s
    sA = (A_a2 - A_a) / |x_a2 - x_a|            x*_A = x_a - dir * A_a / sA
    both slopes      lambda = s / sA
    only s (edge)    sA = A_a / |x_a - x*_n|    (x*_n strictly inside the step)
    only sA (edge)   s  = n_p / |x*_A - x_p|    (x*_A strictly inside the step)

The absent side must be minimised (flag 1, A >= 0) at a (and a2 when used); a
supersaturated neighbour is HeFESTo flicker, not a boundary.

Polymorph (discontinuous) phases: stishovite <-> alpha-PbO2, gamma <-> epsilon iron and
the like appear or vanish with a finite jump, so x*_n falls far outside the step and
s/sA means nothing. A crossing is a jump when n_p > jump_ratio * s * |x_a - x_p|; a phase
whose crossings are mostly jumps (or that is listed in `polymorph_phases`) gets
lambda_j = median(n_p at its jumps) / A_window, i.e. an affinity of A_window maps to a
typical jump. This is a stop-gap, not a model of polymorphism.

Phases with fewer than `min_crossings` usable crossings get the log-median lambda of the
calibrated phases (reported as a fallback; the training side's own `min_crossings`
leaves their absent cells untrusted).

cStats (4, P), same rows as `affinity_labels.CSTATS_ROWS`:
    row 0  c = lambda_j * A_window
    rows 1-2  log10 lambda_j
    row 3  0
cStats.json also carries `present_median` (per-phase median n where present, from the
calibration bundle) for the training-side floor c >= c_floor_frac * present_median.
s is constant per phase, so every finite absent cell passes the training-side slope
filter; trust is decided here (finite or NaN g), and per phase by the crossing count.

Entry points
    load_affinity_table(csv_path)              cached columnar view of <table>_affinity.csv
    filter_disequilibrium_rows(table, aff, t)  drop rows with A < -t from a BigMetaTable
    calibrate_lambda(npy_dir, aff, ...)        lambda from an unpacked (unshuffled) bundle
    make_labeller(aff, calibration, ...)       labeller for affinity_labels.finalize_bundle
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from builder.processing.affinity_labels import (G_NAME, S_NAME, CSTATS_NAME, CSTATS_JSON,
                                                REPORT_JSON, CSTATS_ROWS, PRESENT_MEDIAN_KEY,
                                                present_medians)

SOURCE_NAME = 'source_rows.npy'
LAMBDA_JSON = 'lambda.json'

# Affinity flags (HeFESTo_affinity_import.py)
FLAG_PRESENT, FLAG_MINIMIZED, FLAG_UPPER_BOUND, FLAG_SUPERSATURATED = 0, 1, 2, 8

DEFAULT_HEFESTO_PARAMS = dict(
    present_tol=0.0,          # a phase is present when molar_labels > present_tol
    A_window=0.025,           # kJ/mol-atom; c = lambda * A_window (~25 K on CMB isobars)
    drop_threshold=1.0,       # kJ/mol-atom; rows with any A < -drop_threshold are dropped
    trust_upper_bound=True,   # label flag-2 cells with g = -lambda*A (A is a lower bound)
    min_crossings=20,         # below this a phase gets the pooled (fallback) lambda
    jump_ratio=2.0,           # crossing is a jump when n_p > jump_ratio * s * step
    polymorph_fraction=0.5,   # phase is a polymorph when this fraction of crossings jump
    polymorph_phases=[],      # force these phases to the jump calibration
    edge_frac=0.05,           # one-sided x* must sit this fraction of a step inside it
    rel_tol=1e-6,             # split_scans grouping tolerance
)


def hefesto_params(params=None):
    prm = dict(DEFAULT_HEFESTO_PARAMS)
    for k, v in (params or {}).items():
        if k in prm and v is not None:
            prm[k] = v
    prm['polymorph_phases'] = list(prm['polymorph_phases'] or [])
    return prm


# --------------------------------------------------------------------------------------
# Affinity table
# --------------------------------------------------------------------------------------
def _cache_dir(csv_path):
    p = Path(csv_path)
    return p.with_name(p.name[:-4] + '_cache')


def load_affinity_table(csv_path, chunksize=500_000, verbose=True):
    """Columnar view of a HeFESTo affinity CSV (one row per table row, same order).

    Returns a dict: phases (list), A (N,Q) float32 mmap, flags (N,Q) int8 mmap, and
    sim_id/row (int64), P/T (float64). Built once into <csv stem>_cache/ next to the CSV
    and rebuilt when the CSV is newer than the cache."""
    import pandas as pd
    csv_path = Path(csv_path)
    cdir = _cache_dir(csv_path)
    meta = cdir / 'meta.json'
    fresh = meta.exists() and meta.stat().st_mtime >= csv_path.stat().st_mtime
    if not fresh:
        t = time.time()
        cols = open(csv_path).readline().strip().split(',')
        phases = [c[2:-len('(kJ/mol-atom)')] for c in cols if c.startswith('A_')]
        a_cols = [f'A_{p}(kJ/mol-atom)' for p in phases]
        f_cols = [f'Aflag_{p}' for p in phases]
        for c in f_cols:
            if c not in cols:
                raise ValueError(f'{csv_path.name}: missing column {c}')
        with open(csv_path) as fh:
            N = sum(1 for _ in fh) - 1
        cdir.mkdir(exist_ok=True)
        Q = len(phases)
        A = np.lib.format.open_memmap(cdir / 'A.npy', 'w+', np.float32, (N, Q))
        F = np.lib.format.open_memmap(cdir / 'flags.npy', 'w+', np.int8, (N, Q))
        scal = {k: np.empty(N, np.int64 if k in ('sim_id', 'row') else np.float64)
                for k in ('sim_id', 'row', 'P', 'T')}
        src = {'sim_id': 'sim_id', 'row': 'row', 'P': 'P(GPa)', 'T': 'T(K)'}
        a = 0
        for ch in pd.read_csv(csv_path, usecols=list(src.values()) + a_cols + f_cols,
                              chunksize=chunksize, dtype=np.float64):
            b = a + len(ch)
            A[a:b] = ch[a_cols].to_numpy(np.float32)
            F[a:b] = ch[f_cols].to_numpy(np.float64).astype(np.int8)
            for k, c in src.items():
                scal[k][a:b] = ch[c].to_numpy()
            a = b
        assert a == N, (a, N)
        A.flush(); F.flush()
        del A, F
        for k, v in scal.items():
            np.save(cdir / f'{k}.npy', v)
        meta.write_text(json.dumps(dict(phases=phases, rows=N, source=str(csv_path))))
        if verbose:
            print(f'[hefesto-affinity] cached {N:,} rows x {Q} phases from {csv_path.name} '
                  f'in {time.time() - t:.0f} s -> {cdir}')
    info = json.loads(meta.read_text())
    out = dict(phases=info['phases'], path=str(csv_path),
               A=np.load(cdir / 'A.npy', mmap_mode='r'),
               flags=np.load(cdir / 'flags.npy', mmap_mode='r'))
    for k in ('sim_id', 'row', 'P', 'T'):
        out[k] = np.load(cdir / f'{k}.npy')
    return out


def find_affinity_csv(table_name, read_dir=None, override=None):
    """<table>_affinity.csv next to the table (or in read_dir), unless overridden."""
    cands = [override] if override else []
    cands += [table_name + '_affinity.csv']
    if read_dir:
        cands.append(os.path.join(read_dir, table_name + '_affinity.csv'))
    for c in cands:
        if c and os.path.exists(c):
            return c
    raise FileNotFoundError('No HeFESTo affinity table found; tried:\n  ' + '\n  '.join(map(str, cands))
                            + '\nImport with scripts/import_hefesto_subdirs_affinity.py, or set '
                              'affinity.train_affinity_csv / valid_affinity_csv in the recipe.')


def check_alignment(table, aff):
    """The affinity table must describe the loaded table row for row."""
    N = table.table.shape[0]
    if len(aff['P']) != len(table.source_rows) or N != len(table.source_rows):
        raise ValueError(f'Affinity table has {len(aff["P"]):,} rows but the loaded table has '
                         f'{len(table.source_rows):,} source rows ({N:,} now); import them together.')
    for col, key in (('P(GPa)(System_main)', 'P'), ('T(K)(System_main)', 'T')):
        if col in table.header:
            ci = table.header.index(col)
            idx = np.linspace(0, N - 1, min(N, 20_000)).astype(np.int64)
            d = np.nanmax(np.abs(np.asarray(table.table[idx, ci], np.float64)
                                 - aff[key][table.source_rows[idx]]))
            if not d < 1e-3:
                raise ValueError(f'Affinity table {key} disagrees with the table by {d:.3g}; '
                                 f'rows are not aligned.')


def filter_disequilibrium_rows(table, aff, drop_threshold, chunk=1_000_000):
    """Delete rows whose assemblage has a supersaturated absent phase, A < -drop_threshold.
    Returns the number of rows deleted."""
    src = table.source_rows
    bad = np.zeros(len(src), dtype=bool)
    for a in range(0, len(src), chunk):
        A = np.asarray(aff['A'][src[a:a + chunk]])
        F = np.asarray(aff['flags'][src[a:a + chunk]])
        with np.errstate(invalid='ignore'):
            bad[a:a + chunk] = ((F == FLAG_SUPERSATURATED) & (A < -drop_threshold)).any(axis=1)
    n = int(bad.sum())
    print(f'[hefesto-affinity] disequilibrium filter (A < -{drop_threshold} kJ/mol-atom): '
          f'dropping {n:,} of {len(src):,} rows ({100 * n / max(len(src), 1):.2f}%)')
    if n:
        table.delete(np.flatnonzero(bad))
    return n


# --------------------------------------------------------------------------------------
# Bundle access
# --------------------------------------------------------------------------------------
def _bundle_arrays(npy_dir, ml_indexer, aff):
    npy_dir = Path(npy_dir)
    if not (npy_dir / SOURCE_NAME).exists():
        raise FileNotFoundError(f'{SOURCE_NAME} not in the bundle: export with '
                                f'export_run_ids=True from a table carrying source_rows.')
    src = np.asarray(np.load(npy_dir / SOURCE_NAME))
    M = np.load(npy_dir / 'molar_labels.npy', mmap_mode='r')
    phases = list(ml_indexer.all_phases)
    if M.shape != (len(src), len(phases)):
        raise ValueError(f'molar_labels {M.shape} vs {len(src)} source rows x {len(phases)} phases')
    if src.size and src.max() >= len(aff['P']):
        raise ValueError('source_rows index past the end of the affinity table')
    col = np.array([aff['phases'].index(p) if p in aff['phases'] else -1 for p in phases])
    return src, M, phases, col


def _segments(src, aff):
    """Segment id per bundle row: consecutive rows of one simulation (same sim_id and both
    the source row and the HeFESTo row advance by exactly 1)."""
    sim = aff['sim_id'][src]
    hrow = aff['row'][src]
    new = np.r_[True, (sim[1:] != sim[:-1]) | (np.diff(src) != 1) | (np.diff(hrow) != 1)]
    return np.cumsum(new) - 1


def _scan_order(seg, key, x, rel_tol):
    """Rows sorted into scans (segment, constant key, increasing x); scan id per sorted row."""
    scale = max(float(np.nanmax(np.abs(key))), 1.0) * rel_tol
    kr = np.round(key / scale).astype(np.int64)
    order = np.lexsort((x, kr, seg))
    s, k = seg[order], kr[order]
    scan = np.cumsum(np.r_[True, (s[1:] != s[:-1]) | (k[1:] != k[:-1])]) - 1
    return order, scan


# --------------------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------------------
def find_crossings(n, A, F, x, scan, prm):
    """Vectorised crossings of every phase along sorted scans.

    n, A, F  (m, P) in scan order (A/F already mapped to bundle phase columns, NaN/-1 where
    absent from the affinity table), x (m,), scan (m,). Returns a dict of per-crossing
    arrays: phase, kind ('both'/'n_edge'/'A_edge'/'jump' as 0..3), lam, jump (n_p)."""
    m, P = n.shape
    pres = n > prm['present_tol']
    out = {k: [] for k in ('phase', 'kind', 'lam', 'jump')}
    for j in range(P):
        pj = pres[:, j]
        flip = np.flatnonzero((pj[1:] != pj[:-1]) & (scan[1:] == scan[:-1]))
        if flip.size == 0:
            continue
        p_left = pj[flip]
        ip = np.where(p_left, flip, flip + 1)
        ia = np.where(p_left, flip + 1, flip)
        ip2, ia2 = 2 * ip - ia, 2 * ia - ip
        ip2c, ia2c = np.clip(ip2, 0, m - 1), np.clip(ia2, 0, m - 1)
        ok_p2 = (ip2 >= 0) & (ip2 < m) & (scan[ip2c] == scan[ip]) & pj[ip2c]
        ok_a2 = ((ia2 >= 0) & (ia2 < m) & (scan[ia2c] == scan[ia]) & ~pj[ia2c]
                 & (F[ia2c, j] == FLAG_MINIMIZED))
        flicker_p2 = (ip2 >= 0) & (ip2 < m) & (scan[ip2c] == scan[ip]) & ~pj[ip2c]
        ok_a = (F[ia, j] == FLAG_MINIMIZED) & (A[ia, j] >= 0)
        nj = n[:, j].astype(np.float64)
        Aj = A[:, j].astype(np.float64)
        step = np.abs(x[ia] - x[ip])
        dirn = np.sign(x[ia] - x[ip])
        with np.errstate(divide='ignore', invalid='ignore'):
            s = np.where(ok_p2, np.abs(nj[ip2c] - nj[ip]) / np.abs(x[ip2c] - x[ip]), np.nan)
            sA = np.where(ok_a2, (Aj[ia2c] - Aj[ia]) / np.abs(x[ia2c] - x[ia]), np.nan)
            s = np.where(s > 0, s, np.nan)
            sA = np.where(sA > 0, sA, np.nan)
            xs_n = x[ip] + dirn * nj[ip] / s
            xs_A = x[ia] - dirn * Aj[ia] / sA
            jump = ok_p2 & np.isfinite(s) & (nj[ip] > prm['jump_ratio'] * s * step)
            lo = prm['edge_frac'] * step
            d_a = np.abs(x[ia] - xs_n)   # x*_n to the absent row
            d_p = np.abs(xs_A - x[ip])   # x*_A to the present row
            in_n = np.isfinite(xs_n) & (d_a > lo) & (np.abs(xs_n - x[ip]) < step - lo)
            in_A = np.isfinite(xs_A) & (d_p > lo) & (np.abs(xs_A - x[ia]) < step - lo)
            both = np.isfinite(s) & np.isfinite(sA) & ~jump
            n_edge = np.isfinite(s) & ~np.isfinite(sA) & ~jump & in_n
            A_edge = ~np.isfinite(s) & np.isfinite(sA) & ~flicker_p2 & in_A
            lam = np.full(flip.size, np.nan)
            lam[both] = s[both] / sA[both]
            lam[n_edge] = s[n_edge] * d_a[n_edge] / Aj[ia][n_edge]
            lam[A_edge] = (nj[ip][A_edge] / d_p[A_edge]) / sA[A_edge]
        kind = np.full(flip.size, -1)
        kind[both], kind[n_edge], kind[A_edge] = 0, 1, 2
        kind[jump] = 3
        use = ok_a & (kind >= 0) & ((kind == 3) | (np.isfinite(lam) & (lam > 0)))
        out['phase'].append(np.full(use.sum(), j))
        out['kind'].append(kind[use])
        out['lam'].append(lam[use])
        out['jump'].append(nj[ip][use])
    return {k: (np.concatenate(v) if v else np.array([])) for k, v in out.items()}


def _mapped_affinities(aff, src, col):
    A = np.full((len(src), len(col)), np.nan, np.float32)
    F = np.full((len(src), len(col)), -1, np.int8)
    have = col >= 0
    if src.size:
        A[:, have] = np.asarray(aff['A'][src])[:, col[have]]
        F[:, have] = np.asarray(aff['flags'][src])[:, col[have]]
    return A, F


def collect_crossings(npy_dir, ml_indexer, aff, prm):
    """Crossings of every bundle phase on the isobar and isotherm scans of the bundle."""
    src, M, phases, col = _bundle_arrays(npy_dir, ml_indexer, aff)
    seg = _segments(src, aff)
    P_, T_ = aff['P'][src], aff['T'][src]
    n_all = np.asarray(M, dtype=np.float32)
    A_all, F_all = _mapped_affinities(aff, src, col)
    recs = []
    for axis, key, x in (('isobar', P_, T_), ('isotherm', T_, P_)):
        order, scan = _scan_order(seg, key, x, prm['rel_tol'])
        r = find_crossings(n_all[order], A_all[order], F_all[order], x[order], scan, prm)
        r['axis'] = np.full(r['phase'].size, axis)
        recs.append(r)
    rec = {k: np.concatenate([r[k] for r in recs]) for k in recs[0]}
    return rec, phases, int(seg.max() + 1 if seg.size else 0)


def summarize_crossings(rec, phases, prm):
    """Per-phase lambda from crossing records (see module docstring)."""
    P = len(phases)
    lam = np.full(P, np.nan)
    method = ['none'] * P
    stats = {}
    for j, ph in enumerate(phases):
        m = rec['phase'] == j
        kind = rec['kind'][m]
        n_cont = int((kind != 3).sum())
        n_jump = int((kind == 3).sum())
        ll = np.log10(rec['lam'][m][kind != 3])
        st = dict(continuous=n_cont, jumps=n_jump,
                  both=int((kind == 0).sum()), n_edge=int((kind == 1).sum()),
                  A_edge=int((kind == 2).sum()))
        for ax in ('isobar', 'isotherm'):
            sel = (rec['axis'][m] == ax) & (kind != 3)
            st[f'{ax}_median_log10_lambda'] = (float(np.median(np.log10(rec['lam'][m][sel])))
                                               if sel.any() else None)
        if ll.size:
            st['median_log10_lambda'] = float(np.median(ll))
            st['iqr_log10_lambda'] = float(np.subtract(*np.percentile(ll, [75, 25])))
        tot = n_cont + n_jump
        poly = ph in prm['polymorph_phases'] or (tot >= prm['min_crossings']
                                                 and n_jump >= prm['polymorph_fraction'] * tot)
        if poly and n_jump:
            lam[j] = float(np.median(rec['jump'][m][kind == 3])) / prm['A_window']
            method[j] = 'polymorph'
        elif n_cont >= prm['min_crossings']:
            lam[j] = 10 ** st['median_log10_lambda']
            method[j] = 'crossings'
        st['n_used'] = n_jump if method[j] == 'polymorph' else n_cont
        stats[ph] = st
    cal = np.array([mt == 'crossings' for mt in method])
    if not cal.any():
        raise ValueError('No phase has enough crossings to calibrate lambda '
                         f'(min_crossings={prm["min_crossings"]}).')
    fallback = float(10 ** np.median(np.log10(lam[cal])))
    for j in range(P):
        if method[j] == 'none':
            lam[j] = fallback
            method[j] = 'fallback'
    return dict(phases=phases, lam=lam.tolist(), method=method, fallback_lambda=fallback,
                n_used=[stats[p]['n_used'] for p in phases], stats=stats,
                params={k: prm[k] for k in ('A_window', 'present_tol', 'min_crossings',
                                            'jump_ratio', 'polymorph_fraction',
                                            'polymorph_phases', 'edge_frac')})


def calibrate_lambda(npy_dir, ml_indexer, aff, params=None, verbose=True):
    prm = hefesto_params(params)
    t = time.time()
    rec, phases, n_seg = collect_crossings(npy_dir, ml_indexer, aff, prm)
    cal = summarize_crossings(rec, phases, prm)
    cal[PRESENT_MEDIAN_KEY] = present_medians(np.load(Path(npy_dir) / 'molar_labels.npy', mmap_mode='r'),
                                              prm['present_tol'])
    cal['segments'] = n_seg
    cal['source'] = str(aff['path'])
    if verbose:
        print(f'[hefesto-affinity] lambda calibration: {n_seg:,} segments, '
              f'{rec["phase"].size:,} usable crossings ({time.time() - t:.0f} s)')
        print(format_calibration(cal))
    return cal


def format_calibration(cal):
    lines = [f'  {"phase":<22}{"lambda":>10}  {"method":<10}{"used":>7}{"jumps":>7}'
             f'{"IQR dex":>9}{"isobar":>9}{"isoth.":>9}']
    for j, ph in enumerate(cal['phases']):
        st = cal['stats'][ph]
        f = lambda v: f'{v:9.2f}' if v is not None else f'{"-":>9}'
        lines.append(f'  {ph:<22}{cal["lam"][j]:10.3g}  {cal["method"][j]:<10}'
                     f'{st["n_used"]:7d}{st["jumps"]:7d}{f(st.get("iqr_log10_lambda"))}'
                     f'{f(st["isobar_median_log10_lambda"])}{f(st["isotherm_median_log10_lambda"])}')
    lines.append(f'  fallback lambda = {cal["fallback_lambda"]:.3g}')
    return '\n'.join(lines)


# --------------------------------------------------------------------------------------
# Labelling
# --------------------------------------------------------------------------------------
def label_rows(n, A, F, lam, prm):
    """(m, P) g and s for one chunk (module docstring rules)."""
    m, P = n.shape
    g = np.full((m, P), np.nan, np.float32)
    lam32 = np.asarray(lam, np.float32)
    pres = n > prm['present_tol']
    g[pres] = n[pres]
    ab = ~pres
    mini = ab & (F == FLAG_MINIMIZED) & (A >= 0)
    ub = ab & (F == FLAG_UPPER_BOUND) & np.isfinite(A) if prm['trust_upper_bound'] else np.zeros_like(ab)
    sup = ab & (F == FLAG_SUPERSATURATED) & np.isfinite(A)
    neg = mini | ub
    g = np.where(neg, -lam32 * A, g)
    g = np.where(sup, np.float32(0.0), g)
    s = np.where(np.isfinite(g), lam32, np.float32(np.nan)).astype(np.float32)
    counts = dict(present=int(pres.sum()), minimized=int(mini.sum()), upper_bound=int(ub.sum()),
                  supersaturated_to_zero=int(sup.sum()),
                  absent_unlabelled=int((ab & ~neg & ~sup).sum()),
                  absent_but_flag_present=int((ab & (F == FLAG_PRESENT)).sum()))
    per_phase_nan = (ab & ~neg & ~sup).sum(axis=0)
    return g.astype(np.float32), s, counts, per_phase_nan


def make_labeller(aff, calibration, params=None):
    """Labeller with the `affinity_labels.add_affinity_labels` signature, for
    `affinity_labels.finalize_bundle(..., labeller=...)`. `calibration` is a dict from
    `calibrate_lambda` (or a zero-argument callable returning one, resolved at labelling
    time), or the string 'self' to calibrate on the bundle being labelled
    (only correct before any shuffle - finalize_bundle labels first)."""
    prm = hefesto_params(params)

    def labeller(npy_dir, ml_indexer=None, compute_cstats_flag=False, params=None,
                 chunk_size=1_000_000, verbose=True):
        npy_dir = Path(npy_dir)
        t0 = time.time()
        if ml_indexer is None:
            from ngibbs.config.ml_indexer import load_ml_indexer_from_state
            ml_indexer = load_ml_indexer_from_state(str(npy_dir / 'ml_indexer'))
        cal = calibration() if callable(calibration) else calibration
        if isinstance(cal, str) and cal == 'self':
            cal = calibrate_lambda(npy_dir, ml_indexer, aff, prm, verbose=verbose)
        src, M, phases, col = _bundle_arrays(npy_dir, ml_indexer, aff)
        if list(cal['phases']) != phases:
            raise ValueError(f'lambda calibration phases {cal["phases"]} != bundle phases {phases}')
        lam = np.asarray(cal['lam'], np.float64)

        # Held-out check: crossings of this bundle against the calibration it is labelled with.
        check = None
        if compute_cstats_flag and not (isinstance(calibration, str) and calibration == 'self'):
            rec, _, _ = collect_crossings(npy_dir, ml_indexer, aff, prm)
            check = summarize_crossings(rec, phases, prm) if rec['phase'].size else None

        N, P = M.shape
        G = np.lib.format.open_memmap(npy_dir / G_NAME, mode='w+', dtype=np.float32, shape=(N, P))
        S = np.lib.format.open_memmap(npy_dir / S_NAME, mode='w+', dtype=np.float32, shape=(N, P))
        tot = {}
        nan_pp = np.zeros(P, dtype=np.int64)
        for a in range(0, N, chunk_size):
            b = min(a + chunk_size, N)
            A, F = _mapped_affinities(aff, src[a:b], col)
            g, s, cnt, npp = label_rows(np.asarray(M[a:b], np.float32), A, F, lam, prm)
            G[a:b], S[a:b] = g, s
            nan_pp += npp
            for k, v in cnt.items():
                tot[k] = tot.get(k, 0) + v
        G.flush(); S.flush()
        del G, S

        (npy_dir / LAMBDA_JSON).write_text(json.dumps(cal, indent=1))
        report = dict(rows=int(N), phases=phases, labeller='hefesto_affinity',
                      params=prm, cell_counts=tot,
                      absent_unlabelled_per_phase=dict(zip(phases, nan_pp.tolist())),
                      lambda_=dict(zip(phases, cal['lam'])),
                      lambda_method=dict(zip(phases, cal['method'])),
                      extra_files=[LAMBDA_JSON],
                      seconds=dict(labels=round(time.time() - t0, 2)))
        if check is not None:
            report['holdout_crossings'] = {
                ph: dict(n=check['stats'][ph]['continuous'], jumps=check['stats'][ph]['jumps'],
                         median_log10_lambda=check['stats'][ph].get('median_log10_lambda'),
                         calibration_log10_lambda=float(np.log10(lam[j])))
                for j, ph in enumerate(phases) if check['stats'][ph]['n_used']}

        if compute_cstats_flag:
            cs = np.full((4, P), np.nan)
            cs[0] = lam * prm['A_window']
            cs[1] = cs[2] = np.log10(lam)
            cs[3] = 0.0
            np.save(npy_dir / CSTATS_NAME, cs)
            (npy_dir / CSTATS_JSON).write_text(json.dumps(dict(
                rows=list(CSTATS_ROWS), phases=phases, crossings_per_phase=list(cal['n_used']),
                **{PRESENT_MEDIAN_KEY: cal.get(PRESENT_MEDIAN_KEY) or present_medians(M, prm['present_tol'])},
                lambda_method=cal['method'], A_window=prm['A_window'],
                note='HeFESTo affinity labels: g = -lambda*A, s = lambda; c = lambda*A_window. '
                     'crossings_per_phase counts the crossings lambda was calibrated on; '
                     'fallback-lambda phases carry their (too few) crossings, so the '
                     'training-side min_crossings leaves their absent cells untrusted.'), indent=1))
            report['extra_files'] += [CSTATS_NAME, CSTATS_JSON]
            report['cstats'] = {p: dict(zip(CSTATS_ROWS, map(float, cs[:, j])))
                                for j, p in enumerate(phases)}
        (npy_dir / REPORT_JSON).write_text(json.dumps(report, indent=1))
        if verbose:
            print(f'[hefesto-affinity] {N:,} rows labelled: ' +
                  ', '.join(f'{k} {v:,}' for k, v in tot.items()))
            if check is not None:
                print('[hefesto-affinity] held-out crossings (this bundle) vs calibration, log10 lambda:')
                for ph, d in report['holdout_crossings'].items():
                    med = d['median_log10_lambda']
                    print(f'    {ph:<22} n={d["n"]:6d}  here {"-" if med is None else f"{med:6.2f}"}'
                          f'  calib {d["calibration_log10_lambda"]:6.2f}')
        return report

    return labeller
