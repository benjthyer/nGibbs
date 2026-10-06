"""
Import HeFESTo's phase affinities from ``qout`` as a shadow table.

Where they come from
--------------------
At every P-T point, ``petsub.f`` repeatedly calls ``phaseadd.f``, which evaluates
the affinity of every absent phase against the current component chemical
potentials (``lagcomp.f``) by minimising SLB11 eq. B10 over the phase's
composition from m+1 starting guesses. Each call writes a table to unit 31
(``qout``, opened in ``readin.f``):

    Check for phase addition or subtraction
      phase      affinity  best_guess total_guess  iterations  total_iter     quality composition
      1 plg       13.1808           2           3           0           0      0.0000
      2 sp         8.4544           5           5          58         243      0.0000      0.6194 ...

format ``(i3,1x,a5,f12.4,4i12,99f12.4)``. The affinity is ``fret/fn``: kJ per mole of
ATOMS (fn = atoms per formula unit of the phase's first species), with HeFESTo's
sign convention -- POSITIVE means the phase is unstable / undersaturated. Nothing
here flips the sign; do that when building targets.

The LAST such table in a point's block is the one evaluated at the converged
assemblage (phaseadd found nothing to add, petsub fell through to the final
``lagcomp``). That table is what this module records.

What the final table does NOT contain (flag NOT_EVALUATED, value NaN)
--------------------------------------------------------------------
``phaseadd`` skips phases that are
  * ``allow = .false.``: added earlier at this point and later removed again
    (e.g. tracesub dropped it). These are typically the NEAR-SATURATION phases.
  * spinodally unstable (``spinph``): no valid volume for its species at this P-T.
  * a redundant duplicate of a phase listed directly before it in the control file.
These must be back-calculated (or masked) downstream.

Upper bounds (flag UPPER_BOUND)
-------------------------------
If the best starting guess is already worse than ``alarge`` = 10 kJ/mol-atom,
phaseadd logs that guess with 0 iterations instead of minimising. For a
solution phase (m > 1) the value is then an upper bound on the true affinity.
Single-species phases need no minimisation and are always exact.

Shadow-table design
-------------------
One row per row emitted by ``import_HeFESTo_components`` for the same simulations,
in the same order, using the main import's ``nrows = min(len(fort.56, fort.61,
fort.68, fort.99))`` rule. ``P(GPa)``/``T(K)`` come from fort.56 so the positional
join can be checked. Pass the main import's ``passed_ids`` as ``only_sim_ids`` so a
simulation the main import rejected contributes no rows here either.

Columns
-------
    sim_id, row, P(GPa), T(K)
    qout_status             row status, see ROW_* below
    qout_dP, qout_dT        fort.56 minus qout block (2-decimal) P and final T
    mu_<El>(kJ/mol)         component (element) chemical potentials, lagcomp.f
    A_<phase>(kJ/mol-atom)  affinity; 0 for present phases; NaN where unknown
    Aflag_<phase>           per-phase flag, see FLAG_* below

The ``mu_*`` columns let missing phases be back-calculated from end-member Gibbs
energies alone, without re-fitting the chemical potentials from rounded fort.99
moles.
"""
from __future__ import annotations

import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from ngibbs.utils.file_utils import (
        _safe_read_ws_table, _parse_control_file, _parse_fort56,
        _resolve_component_name_from_abbr, _resolve_component_phase,
        _build_reverse_component_phase_map,
    )
except ImportError:  # pragma: no cover
    from src.ngibbs.utils.file_utils import (
        _safe_read_ws_table, _parse_control_file, _parse_fort56,
        _resolve_component_name_from_abbr, _resolve_component_phase,
        _build_reverse_component_phase_map,
    )

from .HeFESTo_functions import (
    _list_simulation_dirs, _write_block_to_csv, _ensure_existing_csv_headers_match,
)

QOUT = 'qout'
ELEMENT_ORDER: Tuple[str, ...] = ('Si', 'Mg', 'Fe', 'Ca', 'Al', 'Na', 'Cr', 'O')
_NON_PHASE_KEYS = {'System_main', 'Bulk_comp', 'Bulk_comp_elements'}

# ── per-phase flags ──────────────────────────────────────────────────────────
FLAG_PRESENT = 0         # phase present in the final assemblage (fort.99); A := 0
FLAG_MINIMIZED = 1       # absent; affinity minimised over composition (or m == 1)
FLAG_UPPER_BOUND = 2     # absent; m > 1 and phaseadd skipped minimisation (> alarge)
FLAG_NOT_EVALUATED = 3   # absent; not in final table (not allowed / spinodal / redundant)
FLAG_FAILED = 4          # listed, but value unparseable (every minimisation failed -> 1e15)
FLAG_CONFLICT = 5        # listed as absent in qout but fort.99 carries moles
FLAG_NOT_IN_CONTROL = 6  # schema phase not in this simulation's control file
FLAG_NO_DATA = 7         # absent, but no usable qout table for this row
FLAG_NAMES = {
    FLAG_PRESENT: 'present', FLAG_MINIMIZED: 'minimized', FLAG_UPPER_BOUND: 'upper_bound',
    FLAG_NOT_EVALUATED: 'not_evaluated', FLAG_FAILED: 'failed', FLAG_CONFLICT: 'conflict',
    FLAG_NOT_IN_CONTROL: 'not_in_control', FLAG_NO_DATA: 'no_data',
}

# ── row status ───────────────────────────────────────────────────────────────
ROW_OK = 0
ROW_NO_QOUT = 1          # simulation has no qout (or it could not be read)
ROW_NO_BLOCK = 2         # no qout point block matched this fort.56 row
ROW_NO_TABLE = 3         # block found but it holds no affinity table
ROW_STALE_ADD = 4        # last table ADDED a phase and no later table exists
ROW_STALE_RESTORE = 5    # species restored after the last table (assemblage moved)
ROW_INCOMPLETE = 6       # block never reached writeout (killed mid-point / truncated)
ROW_NAMES = {
    ROW_OK: 'ok', ROW_NO_QOUT: 'no_qout', ROW_NO_BLOCK: 'no_block',
    ROW_NO_TABLE: 'no_table', ROW_STALE_ADD: 'stale_add', ROW_STALE_RESTORE: 'stale_restore',
    ROW_INCOMPLETE: 'incomplete',
}

_POINT_HEADER = re.compile(
    r'^-{5,} Pressure \(GPa\), Depth \(km\), Temperature \(K\) -{5,}[ \t]*\r?$', re.M)
_TABLE_MARK = 'Check for phase addition or subtraction'
_TABLE_END = 'Ti at end of phaseadd'
_MU_LABEL = 'Component chemical potentials = '   # a32 in lagcomp.f
_PHASE_EQ = 'Phase equilibria'

PRESENT_TOL = 5.0e-9     # fort.99 prints 8 decimals; anything printed nonzero is present
P_TOL = 0.006            # qout prints P with f16.2
T_TOL = 0.006            # 'Phase equilibria' row prints T with 2 decimals
LOOKAHEAD = 4            # how many qout blocks to skip looking for a row's match


# ════════════════════════════════════════════════════════════════════════════
# qout parsing
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class QoutRow:
    name: str
    affinity: float      # kJ/mol-atom, HeFESTo sign; NaN if the field overflowed
    best_guess: int
    n_guess: int
    iterations: int
    total_iter: int


@dataclass
class QoutPoint:
    P: float
    T: float                                  # final T (NaN if not printed)
    has_table: bool = False
    stale_add: bool = False
    stale_restore: bool = False
    rows: List[QoutRow] = field(default_factory=list)
    mu: Optional[List[float]] = None          # control element order


def _ffloat(s: str) -> float:
    try:
        return float(s)
    except ValueError:
        return math.nan


def _parse_table_row(line: str) -> Optional[QoutRow]:
    """Fixed-width parse of ``(i3,1x,a5,f12.4,4i12,...)``.

    Requiring the four integer fields rejects everything else that can appear in
    the region: WARNING / exsolution lines, and the trailing ``lafmin`` summary
    line (``(i3,1x,a5,90f12.4)``), whose columns 21-33 hold a float.
    """
    if len(line) < 69:
        return None
    try:
        int(line[0:3])
        ints = [int(line[21 + 12 * q: 33 + 12 * q]) for q in range(4)]
    except ValueError:
        return None
    name = line[4:9].strip()
    if not name:
        return None
    return QoutRow(name=name, affinity=_ffloat(line[9:21]),
                   best_guess=ints[0], n_guess=ints[1],
                   iterations=ints[2], total_iter=ints[3])


def _parse_point(block: str) -> QoutPoint:
    P = math.nan
    for ln in block.split('\n', 3)[:3]:
        tok = ln.split()
        if tok:
            P = _ffloat(tok[0])
            break

    T = math.nan
    i = block.rfind(_PHASE_EQ)
    if i >= 0:
        sub = block[i:].split('\n', 3)
        if len(sub) > 2:
            tok = sub[2].split()
            if len(tok) >= 3:
                T = _ffloat(tok[2])

    pt = QoutPoint(P=P, T=T)

    j = block.rfind(_MU_LABEL)
    if j >= 0:
        body = block[j + len(_MU_LABEL):].split('\n', 1)[0].rstrip()
        pt.mu = [_ffloat(body[c:c + 12]) for c in range(0, len(body), 12)]

    t = block.rfind(_TABLE_MARK)
    if t >= 0:
        pt.has_table = True
        end = block.find(_TABLE_END, t)
        region = block[t:end] if end >= 0 else block[t:]
        after = block[end:] if end >= 0 else ''
        pt.stale_add = 'Adding Phase' in region
        pt.stale_restore = 'Restoring species' in after
        for ln in region.split('\n')[2:]:
            r = _parse_table_row(ln.rstrip('\r'))
            if r is not None:
                pt.rows.append(r)
    return pt


def parse_qout(path: str) -> List[QoutPoint]:
    """One ``QoutPoint`` per P-T point block, in file order."""
    with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
        txt = fh.read()
    heads = list(_POINT_HEADER.finditer(txt))
    points = []
    for k, h in enumerate(heads):
        end = heads[k + 1].start() if k + 1 < len(heads) else len(txt)
        points.append(_parse_point(txt[h.end():end]))
    return points


def align_points(P: np.ndarray, T: np.ndarray, points: Sequence[QoutPoint],
                 lookahead: int = LOOKAHEAD) -> List[Optional[int]]:
    """Map each fort.56 row to a qout block by order, verified on P and final T.

    Greedy with a short look-ahead, so a block written for a point that never
    reached fort.56 (or a truncated last block) is skipped instead of shifting
    every later row -- the failure mode the fort.99 warning lines caused.
    """
    out: List[Optional[int]] = []
    j = 0
    for i in range(len(P)):
        hit = None
        for k in range(j, min(j + lookahead + 1, len(points))):
            pt = points[k]
            if not abs(pt.P - P[i]) <= P_TOL:
                continue
            if math.isnan(pt.T) or abs(pt.T - T[i]) <= T_TOL:
                hit = k
                break
        out.append(hit)
        if hit is not None:
            j = hit + 1
    return out


# ════════════════════════════════════════════════════════════════════════════
# schema
# ════════════════════════════════════════════════════════════════════════════
def affinity_phases(indexer) -> List[str]:
    return [p for p in indexer.MELTS_indices.keys() if p not in _NON_PHASE_KEYS]


def affinity_headers(phases: Sequence[str]) -> List[str]:
    return (['sim_id', 'row', 'P(GPa)', 'T(K)', 'qout_status', 'qout_dP', 'qout_dT']
            + [f'mu_{el}(kJ/mol)' for el in ELEMENT_ORDER]
            + [f'A_{p}(kJ/mol-atom)' for p in phases]
            + [f'Aflag_{p}' for p in phases])


def _phase_abbr_to_schema(c2p: Dict[str, str], reverse_map) -> Dict[str, str]:
    """Phase abbreviation -> schema phase name, decided by the same species
    resolution the main import uses to place moles (fea/feg/fee resolve to
    alpha/gamma/epsilon-iron this way; the bare phase-abbreviation lookup says
    'iron' for all three)."""
    votes: Dict[str, Counter] = defaultdict(Counter)
    for sp, ph in c2p.items():
        name = _resolve_component_phase(
            component_abbr=sp,
            component_name=_resolve_component_name_from_abbr(sp),
            reverse_component_phase_map=reverse_map,
            control_component_to_phase_abbr=c2p)
        if name is not None:
            votes[ph][name] += 1
    return {ph: c.most_common(1)[0][0] for ph, c in votes.items()}


def _merge(results: List[Tuple[float, int]]) -> Tuple[float, int]:
    """Combine several control phases mapping to one schema phase (duplicated /
    exsolution copies): present wins, then the most negative finite affinity."""
    flags = [f for _, f in results]
    if FLAG_PRESENT in flags:
        return 0.0, FLAG_PRESENT
    finite = [(a, f) for a, f in results
              if f in (FLAG_MINIMIZED, FLAG_UPPER_BOUND) and np.isfinite(a)]
    if finite:
        return min(finite, key=lambda x: x[0])
    for f in (FLAG_CONFLICT, FLAG_FAILED, FLAG_NOT_EVALUATED, FLAG_NO_DATA):
        for a, g in results:
            if g == f:
                return a, g
    return results[0]


# ════════════════════════════════════════════════════════════════════════════
# per-simulation
# ════════════════════════════════════════════════════════════════════════════
def simulation_affinities(sim_dir: str, sim_id: int, phases: Sequence[str],
                          reverse_map=None) -> Tuple[np.ndarray, dict]:
    """Affinity shadow block for one simulation, or raise if the MAIN import would
    have produced no rows for it."""
    if reverse_map is None:
        reverse_map = _build_reverse_component_phase_map()
    control = os.path.join(sim_dir, 'control')
    f56, f61, f68, f99 = (os.path.join(sim_dir, f) for f in
                          ('fort.56', 'fort.61', 'fort.68', 'fort.99'))

    element_moles, c2p = _parse_control_file(control)
    sys_df = _parse_fort56(f56)
    rho_df = _safe_read_ws_table(f61, skiprows=0)
    vol_df = _safe_read_ws_table(f68, skiprows=0)
    comp_df = _safe_read_ws_table(f99, skiprows=0)
    nrows = min(len(sys_df), len(rho_df), len(vol_df), len(comp_df))
    if nrows <= 0:
        raise ValueError('no rows across required HeFESTo tables')
    sys_df = sys_df.iloc[:nrows]
    comp_df = comp_df.iloc[:nrows]
    P = pd.to_numeric(sys_df.get('P(GPa)'), errors='coerce').fillna(0.0).to_numpy(float)
    T = pd.to_numeric(sys_df.get('T(K)'), errors='coerce').fillna(0.0).to_numpy(float)

    # phase moles from fort.99 (presence test), by the control file's species blocks
    abbr_to_schema = _phase_abbr_to_schema(c2p, reverse_map)
    m_species = Counter(c2p.values())
    ph_moles: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros(nrows))
    for col in comp_df.columns[3:]:
        ph = c2p.get(str(col).strip())
        if ph is None:
            continue
        ph_moles[ph] = ph_moles[ph] + pd.to_numeric(
            comp_df[col], errors='coerce').fillna(0.0).to_numpy(float)
    control_phases = list(dict.fromkeys(c2p.values()))
    schema_to_abbrs: Dict[str, List[str]] = defaultdict(list)
    for ph in control_phases:
        if ph in abbr_to_schema:
            schema_to_abbrs[abbr_to_schema[ph]].append(ph)

    qpath = os.path.join(sim_dir, QOUT)
    points: List[QoutPoint] = []
    if os.path.exists(qpath):
        try:
            points = parse_qout(qpath)
        except OSError:
            points = []
    have_qout = len(points) > 0
    match = align_points(P, T, points) if have_qout else [None] * nrows

    headers = affinity_headers(phases)
    n_ph = len(phases)
    out = np.full((nrows, len(headers)), np.nan)
    out[:, 0] = sim_id
    out[:, 1] = np.arange(nrows)
    out[:, 2] = P
    out[:, 3] = T
    mu0 = 7
    a0 = mu0 + len(ELEMENT_ORDER)
    f0 = a0 + n_ph
    elements = list(element_moles.keys())

    stats = Counter()
    for i in range(nrows):
        k = match[i]
        pt = points[k] if k is not None else None
        if not have_qout:
            status = ROW_NO_QOUT
        elif pt is None:
            status = ROW_NO_BLOCK
        elif math.isnan(pt.T):
            status = ROW_INCOMPLETE
        elif not pt.has_table:
            status = ROW_NO_TABLE
        elif pt.stale_add:
            status = ROW_STALE_ADD
        elif pt.stale_restore:
            status = ROW_STALE_RESTORE
        else:
            status = ROW_OK
        out[i, 4] = status
        stats[f'row_{ROW_NAMES[status]}'] += 1
        if pt is not None:
            out[i, 5] = P[i] - pt.P
            out[i, 6] = T[i] - pt.T
            if pt.mu is not None and len(pt.mu) == len(elements):
                for el, v in zip(elements, pt.mu):
                    if el in ELEMENT_ORDER:
                        out[i, mu0 + ELEMENT_ORDER.index(el)] = v

        listed: Dict[str, List[QoutRow]] = defaultdict(list)
        if pt is not None:
            for r in pt.rows:
                listed[r.name[:5]].append(r)

        for p_idx, schema in enumerate(phases):
            abbrs = schema_to_abbrs.get(schema)
            if not abbrs:
                a, f = math.nan, FLAG_NOT_IN_CONTROL
            else:
                res = []
                for ab in abbrs:
                    present = ph_moles[ab][i] > PRESENT_TOL
                    rows = listed.get(ab[:5], [])
                    if rows:
                        vals = [r for r in rows if np.isfinite(r.affinity)]
                        if present:
                            res.append((min((r.affinity for r in vals), default=math.nan),
                                        FLAG_CONFLICT))
                        elif not vals:
                            res.append((math.nan, FLAG_FAILED))
                        else:
                            best = min(vals, key=lambda r: r.affinity)
                            ub = m_species[ab] > 1 and best.total_iter == 0
                            res.append((best.affinity,
                                        FLAG_UPPER_BOUND if ub else FLAG_MINIMIZED))
                    elif present:
                        res.append((0.0, FLAG_PRESENT))
                    elif pt is None or not pt.has_table:
                        res.append((math.nan, FLAG_NO_DATA))
                    else:
                        res.append((math.nan, FLAG_NOT_EVALUATED))
                a, f = _merge(res)
            out[i, a0 + p_idx] = a
            out[i, f0 + p_idx] = f
            stats[f'flag_{FLAG_NAMES[f]}'] += 1

    n_matched = sum(m is not None for m in match)
    info = dict(sim_id=sim_id, n_rows=nrows, qout_present=have_qout,
                n_qout_points=len(points), n_matched=n_matched,
                n_ok=int(stats['row_ok']),
                n_stale=int(stats['row_stale_add'] + stats['row_stale_restore']),
                fully_matched=bool(have_qout and n_matched == nrows),
                stats=stats)
    return out, info


# ════════════════════════════════════════════════════════════════════════════
# workspace
# ════════════════════════════════════════════════════════════════════════════
def import_HeFESTo_affinities(
    workspace_dir: str,
    indexer,
    affinity_dataname: str = 'DefaultHeFESTo_affinity.csv',
    manifest_name: Optional[str] = None,
    only_sim_ids: Optional[Iterable[int]] = None,
    flush_every: int = 128,
):
    """Write the affinity shadow table for one workspace.

    only_sim_ids : the main import's ``passed_ids``. Strongly recommended: it is the
        only way to guarantee a simulation the main import rejected (for any reason,
        including an exception mid-parse) contributes no rows here either.

    Returns (fully_matched_ids, partial_ids, no_qout_ids, failed_ids, stats).
    A failed simulation that the main import accepted still contributes NaN rows
    (status NO_QOUT) so row parity holds.
    """
    phases = affinity_phases(indexer)
    headers = affinity_headers(phases)
    _ensure_existing_csv_headers_match(affinity_dataname, headers)
    if not os.path.exists(affinity_dataname):
        pd.DataFrame(columns=headers).to_csv(affinity_dataname, index=False)

    keep = None if only_sim_ids is None else set(int(s) for s in only_sim_ids)
    reverse_map = _build_reverse_component_phase_map()
    full_ids: List[int] = []
    partial_ids: List[int] = []
    noq_ids: List[int] = []
    failed_ids: List[int] = []
    totals = Counter()
    manifest: List[dict] = []
    blocks: List[np.ndarray] = []

    def _flush():
        if blocks:
            _write_block_to_csv(affinity_dataname, headers, np.concatenate(blocks, axis=0))
            blocks.clear()

    for sim_id, sim_dir in _list_simulation_dirs(workspace_dir):
        if keep is not None and sim_id not in keep:
            continue
        try:
            block, info = simulation_affinities(sim_dir, sim_id, phases, reverse_map)
        except Exception as exc:
            if keep is None:
                continue      # main import would have produced no rows either
            # Main import accepted it, so its rows exist: hold their place.
            try:
                s56 = _parse_fort56(os.path.join(sim_dir, 'fort.56'))
                nrows = min(len(s56),
                            *(len(_safe_read_ws_table(os.path.join(sim_dir, f), skiprows=0))
                              for f in ('fort.61', 'fort.68', 'fort.99')))
                s56 = s56.iloc[:nrows]
            except Exception:
                print(f'Simulation{sim_id}: affinity import failed AND row count '
                      f'unrecoverable ({exc}); row parity will break.')
                failed_ids.append(sim_id)
                continue
            block = np.full((nrows, len(headers)), np.nan)
            block[:, 0] = sim_id
            block[:, 1] = np.arange(nrows)
            block[:, 2] = pd.to_numeric(s56.get('P(GPa)'), errors='coerce').to_numpy(float)
            block[:, 3] = pd.to_numeric(s56.get('T(K)'), errors='coerce').to_numpy(float)
            block[:, 4] = ROW_NO_QOUT
            block[:, 7 + len(ELEMENT_ORDER) + len(phases):] = FLAG_NO_DATA
            info = dict(sim_id=sim_id, n_rows=nrows, qout_present=False, n_qout_points=0,
                        n_matched=0, n_ok=0, n_stale=0, fully_matched=False,
                        stats=Counter(), error=f'{type(exc).__name__}: {exc}')
            failed_ids.append(sim_id)
        blocks.append(block)
        totals.update(info.pop('stats'))
        manifest.append(info)
        if not info['qout_present']:
            if sim_id not in failed_ids:
                noq_ids.append(sim_id)
        elif info['fully_matched']:
            full_ids.append(sim_id)
        else:
            partial_ids.append(sim_id)
        if len(blocks) >= flush_every:
            _flush()
    _flush()

    if manifest_name is not None and manifest:
        pd.DataFrame(manifest).to_csv(manifest_name, mode='a', index=False,
                                      header=not os.path.exists(manifest_name))
    return full_ids, partial_ids, noq_ids, failed_ids, totals
