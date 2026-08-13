"""
Deep (second-generation) phase-change resampling.

The first resampling pass turns coarse isentropes into P-T *grid* searches bracketing
each phase change. This module handles the pass after that: it treats a grid simulation
not as one continuum but as a stack of independent 1-D scans, finds the phase-change
bounds within each scan, and prepares fine-step control files that traverse them.

Two scan axes:

    isotherm  fixed T, sweep P.   Crosses a phase boundary steeply, so the edges are
                                  sharply located. Natural step 0.01 GPa.
    isobar    fixed P, sweep T.   Crosses the same boundary at a shallow angle, so it
                                  spends many samples inside the two-phase field but
                                  locates its edges less precisely.

Choosing the isobaric step
--------------------------
Do not pick it independently of the pressure step. A boundary with Clapeyron slope
`dP/dT` is crossed over `dT = dP / |dP/dT|` in temperature, so matching resolution
across the boundary means

    dT_step = dP_step / |dP/dT|

At 0.01 GPa and the post-spinel slope of 1.71 MPa/K that is ~5.8 K. For post-perovskite
at ~10 MPa/K it is ~1 K. A single fixed 1 K step therefore oversamples the transition
zone roughly sixfold while being about right for the deep lower mantle, and 0.1 K
oversamples everywhere by another order of magnitude. `deep_temperature_step` computes
it per boundary when a Clapeyron estimate is available and falls back to a constant.

Nothing in `HeFESTo_functions.py` is modified.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from ngibbs.utils.file_utils import (
        _safe_read_ws_table, _parse_control_file, _parse_fort56,
        _resolve_component_name_from_abbr, _resolve_component_phase,
        _build_reverse_component_phase_map,
    )
except ImportError:
    from src.ngibbs.utils.file_utils import (
        _safe_read_ws_table, _parse_control_file, _parse_fort56,
        _resolve_component_name_from_abbr, _resolve_component_phase,
        _build_reverse_component_phase_map,
    )

from .HeFESTo_functions import (
    _list_simulation_dirs, _safe_assign, _write_block_to_csv,
    _ensure_existing_csv_headers_match, _compute_bulk_from_elements,
    _normalize_element_label, _build_control_lines, _next_available_batch_dir,
)

TRANSITION_TOL = 1e-6
DEFAULT_DP = 0.01          # GPa
DEFAULT_DT = 5.0           # K, see the module docstring on why not 1.0
DEFAULT_CLAPEYRON = 1.71   # MPa/K, post-spinel, used when none is supplied


# --------------------------------------------------------------------------- #
#  Scan splitting
# --------------------------------------------------------------------------- #
def split_scans(P: np.ndarray, T: np.ndarray, axis: str,
                rel_tol: float = 1e-6) -> List[np.ndarray]:
    """Group grid rows into 1-D scans.

    axis='isotherm' groups by unique T and orders each group by P;
    axis='isobar'   groups by unique P and orders each group by T.

    Grouping is by value, not by row order, because the grid's traversal order is a
    property of how HeFESTo walked it and should not be assumed.
    """
    key, run = (T, P) if axis == 'isotherm' else (P, T)
    scale = max(float(np.nanmax(np.abs(key))), 1.0)
    rounded = np.round(key / (scale * rel_tol)) * (scale * rel_tol)
    scans = []
    for value in np.unique(rounded):
        idx = np.where(rounded == value)[0]
        if idx.size < 2:
            continue
        scans.append(idx[np.argsort(run[idx])])
    return scans


def boundary_rows_in_scan(phase_moles: np.ndarray, order: np.ndarray,
                          tol: float = TRANSITION_TOL) -> List[int]:
    """Row indices bounding each zero/non-zero phase transition along one scan.

    Same rule as the first-pass collector -- the transition row and the one before it --
    but applied within a scan rather than across the whole simulation, so a jump between
    the end of one isotherm and the start of the next is never mistaken for a
    transition.
    """
    out: List[int] = []
    present = np.abs(phase_moles[order]) > tol
    for j in range(present.shape[1]):
        col = present[:, j]
        for k in (np.where(col[1:] != col[:-1])[0] + 1):
            out.append(int(order[k - 1]))
            out.append(int(order[k]))
    return out


# --------------------------------------------------------------------------- #
#  Collection
# --------------------------------------------------------------------------- #
def collect_deep_phase_changes(
    workspace_dir: str,
    indexer,
    out_csv: str,
    axis: str = 'isotherm',
    tol: float = TRANSITION_TOL,
    flush_every: int = 128,
):
    """Write phase-change bound pairs found within 1-D scans of grid simulations.

    Output schema is `indexer.database_headers`, identical to the first-pass
    `--phase-change-dataname` file, so the prepare scripts consume either.

    axis : 'isotherm', 'isobar', or 'both'. With 'both' each simulation contributes
        bounds found along constant-T scans and along constant-P scans; pairs are
        emitted in that order.

    Returns (passed_ids, malformed_ids, n_pairs).
    """
    axes = ['isotherm', 'isobar'] if axis == 'both' else [axis]
    for a in axes:
        if a not in ('isotherm', 'isobar'):
            raise ValueError(f"axis must be isotherm, isobar or both; got {axis!r}")

    reverse_map = _build_reverse_component_phase_map()
    headers = indexer.database_headers
    _ensure_existing_csv_headers_match(out_csv, headers)
    if not os.path.exists(out_csv):
        pd.DataFrame(columns=headers).to_csv(out_csv, index=False)

    passed: List[int] = []
    malformed: List[int] = []
    blocks: List[np.ndarray] = []
    n_pairs = 0
    since_flush = 0

    component_count = len(indexer.label_names)
    p_to_c = np.asarray(indexer.phaseToCompMap, dtype=float)

    for sim_id, sim_dir in _list_simulation_dirs(workspace_dir):
        control_path = os.path.join(sim_dir, 'control')
        paths = {k: os.path.join(sim_dir, f'fort.{k}') for k in ('56', '61', '68', '99')}
        if not os.path.exists(control_path) or any(
                not os.path.exists(v) for v in paths.values()):
            continue
        try:
            element_moles, control_component_to_phase_abbr = _parse_control_file(control_path)
            bulk_wt, bulk_el, system_mass = _compute_bulk_from_elements(element_moles)
            sys_df = _parse_fort56(paths['56'])
            rho_df = _safe_read_ws_table(paths['61'], skiprows=0)
            vol_df = _safe_read_ws_table(paths['68'], skiprows=0)
            comp_df = _safe_read_ws_table(paths['99'], skiprows=0)
            nrows = min(len(sys_df), len(rho_df), len(vol_df), len(comp_df))
            if nrows < 2:
                continue
            sys_df = sys_df.iloc[:nrows].reset_index(drop=True)
            comp_df = comp_df.iloc[:nrows].reset_index(drop=True)

            out = np.zeros((nrows, indexer.get_max_index() + 1), dtype=float)
            num = lambda s: pd.to_numeric(s, errors='coerce').fillna(0.0).to_numpy(float)
            P = num(sys_df.get('P(GPa)'))
            T = num(sys_df.get('T(K)'))
            _safe_assign(out, indexer, 'System_main', 'P(GPa)', P)
            _safe_assign(out, indexer, 'System_main', 'T(K)', T)
            for attr in ('rho(g/cm^3)', 'VS(km/s)', 'VP(km/s)', 'S(J/g/K)',
                         'cp(J/g/K)', 'KS(GPa)', 'H(kJ/g)', 'alpha(1e5_K^-1)'):
                if attr in sys_df:
                    _safe_assign(out, indexer, 'System_main', attr, num(sys_df[attr]))
            _safe_assign(out, indexer, 'System_main', 'mass (gm)',
                         np.full(nrows, system_mass, dtype=float))
            for oxide, wt in bulk_wt.items():
                _safe_assign(out, indexer, 'Bulk_comp', oxide, np.full(nrows, wt))
            for element, value in bulk_el.items():
                _safe_assign(out, indexer, 'Bulk_comp_elements', element,
                             np.full(nrows, value))

            for abbr in list(comp_df.columns)[3:-2]:
                a = str(abbr).strip()
                cname = _resolve_component_name_from_abbr(a)
                pname = _resolve_component_phase(
                    component_abbr=a, component_name=cname,
                    reverse_component_phase_map=reverse_map,
                    control_component_to_phase_abbr=control_component_to_phase_abbr)
                if pname is not None:
                    _safe_assign(out, indexer, pname, cname, num(comp_df[abbr]))

            component_moles = np.zeros((nrows, component_count), dtype=float)
            for phase_name, c_indices in indexer.label_indices.items():
                phase_map = indexer.MELTS_indices.get(phase_name)
                if phase_map is None:
                    continue
                for c_idx in np.atleast_1d(c_indices):
                    c = int(c_idx)
                    col = phase_map.get(str(indexer.label_names[c]))
                    if col is not None:
                        component_moles[:, c] = out[:, col]
            phase_moles = component_moles @ p_to_c.T

            rows: List[int] = []
            for a in axes:
                for order in split_scans(P, T, a):
                    rows += boundary_rows_in_scan(phase_moles, order, tol)
            if not rows:
                passed.append(sim_id)
                continue

            blocks.append(out[rows, :])
            n_pairs += len(rows) // 2
            passed.append(sim_id)
        except Exception:
            malformed.append(sim_id)
            continue

        since_flush += 1
        if since_flush >= flush_every:
            _write_block_to_csv(out_csv, headers, np.concatenate(blocks, axis=0))
            blocks.clear()
            since_flush = 0

    if blocks:
        _write_block_to_csv(out_csv, headers, np.concatenate(blocks, axis=0))
    return passed, malformed, n_pairs


# --------------------------------------------------------------------------- #
#  Preparation
# --------------------------------------------------------------------------- #
def deep_temperature_step(clapeyron_MPa_per_K: Optional[float] = None,
                          dP: float = DEFAULT_DP,
                          floor: float = 0.25, ceiling: float = 25.0) -> float:
    """dT that resolves a boundary as finely as `dP` does: dT = dP / |dP/dT|."""
    slope = abs(clapeyron_MPa_per_K or DEFAULT_CLAPEYRON)
    return float(np.clip(dP * 1000.0 / slope, floor, ceiling))


def prepare_deep_tree(
    directory: Path,
    phase_path: Path,
    CONTROL_DIR: Path,
    axis: str = 'isotherm',
    dP: float = DEFAULT_DP,
    dT: Optional[float] = None,
    clapeyron_MPa_per_K: Optional[float] = None,
    limit: Optional[int] = None,
    max_steps: int = 4000,
) -> int:
    """Prepare fine-step 1-D scans from phase-boundary pairs.

    Deep mode does not accept grid bounds: each pair must lie on a single scan.
    An isotherm pair must share T (P varies), an isobar pair must share P. A pair that
    varies in both is a grid bound from the first pass and is skipped with a count,
    rather than being silently turned into a grid.

    Emits `[p_min, p_max, n_p, t_min, t_max, n_t, 0, ...]` control run codes with the
    stepped axis resolved at `dP` or `dT` and the other axis pinned.

    Returns the number of simulations written.
    """
    directory, phase_path = Path(directory), Path(phase_path)
    if axis not in ('isotherm', 'isobar'):
        raise ValueError("axis must be 'isotherm' or 'isobar'")
    if not phase_path.is_file():
        raise FileNotFoundError(f'Phase boundary file not found: {phase_path}')

    df = pd.read_csv(phase_path, dtype=str)
    if df.empty or len(df) % 2 != 0:
        raise ValueError('Phase boundary file must hold an even, non-zero number of rows')

    pcol, tcol = 'P(GPa)(System_main)', 'T(K)(System_main)'
    for c in (pcol, tcol):
        if c not in df.columns:
            raise ValueError(f'Phase boundary file must contain {c!r}')
    bulk_cols = [c for c in df.columns if str(c).endswith('(Bulk_comp_elements)')]
    if not bulk_cols:
        raise ValueError('Phase boundary file does not contain Bulk_comp_elements columns')

    n_pairs = len(df) // 2
    order = list(range(n_pairs))
    if limit is not None:
        if limit <= 0:
            raise ValueError('limit must be positive')
        import random
        order = random.sample(order, min(limit, n_pairs))

    if dT is None:
        dT = deep_temperature_step(clapeyron_MPa_per_K, dP)

    directory.mkdir(parents=True, exist_ok=True)
    batch_dir = _next_available_batch_dir(directory, 1)
    batch_dir.mkdir(parents=True, exist_ok=False)
    batch_number = int(batch_dir.name.removeprefix('Batch'))
    in_batch = 0
    written = 0
    skipped_grid = 0

    for pair_idx in order:
        pair = df.iloc[pair_idx * 2:(pair_idx + 1) * 2].reset_index(drop=True)
        p = pd.to_numeric(pair[pcol], errors='coerce').to_numpy(float)
        t = pd.to_numeric(pair[tcol], errors='coerce').to_numpy(float)
        if np.isnan(p).any() or np.isnan(t).any():
            continue

        p_min, p_max = float(p.min()), float(p.max())
        t_min, t_max = float(t.min()), float(t.max())
        p_varies = not np.isclose(p_min, p_max)
        t_varies = not np.isclose(t_min, t_max)

        if axis == 'isotherm':
            if t_varies or not p_varies:
                skipped_grid += 1
                continue
            n_p = int(min(max(round((p_max - p_min) / dP), 1), max_steps))
            run_code = [p_min, p_max, n_p, t_min, t_min, 0, 0, 0, 0, 0, 0]
        else:
            if p_varies or not t_varies:
                skipped_grid += 1
                continue
            n_t = int(min(max(round((t_max - t_min) / dT), 1), max_steps))
            run_code = [p_min, p_min, 0, t_min, t_max, n_t, 0, 0, 0, 0, 0]

        if in_batch >= 1000:
            batch_number += 1
            batch_dir = _next_available_batch_dir(directory, batch_number)
            batch_dir.mkdir(parents=True, exist_ok=False)
            in_batch = 0
        sim_dir = batch_dir / f'Simulation{in_batch + 1}'
        sim_dir.mkdir(parents=True, exist_ok=True)
        in_batch += 1

        if CONTROL_DIR.is_file() or CONTROL_DIR.name in ('shallowHeFESTo', 'deepHeFESTo'):
            template = CONTROL_DIR
        elif p_min < 23:
            template = CONTROL_DIR / 'shallowHeFESTo'
        else:
            template = CONTROL_DIR / 'deepHeFESTo'

        control_path = sim_dir / 'control'
        shutil.copy2(template, control_path)
        with open(control_path, 'r', encoding='utf-8', errors='ignore') as handle:
            lines = [l.rstrip('\n') for l in handle]

        ref = pair.iloc[0]
        element_values = {
            _normalize_element_label(c.split('(', 1)[0]): float(ref[c]) for c in bulk_cols
        }
        lines = _build_control_lines(template_lines=lines,
                                     element_values=element_values,
                                     run_code=run_code)
        with open(control_path, 'w', encoding='utf-8') as handle:
            handle.write('\n'.join(lines) + '\n')
        written += 1

    if skipped_grid:
        print(f'  [deep] skipped {skipped_grid} pairs that vary in both P and T '
              f'(grid bounds, not {axis} bounds)')
    return written
