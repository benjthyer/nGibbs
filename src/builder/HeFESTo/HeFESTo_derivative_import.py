"""
Import HeFESTo's composition derivatives as shadow tables.

`fort.42` is titled `dndt and dndp by species`: one block per pressure step, one line
per species (`index name dndt dndp`), blocks aligned 1:1 with `fort.56` rows. These are
the TRUE partials at fixed T and fixed P -- verified by chain rule against `fort.99`,

    dn/dP|_isentrope = dn/dP|_T + dn/dT|_P * dT/dP        (median ratio 1.000)

so nothing has to be finite-differenced and no cross-run linkage is needed.

Design: **shadow tables**. Each output CSV carries exactly `indexer.database_headers`,
one row per row of the main import, in the same order. Component slots hold the
derivative instead of the abundance; `P(GPa)` and `T(K)` are copied through so the join
can be checked rather than assumed. A loader can therefore read the main table and the
two shadows with the same code and index the same component columns.

`import_HeFESTo_components` is not modified.
"""
from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from ngibbs.utils.file_utils import (
        _safe_read_ws_table, get_dropped_rows, _parse_control_file, _parse_fort56,
        _resolve_component_name_from_abbr, _resolve_component_phase,
        reconcile_component_name,
        reconcile_component_name,
        _build_reverse_component_phase_map,
    )
except ImportError:
    from src.ngibbs.utils.file_utils import (
        _safe_read_ws_table, get_dropped_rows, _parse_control_file, _parse_fort56,
        _resolve_component_name_from_abbr, _resolve_component_phase,
        reconcile_component_name,
        reconcile_component_name,
        _build_reverse_component_phase_map,
    )

from .HeFESTo_functions import (
    _list_simulation_dirs, _safe_assign, _write_block_to_csv,
    _ensure_existing_csv_headers_match,
)
from .HeFESTo_derivative_recovery import recover_derivatives, write_fort42

FORT42 = 'fort.42'
_BLOCK_MARKER = 'dndt and dndp'

# build_tables() parses the whole parameter set, so it is cached per parameter
# directory rather than rebuilt per simulation.
_PARAM_CACHE: Dict[str, tuple] = {}


def _params_and_tables(control_path: str, param_dir: Optional[str]):
    """(params, tables, resolved_param_dir), cached by parameter directory.

    Resolution order: explicit `param_dir`, then the path recorded inside the run's
    own control file, then the copy packaged with ngibbs. The control file records
    the path on the machine that ran it, which often does not exist locally -- hence
    the fallbacks.
    """
    from ngibbs.engine.EOS_arithmetic.hefesto_vec import load_control, build_tables

    candidates = []
    if param_dir:
        candidates.append(param_dir)
    try:
        with open(control_path) as handle:
            for line in handle:
                token = line.strip()
                if os.path.isdir(token):
                    candidates.append(token)
                    break
    except OSError:
        pass
    packaged = os.path.join(
        os.path.dirname(__import__('ngibbs').__file__),
        'engine', 'EOS_arithmetic', 'HeFESTo_Parameters_010123')
    candidates.append(packaged)

    resolved = next((c for c in candidates if c and os.path.isdir(c)), None)
    if resolved is None:
        raise FileNotFoundError(
            'No usable HeFESTo parameter directory. Pass --param-dir.')

    if resolved not in _PARAM_CACHE:
        params = load_control(control_path, param_dir_override=resolved)
        _PARAM_CACHE[resolved] = (params, build_tables(params, resolved))
    params, tables = _PARAM_CACHE[resolved]
    return params, tables, resolved


# --------------------------------------------------------------------------- #
#  fort.42
# --------------------------------------------------------------------------- #
def load_fort42(path: str) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """-> (species_abbrs, dndt (nB, S), dndp (nB, S)) in mol/K and mol/GPa.

    Tolerates a truncated final block. HeFESTo can be killed mid-write, leaving a last
    block with fewer than `nspec` lines; incomplete blocks are dropped rather than
    raising, so one interrupted run does not cost a whole simulation its derivatives.
    """
    with open(path) as handle:
        lines = handle.readlines()
    blocks = [i for i, line in enumerate(lines) if _BLOCK_MARKER in line]
    if len(blocks) == 0:
        raise ValueError(f'{path}: no "{_BLOCK_MARKER}" blocks found')
    nsp = (blocks[1] - blocks[0] - 1) if len(blocks) > 1 else (len(lines) - 1)
    if nsp <= 0:
        raise ValueError(f'{path}: empty blocks')
    names = [lines[blocks[0] + 1 + k].split()[1] for k in range(nsp)]

    complete = [i for i in blocks if i + nsp < len(lines)]
    dndt = np.zeros((len(complete), nsp), dtype=float)
    dndp = np.zeros_like(dndt)
    for b, i in enumerate(complete):
        for k in range(nsp):
            parts = lines[i + 1 + k].split()
            if len(parts) < 4:
                raise ValueError(f'{path}: malformed line {i + 1 + k + 1}')
            dndt[b, k] = float(parts[2])
            dndp[b, k] = float(parts[3])
    return names, dndt, dndp


def element_total_from_control(element_moles: Dict[str, float]) -> float:
    """Total element moles for the run.

    Constant *along* a profile (the bulk is fixed) but NOT constant *across* runs:
    Htz_transition is 23.63948, BENCHMARK is 24.00450. The invariant in both control
    files is cations = 10.00000, with oxygen -- and so the total -- floating with Fe3+
    content and the Si/Mg ratio. Hardcoding 24 introduces a scale error correlated with
    composition, which on an Mg/Si sweep reads as a trend rather than as noise.
    """
    return float(sum(float(v) for v in element_moles.values()))


# --------------------------------------------------------------------------- #
#  Schema
# --------------------------------------------------------------------------- #
def derivative_schema(indexer) -> Tuple[List[int], List[str], List[tuple]]:
    """Column subset the shadow tables carry.

    Only two kinds of column mean anything for a derivative table: the P-T keys, and
    the per-component values. Everything else in the main schema -- the fort.56
    properties, the bulk composition, per-phase rho/V/mass -- is either constant down a
    simulation or not a composition derivative at all, so it would just be dead weight
    repeated twice.

    Returns (column_indices, header_names, placements) where `placements` is the list
    of (phase_name, component_name, column_index) the loader fills.
    """
    headers = list(indexer.database_headers)
    skip = {'System_main', 'Bulk_comp', 'Bulk_comp_elements'}
    component_names = {str(x) for x in getattr(indexer, 'label_names', [])}

    keys = []
    for attr in ('P(GPa)', 'T(K)'):
        idx = indexer.MELTS_indices['System_main'].get(attr)
        if idx is None:
            raise KeyError(f'indexer has no System_main/{attr}')
        keys.append(int(idx))

    placements = []
    for phase_name, phase_map in indexer.MELTS_indices.items():
        if phase_name in skip:
            continue
        for attr, idx in phase_map.items():
            if attr in component_names:
                placements.append((phase_name, attr, int(idx)))
    placements.sort(key=lambda t: t[2])

    cols = keys + [t[2] for t in placements]
    return cols, [headers[i] for i in cols], placements


# --------------------------------------------------------------------------- #
#  Import
# --------------------------------------------------------------------------- #
def import_HeFESTo_derivatives(
    workspace_dir: str,
    indexer,
    dndp_dataname: str = 'DefaultHeFESTo_dndP.csv',
    dndt_dataname: str = 'DefaultHeFESTo_dndT.csv',
    manifest_name: Optional[str] = None,
    normalise: bool = False,
    flush_every: int = 128,
    recover: bool = True,
    param_dir: Optional[str] = None,
    recover_nsmall_rel: float = 0.0,
    write_recovered: bool = False,
):
    """Write dn/dP and dn/dT shadow tables for one workspace.

    **Row parity is guaranteed.** The shadows emit one row for every row
    `import_HeFESTo_components` emits, so the three tables concatenate positionally
    with no join. That means the accept/reject logic here mirrors the main import
    exactly -- same ordering, same required-file checks, same
    `nrows = min(len(table))` truncation -- and a simulation whose *derivatives* fail
    still contributes `nrows` rows, filled with NaN, rather than vanishing. Skipping
    would shift every subsequent row, the same class of bug as the stray fort.99
    diagnostic lines.

    A simulation that the main import rejects is rejected here too, so it contributes
    no rows to either table.

    Columns are the trimmed set from `derivative_schema`: P, T, and the components.

    recover : reconstruct derivatives from fort.99 + fort.56 when fort.42 is absent.
        Verified against a run that does carry one: mean correlation 0.9972, rms error
        2.6e-5 against a signal rms of 0.859.
    write_recovered : also write the reconstruction to `fort.42` in the simulation
        directory, so a re-import reads it instead of recomputing. Off by default --
        it modifies the data directory.

    Returns (passed_ids, malformed_ids, missing_fort42_ids, recovered_ids). `passed`
    counts simulations with real derivative values; `malformed` ones contributed NaN
    rows and still hold their place.
    """
    passed_ids: List[int] = []
    malformed_ids: List[int] = []
    missing42_ids: List[int] = []
    recovered_ids: List[int] = []

    reverse_map = _build_reverse_component_phase_map()
    sim_dirs = _list_simulation_dirs(workspace_dir)
    cols, headers, placements = derivative_schema(indexer)
    width = indexer.get_max_index() + 1

    for name in (dndp_dataname, dndt_dataname):
        _ensure_existing_csv_headers_match(name, headers)
        if not os.path.exists(name):
            pd.DataFrame(columns=headers).to_csv(name, index=False)

    dp_blocks: List[np.ndarray] = []
    dt_blocks: List[np.ndarray] = []
    manifest: List[dict] = []
    since_flush = 0

    def _flush():
        if dp_blocks:
            _write_block_to_csv(dndp_dataname, headers,
                                np.concatenate(dp_blocks, axis=0) if len(dp_blocks) > 1 else dp_blocks[0])
        if dt_blocks:
            _write_block_to_csv(dndt_dataname, headers,
                                np.concatenate(dt_blocks, axis=0) if len(dt_blocks) > 1 else dt_blocks[0])
        dp_blocks.clear()
        dt_blocks.clear()

    for sim_id, sim_dir in sim_dirs:
        control_path = os.path.join(sim_dir, 'control')
        f56 = os.path.join(sim_dir, 'fort.56')
        f61 = os.path.join(sim_dir, 'fort.61')
        f68 = os.path.join(sim_dir, 'fort.68')
        f99 = os.path.join(sim_dir, 'fort.99')
        f42 = os.path.join(sim_dir, FORT42)

        # ---- stage 1: the checks the main import makes. Failing here means the main
        # ---- import contributes no rows either, so neither do we.
        try:
            sim_files = [e for e in os.listdir(sim_dir)
                         if os.path.isfile(os.path.join(sim_dir, e))]
        except OSError:
            continue
        if len(sim_files) == 0:
            continue
        if not os.path.exists(control_path) or any(
                not os.path.exists(q) for q in (f56, f61, f68, f99)):
            continue

        # ---- stage 2: row count, by the main import's rule
        try:
            element_moles, control_component_to_phase_abbr = _parse_control_file(control_path)
            N_el = element_total_from_control(element_moles)
            sys_df = _parse_fort56(f56)
            rho_df = _safe_read_ws_table(f61, skiprows=0)
            vol_df = _safe_read_ws_table(f68, skiprows=0)
            comp_df = _safe_read_ws_table(f99, skiprows=0)
            sim_was_offset = f99 in get_dropped_rows()
            nrows = min(len(sys_df), len(rho_df), len(vol_df), len(comp_df))
            if nrows <= 0:
                continue
            sys_df = sys_df.iloc[:nrows].reset_index(drop=True)
        except Exception:
            continue

        P_col = pd.to_numeric(sys_df.get('P(GPa)'), errors='coerce').fillna(0.0).to_numpy(float)
        T_col = pd.to_numeric(sys_df.get('T(K)'), errors='coerce').fillna(0.0).to_numpy(float)

        def _blank():
            """nrows of NaN with the P-T keys still filled, so the row keeps its place
            and can still be identified."""
            a = np.full((nrows, width), np.nan, dtype=float)
            _safe_assign(a, indexer, 'System_main', 'P(GPa)', P_col)
            _safe_assign(a, indexer, 'System_main', 'T(K)', T_col)
            return a[:, cols]

        # ---- stage 3: derivatives. Any failure yields NaN rows, never a skip.
        have42 = os.path.exists(f42)
        source = 'fort.42' if have42 else ('recovered' if recover else 'unavailable')
        try:
            if have42:
                sp_names, dndt, dndp = load_fort42(f42)
            elif recover:
                # A deterministic function of the assemblage and the P-T state -- the
                # same Hessian system physub.f solves -- so a build that never wrote
                # fort.42 loses nothing recoverable.
                params, tables, _ = _params_and_tables(control_path, param_dir)
                dndt, dndp, sp_names = recover_derivatives(
                    sim_dir, params, nsmall_rel=recover_nsmall_rel, tables=tables)
                if write_recovered:
                    write_fort42(f42, sp_names, dndt, dndp)
            else:
                raise FileNotFoundError('fort.42 absent and recovery disabled')

            if dndt.shape[0] < nrows:
                raise ValueError(f'derivatives have {dndt.shape[0]} blocks < {nrows} rows')
            dndt = dndt[:nrows]
            dndp = dndp[:nrows]
            if normalise:
                dndt = dndt / N_el
                dndp = dndp / N_el

            out_dp = np.zeros((nrows, width), dtype=float)
            out_dt = np.zeros((nrows, width), dtype=float)
            for out in (out_dp, out_dt):
                _safe_assign(out, indexer, 'System_main', 'P(GPa)', P_col)
                _safe_assign(out, indexer, 'System_main', 'T(K)', T_col)

            # Place by NAME. fort.42 uses parameter-file order (an, ab, sp, hc, smag,
            # picr, en, ...), which does not match fort.99's columns positionally.
            placed = 0
            for k, abbr in enumerate(sp_names):
                abbr = str(abbr).strip()
                component_name = _resolve_component_name_from_abbr(abbr)
                phase_name = _resolve_component_phase(
                    component_abbr=abbr,
                    component_name=component_name,
                    reverse_component_phase_map=reverse_map,
                    control_component_to_phase_abbr=control_component_to_phase_abbr,
                )
                if phase_name is None:
                    continue
                # Same reconciliation the component importer needs: 'smag' resolves to
                # 'magnetite-spinel' while the schema key is 'magnetite'. Without it the
                # spinel magnetite derivative column is written nowhere and stays zero,
                # exactly as the abundance column did.
                schema_name = reconcile_component_name(
                    indexer, phase_name, component_name, abbr)
                if schema_name is None:
                    peak = float(np.nanmax(np.abs(dndp[:, k]))) if dndp.shape[0] else 0.0
                    if peak > 0:
                        raise KeyError(
                            f"fort.42 species '{abbr}' resolves to ({component_name}, "
                            f"{phase_name}) with no matching schema column, and carries up "
                            f"to {peak:.4e} mol/GPa. Refusing to drop it silently.")
                    continue
                _safe_assign(out_dp, indexer, phase_name, schema_name, dndp[:, k])
                _safe_assign(out_dt, indexer, phase_name, schema_name, dndt[:, k])
                placed += 1

            dp_blocks.append(out_dp[:, cols])
            dt_blocks.append(out_dt[:, cols])
            passed_ids.append(sim_id)
            if source == 'recovered':
                recovered_ids.append(sim_id)
            n_species = len(sp_names)
        except Exception:
            dp_blocks.append(_blank())
            dt_blocks.append(_blank())
            if have42 or recover:
                malformed_ids.append(sim_id)
            else:
                missing42_ids.append(sim_id)
            source = 'unavailable'
            placed = 0
            n_species = 0

        manifest.append(dict(sim_id=sim_id, n_rows=nrows, N_el=N_el,
                             n_species=n_species, n_placed=placed,
                             deriv_source=source,
                             fort99_offset=bool(sim_was_offset)))

        since_flush += 1
        if since_flush >= flush_every:
            _flush()
            since_flush = 0

    _flush()

    if manifest_name is not None:
        pd.DataFrame(manifest).to_csv(manifest_name, mode='a', index=False,
                                      header=not os.path.exists(manifest_name))

    return passed_ids, malformed_ids, missing42_ids, recovered_ids


# --------------------------------------------------------------------------- #
#  Verification
# --------------------------------------------------------------------------- #
def monotonic_segments(P: np.ndarray, min_len: int = 5) -> List[Tuple[int, int]]:
    """Split rows into runs of strictly increasing pressure.

    A simulation is not necessarily one continuum. A stacked-adiabat run holds many
    profiles end to end -- one observed workspace had 29 isentropes concatenated, with
    P resetting 140 -> 0 GPa at each of 28 seams -- and a P-T grid run resets P at every
    isotherm. Differencing straight through a seam produces a |dP| of order the whole
    pressure range, which then dominates any relative outlier threshold and quietly
    discards the real signal.
    """
    breaks = np.where(np.diff(P) <= 0)[0] + 1
    bounds = np.concatenate([[0], breaks, [len(P)]])
    return [(int(bounds[i]), int(bounds[i + 1]))
            for i in range(len(bounds) - 1)
            if bounds[i + 1] - bounds[i] >= min_len]


def verify_chain_rule(sim_dir: str, species: Optional[List[str]] = None,
                      rel_tol: float = 0.05, dndt=None, dndp=None,
                      names: Optional[Sequence[str]] = None,
                      max_step_frac: float = 1.0) -> dict:
    """Chain-rule acceptance test for one simulation directory.

        dn/dP|_along = dn/dP|_T + dn/dT|_P * dT/dP

    Catches misalignment between the derivatives and `fort.99`, wrong species mapping,
    and normalisation mistakes -- all silent otherwise. Returns per-species ratios;
    each should be 1.000.

    Differencing is **pairwise within monotonic-P segments**, not `np.gradient` over the
    whole file: see `monotonic_segments`. Pass `dndt`/`dndp`/`names` to check an
    in-memory reconstruction instead of the file on disk.
    """
    if dndt is None or dndp is None:
        names, dndt, dndp = load_fort42(os.path.join(sim_dir, FORT42))
    names = list(names)

    numeric = re.compile(r'^[\s\-\d\.eEdD\+]+$')
    with open(os.path.join(sim_dir, 'fort.99')) as handle:
        lines = handle.readlines()
    cols = lines[0].split()
    rows = [l for l in lines[1:]
            if numeric.match(l.rstrip('\n')) and len(l.split()) == len(cols)]
    arr = np.array([[float(x) for x in l.split()] for l in rows])
    # 'Gibbs' and 'Quality' trail the species columns and are NOT species
    sp = [c for c in cols[3:] if c not in ('Gibbs', 'Quality')]
    N = arr[:, [cols.index(c) for c in sp]]
    P, T = arr[:, 0], arr[:, 2]

    n = min(len(P), dndp.shape[0])
    P, T, N, dndt, dndp = P[:n], T[:n], N[:n], dndt[:n], dndp[:n]
    segments = monotonic_segments(P)
    pos = {a: i for i, a in enumerate(names)}

    if species is None:
        moved = np.abs(dndp).max(axis=0)
        species = [names[i] for i in np.argsort(-moved)[:8] if names[i] in sp]

    out: dict = {}
    for s in species:
        if s not in pos or s not in sp:
            continue
        k, j = sp.index(s), pos[s]
        num_all, chain_all = [], []
        for a, b in segments:
            dP = np.diff(P[a:b])
            if not np.all(dP > 0):
                continue
            dTdP = np.diff(T[a:b]) / dP
            n0, n1 = N[a:b - 1, k], N[a + 1:b, k]
            # A secant approximates the derivative at the midpoint to O(h^2), but only
            # where the function is smooth. Across a phase-in or phase-out the error is
            # O(1), so those pairs are dropped -- otherwise a coarse grid makes a
            # correct derivative look wrong.
            # The midpoint form is only a valid reference where the derivative itself
            # is near-linear across the step; otherwise the O(h^2) error becomes O(1)
            # and the check loses its power. Filter on how much the analytic partial
            # changes between the two endpoints, not on how much n changes -- an
            # abundant phase can move slowly in relative terms while its derivative
            # swings wildly. Observed: forsterite at 1 GPa steps reads 1.00 above
            # 3 GPa and 0.52 below, where the plagioclase/spinel/garnet reactions run
            # near-univariantly and dndp changes sign inside a single step.
            d0, d1 = dndp[a:b - 1, j], dndp[a + 1:b, j]
            mean_d = 0.5 * np.abs(d0 + d1)
            smooth = ((n0 > 0) & (n1 > 0)
                      & (np.abs(d1 - d0) <= max_step_frac * np.maximum(mean_d, 1e-30)))
            if smooth.sum() == 0:
                continue
            num_all.append((np.diff(N[a:b, k]) / dP)[smooth])
            chain_all.append((0.5 * (dndp[a:b - 1, j] + dndp[a + 1:b, j])
                              + 0.5 * (dndt[a:b - 1, j] + dndt[a + 1:b, j]) * dTdP)[smooth])
        if not num_all:
            continue
        num = np.concatenate(num_all)
        chain = np.concatenate(chain_all)
        if num.size < 5:
            continue
        # percentile rather than max, so one outlier cannot set the threshold
        thresh = 0.02 * np.percentile(np.abs(num), 99)
        sel = np.abs(num) > max(thresh, 1e-12)
        if sel.sum() < 5:
            continue
        out[s] = float(np.median(chain[sel] / num[sel]))

    out['_segments'] = len(segments)
    out['_median_dP'] = float(np.median(np.concatenate(
        [np.diff(P[a:b]) for a, b in segments]))) if segments else float('nan')
    # The midpoint reference carries O(h^2) error, so a 1 GPa profile lands a few
    # percent off while a 10 MPa one sits at 1.000. Scale the tolerance with the step
    # rather than failing every coarse run. The faults this check exists to catch --
    # block misalignment, wrong species mapping, a normalisation slip -- show up as
    # ratios of 0.001 or 0.5, nowhere near the boundary.
    tol = rel_tol + 0.05 * min(out['_median_dP'] if np.isfinite(out['_median_dP']) else 0.0, 1.0)
    out['_tolerance'] = float(tol)
    out['_pass'] = all(abs(v - 1.0) <= tol
                       for kk, v in out.items() if not kk.startswith('_'))
    return out
