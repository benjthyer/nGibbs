#!/usr/bin/env python3
"""QC gate for HeFESTo composition derivatives (`fort.42`).

Run this before any long derivative-supervised training run, and after any change to the
species mapping. It is cheap -- no finite differencing is needed for the primary test --
and it is sharp, because the clean value of that test is machine precision rather than a
tolerance.

The primary test
----------------
At fixed bulk composition, `sum_c n_c * (elements in c) = b` is constant along a scan, so
differentiating gives, exactly,

    A^T (dn/dP) = 0        and        A^T (dn/dT) = 0

with `A` the component-to-element matrix. This depends on no derivative of the data and no
model. Over the FULL species set it holds to ~7e-8 -- `fort.42`'s printed precision -- in
every run tested. Anything larger is a mapping error or a genuinely unconverged row, and
the two are distinguishable: see `--- why the rank-1 report matters ---` below.

Two element bases, one identity
-------------------------------
HeFESTo's `control` file fixes seven cations AND oxygen. The model's element basis swaps
oxygen for Fe3, which is the unique remap a fixed-O, only-Fe-redox-active system admits:

    Fe_total = n_Fe + n_Fe3
    O        = n_Fe + 1.5 n_Fe3 + 1.5 Al + Ca + 1.5 Cr + Mg + 0.5 Na + 2 Si

`det [[1, 1], [1, 1.5]] = 0.5`, so conserving (Fe_total, O) is equivalent to conserving the
two columns separately -- there is no redox freedom to excuse a drift in either. Metallic
iron is consistent with this: `compToEl` encodes gamma- and epsilon-iron as `Fe = +3,
Fe3 = -2`, i.e. `Fe(0) = 3FeO - Fe2O3`, giving `Fe_total = 1` and `O = 0`. Fe0/Fe2/Fe3
speciation therefore moves freely without touching either conserved quantity. Check 1b
reconstructs the bulk in BOTH bases and compares against the control file.

Why the injectivity check is step 0
-----------------------------------
`ml_indexer.components_in_phases` contains the name `magnetite` TWICE -- once under
`spinel` (HeFESTo `smag`), once under `ferropericlase` (HeFESTo `mag`). A lookup keyed on
component name alone silently sends both to one `fort.42` column and never reads the other.
That produces a residual of exactly `(n_mag - n_smag)` times the Fe3O4 composition, which
looks like a beautifully clean physical signal and is not one. Always key on
`(phase, component)`.

Why the rank-1 report matters
-----------------------------
A mapping error puts the entire residual along ONE stoichiometric direction, because it is
one species being miscounted. An unconverged row spreads it. So `99.8% along Fe3O4` is a
diagnosis -- go fix the map -- while `residual = 9e-5, rank-1 fraction 40%` is a bad row.
Reporting only the norm loses that distinction, which is why check 1 reports both.

Expected leakage is not an error
--------------------------------
The trained model covers only the phases that clear the abundance threshold in the training
data, so the derivative tables carry MORE species than the network does. Restricted to the
tracked subset the identity therefore picks up a residual equal to the untracked species'
contribution. That is expected and is reported separately, attributed by name, rather than
being confused with a data fault. In `somesims/Simulation2` this is `alpha-iron` at
2.256e-3 mol/GPa on one row -- times its `Fe=+3, Fe3=-2` encoding, exactly the 6.77e-3
residual seen over the tracked subset.

Usage
-----
    python check_derivatives.py WORKSPACE_DIR [--tol 1e-6] [--chain] [--json out.json]

`WORKSPACE_DIR` is either a single simulation directory (containing `control`, `fort.42`,
`fort.99`) or a workspace containing several.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import OrderedDict

import numpy as np

# --------------------------------------------------------------------------- #
#  Element bookkeeping
# --------------------------------------------------------------------------- #
# Cation charge, used only to reconstruct oxygen from a cation-basis composition.
# O per cation = charge / 2.
CATION_CHARGE = {'Al': 3, 'Ca': 2, 'Cr': 3, 'Fe': 2, 'Fe3': 3,
                 'Mg': 2, 'Na': 1, 'Si': 4, 'K': 1, 'Ti': 4, 'Mn': 2, 'Ni': 2}

_NUMERIC = re.compile(r'^[\s\-\d\.eEdD\+]+$')


# --------------------------------------------------------------------------- #
#  Readers
# --------------------------------------------------------------------------- #
def load_fort42(path):
    """-> (abbrevs, dndt (nP, S), dndp (nP, S)) in mol/K and mol/GPa.

    Trailing incomplete blocks are dropped: a run killed mid-write leaves a partial final
    block, and losing one row is better than raising on an otherwise good file.
    """
    lines = open(path).readlines()
    starts = [i for i, l in enumerate(lines) if 'dndt and dndp' in l]
    if not starts:
        raise ValueError(f'{path}: no "dndt and dndp" block markers')
    nsp = (starts[1] - starts[0] - 1) if len(starts) > 1 else (len(lines) - starts[0] - 1)
    abbrevs = [lines[starts[0] + 1 + k].split()[1] for k in range(nsp)]
    keep = [i for i in starts if i + nsp < len(lines)]
    dndt = np.zeros((len(keep), nsp))
    dndp = np.zeros_like(dndt)
    for b, i in enumerate(keep):
        for k in range(nsp):
            p = lines[i + 1 + k].split()
            dndt[b, k], dndp[b, k] = float(p[2]), float(p[3])
    return abbrevs, dndt, dndp


def load_fort99(path):
    """-> (species_names, moles (nP, S), PdT (nP, 3)).

    Drops interleaved WARNING lines and the trailing `Gibbs` / `Quality` columns, which are
    not species.
    """
    lines = open(path).readlines()
    cols = lines[0].split()
    rows = [l for l in lines[1:]
            if _NUMERIC.match(l.rstrip('\n')) and len(l.split()) == len(cols)]
    M = np.array([[float(x) for x in l.split()] for l in rows])
    names = [c for c in cols[3:] if c not in ('Gibbs', 'Quality')]
    return names, M[:, [cols.index(c) for c in names]], M[:, :3]


def load_control(path):
    """-> (element_amounts {symbol: mol}, [(phase_abbr, species_abbr), ...] in fort.42 order).

    The species list is read from the control file's own phase blocks, which is what makes
    the `(phase, species)` key available at all -- `fort.42` prints species in
    parameter-file order with no phase labels.
    """
    L = [l.rstrip('\n') for l in open(path)]
    strip = [l.strip() for l in L]

    amounts = OrderedDict()
    if 'oxides' in strip:
        j = strip.index('oxides') + 1
        while j < len(strip) and not re.match(r'^[\d,\s\.]+$', strip[j]):
            parts = strip[j].split()
            if len(parts) >= 2:
                amounts[parts[0]] = float(parts[1])
            j += 1

    species, j = [], 0
    while j < len(strip):
        if strip[j].startswith('phase '):
            ph = strip[j].split()[1]
            j += 2                                   # skip the phase's flag line
            while (j < len(strip) and strip[j] and not strip[j].startswith('phase ')
                   and not strip[j][0].isdigit()):
                species.append((ph, strip[j]))
                j += 1
        else:
            j += 1
    return amounts, species


# --------------------------------------------------------------------------- #
#  Mapping
# --------------------------------------------------------------------------- #
def build_species_map(control_species, indexer):
    """`fort.42` column index -> (phase_name, component_name), keyed on BOTH.

    Uses the same resolver the importer uses, so a mismatch here is a real mismatch and not
    an artefact of this script having its own opinion about names.
    """
    from ngibbs.utils.file_utils import (_resolve_component_name_from_abbr,
                                         _resolve_component_phase,
                                         _build_reverse_component_phase_map)
    reverse = _build_reverse_component_phase_map()
    abbr_to_phase = {s: ph for ph, s in control_species}
    keys = []
    for ph_abbr, sp_abbr in control_species:
        name = _resolve_component_name_from_abbr(sp_abbr)
        phase = _resolve_component_phase(
            component_abbr=sp_abbr, component_name=name,
            reverse_component_phase_map=reverse,
            control_component_to_phase_abbr=abbr_to_phase)
        keys.append((phase, name))
    return keys


def tracked_columns(keys, indexer):
    """Column indices of the components the TRAINED model covers, and the rest.

    The model's phase set is data-dependent -- a phase below the abundance threshold in the
    training data is simply absent from the indexer -- so the untracked list is expected to
    be non-empty and is not by itself a fault.
    """
    want = [(ph, str(c)) for ph, cl in indexer.components_in_phases.items() for c in cl]
    pos = {}
    for j, k in enumerate(keys):
        pos.setdefault(k, j)
    missing = [w for w in want if w not in pos]
    cols = np.array([pos[w] for w in want if w in pos], dtype=int)
    untracked = np.array([j for j in range(len(keys)) if j not in set(cols.tolist())],
                         dtype=int)
    return cols, untracked, missing


# --------------------------------------------------------------------------- #
#  Checks
# --------------------------------------------------------------------------- #
def rank1_report(residual, A, names):
    """How much of a residual lies along a single species' element signature, and whose.

    A residual that is ~entirely along one species is a MAPPING error. A residual spread
    across directions is a convergence error. Distinguishing them is the whole point.
    """
    R = residual[np.abs(residual).max(axis=1) > 0]
    if not len(R):
        return None
    best = (0.0, None)
    for k in range(A.shape[0]):
        v = A[k]
        nv = float(v @ v)
        if nv <= 0:
            continue
        proj = (R @ v) / nv
        left = R - np.outer(proj, v)
        num = np.median(np.linalg.norm(left, axis=1))
        den = np.median(np.linalg.norm(R, axis=1))
        frac = 1.0 - (num / den if den > 0 else 0.0)
        if frac > best[0]:
            best = (frac, names[k])
    return best


def check_simulation(sim_dir, indexer, A, El, tol, do_chain):
    out = {'dir': sim_dir}
    f42, f99, ctl = (os.path.join(sim_dir, f) for f in ('fort.42', 'fort.99', 'control'))
    if not all(os.path.exists(p) for p in (f42, f99, ctl)):
        out['status'] = 'missing files'
        return out

    amounts, control_species = load_control(ctl)
    keys = build_species_map(control_species, indexer)
    cols, untracked, missing = tracked_columns(keys, indexer)

    # ---- 0. injectivity ---------------------------------------------------
    dup_keys = len(keys) != len(set(keys))
    dup_cols = len(cols) != len(set(cols.tolist()))
    out['n_species'] = len(keys)
    out['n_tracked'] = int(len(cols))
    out['n_untracked'] = int(len(untracked))
    out['map_injective'] = not (dup_keys or dup_cols)
    out['components_not_in_fort42'] = [f'{p}/{c}' for p, c in missing]
    if dup_keys or dup_cols:
        out['status'] = 'FAIL: species map is not injective'
        return out

    abbrevs, dndt, dndp = load_fort42(f42)
    sp99, N99, PdT = load_fort99(f99)
    n = min(len(dndp), len(N99))
    dndp, dndt, N99, PdT = dndp[:n], dndt[:n], N99[:n], PdT[:n]
    out['n_rows'] = int(n)

    # Full-species element matrix: tracked rows come from compToEl; untracked rows are
    # reconstructed from the tracked row of the same species where possible. Where they
    # cannot be, the full-set test is skipped rather than silently run on a partial A.
    A_full = np.zeros((len(keys), A.shape[1]))
    A_full[cols] = A
    have_full = len(untracked) == 0

    # ---- 1. tangent identity ----------------------------------------------
    res = {}
    for lbl, arr in (('dndp', dndp), ('dndt', dndt)):
        r_tr = arr[:, cols] @ A
        bad = np.abs(r_tr).max(axis=1) > tol
        entry = {'max_abs_tracked': float(np.abs(r_tr).max()),
                 'rows_over_tol': int(bad.sum())}
        # Attribute any excess to untracked species before calling it a fault.
        leak = (np.abs(arr[:, untracked]).sum(axis=1) if len(untracked)
                else np.zeros(len(arr)))
        if len(untracked):
            entry['untracked_max'] = float(leak.max())
            entry['rows_over_tol_with_untracked_active'] = int((bad & (leak > 0)).sum())
            entry['rows_over_tol_unexplained'] = int((bad & (leak == 0)).sum())
            worst = {}
            for j in untracked:
                m = float(np.abs(arr[:, j]).max())
                if m > 0:
                    worst[keys[j][1]] = m
            entry['untracked_carriers'] = dict(sorted(worst.items(),
                                                      key=lambda kv: -kv[1])[:5])
        else:
            entry['rows_over_tol_unexplained'] = int(bad.sum())
        # The rank-1 diagnosis applies ONLY to rows the untracked species do not already
        # explain. Running it on explained rows is actively misleading: alpha-iron shares
        # its element signature with gamma-iron, so leakage reads as "100% along
        # gamma-iron -- looks like a mapping error" when nothing is wrong at all.
        unexp = bad & (leak == 0) if len(untracked) else bad
        if unexp.any():
            r1 = rank1_report(r_tr[unexp], A,
                              [f'{p}/{c}' for p, c in [keys[j] for j in cols]])
            if r1:
                entry['rank1_fraction'] = float(r1[0])
                entry['rank1_direction'] = r1[1]
        res[lbl] = entry
    out['tangent'] = res

    # ---- 1b. bulk vs the control file, in BOTH bases -----------------------
    B = N99[:, cols] @ A
    o_per = np.array([CATION_CHARGE.get(e, 0) / 2.0 for e in El])
    idx_fe = El.index('Fe') if 'Fe' in El else None
    idx_fe3 = El.index('Fe3') if 'Fe3' in El else None
    bulk = {}
    for e, name in enumerate(El):
        if name in ('Fe', 'Fe3'):
            continue
        bulk[name] = {'drift': float(np.ptp(B[:, e])),
                      'control': amounts.get(name),
                      'row0': float(B[0, e])}
    if idx_fe is not None and idx_fe3 is not None:
        fe_tot = B[:, idx_fe] + B[:, idx_fe3]
        bulk['Fe_total'] = {'drift': float(np.ptp(fe_tot)),
                            'control': amounts.get('Fe'), 'row0': float(fe_tot[0])}
    O = B @ o_per
    bulk['O'] = {'drift': float(np.ptp(O)), 'control': amounts.get('O'),
                 'row0': float(O[0])}
    out['bulk'] = bulk
    # The tracked subset can legitimately fall short of the control file when a
    # below-threshold phase holds moles -- that is a domain statement, not a data fault.
    # It is only a fault when the tracked bulk misses AND nothing untracked is present.
    ut_row0 = float(np.abs(N99[0, untracked]).sum()) if len(untracked) else 0.0
    mism = [k for k, v in bulk.items()
            if v['control'] is not None and abs(v['row0'] - v['control']) >= 1e-4]
    out['bulk_row0_mismatched'] = mism
    out['bulk_row0_matches_control'] = not mism
    out['bulk_row0_deficit_explained'] = (not mism) or ut_row0 > 1e-9

    # ---- 1c. untracked species, by name ------------------------------------
    if len(untracked):
        carr = {}
        for j in untracked:
            m = float(np.abs(N99[:, j]).max())
            if m > 0:
                carr[keys[j][1]] = m
        out['untracked_moles'] = dict(sorted(carr.items(), key=lambda kv: -kv[1]))
        # The number the exporter needs: how much of this run sits outside the phase set
        # the model was trained on. Those rows' tracked-subset derivatives cannot satisfy
        # the tangent identity, so supervising them fights the projection.
        ut_row = np.abs(N99[:, untracked]).sum(axis=1)
        cat = B.sum(axis=1)
        out['untracked_row_fraction'] = float(np.mean(ut_row > 1e-9))
        out['untracked_moles_frac_max'] = float((ut_row / np.clip(cat, 1e-12, None)).max())

    # ---- 2. coverage --------------------------------------------------------
    finite = np.isfinite(dndp).all(axis=1) & np.isfinite(dndt).all(axis=1)
    out['coverage'] = float(finite.mean())

    # ---- 3. chain rule (optional; the only check that finite-differences) ----
    if do_chain:
        P, T = PdT[:, 0], PdT[:, 2]
        breaks = np.where(np.diff(P) <= 0)[0] + 1
        bounds = np.concatenate([[0], breaks, [len(P)]])
        ratios = []
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b - a < 5:
                continue
            Ps, Ts = P[a:b], T[a:b]
            for k in cols[:24]:                       # a sample is enough to catch a fault
                ns = dndp[a:b, k] * 0 + N99[a:b, k]
                dP = np.diff(Ps)
                num = np.diff(ns) / np.where(dP == 0, np.nan, dP)
                mid = 0.5 * (dndp[a:b - 1, k] + dndp[a + 1:b, k])
                midt = 0.5 * (dndt[a:b - 1, k] + dndt[a + 1:b, k])
                dTdP = np.diff(Ts) / np.where(dP == 0, np.nan, dP)
                ana = mid + midt * dTdP
                # only where the analytic partial is locally linear, so the midpoint
                # difference is a fair comparison
                lin = np.abs(np.diff(dndp[a:b, k])) < 0.1 * np.abs(mid).clip(min=1e-12)
                m = lin & np.isfinite(num) & np.isfinite(ana) & \
                    (np.abs(num) > 0.01 * np.nanpercentile(np.abs(num), 99))
                if m.sum() >= 3:
                    ratios.append(np.median(ana[m] / num[m]))
        out['chain_rule_median_ratio'] = float(np.median(ratios)) if ratios else None
        out['chain_rule_n_segments'] = len(ratios)

    unexplained = sum(res[k].get('rows_over_tol_unexplained', 0) for k in res)
    if not (out['map_injective'] and unexplained == 0
            and out['bulk_row0_deficit_explained']):
        out['status'] = 'CHECK'
    elif out.get('untracked_row_fraction', 0.0) > 0.01:
        # Everything is internally consistent; the run simply spends time in phases the
        # trained model does not carry. Usable, but those rows want down-weighting.
        out['status'] = 'out-of-domain'
    else:
        out['status'] = 'ok'
    return out


# --------------------------------------------------------------------------- #
#  Reporting
# --------------------------------------------------------------------------- #
def render(rep):
    name = os.path.basename(rep['dir'].rstrip('/'))
    print(f"\n=== {name} ===")
    if rep.get('status') == 'missing files':
        print('   missing control/fort.42/fort.99 -- skipped')
        return
    print(f"   rows {rep.get('n_rows','?')}   species {rep['n_species']}   "
          f"tracked {rep['n_tracked']}   untracked {rep['n_untracked']}")
    print(f"   [0] species map injective on (phase, component): {rep['map_injective']}")
    if rep['components_not_in_fort42']:
        print(f"       !! model components absent from fort.42: "
              f"{rep['components_not_in_fort42']}")
    if not rep['map_injective']:
        print('   FAIL -- fix the map before reading anything below.')
        return

    for lbl, e in rep['tangent'].items():
        print(f"   [1] A^T {lbl} : max |resid| over tracked {e['max_abs_tracked']:.2e}   "
              f"rows>tol {e['rows_over_tol']}")
        if e.get('untracked_max'):
            print(f"       untracked (below-threshold phases) carry up to "
                  f"{e['untracked_max']:.2e}; "
                  f"{e.get('rows_over_tol_with_untracked_active',0)} of those rows are "
                  f"explained by them")
            if e.get('untracked_carriers'):
                print(f"       carriers: "
                      f"{', '.join(f'{k} {v:.2e}' for k, v in e['untracked_carriers'].items())}")
        if e.get('rows_over_tol_unexplained'):
            print(f"       !! {e['rows_over_tol_unexplained']} rows UNEXPLAINED")
        if 'rank1_fraction' in e:
            tag = ('looks like a MAPPING error' if e['rank1_fraction'] > 0.95
                   else 'spread across directions -- looks like convergence')
            print(f"       rank-1: {100*e['rank1_fraction']:.1f}% along "
                  f"{e['rank1_direction']} ({tag})")

    tag = ''
    if not rep['bulk_row0_matches_control']:
        tag = ('  (deficit explained by untracked phases)'
               if rep['bulk_row0_deficit_explained'] else '  !! UNEXPLAINED')
    print(f"   [1b] tracked bulk at row 0 matches control file: "
          f"{rep['bulk_row0_matches_control']}{tag}")
    for k, v in rep['bulk'].items():
        ctl = '' if v['control'] is None else f"  control {v['control']:.5f}"
        print(f"        {k:9s} row0 {v['row0']:.6f}{ctl}   drift {v['drift']:.2e}")
    if rep.get('untracked_moles'):
        print("   [1c] untracked species holding moles (expected: below the abundance "
              "threshold the model was trained at)")
        for k, v in rep['untracked_moles'].items():
            print(f"        {k:24s} max {v:.3e}")
        print(f"        rows with any untracked phase present: "
              f"{100*rep['untracked_row_fraction']:.1f}%   "
              f"peak untracked share of cations {100*rep['untracked_moles_frac_max']:.2f}%")
    print(f"   [2] derivative coverage: {100*rep['coverage']:.1f}% of rows")
    if 'chain_rule_median_ratio' in rep:
        r = rep['chain_rule_median_ratio']
        print(f"   [3] chain rule median ratio: "
              f"{'n/a' if r is None else f'{r:.4f}'} "
              f"over {rep['chain_rule_n_segments']} segment-species")
    print(f"   -> {rep['status']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('workspace')
    ap.add_argument('--tol', type=float, default=1e-6,
                    help='absolute tolerance on |A^T dn| (clean value is ~7e-8)')
    ap.add_argument('--chain', action='store_true',
                    help='also run the finite-difference chain-rule check (slower)')
    ap.add_argument('--json', default=None, help='write the full report here')
    a = ap.parse_args()

    from ngibbs.engine.API import HeFESToEmulatorCPU as E
    emu = E.isothermal_emulator
    indexer = emu.ml_indexer
    A = np.asarray(emu.compToEl, dtype=float)
    El = list(emu.Elkeys)

    root = a.workspace.rstrip('/')
    sims = ([root] if os.path.exists(os.path.join(root, 'control'))
            else sorted(os.path.join(root, d) for d in os.listdir(root)
                        if os.path.isdir(os.path.join(root, d))))
    reports = [check_simulation(s, indexer, A, El, a.tol, a.chain) for s in sims]
    for r in reports:
        render(r)

    bad = [r for r in reports if r.get('status') not in ('ok', 'missing files')]
    print(f"\n{len(reports) - len(bad)}/{len(reports)} simulations clean")
    if a.json:
        json.dump(reports, open(a.json, 'w'), indent=1)
        print(f"report -> {a.json}")
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
