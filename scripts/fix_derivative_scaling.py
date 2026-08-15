"""
Retroactively undo the element-total normalisation on derivative shadow tables.

Shadow tables built before the fix divided dn/dP and dn/dT by the run's element total
while `import_HeFESTo_components` wrote fort.99 moles raw, leaving the derivatives a
factor of N_el below their own moles.

The correction is exactly invertible -- multiply each simulation's block of rows by that
run's N_el -- but **N_el is per run, not a constant**. Measured across four control
files: 23.63948 (Mg/Si 1.58), 24.00450 (Mg/Si 1.29), 25.35841 (Mg/Si 0.29). Cations are
pinned at 10 and oxygen follows Si stoichiometry, so the total spans 7.1% and tracks
Mg/Si directly. A single global 24x would leave a composition-correlated residual --
the worst kind, since on an Mg/Si sweep it reads as a trend rather than as noise.

So the per-simulation N_el recorded in the manifest is required, and the row blocks are
taken from the manifest's `n_rows` in write order.

By default the script DETECTS whether each simulation was normalised, by comparing the
shadow values against the source fort.42 (or a reconstruction). Detection beats
assumption here: applying the correction twice is as wrong as not applying it, and
nothing downstream would show it.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / 'src'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from builder.HeFESTo.HeFESTo_derivative_import import load_fort42  # noqa: E402

KEY_COLS = ('P(GPa)(System_main)', 'T(K)(System_main)')


def detect_scale(shadow_block: pd.DataFrame, sim_dir: Path, N_el: float,
                 which: str = 'dndp', tol: float = 0.02):
    """-> ('raw' | 'normalised' | 'unknown', measured_ratio).

    Compares the largest-magnitude shadow value against the same entry in the source
    fort.42. A ratio near 1 means the block is already raw; near 1/N_el means it was
    normalised.
    """
    f42 = sim_dir / 'fort.42'
    if not f42.exists():
        return 'unknown', float('nan')
    try:
        names, dndt, dndp = load_fort42(str(f42))
    except Exception:
        return 'unknown', float('nan')
    comp = [c for c in shadow_block.columns if c not in KEY_COLS]
    vals = shadow_block[comp].to_numpy(dtype=float)
    n = min(len(vals), dndp.shape[0])
    if n == 0:
        return 'unknown', float('nan')
    raw = (dndp if which == 'dndp' else dndt)[:n]
    big = np.unravel_index(np.nanargmax(np.abs(raw)), raw.shape)
    denom = raw[big]
    if abs(denom) < 1e-12:
        return 'unknown', float('nan')
    # find the shadow column holding that species by magnitude match
    row = vals[big[0]]
    j = int(np.nanargmax(np.abs(row)))
    ratio = float(row[j] / denom)
    if abs(ratio - 1.0) < tol:
        return 'raw', ratio
    if abs(ratio - 1.0 / N_el) < tol / N_el:
        return 'normalised', ratio
    return 'unknown', ratio


def fix_table(path: Path, manifest: pd.DataFrame, workspace: Path | None,
              apply: bool, force: bool, which: str = 'dndp') -> dict:
    df = pd.read_csv(path)
    total = int(manifest['n_rows'].sum())
    if len(df) != total:
        raise SystemExit(
            f'{path.name}: {len(df)} rows but manifest sums to {total}. '
            'The manifest must correspond to this table, in write order.')

    comp = [c for c in df.columns if c not in KEY_COLS]
    start = 0
    counts = {'raw': 0, 'normalised': 0, 'unknown': 0}
    for _, rec in manifest.iterrows():
        n = int(rec['n_rows'])
        block = df.iloc[start:start + n]
        N_el = float(rec['N_el'])
        state, ratio = ('unknown', float('nan'))
        if workspace is not None:
            state, ratio = detect_scale(block, workspace / f"Simulation{int(rec['sim_id'])}",
                                        N_el, which=which)
        if state == 'unknown' and force:
            state = 'normalised'
        counts[state] += 1
        if state == 'normalised' and apply:
            df.iloc[start:start + n, [df.columns.get_loc(c) for c in comp]] = \
                block[comp].to_numpy(dtype=float) * N_el
        start += n

    if apply and counts['normalised']:
        backup = path.with_suffix(path.suffix + '.prescale.bak')
        if not backup.exists():
            os.replace(path, backup)
        df.to_csv(path, index=False)
    return counts


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dndp', type=Path, required=True)
    p.add_argument('--dndt', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True,
                   help='Manifest written alongside the shadows. Carries the per-run '
                        'N_el and the row blocks, both of which are required.')
    p.add_argument('--workspace', type=Path, default=None,
                   help='Workspace holding SimulationN dirs. Used to DETECT which '
                        'simulations were normalised by comparing against fort.42. '
                        'Strongly recommended -- applying the correction twice is as '
                        'wrong as not applying it.')
    p.add_argument('--apply', action='store_true',
                   help='Write the corrected tables. Without it, reports only. The '
                        'original is kept as <name>.prescale.bak.')
    p.add_argument('--force-normalised', action='store_true',
                   help='Treat simulations whose state cannot be detected as '
                        'normalised. Only use when you know every row was built '
                        'before the fix.')
    args = p.parse_args()

    manifest = pd.read_csv(args.manifest)
    for col in ('sim_id', 'n_rows', 'N_el'):
        if col not in manifest.columns:
            raise SystemExit(f'manifest lacks required column {col!r}')

    print(f'manifest: {len(manifest)} simulations, {int(manifest["n_rows"].sum())} rows')
    print(f'N_el range: {manifest["N_el"].min():.5f} to {manifest["N_el"].max():.5f} '
          f'({100 * (manifest["N_el"].max() - manifest["N_el"].min()) / manifest["N_el"].mean():.2f}% '
          f'spread -- this is why a global constant will not do)')
    if args.workspace is None and not args.force_normalised:
        print('\nNo --workspace given, so nothing can be detected and nothing will be '
              'changed.\nPass --workspace to detect, or --force-normalised to assert it.')

    for tag, path, which in (('dndP', args.dndp, 'dndp'), ('dndT', args.dndt, 'dndt')):
        counts = fix_table(path, manifest, args.workspace, args.apply,
                           args.force_normalised, which=which)
        verb = 'corrected' if args.apply else 'would correct'
        print(f'  {tag} {path.name}: raw={counts["raw"]}  normalised={counts["normalised"]}  '
              f'undetermined={counts["unknown"]}  ->  {verb} {counts["normalised"]} sims')
    if not args.apply:
        print('\nDry run. Re-run with --apply to write.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
