#!/usr/bin/env python
"""Bulk-reconstruction figures (PDF) for the shipped nGibbs emulators.

For each emulator: b_hat vs b and (b - b_hat)/b vs b, one panel per element,
the raw network reconstruction (mass_balance='none') in orange with the
mass-balanced one in blue on top.  Also writes the per-element statistics table
for each emulator.

Everything is written to ``generated/`` next to this file:

    generated/<family>[_<variant>]_<kind>_bulk_parity.pdf
    generated/<family>[_<variant>]_<kind>_bulk_relerr.pdf
    generated/<family>[_<variant>]_<kind>_bulk_reconstruction.csv

Only the installed ``ngibbs`` package is used (its shipped TrainedModels and
Test_subset bundles), so this file can live in any repository.

Examples
--------
    python make_bulk_reconstruction_figures.py                       # every shipped emulator
    python make_bulk_reconstruction_figures.py --families 102 120 --kinds isothermal
    python make_bulk_reconstruction_figures.py --pair my_model.tar my_Test_subset.tar.gz
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from ngibbs.deployment_tests.bulk_reconstruction import (
    evaluate_bulk_reconstruction, plot_bulk_reconstruction, write_summary,
)
from ngibbs.deployment_tests.discovery import shipped_emulators
from ngibbs.engine.NN import rebuild_MELTS_model
from ngibbs.engine.emulator import NN_MELTS

OUT_DIR = Path(__file__).resolve().parent / 'generated'


def _jobs(args):
    if args.pair:
        for ckpt, bundle in args.pair:
            ckpt, bundle = Path(ckpt), Path(bundle)
            yield ckpt.stem, ckpt, bundle
        return
    for e in shipped_emulators(families=args.families, kinds=args.kinds):
        if e.bundle is None:
            print(f'[skip] {e.label}: no Test_subset bundle found for {e.checkpoint.name}')
            continue
        yield e.label, e.checkpoint, e.bundle


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--families', nargs='+', help='TrainedModels sub-directories to include')
    ap.add_argument('--kinds', nargs='+', choices=['isothermal', 'isentropic', 'openox'])
    ap.add_argument('--pair', nargs=2, action='append', metavar=('CHECKPOINT', 'BUNDLE'),
                    help='explicit checkpoint + Test_subset bundle (repeatable); '
                         'replaces the shipped-emulator discovery')
    ap.add_argument('--max-samples', type=int, default=None, help='random row cap per bundle')
    ap.add_argument('--seed', type=int, default=1337)
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    ap.add_argument('--mass-balance', default='iterative', choices=['iterative', 'pinv'],
                    help='correction shown in blue (raw is always mass_balance=none)')
    ap.add_argument('--fmt', default='pdf', help='output format (pdf, png, svg)')
    ap.add_argument('--dpi', type=int, default=300, help='raster resolution of the scatter layers')
    args = ap.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_done = 0
    for label, ckpt, bundle in _jobs(args):
        t0 = time.time()
        emu = NN_MELTS(rebuild_MELTS_model(str(ckpt)), cuda=(args.device == 'cuda'),
                       mass_balance=args.mass_balance)
        r = evaluate_bulk_reconstruction(emu, bundle, max_samples=args.max_samples, seed=args.seed)
        paths = plot_bulk_reconstruction(r, OUT_DIR, label, fmt=args.fmt, dpi=args.dpi)
        write_summary(r, OUT_DIR / f'{label}_bulk_reconstruction.csv')
        s = r['summary']
        print(f'[done] {label}  ({time.time() - t0:.1f} s, n={r["meta"]["n_samples"]:,})  '
              + '  '.join(f"{k}: med|rel| {s.at[f'{k}_relerr_median_pct', 'all']:.2g}% "
                          f"p98 {s.at[f'{k}_relerr_p98_pct', 'all']:.2g}%" for k in r['b_hat'])
              + f"  -> {', '.join(p.name for p in paths.values())}")
        n_done += 1
    print(f'{n_done} emulator(s) written to {OUT_DIR}')


if __name__ == '__main__':
    main()
