"""
Re-draw cumulate-inversion figures from saved `<tag>_nodes.csv(.gz)` files (as
written by cumulate_inversion_demo.py), without re-running the inversion.

For every nodes file in --results it writes, next to it:
    <tag>_density.png          accepted final states (liquid composition + P-T)
    <tag>_prior.png            the same figure for the STARTS of all nodes (the prior)
    <tag>_prior_vs_final.png   prior (top) vs accepted final states (bottom), shared axes
    <tag>_convergence.png      loss vs iteration (only if <tag>_history.csv exists)
Composition axes: SiO2 vs MgO, or SiO2 vs molar (Na2O+K2O+2CaO)/(2Al2O3) when the
true liquid has < 4 wt% MgO.

Example:
    python scripts/cumulate_inversion_replot.py --model-dir src/ngibbs/engine/TrainedModels/102 \
        --results "Claude outputs/cumulate_inversion_v2/gpu/102_NoCr"
"""
from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'scripts'))

import matplotlib                                                       # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                         # noqa: E402

from ngibbs.engine.NN import rebuild_MELTS_model                        # noqa: E402
from ngibbs.engine.emulator import NN_MELTS, TrainingRangeWarning       # noqa: E402
from ngibbs.engine import cumulate_inversion as ci                      # noqa: E402
from cumulate_inversion_demo import find_checkpoint                     # noqa: E402


def load_result(em, bundle, nodes_path: Path, cfg=None) -> ci.InversionResult:
    m = re.match(r'row(\d+)_', nodes_path.name)
    if not m:
        raise ValueError(f"Cannot read the bundle row from {nodes_path.name}")
    tgt = ci.CumulateTarget.from_bundle(em, bundle, int(m.group(1)))
    inv = ci.CumulateInverter(em, tgt, cfg or ci.InversionConfig())
    hist = nodes_path.with_name(nodes_path.name.split('_nodes.csv')[0] + '_history.csv')
    return ci.InversionResult(nodes=pd.read_csv(nodes_path), target=tgt, config=inv.cfg,
                              history=pd.read_csv(hist) if hist.exists() else pd.DataFrame(),
                              oxides=list(inv.geo.Oxides),
                              basis_oxides=list(inv.basis_oxides),
                              constrained_oxides=list(inv.constrained_oxides))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--variant', default='NoCr', choices=['NoCr', 'Cr'])
    ap.add_argument('--results', required=True, help='folder holding <tag>_nodes.csv(.gz)')
    ap.add_argument('--accept-quantile', type=float, default=0.25)
    args = ap.parse_args(argv)

    warnings.simplefilter('ignore', TrainingRangeWarning)
    ckpt, bundle_path = find_checkpoint(Path(args.model_dir), args.variant)
    em = NN_MELTS(rebuild_MELTS_model(str(ckpt)))
    bundle = ci.load_ml_bundle(bundle_path)
    res_dir = Path(args.results)
    files = sorted(res_dir.glob('*_nodes.csv.gz')) + sorted(res_dir.glob('*_nodes.csv'))
    rows = []
    for f in files:
        tag = f.name.split('_nodes.csv')[0]
        res = load_result(em, bundle, f)
        acc_kw = {'loss_quantile': args.accept_quantile}
        acc = res.accepted(**acc_kw)
        plots = [('density', lambda p: res.plot_density(subset=acc, path=p)),
                 ('prior', lambda p: res.plot_prior(which='all', path=p)),
                 ('prior_vs_final', lambda p: res.plot_prior_vs_final(path=p, **acc_kw))]
        if len(res.history):
            plots.append(('convergence', lambda p: res.plot_convergence(path=p)))
        for name, make in plots:
            plt.close(make(res_dir / f'{tag}_{name}.png'))
        dall, dacc = res.drift(), res.drift(acc)
        rows.append({'target': res.target.name,
                     **{f'all_{k}_median': dall[k].median() for k in ('dliq_rmse_wt', 'dT_C', 'dP_bar')},
                     **{f'acc_{k}_median': dacc[k].median() for k in ('dliq_rmse_wt', 'dT_C', 'dP_bar')},
                     **{f'acc_abs_{k}_median': dacc[k].abs().median() for k in ('dT_C', 'dP_bar')}})
        print(f"  {tag}: plots written")
    if rows:
        out = pd.DataFrame(rows)
        out.to_csv(res_dir / 'drift_summary.csv', index=False)
        with pd.option_context('display.width', 200, 'display.max_columns', 20):
            print(out.round(2))


if __name__ == '__main__':
    main()
