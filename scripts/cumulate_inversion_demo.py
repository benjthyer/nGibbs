"""
Cumulate-inversion prototype: recover (liquid, P, T) from cumulates taken out of an
ML bundle, and compare against the bundle's own truth.

Targets: `--n-assemblages` (default 20) bundle rows drawn at random among
liquid-bearing rows with >= 2 solid phases (fluid ignored), skipping any row whose
solid assemblage was already drawn, so every target has a different assemblage.
`--fixed-assemblages` uses the old hand-picked list instead. For each row the
liquid is stripped off and the remaining solids (assemblage + phase compositions)
become the target cumulate. `CumulateInverter`
then optimises `--nodes` independent starts drawn from the cumulate-derived prior
(random cumulus proportions + training-data volatiles, forward-modelled onto a
liquidus; see CumulateInverter.initial_nodes) and keeps each node's best state.

Comparisons with the truth are made only in the chemical space the cumulate can
speak for: the true liquid is renormalised to 100 over the basis oxides
(represented in the cumulate + H2O/CO2), and liquid errors are reported over the
represented ("constrained") oxides only.

Outputs (per target, in --out):
    <tag>_nodes.csv         best state of every node
    <tag>_summary.csv       accepted-ensemble mean/std/quantiles vs truth
    <tag>_covariance.csv    covariance of (P, T, liquid oxides) over accepted nodes
    <tag>_correlation.csv   the same as correlations
    <tag>_history.csv       per-iteration loss terms / quantiles
    <tag>_density.png       accepted liquids (SiO2 vs MgO, or vs molar
                            (Na2O+K2O+2CaO)/(2Al2O3) when the liquid has < 4 wt% MgO) + P-T
    <tag>_prior.png         the same figure for the starting states of all nodes (the prior)
    <tag>_prior_vs_final.png  prior (top) vs accepted final states (bottom) on shared axes
    <tag>_convergence.png   loss against iteration
and recovery.csv, one row per target: best node, accepted median and prior (the
starts, no inversion) against the truth, per oxide for every model oxide ('--' for
oxides outside that target's basis), plus the emulator baseline ('emu_*': nGibbs
run forward at the row's true P, T and bulk -- its error is emulator error, before
any inversion).

Example (GPU box):
    python scripts/cumulate_inversion_demo.py --model-dir src/ngibbs/engine/TrainedModels/102 \
        --variant NoCr --nodes 32768 --steps 600 --device cuda
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))

from ngibbs.engine.NN import rebuild_MELTS_model                       # noqa: E402
from ngibbs.engine.emulator import NN_MELTS, TrainingRangeWarning      # noqa: E402
from ngibbs.engine import cumulate_inversion as ci                     # noqa: E402

DEFAULT_ASSEMBLAGES = [
    ('olivine',),
    ('olivine', 'clinopyroxene'),
    ('olivine', 'clinopyroxene', 'plagioclase'),
    ('clinopyroxene', 'plagioclase'),
    ('olivine', 'orthopyroxene'),
    ('olivine', 'orthopyroxene', 'clinopyroxene', 'spinel'),
    ('orthopyroxene', 'clinopyroxene', 'plagioclase'),
]


def random_unique_targets(em, bundle, n, rng, min_liquid_mass=20.0, min_solids=2,
                          ignore=('fluid',)):
    """Draw bundle rows at random (liquid present, >= min_liquid_mass %% liquid,
    >= min_solids solids excluding `ignore`), keeping only the first row seen for
    each solid assemblage, until `n` distinct assemblages are collected."""
    geo = ci._Geometry(em)
    B = bundle.binary_labels > 0.5
    solids = [p for p in geo.phases if p != ci.LIQUID and p not in ignore]
    cols = np.array([geo.phase_col[p] for p in solids])
    ok = B[:, geo.liq] & (bundle.mass_labels[:, geo.liq] >= min_liquid_mass) & \
        (B[:, cols].sum(1) >= min_solids)
    seen, rows = set(), []
    for r in rng.permutation(np.nonzero(ok)[0]):
        asm = tuple(p for p, c in zip(solids, cols) if B[r, c])
        if asm in seen:
            continue
        seen.add(asm)
        rows.append(int(r))
        if len(rows) == n:
            break
    return rows


def find_checkpoint(model_dir: Path, variant: str):
    tars = [p for p in model_dir.glob('*.tar') if 'Closed' in p.name and p.stem.endswith('_NPT')
            and f'_{variant}_' in p.name]
    bundles = [p for p in model_dir.glob('*_Test_subset*.tar.gz') if 'Closed' in p.name
               and '_NPT_' in p.name and f'_{variant}_' in p.name]
    if len(tars) != 1 or len(bundles) != 1:
        raise FileNotFoundError(f"Expected one closed {variant} NPT checkpoint and test bundle in "
                                f"{model_dir}, found {tars} / {bundles}")
    return tars[0], bundles[0]


def truth_basin(inv: ci.CumulateInverter, n: int = 64, seed: int = 0):
    """Re-run the inversion from small perturbations of the TRUE (liquid, P, T).

    If this basin's best loss is no lower than the prior ensemble's, a miss by the
    ensemble is emulator degeneracy (other minima are as good as the truth), not an
    optimiser failure."""
    tr = inv.target.truth
    rng = np.random.default_rng(seed)
    x = np.repeat(tr['liquid_elements'][None], n, 0) * np.exp(0.03 * rng.standard_normal((n, inv.geo.E)))
    x /= x.sum(1, keepdims=True)
    init = {'x': x.astype(np.float32),
            'P': np.clip(tr['P_bar'] + 300 * rng.standard_normal(n), *inv.P_bounds).astype(np.float32),
            'T': np.clip(tr['T_C'] + 10 * rng.standard_normal(n), *inv.T_bounds).astype(np.float32)}
    res = inv.run(init=init, verbose=False)
    return res.nodes.loc[res.nodes['loss'].idxmin()]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model-dir', default=str(REPO / 'src/ngibbs/engine/TrainedModels/102'))
    ap.add_argument('--variant', default='NoCr', choices=['NoCr', 'Cr'])
    ap.add_argument('--out', default=str(REPO / 'plots/cumulate_inversion'))
    ap.add_argument('--nodes', type=int, default=2 ** 15)
    ap.add_argument('--steps', type=int, default=600)
    ap.add_argument('--batch-size', type=int, default=8192)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--n-assemblages', type=int, default=20,
                    help='number of random targets, each with a different solid assemblage')
    ap.add_argument('--fixed-assemblages', action='store_true',
                    help='use the hand-picked DEFAULT_ASSEMBLAGES list instead')
    ap.add_argument('--min-liquid-mass', type=float, default=20.0,
                    help='only pick bundle rows with at least this much liquid (mass %%)')
    ap.add_argument('--accept-quantile', type=float, default=0.25,
                    help='accepted = cumulus phases + liquid present AND loss in this lowest quantile')
    ap.add_argument('--extra-phases', default='free', choices=['free', 'forbid'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--no-truth-check', action='store_true',
                    help='skip the truth-seeded control run (see truth_basin)')
    args = ap.parse_args(argv)

    warnings.simplefilter('ignore', TrainingRangeWarning)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt, bundle_path = find_checkpoint(Path(args.model_dir), args.variant)
    dev_name = torch.cuda.get_device_name(0) if args.device == 'cuda' else 'cpu'
    print(f"checkpoint {ckpt.name}\nbundle     {bundle_path.name}\ndevice     {args.device} ({dev_name})",
          flush=True)
    em = NN_MELTS(rebuild_MELTS_model(str(ckpt)), cuda=(args.device == 'cuda'))
    bundle = ci.load_ml_bundle(bundle_path)
    geo = ci._Geometry(em)
    rng = np.random.default_rng(args.seed)

    if args.fixed_assemblages:
        rows = []
        for asm in DEFAULT_ASSEMBLAGES:
            cand = ci.select_cumulate_rows(em, bundle, asm)
            cand = cand[bundle.mass_labels[cand, geo.liq] >= args.min_liquid_mass]
            if len(cand):
                rows.append(int(rng.choice(cand)))
    else:
        rows = random_unique_targets(em, bundle, args.n_assemblages, rng, args.min_liquid_mass)
    targets = [ci.CumulateTarget.from_bundle(em, bundle, r) for r in rows]
    print(f"{len(targets)} targets: " + ', '.join(t.name for t in targets), flush=True)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    recovery = []
    for k, tgt in enumerate(targets):
        tag = tgt.name.replace(':', '_').replace('+', '-')
        print(f"\n=== target {k + 1}/{len(targets)}", flush=True)
        cfg = ci.InversionConfig(steps=args.steps, batch_size=args.batch_size, seed=args.seed,
                                 log_every=100, extra_phases=args.extra_phases)
        inv = ci.CumulateInverter(em, tgt, cfg)
        t0 = time.time()
        res = inv.run(args.nodes)
        seconds = time.time() - t0
        acc = res.accepted(loss_quantile=args.accept_quantile)
        res.nodes.to_csv(out / f'{tag}_nodes.csv.gz', index=False)
        res.history.to_csv(out / f'{tag}_history.csv', index=False)
        res.summary(subset=acc).to_csv(out / f'{tag}_summary.csv')
        res.covariance(subset=acc).to_csv(out / f'{tag}_covariance.csv')
        res.covariance(subset=acc, correlation=True).to_csv(out / f'{tag}_correlation.csv')
        for name, make in (('density', lambda p: res.plot_density(subset=acc, path=p)),
                           ('prior', lambda p: res.plot_prior(which='all', path=p)),
                           ('prior_vs_final', lambda p: res.plot_prior_vs_final(
                               path=p, loss_quantile=args.accept_quantile)),
                           ('convergence', lambda p: res.plot_convergence(path=p))):
            try:
                plt.close(make(out / f'{tag}_{name}.png'))
            except Exception as exc:   # plotting is optional
                print(f"  ({name} plot skipped: {exc})")

        rec = recovery_row(res, acc, seconds)
        if not args.no_truth_check:
            tr = tgt.truth
            tb = truth_basin(inv)
            x = np.asarray(tr['liquid_elements'], np.float32)[None]
            z, u, xfix = inv._encode(torch.tensor([tr['P_bar']], device=geo.dev),
                                     torch.tensor([tr['T_C']], device=geo.dev),
                                     torch.as_tensor(x, device=geo.dev))
            P, T, xb = inv._decode(z, u, xfix)
            rec['loss_at_truth'] = float(inv.evaluate(P, T, xb)['total'][0])
            rec['truth_basin_loss'] = tb['loss']
            rec['ensemble_best_loss'] = res.nodes['loss'].min()
            rec['truth_basin_P_bar'], rec['truth_basin_T_C'] = tb['P_bar'], tb['T_C']
            tl = res.truth_liquid()
            rec['truth_basin_liq_rmse_wt'] = float(np.sqrt(np.mean(
                [(tb[f'liq_{o}'] - tl[o]) ** 2 for o in res.constrained_oxides])))
        recovery.append(rec)
        pd.DataFrame(recovery).to_csv(out / 'recovery.csv', index=False)   # keep it current
        print(pd.Series({k_: v for k_, v in rec.items() if not str(k_).startswith(('truth_', 'best_', 'acc_', 'prior_'))
                         or k_.endswith(('rmse_wt', 'P_bar', 'T_C'))}).to_string(), flush=True)

    rec = pd.DataFrame(recovery)
    rec.to_csv(out / 'recovery.csv', index=False)
    with pd.option_context('display.width', 250, 'display.max_columns', 40):
        cols = ['target', 'truth_P_bar', 'best_P_bar', 'acc_P_bar_p50', 'prior_P_bar_p50',
                'truth_T_C', 'best_T_C', 'acc_T_C_p50', 'prior_T_C_p50',
                'emu_liq_rmse_wt', 'best_liq_rmse_wt', 'acc_liq_rmse_wt', 'prior_liq_rmse_wt',
                'acc_node_liq_rmse_p50', 'prior_node_liq_rmse_p50',
                'best_f_liquid', 'ensemble_best_loss', 'truth_basin_loss', 'seconds']
        print(rec[[c for c in cols if c in rec]].round(2))


def recovery_row(res: ci.InversionResult, acc: pd.DataFrame, seconds: float) -> dict:
    """One recovery.csv row. Liquids are compared on the target's basis (true
    liquid renormalised over it); RMSEs are over the constrained oxides. Every
    model oxide gets truth / best / accepted-median / prior-median columns and
    their errors; oxides outside this target's basis are '--'. The 'prior' is the
    starting states (forward-modelled liquids at their start P, T): what you get
    with no inversion at all."""
    tgt, tr = res.target, res.target.truth
    tl = res.truth_liquid()
    con, basis = res.constrained_oxides, res.basis_oxides
    d, best = res.nodes, res.nodes.loc[res.nodes['loss'].idxmin()]
    med, prior = acc.median(numeric_only=True), d.median(numeric_only=True)

    def rmse(row, pre):
        return float(np.sqrt(np.mean([(row[f'{pre}_{o}'] - tl[o]) ** 2 for o in con])))

    def node_rmse(frame, pre):
        return np.sqrt(np.mean([(frame[f'{pre}_{o}'].values - tl[o]) ** 2 for o in con], axis=0))

    rec = {'target': tgt.name, 'assemblage': '+'.join(tgt.phases), 'n_nodes': len(d), 'n_accepted': len(acc),
           'constrained_oxides': '+'.join(con), 'basis_oxides': '+'.join(basis),
           'frac_assemblage_ok': d['assemblage_ok'].mean(),
           'frac_with_extra_phases': float(np.mean(d['n_extra_phases'] > 0)),
           'truth_liq_mass_pct': tr['mass_pct']['melts-liquid'],
           'truth_P_bar': tr['P_bar'], 'best_P_bar': best['P_bar'],
           'acc_P_bar_p50': med['P_bar'], 'prior_P_bar_p50': prior['init_P_bar'],
           'acc_P_bar_std': acc['P_bar'].std(), 'prior_P_bar_std': d['init_P_bar'].std(),
           'acc_abs_dP_bar_p50': float(np.median(np.abs(acc['P_bar'] - tr['P_bar']))),
           'prior_abs_dP_bar_p50': float(np.median(np.abs(d['init_P_bar'] - tr['P_bar']))),
           'truth_T_C': tr['T_C'], 'best_T_C': best['T_C'],
           'acc_T_C_p50': med['T_C'], 'prior_T_C_p50': prior['init_T_C'],
           'acc_T_C_std': acc['T_C'].std(), 'prior_T_C_std': d['init_T_C'].std(),
           'acc_abs_dT_C_p50': float(np.median(np.abs(acc['T_C'] - tr['T_C']))),
           'prior_abs_dT_C_p50': float(np.median(np.abs(d['init_T_C'] - tr['T_C']))),
           'best_liq_rmse_wt': rmse(best, 'liq'), 'acc_liq_rmse_wt': rmse(med, 'liq'),
           'prior_liq_rmse_wt': rmse(prior, 'init'),
           'acc_node_liq_rmse_p50': float(np.median(node_rmse(acc, 'liq'))),
           'prior_node_liq_rmse_p50': float(np.median(node_rmse(d, 'init'))),
           'best_comp_rmse_wt': best['comp_rmse_wt'], 'best_f_liquid': best['f_liquid'],
           'acc_f_liquid_p50': med['f_liquid'], 'best_extra_phases': best['extra_phases'],
           'seconds': seconds}
    if 'liqMB_SiO2' in d:
        rec['best_liqMB_rmse_wt'] = rmse(best, 'liqMB')
    # Emulator baseline: nGibbs forward model at the row's TRUE (P, T, bulk). Its
    # liquid error is what the emulator gets wrong before any inversion happens.
    emu = tr.get('emulator') or {}
    el = res.truth_liquid('emulator')
    rec['emu_liq_rmse_wt'] = float(np.sqrt(np.mean([(el[o] - tl[o]) ** 2 for o in con]))) if el else np.nan
    rec['emu_assemblage_match'] = emu.get('assemblage_match')
    rec['emu_solids_predicted'] = '+'.join(emu.get('solids_predicted', []))
    rec['true_solids'] = '+'.join(emu.get('solids_true', []))
    rec['emu_f_liquid'] = emu.get('f_liquid')
    rec['emu_cumulus_rmse_wt'] = emu.get('cumulus_rmse_wt')
    for o in res.oxides:
        inb = o in basis
        rec[f'role_{o}'] = ('constrained' if o in con else 'carried') if inb else '--'
        emu_vals = ((f'emu_{o}', el[o]), (f'emu_err_{o}', el[o] - tl[o])) if el else ()
        for key, val in ((f'truth_{o}', tl[o]), *emu_vals, (f'best_{o}', best[f'liq_{o}']),
                         (f'acc_{o}_p50', med[f'liq_{o}']), (f'prior_{o}_p50', prior[f'init_{o}']),
                         (f'best_err_{o}', best[f'liq_{o}'] - tl[o]),
                         (f'acc_err_{o}', med[f'liq_{o}'] - tl[o]),
                         (f'prior_err_{o}', prior[f'init_{o}'] - tl[o])):
            rec[key] = float(val) if inb else '--'
    return rec


if __name__ == '__main__':
    main()
