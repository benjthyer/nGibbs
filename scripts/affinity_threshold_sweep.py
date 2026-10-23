"""
Per-phase precision/recall as a function of the presence threshold on u = m/c.

The shipped presence call is `n = clamp(m, 0) > 0`, i.e. u > 0. Sweeping the threshold
tau across zero shows how much of each model's recall/precision gap is a systematic
offset at the boundary (a curve crossing that would sit at tau != 0) rather than a
capacity or label-noise problem (curves that never meet). Diagnostic only: tau is not a
knob a deployed model can use, since m is a real molar abundance above zero.

`m` is the signed mole-head output -- y_hat (the activated affinity) for an
AffinityModel, the raw mole-head output g for a ContinuousModel -- and c is the per-phase affinity
scale: the model's own `aff_c` for an AffinityModel, otherwise the Test bundle's cStats
(falling back to ml_indexer.T0 when the bundle has none) so a ContinuousModel scored on
the same bundle reads on the same axis. Ground truth is the bundle's binary labels.

    python scripts/affinity_threshold_sweep.py \
        --models 102ClosedFxBatch_Cr_NPT_train2_small 102ClosedFxBatch_Cr_NPT_train1 \
        --tarname 102ClosedFxBatch_Cr_NPT
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'src'))

import matplotlib                                                       # noqa: E402
# matplotlib.use('Agg')  # disabled: a global backend switch breaks interactive plotting (e.g. PetThermoTools)
import matplotlib.pyplot as plt                                         # noqa: E402

from ngibbs.engine.NN import rebuild_MELTS_model                        # noqa: E402
from builder.training.main import _resolve_train_roots, _resolve_bundle_path   # noqa: E402
from builder.training.loadTrainData import load_ML_data, load_ML_test_affinity  # noqa: E402
from builder.training.affinity_targets import (AffinityTargets,         # noqa: E402
                                               bundle_has_affinity_labels)

TRAIN_DIR = REPO / 'src' / 'builder' / 'training'
CKPT_DIR = TRAIN_DIR / 'temp_models'


def _checkpoint(name):
    p = Path(name)
    if p.exists():
        return p
    for cand in (CKPT_DIR / name, CKPT_DIR / f'{name}.tar'):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"No checkpoint {name!r} (looked in {CKPT_DIR})")


def _phase_names(ml_indexer):
    return sorted(ml_indexer.mass_phasedict, key=ml_indexer.mass_phasedict.get)


def _test_tensors(test_bundle, model, bundle_targets):
    """(features, binary labels) normalized with the model's own Train bounds."""
    normalizer = getattr(model.ml_indexer, 'feature_normalizer', None)
    if normalizer is None:
        raise ValueError("Checkpoint's ml_indexer has no feature_normalizer; cannot "
                         "normalize the Test bundle the way the model was trained.")
    if bundle_targets is not None:
        ds, _ = load_ML_test_affinity(test_bundle, feature_normalizer=normalizer,
                                      affinity=bundle_targets)
    else:
        ds, _ = load_ML_data(test_bundle, feature_normalizer=normalizer,
                             with_derivatives=False)
    return ds.features, ds.binarylabels


def _scale(model, bundle_targets):
    if hasattr(model, 'aff_c'):
        return model.aff_c.detach().float().cpu().numpy().ravel(), 'model aff_c'
    if bundle_targets is not None:
        return np.asarray(bundle_targets.c, dtype=np.float32), 'bundle cStats c'
    T0 = getattr(model.ml_indexer, 'T0', None)
    if T0 is not None:
        return np.asarray(T0, dtype=np.float32).ravel(), 'ml_indexer.T0'
    raise ValueError("No per-phase scale: not an AffinityModel, bundle has no cStats, "
                     "and the checkpoint has no T0.")


@torch.no_grad()
def sweep(model, X, B, c, taus, device, batch_size):
    """tp/fp/fn counts, shape (n_phases, n_taus), for presence = u > tau."""
    model.eval().to(device)
    c_t = torch.as_tensor(c, dtype=torch.float32, device=device).reshape(1, -1, 1)
    tau_t = torch.as_tensor(taus, dtype=torch.float32, device=device).reshape(1, 1, -1)
    P, K = c.size, taus.size
    tp = torch.zeros(P, K, dtype=torch.float64, device=device)
    fp, fn = torch.zeros_like(tp), torch.zeros_like(tp)
    u_present = []      # u of GT-present cells, for the boundary histogram
    for i in range(0, X.shape[0], batch_size):
        x = X[i:i + batch_size].to(device, non_blocking=True)
        truth = (B[i:i + batch_size].to(device) > 0.5).unsqueeze(-1)
        _, _, m, _, _ = model.network_component_moles(x)   # raw g
        if hasattr(model, '_mole_activation'):                 # AffinityModel: y_hat
            m = model._mole_activation(m)
        u = m.float().unsqueeze(-1) / c_t
        pred = u > tau_t
        tp += (pred & truth).sum(0)
        fp += (pred & ~truth).sum(0)
        fn += (~pred & truth).sum(0)
        u_present.append(torch.where(truth[..., 0], u[..., 0], torch.nan).cpu())
    return (tp.cpu().numpy(), fp.cpu().numpy(), fn.cpu().numpy(),
            torch.cat(u_present).numpy())


def _pr(tp, fp, fn):
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / np.maximum(tp + fn, 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return prec, rec, f1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--models', nargs='+', required=True,
                    help="checkpoint names in temp_models/ (with or without .tar) or paths")
    ap.add_argument('--tarname', required=True, help="bundle base name; <tarname>_Test is scored")
    ap.add_argument('--tau-min', type=float, default=-0.1)
    ap.add_argument('--tau-max', type=float, default=0.05)
    ap.add_argument('--n-tau', type=int, default=61)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--batch-size', type=int, default=32768)
    ap.add_argument('--min-present', type=int, default=50,
                    help="skip phases with fewer GT-present test rows than this")
    ap.add_argument('--recipe', default=None,
                    help="training recipe whose `affinity:` block built the Test workspace; "
                         "pass it so the cached workspace is reused rather than rebuilt "
                         "(the trust mask does not affect this sweep, only the cache key)")
    ap.add_argument('--out', default=None,
                    help="output dir (default temp_models/comparison/<tarname>_threshold)")
    args = ap.parse_args()

    taus = np.linspace(args.tau_min, args.tau_max, args.n_tau)
    if not np.any(np.isclose(taus, 0.0)):
        taus = np.sort(np.append(taus, 0.0))
    i0 = int(np.argmin(np.abs(taus)))

    test_bundle = _resolve_bundle_path(f'{args.tarname}_Test', _resolve_train_roots(args.tarname))
    aff_params = None
    if args.recipe:
        import yaml
        with open(args.recipe, 'r', encoding='utf-8') as fh:
            aff_params = (yaml.safe_load(fh) or {}).get('affinity')
    bundle_targets = (AffinityTargets.from_bundle(test_bundle, params=aff_params)
                      if bundle_has_affinity_labels(test_bundle) else None)
    out = Path(args.out) if args.out else CKPT_DIR / 'comparison' / f'{args.tarname}_threshold'
    out.mkdir(parents=True, exist_ok=True)

    results = {}
    phases = None
    for name in args.models:
        ckpt = _checkpoint(name)
        model = rebuild_MELTS_model(str(ckpt))
        names = _phase_names(model.ml_indexer)
        if phases is not None and names != phases:
            raise ValueError(f"{ckpt.name} has a different phase order from the first model")
        phases = names
        c, c_src = _scale(model, bundle_targets)
        X, B = _test_tensors(test_bundle, model, bundle_targets)
        print(f"[{ckpt.stem}] {type(model).__name__}, scale from {c_src}, n={X.shape[0]:,}")
        tp, fp, fn, u_pres = sweep(model, X, B, c, taus, args.device, args.batch_size)
        results[ckpt.stem] = dict(tp=tp, fp=fp, fn=fn, u_present=u_pres,
                                  cls=type(model).__name__)
        del model
        torch.cuda.empty_cache() if args.device.startswith('cuda') else None

    # ------------------------------------------------------------------ table + csv
    rows = []
    for label, r in results.items():
        prec, rec, f1 = _pr(r['tp'], r['fp'], r['fn'])
        n_pres = r['tp'][:, 0] + r['fn'][:, 0]
        print(f"\n{label} ({r['cls']})")
        print(f"  {'phase':<18}{'n_pres':>9}{'P@0':>8}{'R@0':>8}{'tau*':>9}"
              f"{'P@tau*':>8}{'R@tau*':>8}{'tau(P=R)':>10}{'%pres u<0':>11}")
        for j, ph in enumerate(phases):
            if n_pres[j] < args.min_present:
                continue
            jbest = int(np.argmax(f1[j]))
            # Where precision and recall cross: the threshold that would balance them.
            d = prec[j] - rec[j]
            cross = np.nan
            sgn = np.where(np.diff(np.sign(d)) != 0)[0]
            if sgn.size:
                k = sgn[np.argmin(np.abs(taus[sgn]))]
                cross = taus[k] - d[k] * (taus[k + 1] - taus[k]) / (d[k + 1] - d[k])
            u = r['u_present'][:, j]
            u = u[np.isfinite(u)]
            frac_neg = 100 * np.mean((u <= 0) & (u > args.tau_min)) if u.size else np.nan
            print(f"  {ph:<18}{int(n_pres[j]):>9}{prec[j, i0]:>8.3f}{rec[j, i0]:>8.3f}"
                  f"{taus[jbest]:>9.3f}{prec[j, jbest]:>8.3f}{rec[j, jbest]:>8.3f}"
                  f"{cross:>10.3f}{frac_neg:>11.2f}")
            for k, t in enumerate(taus):
                rows.append((label, ph, t, prec[j, k], rec[j, k], f1[j, k], int(n_pres[j])))
    csv = out / 'threshold_sweep.csv'
    with open(csv, 'w') as fh:
        fh.write('model,phase,tau,precision,recall,f1,n_present\n')
        for row in rows:
            fh.write(f"{row[0]},{row[1]},{row[2]:.6g},{row[3]:.6g},{row[4]:.6g},"
                     f"{row[5]:.6g},{row[6]}\n")

    # ------------------------------------------------------------------ plot
    first = next(iter(results.values()))
    n_pres = first['tp'][:, 0] + first['fn'][:, 0]
    keep = [j for j in range(len(phases)) if n_pres[j] >= args.min_present]
    ncol = 4
    nrow = int(np.ceil(len(keep) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.1 * nrow),
                             sharex=True, sharey=True, squeeze=False)
    styles = ['-', '--', ':', '-.']
    for a, j in enumerate(keep):
        ax = axes.flat[a]
        for s, (label, r) in enumerate(results.items()):
            prec, rec, _ = _pr(r['tp'], r['fp'], r['fn'])
            ls = styles[s % len(styles)]
            ax.plot(taus, prec[j], ls, color='#2a6fdb', lw=1.6,
                    label=f'precision · {label}' if a == 0 else None)
            ax.plot(taus, rec[j], ls, color='#d9534f', lw=1.6,
                    label=f'recall · {label}' if a == 0 else None)
        ax.axvline(0.0, color='0.4', lw=0.8)
        ax.set_title(f'{phases[j]}  (n={int(n_pres[j]):,})', fontsize=10)
        ax.grid(alpha=0.3)
    for a in range(len(keep), axes.size):
        axes.flat[a].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel(r'threshold $\tau$ on $u = m/c$')
    for ax in axes[:, 0]:
        ax.set_ylabel('score')
    lo = min(min(_pr(r['tp'], r['fp'], r['fn'])[k][keep].min() for k in (0, 1))
             for r in results.values())
    for ax in axes.flat:
        ax.set_ylim(max(0.0, lo - 0.02), 1.005)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.09 / nrow))
    fig.legend(loc='lower center', bbox_to_anchor=(0.5, 1 - 0.09 / nrow), ncol=2,
               fontsize=9, frameon=False,
               title=f'{args.tarname}_Test: presence = u > τ', title_fontsize=11)
    png = out / 'threshold_sweep.png'
    fig.savefig(png, dpi=130, bbox_inches='tight')
    print(f"\nWrote {png}\nWrote {csv}")


if __name__ == '__main__':
    main()
