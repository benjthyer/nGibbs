"""Reproduce (or rule out) the CUDA failure of the one-shot 'pinv' mass-balance solver.

`NN_MELTS.polish_masses` (mass_balance='pinv') calls `torch.linalg.pinv` on
batched (B, E, P) and (B, E, C) abundance-weighted matrices, and is pinned to the
CPU. The hypothesis under test: on CUDA, the batched SVD behind `pinv` fails to
converge on these sparse, frequently rank-deficient matrices in float32, and
float64 may or may not sidestep it.

What it does
  1. CAPTURE  -- runs the real HeFESTo emulator on CPU with mass_balance='pinv'
     and records every matrix handed to torch.linalg.pinv (exact production
     inputs, no reconstruction). Inputs are a P sweep at a fixed bulk, or rows
     sampled from an ML-ready bundle's features (`--bundle`). `--synthetic` skips the model and builds
     matrices with a controllable zero fraction instead.
  2. DESCRIBE -- zero fraction, all-zero rows (uncarried elements), numerical
     rank and condition number, NaN/Inf count.
  3. REPLAY   -- on each device (CUDA if visible, plus CPU) and dtype
     (float32, float64): torch.linalg.pinv, torch.linalg.svd with each CUDA
     driver, and the Lagrange/normal-equation solve the MassBalanceProjector
     uses. Reports success / exception, wall time, NaN outputs, and error
     relative to a CPU float64 pinv reference. For a convergence failure it
     prints the offending batch element.

Usage
  python scripts/diagnose_pinv_cuda.py                        # real HeFESTo matrices
  python scripts/diagnose_pinv_cuda.py --batch 32768
  python scripts/diagnose_pinv_cuda.py --bundle data/MLready/HeFESTo/<name>_Valid.tar.gz
  python scripts/diagnose_pinv_cuda.py --synthetic --zero-frac 0.8
  python scripts/diagnose_pinv_cuda.py --save pinv_inputs.pt  # keep captured matrices
  python scripts/diagnose_pinv_cuda.py --load pinv_inputs.pt  # replay without the model
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
for p in (REPO / "src", REPO / "src" / "ngibbs"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

DEFAULT_MODEL_DIR = REPO / "src" / "ngibbs" / "engine" / "TrainedModels" / "HeFESToEarthAdiabat"

# Same bulk as scripts/performance_benchmarking.py (isentropic HeFESTo input).
BULK = {
    'S(J/g/K)(System_main)': 2.5,
    'Si': 3.57887, 'Mg': 5.64233, 'Fe': 0.58052, 'Ca': 0.07969,
    'Al': 0.09740, 'Na': 0.00160, 'Cr': 0.01960, 'O': 13.63947,
}


# ----------------------------------------------------------------------------- capture
def build_inputs(batch, jitter, seed):
    """P swept 0-140 GPa so the assemblage (and hence the zero pattern) changes
    along the batch; optional multiplicative jitter on the element amounts."""
    rng = np.random.default_rng(seed)
    P = np.linspace(0.0, 140.0, batch)
    vals = np.tile(np.array(list(BULK.values()), dtype=np.float64), (batch, 1))
    if jitter > 0:
        vals[:, 1:] *= np.exp(rng.normal(0.0, jitter, size=vals[:, 1:].shape))
    table = np.concatenate([P[:, None], vals], axis=1).astype(np.float32)
    headers = ['P(GPa)'] + list(BULK.keys())
    return table, headers


def bundle_inputs(bundle_path, batch, seed):
    """Raw feature rows from an ML-ready bundle (labels are never loaded); a
    random `batch` rows are drawn without replacement, or all if fewer."""
    from ngibbs.utils.file_utils import load_ml_bundle

    print(f"Reading features from {bundle_path}")
    bundle = load_ml_bundle(bundle_path, arrays=['features'])
    features = np.asarray(bundle.features)
    # Bundle feature columns: intensive featureNames, then the element amounts.
    ix = bundle.ml_indexer
    headers = [str(h) for h in list(ix.featureNames) + list(ix.Elkeys)]
    if len(headers) != features.shape[1]:
        raise ValueError(f"{len(headers)} feature names {headers} for {features.shape[1]} feature columns")
    n = features.shape[0]
    if batch < n:
        rows = np.sort(np.random.default_rng(seed).choice(n, size=batch, replace=False))
        features = features[rows]
    print(f"  using {features.shape[0]} of {n} rows; features: {headers}")
    return features.astype(np.float32), headers


def capture_real(model_dir, batch, jitter, seed, bundle=None):
    from ngibbs.engine.API import HeFESToAPI  # avoid models.py: it loads every model

    print(f"Loading HeFESTo emulator (CPU, mass_balance='pinv') from {model_dir}")
    api = HeFESToAPI(model_dir=str(model_dir), device='cpu', mass_balance='pinv')
    if bundle is not None:
        table, headers = bundle_inputs(bundle, batch, seed)
    else:
        table, headers = build_inputs(batch, jitter, seed)

    captured = []
    orig = torch.linalg.pinv

    def recording_pinv(A, *args, **kwargs):
        captured.append(A.detach().to('cpu').clone())
        return orig(A, *args, **kwargs)

    torch.linalg.pinv = recording_pinv
    try:
        api.ForwardMB(torch.tensor(table), headers=headers, outputs=['component_moles'])
    finally:
        torch.linalg.pinv = orig

    if not captured:
        raise RuntimeError("No torch.linalg.pinv calls were captured -- is mass_balance='pinv' "
                           "reaching polish_masses on this code path?")
    # ForwardMB runs in internal sub-batches; merge calls that share a matrix shape.
    groups = {}
    for A in captured:
        groups.setdefault(tuple(A.shape[1:]), []).append(A)
    # polish_masses makes two pinv calls per batch: phase-level (E x P) when
    # optimize_masses is on, then component-level (E x C). Groups keep call order.
    return {f"pinv input group {i + 1}: (B, E, K) = (*, {E}, {K})": torch.cat(mats, dim=0)
            for i, ((E, K), mats) in enumerate(groups.items())}


def synthetic(batch, E, C, zero_frac, seed):
    """Abundance-weighted A = (compToEl * w)^T with a fixed sparse stoichiometry
    and a fraction `zero_frac` of components absent per row."""
    g = torch.Generator().manual_seed(seed)
    stoich = torch.zeros(C, E, dtype=torch.float64)
    for c in range(C):
        k = int(torch.randint(1, 4, (1,), generator=g))
        idx = torch.randperm(E, generator=g)[:k]
        stoich[c, idx] = torch.randint(1, 4, (k,), generator=g).double()
    w = torch.exp(torch.randn(batch, C, generator=g, dtype=torch.float64) * 2.0)
    w = w * (torch.rand(batch, C, generator=g, dtype=torch.float64) > zero_frac)
    A = (w.unsqueeze(-1) * stoich.unsqueeze(0)).transpose(1, 2)  # (B, E, C)
    return {f"synthetic ({E}x{C}, {zero_frac:.0%} absent)": A.float()}


# ----------------------------------------------------------------------------- describe
def describe(A):
    A64 = A.double()
    finite = torch.isfinite(A64).all(dim=(1, 2))
    Af = A64[finite]
    zero_frac = (Af == 0).double().mean().item()
    zero_rows = (Af.abs().sum(dim=2) == 0).double().sum(dim=1)        # uncarried elements
    zero_cols = (Af.abs().sum(dim=1) == 0).double().mean().item()       # absent components
    s = torch.linalg.svdvals(Af)
    tol = s[:, :1] * max(Af.shape[1:]) * torch.finfo(torch.float64).eps
    rank = (s > tol).sum(dim=1)
    smin = torch.where(s > tol, s, torch.full_like(s, float('inf'))).min(dim=1).values
    cond = (s[:, 0] / smin).clamp(max=1e300)
    q = lambda x: np.percentile(x.cpu().numpy(), [50, 90, 99, 100])
    E = A.shape[1]
    print(f"    batch {A.shape[0]}, non-finite matrices: {int((~finite).sum())}")
    print(f"    zero entries {zero_frac:.1%}; absent (all-zero) columns {zero_cols:.1%}")
    print(f"    rows with an uncarried element: {(zero_rows > 0).double().mean().item():.1%} of batch")
    print(f"    numerical rank (of max {E}): " +
          ", ".join(f"{r}:{int((rank == r).sum())}" for r in torch.unique(rank).tolist()))
    print(f"    cond (sigma_max/sigma_min, nonzero sigmas) p50/p90/p99/max: "
          + " / ".join(f"{v:.2e}" for v in q(cond)))


# ----------------------------------------------------------------------------- replay
def _sync(dev):
    if dev == 'cuda':
        torch.cuda.synchronize()


def _attempt(fn, dev):
    try:
        _sync(dev)
        t0 = time.perf_counter()
        out = fn()
        _sync(dev)
        return 'ok', time.perf_counter() - t0, out
    except Exception as e:  # LinAlgError, RuntimeError from cuSOLVER, ...
        msg = str(e).strip().splitlines()[0] if str(e).strip() else ''
        return f"{type(e).__name__}: {msg}", None, None


def _batch_element(msg):
    m = re.search(r"[Bb]atch element (\d+)", msg)
    return int(m.group(1)) if m else None


def replay(label, A, devices, dtypes, tikhonov):
    ref = torch.linalg.pinv(A.double())                            # CPU float64 reference
    print(f"\n  {'device':6s} {'dtype':8s} {'operation':22s} {'result':10s} {'time':>9s} "
          f"{'NaN rows':>8s} {'rel.err p50 / max':>20s}")
    for dev in devices:
        for dt in dtypes:
            Ad = A.to(device=dev, dtype=dt)
            ops = {'pinv (default)': lambda: torch.linalg.pinv(Ad)}
            drivers = ['gesvd', 'gesvdj', 'gesvda'] if dev == 'cuda' else []
            ops['svd (default driver)'] = lambda: torch.linalg.svd(Ad, full_matrices=False)
            for drv in drivers:
                ops[f'svd driver={drv}'] = (lambda d=drv: torch.linalg.svd(Ad, full_matrices=False, driver=d))

            E = Ad.shape[1]
            eye = torch.eye(E, device=dev, dtype=dt)

            def lagrange():
                # MassBalanceProjector's route: A^T (A A^T + tau I)^-1, here as the full
                # right-inverse so it can be compared with pinv entry for entry.
                M = Ad @ Ad.transpose(1, 2) + tikhonov * eye
                return Ad.transpose(1, 2) @ torch.linalg.solve(M, eye)
            ops[f'solve A A^T+{tikhonov:g}I'] = lagrange

            for name, fn in ops.items():
                status, t, out = _attempt(fn, dev)
                if status != 'ok':
                    print(f"  {dev:6s} {str(dt)[6:]:8s} {name:22s} FAILED")
                    print(f"         -> {status[:160]}")
                    k = _batch_element(status)
                    if k is not None and k < A.shape[0]:
                        Ak = A[k].double()
                        print(f"         failing element {k}: zero frac {(Ak == 0).double().mean():.0%}, "
                              f"sigma = {np.array2string(torch.linalg.svdvals(Ak).numpy(), precision=2)}")
                    continue
                if isinstance(out, tuple):          # svd: reconstruct pinv for the comparison
                    U, S, Vh = out
                    rtol = max(A.shape[1:]) * torch.finfo(dt).eps
                    keep = S > rtol * S[:, :1]
                    Sinv = torch.where(keep, 1.0 / torch.where(keep, S, torch.ones_like(S)), torch.zeros_like(S))
                    out = Vh.transpose(1, 2) @ (Sinv.unsqueeze(-1) * U.transpose(1, 2))
                out = out.double().cpu()
                nan_rows = int((~torch.isfinite(out).all(dim=(1, 2))).sum())
                ok = torch.isfinite(out).all(dim=(1, 2))
                err = ((out - ref).flatten(1).norm(dim=1) / ref.flatten(1).norm(dim=1).clamp(min=1e-300))[ok]
                e = f"{err.median():.1e} / {err.max():.1e}" if err.numel() else "n/a"
                print(f"  {dev:6s} {str(dt)[6:]:8s} {name:22s} {'ok':10s} {t*1e3:7.1f}ms {nan_rows:8d} {e:>20s}")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--batch', type=int, default=2**14)
    ap.add_argument('--model-dir', type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument('--bundle', type=Path,
                    help='ML-ready .tar.gz bundle: sample --batch rows of its features as model inputs')
    ap.add_argument('--jitter', type=float, default=0.0, help='log-normal sigma on bulk elements')
    ap.add_argument('--synthetic', action='store_true')
    ap.add_argument('--E', type=int, default=8)
    ap.add_argument('--C', type=int, default=32)
    ap.add_argument('--zero-frac', type=float, default=0.8)
    ap.add_argument('--tikhonov', type=float, default=1e-12)
    ap.add_argument('--save', type=Path)
    ap.add_argument('--load', type=Path)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--dtypes', default='float32,float64',
                    help='comma list; run float64 alone if a float32 failure corrupts the CUDA context')
    args = ap.parse_args()

    print(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device: {torch.cuda.get_device_name(0)}, CUDA {torch.version.cuda}, "
              f"preferred linalg backend: {torch.backends.cuda.preferred_linalg_library()}")

    if args.load:
        mats = torch.load(args.load)
    elif args.synthetic:
        mats = synthetic(args.batch, args.E, args.C, args.zero_frac, args.seed)
    else:
        mats = capture_real(args.model_dir, args.batch, args.jitter, args.seed, bundle=args.bundle)
    if args.save:
        torch.save(mats, args.save)
        print(f"saved captured matrices to {args.save}")

    devices = (['cuda'] if torch.cuda.is_available() else []) + ['cpu']
    if devices == ['cpu']:
        print("\n[warn] No CUDA device visible: replaying on CPU only, so the GPU failure "
              "cannot be reproduced here. Use --save, then --load on a GPU node.")
    for label, A in mats.items():
        print(f"\n=== {label} ===")
        describe(A)
        dtypes = tuple(getattr(torch, d.strip()) for d in args.dtypes.split(','))
        replay(label, A, devices, dtypes, args.tikhonov)


if __name__ == '__main__':
    main()
