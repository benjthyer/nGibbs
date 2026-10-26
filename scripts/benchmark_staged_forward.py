"""A/B benchmark: legacy vs current EmulatorAPI._staged_forward.

The legacy loop (pre 2026-10-26) called gc.collect() + torch.cuda.empty_cache()
after every batch and re-concatenated the growing result each batch
(quadratic copying). The current loop moves each batch to host once and
concatenates once at the end.

Both versions are run on the same inputs with the same batch boundaries, so
outputs should be bit-identical; this script checks that, then reports wall
time (median of --repeats) and peak VRAM for:
  A) ForwardMB (isentropic emulator), many chunks
  B) isomorphic EOS properties (rho, Vp, Vs, S) via _compute_bulk_EOS_properties

Usage:
    python scripts/benchmark_staged_forward.py [--n 262144] [--nn-batch 16384] [--repeats 3]
"""
import argparse
import gc
import statistics
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import tqdm

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from ngibbs.engine.models import HeFESToEmulatorGPU  # noqa: E402


def _legacy_staged_forward(self, func, input_tensor, batch_size, **kwargs):
    """Verbatim copy of the pre-change _staged_forward, for A/B timing."""
    cpu_fallback = kwargs.pop('_cpu_fallback', None)

    def _merge_outputs(existing, batch):
        if isinstance(existing, dict) and isinstance(batch, dict):
            merged = {}
            for key in existing:
                if key not in batch:
                    raise KeyError(f"Missing key '{key}' in staged batch output")
                merged[key] = _merge_outputs(existing[key], batch[key])
            return merged
        if isinstance(existing, list) and isinstance(batch, (list, tuple)):
            if len(existing) != len(batch):
                raise ValueError("Staged batch output list length mismatch")
            return [_merge_outputs(existing[i], batch[i]) for i in range(len(existing))]
        if isinstance(existing, tuple) and isinstance(batch, (list, tuple)):
            if len(existing) != len(batch):
                raise ValueError("Staged batch output tuple length mismatch")
            return tuple(_merge_outputs(existing[i], batch[i]) for i in range(len(existing)))
        if isinstance(existing, torch.Tensor) and isinstance(batch, torch.Tensor):
            return torch.cat([existing, batch.detach().cpu()], dim=0)
        if isinstance(existing, np.ndarray) and isinstance(batch, np.ndarray):
            return np.concatenate([existing, batch], axis=0)
        return batch

    n_samples = input_tensor.size(0)
    if n_samples < self._CPU_BATCH_THRESHOLD:
        cpu_fn = cpu_fallback or self._get_cpu_func(func)
        if cpu_fn is not None:
            return cpu_fn(input_tensor.detach().cpu(), **kwargs)
    if n_samples <= batch_size:
        return func(input_tensor, **kwargs)
    with torch.no_grad():
        outputs = func(input_tensor[:batch_size], **kwargs)
        n_batches = (n_samples + batch_size - 1) // batch_size
        for batch_idx in tqdm.tqdm(range(1, n_batches), desc="Processing batches of size " + str(batch_size)):
            start = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, n_samples)
            batch_outputs = func(input_tensor[start:end], **kwargs)
            outputs = _merge_outputs(outputs, batch_outputs)
            del batch_outputs
            gc.collect()
            torch.cuda.empty_cache()
        return outputs


INPUT = {
    'S(J/g/K)(System_main)': 2.5, 'Si': 3.57887, 'Mg': 5.64233, 'Fe': 0.58052,
    'Ca': 0.07969, 'Al': 0.09740, 'Na': 0.00160, 'Cr': 0.01960, 'O': 13.63947,
}


def example_array(n):
    p = np.linspace(0, 140, n, dtype=np.float32).reshape(-1, 1)
    comp = np.array(list(INPUT.values()), dtype=np.float32)[None, :]
    return np.concatenate([p, np.tile(comp, (n, 1))], axis=1), ['P(GPa)(System_main)'] + list(INPUT)


def timed(fn, repeats):
    times, peak, out = [], 0, None
    for _ in range(repeats):
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        t0 = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        peak = max(peak, torch.cuda.max_memory_allocated() - base)
    return statistics.median(times), times, peak, out


def to_np(x):
    if isinstance(x, dict):
        return {k: to_np(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_np(v) for v in x]
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return x


def identical(a, b):
    a, b = to_np(a), to_np(b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(identical(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(identical(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray):
        return a.shape == b.shape and np.array_equal(a, b, equal_nan=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=2 ** 18)
    ap.add_argument('--nn-batch', type=int, default=2 ** 14)
    ap.add_argument('--repeats', type=int, default=3)
    a = ap.parse_args()

    api = HeFESToEmulatorGPU
    if api is None or not torch.cuda.is_available():
        sys.exit("No CUDA device / GPU emulator available.")
    print(f"GPU: {torch.cuda.get_device_name()}  torch {torch.__version__}")
    print(f"N = {a.n}, NN batch = {a.nn_batch}, EOS batch = {api._EOS_BATCH_SIZE}, repeats = {a.repeats}\n")

    current = api._staged_forward  # bound method from the class
    legacy = types.MethodType(_legacy_staged_forward, api)

    X, headers = example_array(a.n)
    Xt = torch.tensor(X, device='cuda')
    outs = ['component_moles', 'phase_moles', 'chem_out']

    def run_nn():
        return api.ForwardMB(Xt, headers=headers, batch_size=a.nn_batch, outputs=list(outs))

    # warm-up (CUDA context, cuBLAS handles, allocator)
    api.ForwardMB(Xt[:a.nn_batch * 2], headers=headers, batch_size=a.nn_batch, outputs=list(outs))

    rows, results = [], {}
    for label, impl in (('legacy', legacy), ('current', current)):
        api._staged_forward = impl
        try:
            med, ts, peak, out = timed(run_nn, a.repeats)
            results[('nn', label)] = out
            rows.append(('ForwardMB', label, med, ts, peak))
        except Exception as e:  # legacy may fail on mixed-device cat
            rows.append(('ForwardMB', label, float('nan'), [repr(e)[:120]], 0))
    api._staged_forward = current

    cm = results.get(('nn', 'current'))
    cm = cm['component_moles'].to('cuda', torch.float64)
    PT = torch.stack([Xt[:, 0].double(), torch.full((a.n,), 1600.0, device='cuda', dtype=torch.float64)], 1)
    iso = ('rho', 'Vp', 'Vs', 'S')

    def run_eos():
        return api._compute_bulk_EOS_properties(cm, PT, property_names=iso)

    api._compute_bulk_EOS_properties(cm[:api._EOS_BATCH_SIZE * 2], PT[:api._EOS_BATCH_SIZE * 2], property_names=iso)
    for label, impl in (('legacy', legacy), ('current', current)):
        api._staged_forward = impl
        try:
            med, ts, peak, out = timed(run_eos, a.repeats)
            results[('eos', label)] = out
            rows.append(('EOS isomorphic', label, med, ts, peak))
        except Exception as e:
            rows.append(('EOS isomorphic', label, float('nan'), [repr(e)[:120]], 0))
    api._staged_forward = current

    print("\n{:<16}{:<9}{:>10}{:>12}   runs".format('workload', 'impl', 'median s', 'peak MiB'))
    for w, l, med, ts, peak in rows:
        runs = ', '.join(f"{t:.3f}" if isinstance(t, float) else t for t in ts)
        print(f"{w:<16}{l:<9}{med:>10.3f}{peak / 2**20:>12.0f}   {runs}")
    for kind in ('nn', 'eos'):
        if (kind, 'legacy') in results and (kind, 'current') in results:
            same = identical(results[(kind, 'legacy')], results[(kind, 'current')])
            print(f"{kind}: outputs bit-identical = {same}")


if __name__ == '__main__':
    main()
