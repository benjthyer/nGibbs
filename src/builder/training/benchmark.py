"""One training-speed + inference-speed benchmark record per training loop.

Fired once, automatically, from inside `train_Upper_MELTS`/`train_Upper_Sobolev` (so it
covers every tune-trial AND every full train episode, since both go through these same
two functions -- "any training loop" in one place), right after the 5th epoch (0-indexed
`epoch == 4`) has produced enough batches to cover `SAMPLE_COUNT` (2**20 = 1,048,576)
samples. Waiting for epoch 5 rather than epoch 1 sidesteps first-epoch JIT/cuDNN
autotune/allocator-warmup noise, which otherwise makes early-epoch timings incomparable
across runs.

Two numbers, not one, because they answer different questions:
  - train_samples_per_sec: observed from the ACTUAL training steps already
    happening in the loop (forward + backward + optimizer.step, real data, real
    DataLoader overhead) -- this is pure measurement, no extra compute is run, so it
    costs nothing beyond one `torch.cuda.synchronize()` pair per batch during that one
    epoch.
  - infer_samples_per_sec: a SEPARATE, dedicated forward-only pass (`model.eval()`,
    `torch.no_grad()`) through the REAL deployment path -- the raw model wrapped in
    `NN_MELTS` and driven via `forwardMB`, so the timing includes the mass-balance
    correction (`NN_MELTS.mass_balance`, default 'iterative') that inference actually
    pays and the training-time `upper_forward` shortcut skips. This one is deliberately
    synthetic/extra: inference never naturally happens mid-training, so there is no
    "just observe it" option, but it costs only a few hundred batches with no gradients
    and does not touch the model's parameters or the training run's own data budget.

Every record also carries the full model config (architecture), the loss/training
config the caller supplies, a parameter count, and an instantaneous system-resource
snapshot (`ngibbs.utils.system_stats`) taken right when the record is written.

Records are appended as one JSON object per line (JSONL) rather than maintained as a
single JSON array: an array would need the whole file read, appended to, and rewritten
on every training loop, which both scales badly over a long tuning run (hundreds of
trials) and risks corrupting the entire history if a run is killed mid-write. JSONL only
ever appends a line, so a single interrupted run costs at most one partial record, and
every prior one stays intact and independently parseable.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from ngibbs.utils.system_stats import get_system_stats

SAMPLE_COUNT = 1_048_576  # 2**20
TARGET_EPOCH = 4          # 0-indexed -- "after only the 5th epoch"
DEFAULT_LOG_PATH = Path(__file__).parent / "logs" / "benchmark_log.jsonl"

# One throwaway call so the first real reading later isn't psutil's meaningless "0.0,
# no prior call to compare against" first-sample artifact.
try:
    import psutil
    psutil.cpu_percent(interval=None)
except ImportError:
    pass


class ThroughputBenchmark:
    """Accumulates wall-clock time and sample count during epoch `TARGET_EPOCH` only.

    Call `.tick(epoch, n_samples, dt)` once per training batch, unconditionally -- it is
    a no-op outside the target epoch and after `.finalize()` has already run once, so
    the call site in the training loop never needs its own epoch check. `dt` must be
    measured with CUDA already synchronized on both ends (see `timed_step`) or GPU
    training will report a meaningless, far-too-fast rate (async kernel-launch time,
    not actual compute time).
    """

    def __init__(self) -> None:
        self._elapsed = 0.0
        self._count = 0
        self.ready = False
        self._done = False

    def should_time(self, epoch: int) -> bool:
        """Whether the caller should wrap this batch's training step in a timed,
        CUDA-synchronized region at all -- False every epoch except `TARGET_EPOCH`, and
        False forever once `.finalize()` has already run, so a training loop can call
        this unconditionally every batch without its own epoch bookkeeping."""
        return not self._done and epoch == TARGET_EPOCH

    def tick(self, epoch: int, n_samples: int, dt: float) -> None:
        if self._done or epoch != TARGET_EPOCH or self.ready:
            return
        self._elapsed += dt
        self._count += n_samples
        if self._count >= SAMPLE_COUNT:
            self.ready = True

    @property
    def train_samples_per_sec(self) -> float:
        return self._count / self._elapsed if self._elapsed > 0 else float('nan')

    def finalize(
        self,
        *,
        model,
        infer_batches,
        batch_size: int,
        device: str,
        episode_label: str,
        training_config: Dict[str, Any],
        log_path: Optional[Path] = None,
        amp_dtype: "torch.dtype" = torch.float32,
        use_amp: bool = False,
        mass_balance: str = 'iterative',
    ) -> Dict[str, Any]:
        """Run the inference benchmark, assemble one record, append it, and return it.

        `infer_batches`: an iterable of normalized model-space feature tensors (already
        on `device`, or movable to it) fed through `NN_MELTS.forwardMB` -- typically the
        test-set loader already in scope where this is called from, which is why this
        takes an iterable rather than a fixed tensor: the caller doesn't need to
        materialize a separate benchmark dataset.

        Safe to call only once per instance -- a second call is a no-op returning {}
        (mirrors `.tick`'s own one-shot guard, so a caller that races both epoch-loop
        exit and mid-loop `.ready` checks can't double-fire).
        """
        if self._done:
            return {}
        self._done = True

        # n_samples passed explicitly (not left to measure_inference_throughput's own
        # default): that default is bound to SAMPLE_COUNT at module-import time, so it
        # would silently stop tracking any later rebinding of the module-level constant.
        # Reading SAMPLE_COUNT here, by name, at call time, is what actually keeps the
        # train-side threshold (tick()'s "self._count >= SAMPLE_COUNT", looked up fresh
        # every call) and the inference-side one consistent with each other.
        # Measured under the same precision training/eval just ran under, so the
        # reported inference cost reflects what deployment at this precision would
        # actually see, not always a float32 number regardless of constants.TRAIN_PRECISION.
        infer_rate = measure_inference_throughput(model, infer_batches, batch_size, device,
                                                   n_samples=SAMPLE_COUNT,
                                                   amp_dtype=amp_dtype, use_amp=use_amp,
                                                   mass_balance=mass_balance)
        train_rate = self.train_samples_per_sec

        record: Dict[str, Any] = {
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'episode': episode_label,
            'device': device,
            'model_class': type(model).__name__,
            'model_config': dict(getattr(model, 'config', {})),
            'param_count': sum(p.numel() for p in model.parameters()),
            'training_config': training_config,
            'train_samples_per_sec': train_rate,
            'train_time_for_1048576_s': SAMPLE_COUNT / train_rate if train_rate > 0 else None,
            'infer_samples_per_sec': infer_rate,
            'infer_time_for_1048576_s': SAMPLE_COUNT / infer_rate if infer_rate and infer_rate > 0 else None,
            'system_stats': get_system_stats(),
        }
        append_json_record(record, log_path)
        return record


def timed_step(device: str):
    """Context manager: synchronizes CUDA on entry and exit (a no-op on CPU) and hands
    back a zero-argument function returning the elapsed seconds. Wrap exactly the region
    you want timed (e.g. forward + backward + optimizer.step) -- without the enclosing
    syncs, GPU work is asynchronous and CPU-side wall time understates real cost by
    however much of the kernel queue hadn't drained yet.

    Usage:
        with timed_step(device) as get_dt:
            ...forward/backward/step...
        dt = get_dt()
    """
    class _Timer:
        def __enter__(self):
            if device == 'cuda' and torch.cuda.is_available():
                torch.cuda.synchronize()
            self._t0 = time.perf_counter()
            return lambda: self._t1 - self._t0

        def __exit__(self, *exc):
            if device == 'cuda' and torch.cuda.is_available():
                torch.cuda.synchronize()
            self._t1 = time.perf_counter()
            return False

    return _Timer()


def measure_inference_throughput(model, batches, batch_size: int, device: str,
                                 n_samples: Optional[int] = None,
                                 amp_dtype: "torch.dtype" = torch.float32,
                                 use_amp: bool = False,
                                 mass_balance: str = 'iterative') -> float:
    """Dedicated forward-only timed pass over real feature batches through the REAL
    deployment path -- the raw model wrapped in `NN_MELTS`, driven via
    `forwardMB(..., Normalize=False)` -- so the timing includes the `mass_balance`
    correction inference actually pays. `model.eval()` + `no_grad`, cycling through
    `batches` as many times as needed to cover `n_samples`. Returns samples/sec, or
    `float('nan')` if `batches` yields nothing (an empty loader).

    `batches` yields the same normalized model-space feature tensors the training loop
    feeds `model.forward`, hence `Normalize=False`.

    `n_samples=None` (the default) resolves to `SAMPLE_COUNT` INSIDE the function body,
    not as a bound default value -- a `def f(n=SAMPLE_COUNT)` default is evaluated once,
    at import time, and would silently ignore any later rebinding of the module-level
    constant (e.g. a test shrinking it for a quick run). Resolving it in the body reads
    the name fresh on every call instead.
    """
    from ngibbs.engine.emulator import NN_MELTS

    n_samples = SAMPLE_COUNT if n_samples is None else n_samples
    was_training = model.training
    emu = NN_MELTS(model, cuda=(str(device) == 'cuda'), mass_balance=mass_balance)
    seen = 0
    elapsed = 0.0
    try:
        with torch.no_grad():
            while seen < n_samples:
                progressed = False
                for batch in batches:
                    x = batch[0] if isinstance(batch, (list, tuple)) else batch
                    x = x.to(device, non_blocking=True)
                    with timed_step(device) as get_dt:
                        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                            emu.forwardMB(x, Normalize=False, outputs=['phase_tables'])
                    elapsed += get_dt()
                    seen += x.size(0)
                    progressed = True
                    if seen >= n_samples:
                        break
                if not progressed:
                    break  # batches was empty -- avoid spinning forever
    finally:
        if was_training:
            model.train()
    return seen / elapsed if elapsed > 0 else float('nan')


def append_json_record(record: Dict[str, Any], log_path: Optional[Path] = None) -> None:
    log_path = Path(log_path) if log_path is not None else DEFAULT_LOG_PATH
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(record) + "\n")
