"""VRAM-aware, persistent batch-size tuning for staged GPU inference.

Instead of persisting a single batch size, this stores a small *memory model*
per workload (a GPU-resident callable such as ``forwardMB`` on one emulator, or
one tier of the HeFESTo EOS):

    peak_bytes(b) ~= fixed_bytes + bytes_per_row * b

plus ``plateau_rows`` -- the smallest batch whose throughput is within
``plateau_tol`` of the best measured. At call time the batch size is

    min(plateau_rows, (available_vram * safety - fixed_bytes) / bytes_per_row)

using the VRAM that is free *right now*, so whatever is already resident (e.g.
component moles kept on the GPU between the NN and EOS stages of
``get_isentrope``) is accounted for automatically.

Profiles are saved as JSON in a per-user config directory and keyed by a GPU
fingerprint (device name, total memory, torch / CUDA versions); a profile made
on another GPU or software stack is ignored. Precedence everywhere:

    explicit ``batch_size=`` argument  >  tuned profile  >  built-in default

Environment variables
---------------------
NGIBBS_CONFIG_DIR       directory for ``batch_tuning.json`` (overrides default)
NGIBBS_BATCH_TUNING=0   ignore tuned profiles (always use the defaults)

Typical use::

    api = HeFESToAPI(model_dir, device='cuda')
    api.tune_batch_size()          # once per machine; ~1-2 min
    # every later session on this GPU picks the tuned values up automatically
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import torch

SCHEMA_VERSION = 1
FILENAME = "batch_tuning.json"

#: Smallest batch the resolver or the OOM back-off will ever use.
MIN_ROWS = 1024
#: Fraction of currently available VRAM a batch may plan to use.
DEFAULT_SAFETY = 0.8
#: A batch is "on the plateau" when its throughput is within this of the best.
DEFAULT_PLATEAU_TOL = 0.05
#: Throughput falling below this fraction of the best ends the sweep (cliff,
#: e.g. the Windows driver spilling into shared system memory).
CLIFF_FRACTION = 0.6

#: EOS property tiers, cheapest first. A call uses the most expensive tier it
#: requests (see :func:`eos_tier`).
EOS_TIER_PROPERTIES = {
    'isomorphic': ('rho', 'Vp', 'Vs', 'S'),
    'metamorphic': ('rho', 'Vp', 'Vs', 'S', 'cptot', 'KStot'),
    'fast': ('rho', 'Vp_fast', 'Vs', 'S', 'cptot', 'KStot'),
}

# Default HeFESTo bulk (pyrolite-like, elemental moles) used by the tuner when
# the caller supplies no composition.
_DEFAULT_ELEMENT_COMPOSITION = {
    'Si': 3.57887, 'Mg': 5.64233, 'Fe': 0.58052, 'Ca': 0.07969,
    'Al': 0.09740, 'Na': 0.00160, 'Cr': 0.01960, 'O': 13.63947,
}

_NN_KINDS = (
    ('isothermal', 'isothermal_emulator', '_iso_path'),
    ('isentropic', 'isentropic_emulator', '_isen_path'),
    ('openox', 'open_emulator', '_open_path'),
)

# --------------------------------------------------------------------------- #
# Persistent state (module-global, loaded lazily once per process)
# --------------------------------------------------------------------------- #
_STATE: Optional[dict] = None


def config_dir() -> Path:
    env = os.environ.get('NGIBBS_CONFIG_DIR')
    if env:
        return Path(env).expanduser()
    if sys.platform.startswith('win'):
        base = os.environ.get('LOCALAPPDATA') or (Path.home() / 'AppData' / 'Local')
        return Path(base) / 'ngibbs'
    base = os.environ.get('XDG_CONFIG_HOME') or (Path.home() / '.config')
    return Path(base) / 'ngibbs'


def config_path() -> Path:
    return config_dir() / FILENAME


def enabled() -> bool:
    return os.environ.get('NGIBBS_BATCH_TUNING', '1').strip().lower() not in ('0', 'false', 'off', 'no')


def load(force: bool = False) -> dict:
    """Return the persisted state (reading the JSON file on first use)."""
    global _STATE
    if _STATE is not None and not force:
        return _STATE
    state = {'version': SCHEMA_VERSION, 'profiles': {}}
    path = config_path()
    if path.exists():
        try:
            with open(path, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
            if isinstance(data, dict) and data.get('version') == SCHEMA_VERSION:
                state = data
                state.setdefault('profiles', {})
        except (OSError, ValueError) as exc:
            warnings.warn(f"[nGibbs] Ignoring unreadable batch tuning file {path}: {exc}")
    _STATE = state
    return _STATE


def save() -> Path:
    """Atomically write the in-memory state to :func:`config_path`."""
    state = load()
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.batch_tuning.', suffix='.json', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as fh:
            json.dump(state, fh, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def _is_cuda(device) -> bool:
    return torch.device(device).type == 'cuda' and torch.cuda.is_available()


def _cuda_index(device) -> int:
    d = torch.device(device)
    return d.index if d.index is not None else torch.cuda.current_device()


def fingerprint(device) -> Optional[str]:
    """Stable identity of a GPU + software stack, or None for non-CUDA devices."""
    if not _is_cuda(device):
        return None
    idx = _cuda_index(device)
    props = torch.cuda.get_device_properties(idx)
    total_mib = int(props.total_memory // 2 ** 20)
    return f"{props.name}|{total_mib}MiB|torch{torch.__version__}|cuda{torch.version.cuda}"


def profile(device, create: bool = False) -> Optional[dict]:
    """Workload table for ``device``'s fingerprint (``None`` if untuned)."""
    fp = fingerprint(device)
    if fp is None:
        return None
    profiles = load()['profiles']
    if fp not in profiles and create:
        profiles[fp] = {'workloads': {}}
    return profiles.get(fp)


def clear(device=None, workloads: Optional[Iterable[str]] = None, persist: bool = True) -> None:
    """Forget tuned values: everything, one device's profile, or some workloads."""
    state = load()
    if device is None:
        state['profiles'] = {}
    else:
        fp = fingerprint(device)
        if fp in state['profiles']:
            if workloads is None:
                del state['profiles'][fp]
            else:
                for key in workloads:
                    state['profiles'][fp]['workloads'].pop(key, None)
    if persist:
        save()


def summary(device=None) -> Dict[str, dict]:
    """{workload: entry} for ``device`` (default: current CUDA device)."""
    if device is None:
        if not torch.cuda.is_available():
            return {}
        device = 'cuda'
    prof = profile(device)
    return dict(prof['workloads']) if prof else {}


# --------------------------------------------------------------------------- #
# Workload keys
# --------------------------------------------------------------------------- #
def nn_key(kind: str, checkpoint: Optional[str], method: str, mass_balance: Optional[str]) -> str:
    ckpt = Path(checkpoint).name if checkpoint else 'unknown'
    return f"nn/{method}/{kind}/{ckpt}/mb={mass_balance}"


def eos_key(tier: str, backend: str) -> str:
    return f"eos/{tier}/{backend}"


def eos_tier(property_names: Sequence[str], metamorphic_keys=(), fast_keys=()) -> str:
    names = set(property_names)
    if names & set(fast_keys):
        return 'fast'
    if names & set(metamorphic_keys):
        return 'metamorphic'
    return 'isomorphic'


# --------------------------------------------------------------------------- #
# Call-time resolution
# --------------------------------------------------------------------------- #
def available_bytes(device) -> int:
    """Free VRAM now, counting memory torch has cached but is not using."""
    idx = _cuda_index(device)
    free, _total = torch.cuda.mem_get_info(idx)
    cached_unused = torch.cuda.memory_reserved(idx) - torch.cuda.memory_allocated(idx)
    return int(free + max(cached_unused, 0))


def rows_from_entry(entry: dict, available: int, safety: Optional[float] = None) -> int:
    safety = entry.get('safety', DEFAULT_SAFETY) if safety is None else safety
    per_row = max(float(entry['bytes_per_row']), 1.0)
    fixed = max(float(entry.get('fixed_bytes', 0.0)), 0.0)
    mem_rows = int((available * safety - fixed) / per_row)
    rows = min(mem_rows, int(entry['plateau_rows']))
    rows = (rows // MIN_ROWS) * MIN_ROWS
    return max(rows, MIN_ROWS)


def resolve_batch_size(keys: Union[str, Sequence[Optional[str]], None], device, default: int) -> int:
    """Batch size for the first key that has a tuned entry, else ``default``."""
    if keys is None or not enabled() or not _is_cuda(device):
        return default
    if isinstance(keys, str):
        keys = [keys]
    prof = profile(device)
    if not prof:
        return default
    for key in keys:
        entry = prof['workloads'].get(key) if key else None
        if entry:
            return rows_from_entry(entry, available_bytes(device))
    return default


def is_oom(exc: BaseException) -> bool:
    oom_cls = getattr(torch.cuda, 'OutOfMemoryError', None)
    if oom_cls is not None and isinstance(exc, oom_cls):
        return True
    return isinstance(exc, RuntimeError) and 'out of memory' in str(exc).lower()


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #
def _measure(fn: Callable[[int], object], b: int, device, repeats: int):
    """(best wall time, peak extra bytes) for one call of fn(b)."""
    idx = _cuda_index(device)
    times = []
    torch.cuda.synchronize(idx)
    torch.cuda.reset_peak_memory_stats(idx)
    base = torch.cuda.memory_allocated(idx)
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(b)
        torch.cuda.synchronize(idx)
        times.append(time.perf_counter() - t0)
        del out
    peak = torch.cuda.max_memory_allocated(idx) - base
    return min(times), int(peak)


def _fit(points):
    """Least-squares peak = fixed + per_row * b over measured points."""
    b = np.array([p['rows'] for p in points], dtype=np.float64)
    m = np.array([p['peak_bytes'] for p in points], dtype=np.float64)
    if len(b) < 2 or np.ptp(b) == 0:
        return 0.0, float(m.max() / max(b.max(), 1))
    A = np.stack([np.ones_like(b), b], axis=1)
    (fixed, per_row), *_ = np.linalg.lstsq(A, m, rcond=None)
    if per_row <= 0:  # degenerate; fall back to a conservative ratio
        per_row, fixed = float(m.max() / b.max()), 0.0
    return max(float(fixed), 0.0), float(per_row)


def sweep(fn: Callable[[int], object], device, *, max_rows: int = 2 ** 20,
          start_rows: int = 2 ** 12, safety: float = DEFAULT_SAFETY,
          plateau_tol: float = DEFAULT_PLATEAU_TOL, repeats: int = 2,
          label: str = '', verbose: bool = True) -> dict:
    """Measure ``fn(b)`` at doubling batch sizes and build a memory-model entry.

    ``fn(b)`` must run exactly one GPU batch of ``b`` rows and return its output.
    The sweep never plans a batch above the predicted VRAM limit, stops on an
    OOM or on a throughput cliff, and records every point it measured.
    """
    say = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    with torch.no_grad():
        fn(min(MIN_ROWS, max_rows))  # warm-up: kernels, handles, allocator
    torch.cuda.empty_cache()

    points: List[dict] = []
    stop_reason = 'max_rows'
    b = start_rows
    while b <= max_rows:
        if len(points) >= 2:
            fixed, per_row = _fit(points)
            limit = (available_bytes(device) * safety - fixed) / per_row
            if b > limit:
                stop_reason = 'memory'
                break
        try:
            with torch.no_grad():
                t, peak = _measure(fn, b, device, repeats)
        except RuntimeError as exc:
            if not is_oom(exc):
                raise
            torch.cuda.empty_cache()
            stop_reason = 'oom'
            say(f"  [{label}] {b:>8d} rows: out of memory, stopping")
            break
        thr = b / t
        points.append({'rows': b, 'seconds': t, 'peak_bytes': peak, 'rows_per_s': thr})
        say(f"  [{label}] {b:>8d} rows: {t:8.3f} s  {thr:12.0f} rows/s  peak {peak / 2**20:8.1f} MiB")
        best = max(p['rows_per_s'] for p in points)
        if thr < CLIFF_FRACTION * best:
            stop_reason = 'cliff'
            break
        b *= 2

    if not points:
        raise RuntimeError(f"[{label}] no batch size could be measured")
    fixed, per_row = _fit(points)
    good = [p for p in points if p['rows_per_s'] >= CLIFF_FRACTION * max(q['rows_per_s'] for q in points)]
    best = max(p['rows_per_s'] for p in good)
    plateau = min(p['rows'] for p in good if p['rows_per_s'] >= (1 - plateau_tol) * best)
    return {
        'fixed_bytes': fixed,
        'bytes_per_row': per_row,
        'plateau_rows': int(plateau),
        'safety': safety,
        'max_rows_tested': int(max(p['rows'] for p in points)),
        'stop_reason': stop_reason,
        'points': points,
        'tuned_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
    }


# --------------------------------------------------------------------------- #
# Tuner
# --------------------------------------------------------------------------- #
def _condition_bounds(emulator, inner: float = 0.05):
    names = list(emulator.ml_indexer.featureNames)
    norm = emulator.ml_indexer.feature_normalizer
    lo = norm.miner.detach().cpu().numpy().astype(np.float64)
    span = norm.ranger.detach().cpu().numpy().astype(np.float64)
    return [(n, lo[i] + inner * span[i], lo[i] + (1 - inner) * span[i]) for i, n in enumerate(names)]


def _synthetic_features(api, emulator, n: int, composition: Optional[Dict[str, float]], seed: int):
    """Parsed (device, model-ordered) features: conditions uniform inside the
    training bounds, composition fixed. Goes through ``api.parse_input``."""
    if composition is None:
        if getattr(api, '_composition_space', None) != 'elements':
            raise ValueError(
                "tune_batch_size needs `composition=` (a {header: value} dict in this "
                "API's input convention) for non-HeFESTo models.")
        composition = _DEFAULT_ELEMENT_COMPOSITION
    rng = np.random.default_rng(seed)
    cond = _condition_bounds(emulator)
    cols, headers = [], []
    for name, lo, hi in cond:
        cols.append(rng.uniform(lo, hi, n).astype(np.float32))
        headers.append(name)
    for name, value in composition.items():
        cols.append(np.full(n, value, dtype=np.float32))
        headers.append(name)
    table = np.stack(cols, axis=1)
    features, _modeltype = api.parse_input(table, headers=headers)
    return features


def tune_batch_size(api, workloads: Sequence[str] = ('nn', 'eos_isomorphic'), *,
                    composition: Optional[Dict[str, float]] = None,
                    max_rows: Optional[Dict[str, int]] = None,
                    safety: float = DEFAULT_SAFETY,
                    plateau_tol: float = DEFAULT_PLATEAU_TOL,
                    repeats: int = 2, save_results: bool = True,
                    verbose: bool = True, seed: int = 0) -> Dict[str, dict]:
    """Tune and persist batch sizes for ``api`` on its CUDA device.

    Parameters
    ----------
    api : EmulatorAPI (e.g. HeFESToAPI) on a CUDA device.
    workloads : any of 'nn' (forwardMB of every loaded emulator in its default
        mass-balance mode, plus the get_T temperature FCNN), 'eos_isomorphic', 'eos_metamorphic', 'eos_fast'.
        EOS tiers need the HeFESTo EOS backend. Untuned EOS tiers keep using
        ``api._EOS_BATCH_SIZE``.
    composition : {header: value} bulk used for the synthetic inputs; defaults
        to a pyrolite-like elemental bulk for HeFESTo.
    max_rows : per-family caps, default {'nn': 2**20, 'eos': 2**19}.
    safety : fraction of available VRAM a batch may plan to use.
    save_results : write the profile to disk (see :func:`config_path`).

    Returns {workload_key: entry}.
    """
    device = api.device
    if not _is_cuda(device):
        raise RuntimeError("tune_batch_size needs an API on a CUDA device (device='cuda').")
    caps = {'nn': 2 ** 20, 'eos': 2 ** 19}
    caps.update(max_rows or {})
    unknown = set(workloads) - {'nn'} - {f'eos_{t}' for t in EOS_TIER_PROPERTIES}
    if unknown:
        raise ValueError(f"Unknown workloads {sorted(unknown)}")
    say = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    say(f"[nGibbs] Tuning batch sizes on {fingerprint(device)}")

    results: Dict[str, dict] = {}
    prof = profile(device, create=True)
    eos_tiers = [w.split('_', 1)[1] for w in workloads if w.startswith('eos_')]
    if eos_tiers and not hasattr(api, 'get_property_hefesto_vectorized_from_assemblage'):
        say("  EOS tiers skipped: this API has no HeFESTo EOS backend.")
        eos_tiers = []

    n_feats = max(caps['nn'] if 'nn' in workloads else 0, caps['eos'] if eos_tiers else 0)
    feats_by_kind = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # training-bound notices on synthetic rows
        for kind, attr, path_attr in _NN_KINDS:
            emulator = getattr(api, attr, None)
            if emulator is None:
                continue
            if kind != 'isothermal' and 'nn' not in workloads:
                continue
            feats_by_kind[kind] = _synthetic_features(api, emulator, n_feats, composition, seed)

        if 'nn' in workloads:
            for kind, attr, path_attr in _NN_KINDS:
                if kind not in feats_by_kind:
                    continue
                emulator = getattr(api, attr)
                feats = feats_by_kind[kind]
                outs = ['component_moles', 'phase_moles', 'chem_out']
                fn = lambda b, e=emulator, x=feats: e.forwardMB(x[:b], Normalize=True, outputs=list(outs))
                key = nn_key(kind, getattr(api, path_attr, None), 'forwardMB', emulator.mass_balance)
                say(f"- {key}")
                entry = sweep(fn, device, max_rows=min(caps['nn'], feats.shape[0]), safety=safety,
                              plateau_tol=plateau_tol, repeats=repeats, label=kind, verbose=verbose)
                prof['workloads'][key] = results[key] = entry
                say(f"  -> plateau {entry['plateau_rows']} rows, {entry['bytes_per_row']:.0f} B/row, "
                    f"stop: {entry['stop_reason']}")
                torch.cuda.empty_cache()

            # Temperature FCNN (get_T): fed [isentropic features, phase_moles,
            # chem_out] exactly as ForwardMB / get_isentrope build it.
            if getattr(api, 'temperature_model', None) is not None and 'isentropic' in feats_by_kind:
                feats = feats_by_kind['isentropic']
                out = api._staged_forward(api.isentropic_emulator.forwardMB, feats, None, Normalize=True,
                                          outputs=['phase_moles', 'chem_out'])
                t_in = torch.cat([feats, out['phase_moles'].to(device), out['chem_out'].to(device)], dim=1)
                del out
                fn = lambda b, x=t_in: api._get_T_unstaged(x[:b])
                key = api._temperature_workload_key()
                say(f"- {key}")
                entry = sweep(fn, device, max_rows=min(caps['nn'], t_in.shape[0]), safety=safety,
                              plateau_tol=plateau_tol, repeats=repeats, label='get_T', verbose=verbose)
                prof['workloads'][key] = results[key] = entry
                say(f"  -> plateau {entry['plateau_rows']} rows, {entry['bytes_per_row']:.0f} B/row, "
                    f"stop: {entry['stop_reason']}")
                del t_in
                torch.cuda.empty_cache()

        if eos_tiers:
            if 'isothermal' not in feats_by_kind:
                say("  EOS tiers skipped: needs the isothermal emulator for (P, T, moles) inputs.")
            else:
                emulator = api.isothermal_emulator
                feats = feats_by_kind['isothermal'][:caps['eos']]
                names = list(emulator.ml_indexer.featureNames)
                p_idx = names.index('P(GPa)(System_main)')
                t_idx = next((i for i, n in enumerate(names) if n.startswith('T(')), None)
                if t_idx is None:
                    raise ValueError(f"No temperature feature among isothermal conditions {names}")
                out = api._staged_forward(emulator.forwardMB, feats, None, Normalize=True,
                                          outputs=['component_moles'])
                cm = out['component_moles'].to(device=device, dtype=torch.float64)
                PT = torch.stack([feats[:, p_idx], feats[:, t_idx]], dim=1).to(torch.float64)
                del out
                backend = type(api).__name__
                for tier in eos_tiers:
                    props = EOS_TIER_PROPERTIES[tier]
                    fn = (lambda b, p=props:
                          api.get_property_hefesto_vectorized_from_assemblage(cm[:b], PT[:b], p))
                    key = eos_key(tier, backend)
                    say(f"- {key}")
                    entry = sweep(fn, device, max_rows=min(caps['eos'], cm.shape[0]), safety=safety,
                                  plateau_tol=plateau_tol, repeats=repeats, label=f"eos {tier}",
                                  verbose=verbose)
                    prof['workloads'][key] = results[key] = entry
                    say(f"  -> plateau {entry['plateau_rows']} rows, {entry['bytes_per_row']:.0f} B/row, "
                        f"stop: {entry['stop_reason']}")
                    torch.cuda.empty_cache()
                del cm, PT
    del feats_by_kind
    torch.cuda.empty_cache()

    if save_results:
        path = save()
        say(f"[nGibbs] Saved batch tuning profile to {path}")
    return results
