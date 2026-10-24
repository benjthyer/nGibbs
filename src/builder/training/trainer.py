"""Base trainer implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
import inspect
from typing import Dict, NamedTuple, Optional

import torch
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW, Adam
import time
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
import gc
import sys
from pathlib import Path

src_path = str(Path(__file__).parent.parent.parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from ngibbs.utils.string_utils import pull_number_range, _number_pattern as _DROPOUT_NUMBER
from builder.training.optimizer_factory import create_optimizer, create_scheduler, SchedulerWrapper
from builder.training.benchmark import ThroughputBenchmark
from builder.training.history import upper_terms
from ngibbs.config import constants
from ngibbs.utils.file_utils import load_ml_bundle
import ngibbs.engine.NN as NN
from ngibbs.engine.NN_continuous import (AFFINITY_LOSS_DEFAULTS, ACTIVATION_TRAIN_DEFAULTS,
                                         split_regularization)

# Mixed precision: constants.TRAIN_PRECISION selects the autocast dtype. See that
# constant's docstring in constants.py for the float32/bfloat16/float16 tradeoffs.
_PRECISION_DTYPES = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}

# Factor on the upper objective (see _upper_loss) keeping it large relative to Adam's eps.
# Training-history rows report the weighted loss terms at this same scale.
LOSS_SCALE = 1E4


def _resolve_precision(device: str):
    """(amp_dtype, use_amp), read fresh from constants.TRAIN_PRECISION on every call --
    not bound at import time -- so flipping the constant takes effect on the next
    training run without reloading this module. Mixed precision only ever runs on
    cuda: float16 autocast on CPU is not well supported, and bfloat16 on CPU gets none
    of the tensor-core speedup it exists for here, so a non-cuda device always trains
    in float32 regardless of the setting (with a one-line notice, not a silent no-op)."""
    name = getattr(constants, 'TRAIN_PRECISION', 'float32')
    if name not in _PRECISION_DTYPES:
        print(f"constants.TRAIN_PRECISION={name!r} is not one of {list(_PRECISION_DTYPES)} -- "
              f"continuing in float32. (Check for a typo, e.g. 'bfloat32' is not a real dtype.)")
        return torch.float32, False
    dtype = _PRECISION_DTYPES[name]
    use_amp = dtype != torch.float32 and device == 'cuda' and torch.cuda.is_available()
    if dtype != torch.float32 and not use_amp:
        print(f"constants.TRAIN_PRECISION={name!r} requested but device={device!r} -- "
              f"mixed precision only runs on cuda. Continuing in float32.")
        dtype = torch.float32
    return dtype, use_amp

# Set up temp models directory
TEMP_MODELS_DIR = Path(__file__).parent / "temp_models"
TEMP_MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _read_text_file(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError:
        return None


def _make_train_loader(trainData, batch_size, num_workers):
    """Build (or reuse) the training batch iterator.

    `trainData` is either a `torch.utils.data.Dataset` (the usual, fully
    in-RAM case - e.g. TensorDatasetFour from load_ML_data/load_ML_data_auto
    below the RAM threshold), in which case it's wrapped in a normal
    DataLoader exactly as before, or an already-iterable batch source (e.g.
    ChunkedMemmapTrainLoader from load_ML_data_auto above the RAM threshold -
    see builder.training.dataset_workspace), in which case it's used as-is.

    Episodes can each configure their own batch_size (see main.py's episode
    loop), but a ChunkedMemmapTrainLoader is constructed once, up front, by
    load_ML_data_auto - so its batch_size is kept in sync with whatever the
    *current* call configured, rather than whatever it happened to be built
    with.
    """
    if isinstance(trainData, torch.utils.data.Dataset):
        return DataLoader(trainData, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    if hasattr(trainData, 'batch_size'):
        trainData.batch_size = batch_size
    return trainData


# --------------------------------------------------------------------------- #
#  Fit checks (validation + adaptive dropout, several times per epoch)
# --------------------------------------------------------------------------- #
# The over/underfitting ratio used to compare the epoch's running train loss (train mode,
# so dropout and noise active, averaged over a model that improved all epoch) against an
# eval-mode test loss from the end-of-epoch model. Dropout alone inflated the train side
# enough to flip the verdict every epoch. A fit check instead evaluates BOTH sides now, in
# eval mode, on the same model snapshot: the validation set and a fixed random sample of
# the training set of similar size.

def _n_train_rows(trainData):
    if isinstance(trainData, torch.utils.data.Dataset):
        return len(trainData)
    return trainData.workspace.n_rows


def _fit_check_rows(trainData, max_N, checks_per_epoch):
    """Rows-seen thresholds at which the in-epoch fit checks fire. The final check of an
    epoch runs after the batch loop, so only the `checks_per_epoch - 1` inner ones are
    listed here."""
    rows = min(_n_train_rows(trainData), max_N)
    return [rows * k / checks_per_epoch for k in range(1, checks_per_epoch)]


def _train_reference_loader(trainData, n_rows, batch_size, seed=0):
    """A fixed sample of ~n_rows training rows drawn uniformly from the WHOLE training set,
    for the train side of the fit check.

    Drawn once per episode, not per check: the same rows every check means successive
    ratios differ because the model changed, not because the sample did. Uniform over
    the full set (not the most recently trained chunk) so it isn't biased toward rows the
    optimizer just saw. These rows are still trained on -- this measures fit to the
    training distribution, it is not a holdout."""
    n_total = _n_train_rows(trainData)
    rng = np.random.default_rng(seed)
    # Sampling with replacement + unique: ~n_rows distinct rows without allocating an
    # n_total-long permutation (80M rows -> 640 MB) the way choice(replace=False) would.
    idx = np.unique(rng.integers(0, n_total, size=min(int(n_rows), n_total)))
    t0 = time.time()
    if isinstance(trainData, torch.utils.data.Dataset):
        subset = torch.utils.data.Subset(trainData, idx.tolist())
    else:
        from builder.training.torchDataClass import TrainingTensorDataset
        ws = trainData.workspace
        subset = TrainingTensorDataset(**{
            name: torch.from_numpy(np.asarray(ws.arrays[name][idx])) for name in ws.array_names})
    print(f"[fit check] train reference: {len(idx):,} random rows of {n_total:,} "
          f"(built in {time.time()-t0:.1f}s)")
    return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


FIT_TRAIN_MODES = ('train', 'eval')


def _check_fit_train_mode(mode):
    """'train' (default, the original behavior): the train side of Test/Train is the running
    train-mode loss (dropout/noise active) over the batches since the previous check -- with
    one check per epoch, exactly the epoch's running train loss. 'eval': a fixed random
    training sample evaluated in eval mode on the current weights."""
    mode = str(mode).lower()
    if mode not in FIT_TRAIN_MODES:
        raise ValueError(f"fit_train_mode must be one of {FIT_TRAIN_MODES}, got {mode!r}")
    return mode


def _adapt_regularization(model, val_loss, ref_loss, dropout_rate, max_drop, noise,
                          dropout_step_up, dropout_step_down, noise_step_up, noise_step_down,
                          overfit_ratio, underfit_ratio):
    """One step of the adaptive dropout / noise controller. Noise reacts only when dropout
    is already at its ceiling (overfitting) or floor (underfitting). Ratios inside
    [underfit_ratio, overfit_ratio] are a dead band: no change. Returns (dropout_rate, noise)."""
    ratio = val_loss / max(ref_loss, 1e-30)
    if ratio > overfit_ratio:
        if min(dropout_rate + dropout_step_up, max_drop) > dropout_rate:
            old = dropout_rate
            dropout_rate = min(dropout_rate + dropout_step_up, max_drop)
            _set_adaptive_dropout_rate(model, dropout_rate)
            print(f"Overfitting (val/train {ratio:.3f} > {overfit_ratio}). Increasing Dropout: {old} -> {dropout_rate}")
        else:
            old = noise
            noise = noise + noise_step_up
            print(f"Overfitting (val/train {ratio:.3f}), but dropout_rate is at the maximum: {dropout_rate}. "
                  f"Increasing noise instead: {old} -> {noise}")
    elif ratio < underfit_ratio:
        if max(dropout_rate - dropout_step_down, 0) < dropout_rate:
            old = dropout_rate
            dropout_rate = max(dropout_rate - dropout_step_down, 0)
            _set_adaptive_dropout_rate(model, dropout_rate)
            print(f"Underfitting (val/train {ratio:.3f} < {underfit_ratio}). Decreasing Dropout: {old} -> {dropout_rate}")
        else:
            old = noise
            noise = max(noise - noise_step_down, 0)
            print(f"Underfitting (val/train {ratio:.3f}), but dropout_rate is at the minimum: {dropout_rate}. "
                  f"Decreasing noise instead: {old} -> {noise}")
    else:
        print(f"val/train {ratio:.3f} within [{underfit_ratio}, {overfit_ratio}]: dropout stays {dropout_rate}")
    return dropout_rate, noise


def _reduce(loss, reduction):
    if reduction == 'none':
        return loss
    if reduction == 'mean':
        return torch.mean(loss)
    if reduction == 'sum':
        return torch.sum(loss)
    raise ValueError(f"reduction must be 'none', 'mean' or 'sum', got {reduction!r}")


# Elementwise by default. `_upper_loss` applies presence masks and per-phase/component
# weights AFTER the criterion, so the criterion must hand back one value per entry: a
# mean-reduced scalar makes `(scalar * mask * w).sum() / (mask * w).sum()` collapse back
# to the scalar, i.e. masks and weights silently do nothing (which is what happened
# before this default changed). `_upper_loss` now checks the shape and raises.
def symmetric_rel_l1(pred, target, eps=1e-6, reduction='none'):
    denom = torch.clamp(torch.abs(pred) + torch.abs(target), min=eps)
    return _reduce(torch.abs(pred - target) / denom, reduction)

def symmetric_rel_l2(pred, target, eps=1e-6, reduction='none'):
    denom = torch.clamp(torch.abs(pred) + torch.abs(target), min=eps)
    return _reduce((pred - target)**2 / denom, reduction)


def _elementwise(criterion, pred, target, name):
    """Call `criterion` and insist on an elementwise result. A scalar here means the
    masks/weights in `_upper_loss` would cancel out -- fail loudly instead."""
    raw = criterion(pred, target)
    if raw.shape != pred.shape:
        raise ValueError(
            f"{name} criterion returned shape {tuple(raw.shape)} for predictions of shape "
            f"{tuple(pred.shape)}. _upper_loss needs an elementwise (reduction='none') "
            f"criterion so its presence masks and phase/component weights take effect.")
    return raw


def _iter_adaptive_dropout_modules(model: nn.Module):
    # MidLevelNetwork keeps its mole head out of adaptive dropout; a model that shares
    # one rate across every part (ContinuousModel) declares an empty exclusion list.
    excluded_ids = set()
    for attr_name in getattr(model, 'adaptive_dropout_exclude', ('mole_head',)):
        if hasattr(model, attr_name):
            head_module = getattr(model, attr_name)
            excluded_ids.update(id(module) for module in head_module.modules())

    for module in model.modules():
        if isinstance(module, nn.Dropout) and id(module) not in excluded_ids:
            yield module


def _set_adaptive_dropout_rate(model: nn.Module, dropout_rate: float) -> None:
    for module in _iter_adaptive_dropout_modules(model):
        module.p = dropout_rate


def _describe_adaptive_dropout(model: nn.Module) -> str:
    """'138 layers at p=0.01 (encoder 6, mole_head 132)': where the adapted Dropout
    modules sit, by top-level submodule, and their current rate(s)."""
    counts, rates = {}, set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout) and any(module is m for m in _iter_adaptive_dropout_modules(model)):
            top = name.split('.')[0]
            counts[top] = counts.get(top, 0) + 1
            rates.add(round(float(module.p), 6))
    if not counts:
        return "no adaptive dropout layers"
    where = ", ".join(f"{k} {v}" for k, v in counts.items())
    return f"{sum(counts.values())} layers at p={sorted(rates)} ({where})"


def _with_initial_rate(spec, dropout_rate):
    """'0.0dropout0.3' at rate 0.004 -> '0.004dropout0.3' (3 d.p.; maximum kept as
    written). A spec without dropout is returned unchanged."""
    if not isinstance(spec, str) or 'dropout' not in spec.lower():
        return spec
    rate = f"{float(dropout_rate):.3f}"
    if len(_DROPOUT_NUMBER.findall(spec)) >= 2:
        return _DROPOUT_NUMBER.sub(rate, spec, count=1)  # replace the leading initial value
    return rate + spec  # 'dropout0.3' (initial implicitly 0) -> '0.004dropout0.3'


def _record_dropout_state(model, which, dropout_rate, noise, train_params, carry_state):
    """At a new best checkpoint, remember the live adaptive dropout rate and noise so the
    next episode resumes from them. They go into `carry_state`, which main.py folds into
    the recipe config -- never into the model, so a warm start cannot load them. (A
    MidLevelNetwork, whose dropout spec is still part of its config, also has that spec
    rewritten, as before.)"""
    if getattr(model, 'dropout_is_training_setting', False):
        spec = _with_initial_rate(train_params['dropout'], dropout_rate)
    else:
        key = _dropout_config_key(model, which)
        spec = _with_initial_rate(model.config.get(key), dropout_rate)
        if spec is not None:
            model.config[key] = spec
    if carry_state is not None:
        if getattr(model, 'dropout_is_training_setting', False):
            carry_state['dropout'] = spec
        carry_state['noise'] = float(noise)



# --------------------------------------------------------------------------- #
#  Model adapters
# --------------------------------------------------------------------------- #
# The upper loop below drives an arbitrary network through the three hooks in this
# section. A model opts into the general path by defining `upper_forward`; a model that
# does not (i.e. MidLevelNetwork) keeps the exact legacy call, unpack and masks, so its
# behaviour is unchanged. Nothing here inspects a class name, so a new architecture is
# supported by adding methods to that class, not by editing this file.


class UpperBatch(NamedTuple):
    """One forward pass, named. `logits=None` means the architecture has no saturation
    head, and the saturation term is then dropped from the loss rather than zeroed with
    a fake tensor -- a zero BCE would read as 'perfectly classified' in the printout."""
    logits: Optional[torch.Tensor]
    chem: torch.Tensor
    chem_mask: torch.Tensor
    mole: torch.Tensor
    bulk: torch.Tensor
    mole_mask: Optional[torch.Tensor]   # None -> derive from ground-truth binaries
    sat_weight: Optional[torch.Tensor] = None  # None -> uniform (see ContinuousModel.upper_forward's `T`)
    g_phi: Optional[torch.Tensor] = None  # None for a model with no such concept (e.g. MidLevelNetwork)


def _upper_forward(model, x_batch, b_batch, T=None, act_params=None) -> UpperBatch:
    fn = getattr(model, 'upper_forward', None)
    if fn is None:
        logits, chem, chem_mask, mole, bulk = model(x_batch, binaries=b_batch, NN_only=True)
        return UpperBatch(logits, chem, chem_mask, mole, bulk, None)

    # T (annealed complementarity-smoothing temperature) is a ContinuousModel-specific
    # concept; only pass it to models that actually declare support for it, so a plain
    # MidLevelNetwork's upper_forward (if one is ever added) isn't forced to accept an
    # argument it has no use for.
    kwargs = {'binaries': b_batch}
    params = inspect.signature(fn).parameters
    if T is not None and 'T' in params:
        kwargs['T'] = T
    if act_params is not None and 'act_params' in params:
        kwargs['act_params'] = act_params
    out = fn(x_batch, **kwargs)
    missing = {'chem', 'chem_mask', 'mole', 'bulk'} - set(out)
    if missing:
        raise KeyError(f"{type(model).__name__}.upper_forward() omitted {sorted(missing)}")
    return UpperBatch(out.get('logits'), out['chem'], out['chem_mask'],
                      out['mole'], out['bulk'], out.get('mole_mask'), out.get('sat_weight'),
                      out.get('g_phi'))


def make_horizontal_bar(percent, max_width=50):
    """Create horizontal bar: '19.75% ||||||||||||'"""
    n_bars = int(round(percent / 100.0 * max_width))
    return '|' * n_bars


def _print_histogram(values, title, n_bins=20, max_width=40):
    """Percentile-clipped (2nd-98th) histogram of a flat 1D array, one row per bin,
    rendered with `make_horizontal_bar`. A diagnostic, not a correctness check -- an
    empty/degenerate input prints a note rather than raising."""
    values = values.detach().cpu().numpy() if torch.is_tensor(values) else np.asarray(values)
    values = values[np.isfinite(values)]
    print(f"\n[{title}] n={values.size:,}")
    if values.size == 0:
        print("  (no data)")
        return
    lo, hi = np.percentile(values, [2, 98])
    if hi <= lo:
        print(f"  (degenerate range: [{lo:.3g}, {hi:.3g}])")
        return
    edges = np.linspace(lo, hi, n_bins + 1)
    counts, _ = np.histogram(values, bins=edges)
    total = counts.sum()
    for i in range(n_bins):
        pct = 100.0 * counts[i] / total if total else 0.0
        bar = make_horizontal_bar(pct, max_width=max_width)
        print(f"  [{edges[i]:>10.3g}, {edges[i+1]:>10.3g})  {pct:5.2f}% {bar}")


def _anneal_a(epoch, epochs, a_start=1.0, a_end=1e-4):
    """Linear schedule for the complementarity-smoothing scale factor `a` (`T = a*T0`),
    from `a_start` at epoch 0 to `a_end` at this episode's last epoch. Per-episode, not
    global across the whole main.py run: each episode that enables `boundary_temperature`
    starts its own schedule fresh, the same way every other per-episode config (epochs,
    scheduler, loss weights) already works."""
    if epochs <= 1:
        return float(a_end)
    frac = min(max(epoch, 0), epochs - 1) / (epochs - 1)
    return float(a_start + (a_end - a_start) * frac)


BOUNDARY_TEMPERATURE_DEFAULTS = {'a_start': 1.0, 'a_end': 1e-4}


def _required_boundary_temperature(model, boundary_temperature):
    """A model that declares `requires_boundary_temperature` (ContinuousModel) has no
    un-annealed training path: when the recipe gives no schedule (block absent or
    `enabled: false`), use the default one rather than fail or silently train another
    way."""
    if boundary_temperature or not getattr(model, 'requires_boundary_temperature', False):
        return boundary_temperature
    print(f"[boundary_temperature] {type(model).__name__} always trains with annealed boundary "
          f"temperature; the recipe gives no schedule, so using the default "
          f"{BOUNDARY_TEMPERATURE_DEFAULTS}.")
    return dict(BOUNDARY_TEMPERATURE_DEFAULTS)


def _boundary_T(ml_indexer, a, device, dtype=torch.float32):
    """Build this epoch's `(1, P)` annealed temperature tensor `T = a*T0`."""
    T0 = getattr(ml_indexer, 'T0', None)
    if T0 is None:
        raise ValueError(
            "boundary temperature needs ml_indexer.T0, which is missing. main.py "
            "computes it from the validation set when the bundles lack it; otherwise "
            "backfill with scripts/compute_T0.py.")
    T0 = np.asarray(T0, dtype=np.float64)
    if (T0 == 0).any(): raise ValueError("The training dataset contains an empty column!")
    T = a * T0
    return torch.as_tensor(T, dtype=dtype, device=device).reshape(1, -1)


def _mole_targets(model, m_batch):
    """Datasets store moles in MidLevelNetwork's output space (log10(n + molar_epsilon)).
    A model that predicts something else declares the conversion instead of the dataset
    being rebuilt per architecture."""
    fn = getattr(model, 'transform_mole_targets', None)
    return m_batch if fn is None else fn(m_batch)


def _affinity_mask_from_batch(model, batch, device):
    """Affinity models (see NN_continuous.AffinityModel) train against batches of
    (features, binary_labels, labels, affinity_y, affinity_mask); everything else
    returns None here and keeps the historical 4-tuple semantics."""
    if getattr(model, 'affinity_mole_loss', None) is None:
        return None
    if len(batch) < 5:
        raise ValueError(
            f"{type(model).__name__} trains on affinity labels, but this dataset yields "
            f"{len(batch)}-tuples with no trust mask. Set `affinity: {{enabled: true}}` in "
            f"the recipe so the loaders build affinity_y / affinity_mask.")
    return batch[4].to(device, non_blocking=True)


def _dropout_config_key(model, which='upper'):
    """MidLevelNetwork only: the config key whose combined spec holds its dropout."""
    key = getattr(model, f'{which}_regularization_config_key', None)
    if key is None:
        key = 'high_regularization' if which == 'upper' else 'low_regularization'
    return key


def _regularization_spec(model, which='upper', train_params=None):
    """The dropout spec for this half of training: the episode's `dropout` training
    setting for a model that declares `dropout_is_training_setting`, else
    MidLevelNetwork's combined config spec."""
    if getattr(model, 'dropout_is_training_setting', False):
        return str((train_params or {}).get('dropout', 'none'))
    return str(model.config.get(_dropout_config_key(model, which), 'none'))


def _resolve_heads_to_freeze(model, names):
    """Map requested head names onto this model's modules.

    A model declares `head_aliases` to rename its equivalents, or maps a name to None to
    say 'this architecture has no such head, by design' -- e.g. a continuous-saturation
    model has no `sat_head` because saturation is not a separate output. A name that is
    neither present nor declared absent still raises, so a typo in a recipe is still an
    error rather than a silently unfrozen head.
    """
    aliases = getattr(model, 'head_aliases', {})
    resolved = []
    for name in names:
        target = aliases.get(name, name)
        if target is None:
            print(f"Head '{name}' is absent by design in {type(model).__name__}; nothing to freeze.")
            continue
        if not hasattr(model, target):
            raise ValueError(f"{type(model).__name__} has no head named '{target}'"
                             f"{'' if target == name else f' (requested as {name!r})'} to freeze.")
        resolved.append(target)
    return resolved


# --------------------------------------------------------------------------- #
#  Training settings
# --------------------------------------------------------------------------- #
# How a model is trained, as opposed to what it is: these come from the episode config
# (main.py passes them as `train_params`) and are never saved with the model, so a
# warm-start checkpoint cannot override the recipe's values. The affinity loss weights
# (AFFINITY_LOSS_DEFAULTS) and training-only output activations
# (ACTIVATION_TRAIN_DEFAULTS) are included; a model ignores the ones it has no use for.
# `dropout` is the shared '<initial>dropout<max>' spec ('none' = no adaptive dropout).
# ContinuousModel's absent-side margin hinge (see `_absent_hinge_loss`); weight 0 = off.
ABSENT_HINGE_DEFAULTS = {'absent_hinge_weight': 0.0, 'absent_hinge_margin': 1.0}
TRAIN_PARAM_DEFAULTS = {'lowWD': 0.0, 'highWD': 0.0, 'noise': 0.0, 'dropout': 'none',
                        **ACTIVATION_TRAIN_DEFAULTS, **AFFINITY_LOSS_DEFAULTS,
                        **ABSENT_HINGE_DEFAULTS}


def _resolve_train_params(model, train_params=None):
    """Each setting from `train_params` if given there, else from the model's config (a
    legacy MidLevelNetwork still records lowWD/highWD/noise), else the default."""
    train_params = train_params or {}
    unknown = set(train_params) - set(TRAIN_PARAM_DEFAULTS)
    if unknown:
        raise KeyError(f"Unknown training settings {sorted(unknown)}; "
                       f"known: {sorted(TRAIN_PARAM_DEFAULTS)}")
    resolved = {}
    for key, default in TRAIN_PARAM_DEFAULTS.items():
        value = train_params.get(key)
        if value is None:
            value = model.config.get(key, default)
        resolved[key] = (split_regularization(dropout=value)[2] if key == 'dropout'
                         else float(value))
    print(f"Training settings: {resolved}")
    return resolved


def _affinity_params(train_params):
    return {k: train_params[k] for k in AFFINITY_LOSS_DEFAULTS}


def _activation_params(train_params):
    return {k: train_params[k] for k in ACTIVATION_TRAIN_DEFAULTS}


def _hinge_params(train_params):
    return {k: train_params[k] for k in ABSENT_HINGE_DEFAULTS}


def _absent_hinge_loss(model, out: UpperBatch, b_batch, hinge_params):
    """Mean over GT-absent cells of `relu(g_phi/T0 + m)^2`, or None when the weight is 0.

    The BCE on `5*g_phi/T0` is two-sided: besides pushing absent phases below the boundary
    it pulls present trace phases (t < T0) up past it, inflating their abundance. This
    acts on GT-absent cells only, so it opens a margin of `m*T0` on the side the false
    positives come from without biasing present phases. Dividing by T0 makes `m` mean
    the same thing for every phase, as the BCE logit's scaling does."""
    if not hinge_params or not float(hinge_params['absent_hinge_weight']):
        return None
    if out.g_phi is None or getattr(model, 'affinity_mole_loss', None) is not None:
        raise ValueError(f"absent_hinge_weight > 0 needs a ContinuousModel's raw g_phi; "
                         f"{type(model).__name__} does not provide one (AffinityModel has "
                         f"its own hinge).")
    T0 = getattr(model.ml_indexer, 'T0', None)
    if T0 is None:
        raise ValueError("absent_hinge_weight > 0 needs ml_indexer.T0 (see compute_T0).")
    # float32 for the same reason as upper_forward: T0 is as small as ~1e-5.
    with torch.autocast(device_type=out.g_phi.device.type, enabled=False):
        T0_t = torch.as_tensor(T0, dtype=torch.float32,
                               device=out.g_phi.device).reshape(1, -1).clamp(min=1e-30)
        absent = (b_batch <= 0.5).to(torch.float32)
        u = out.g_phi.float() / T0_t
        h = torch.relu(u + float(hinge_params['absent_hinge_margin'])) ** 2
        return (h * absent).sum() / absent.sum().clamp(min=1.0)


def _upper_loss(model, out: UpperBatch, x_batch, b_batch, y_batch, m_batch, feature_offset,
                criterion_sat, criterion_chem, criterion_mole, criterion_bulk,
                compWeights, binWeights, sat_alpha, chem_alpha, mole_alpha, bulk_alpha,
                mask_batch=None, affinity_params=None, hinge_params=None):
    """Single definition of the upper objective, shared by the training step and the
    evaluation pass. These were two copies of the same twenty lines; a change to one that
    missed the other would silently score models against a different loss than it trained
    them on.

    `binWeights` (a recipe's `binweights:` block -- upweighting rare phases the mole
    regression under-serves) now also scales the binary/saturation term via
    `_weighted_binary_loss_gt_positive_only`, not just `mole_loss` below: `criterion_sat`
    is no longer called directly (it was always `nn.BCEWithLogitsLoss()` in practice --
    no call site overrides it -- and that helper already IS that loss, phase-weighted).
    Weighting only the GT-positive term, not GT-negative, is deliberate and matches how
    `_evaluate_binary_model` already weights the lower-stage sat_head loss elsewhere: a
    rare phase's positive examples are the ones actually starved of gradient, and
    upweighting its (already numerous) negatives too would just rescale the loss without
    changing what the network is pushed toward."""
    loss_sat = (_weighted_binary_loss_gt_positive_only(out.logits, b_batch, binWeights,
                                                       extra_weight=out.sat_weight)
                if out.logits is not None
                else torch.zeros((), device=x_batch.device, dtype=x_batch.dtype))

    bulk_target = x_batch[:, feature_offset:]
    chem_loss_raw = _elementwise(criterion_chem, out.chem, y_batch, 'chemistry')
    affinity_loss = getattr(model, 'affinity_mole_loss', None)
    if affinity_loss is None:
        mole_loss_raw = _elementwise(criterion_mole, out.mole, _mole_targets(model, m_batch), 'moles')
    bulk_loss_raw = _elementwise(criterion_bulk, out.bulk, bulk_target, 'bulk')

    bulk_zero_mask = (bulk_target != 0).to(torch.float)
    if affinity_loss is None:
        mole_zero_mask = (out.mole_mask if out.mole_mask is not None
                          else (b_batch > 0.5).to(torch.float)).detach()

    chem_loss = (chem_loss_raw * out.chem_mask * compWeights).sum() / (out.chem_mask * compWeights).sum().clamp(min=1)
    if affinity_loss is None:
        mole_loss = (mole_loss_raw * mole_zero_mask * binWeights).sum() / (mole_zero_mask * binWeights).sum().clamp(min=1)
    else:
        # Signed affinity labels: every cell carries a target (trusted) or a one-sided
        # bound (untrusted), so there is no present-only mask here -- see AffinityModel.
        mole_loss = affinity_loss(out.mole, m_batch, mask_batch, binWeights,
                                  params=affinity_params)
    bulk_loss = (bulk_loss_raw * bulk_zero_mask).sum() / bulk_zero_mask.sum().clamp(min=1)

    # Scale loss to be large wrt Epsilon for optimizer stability.
    total = LOSS_SCALE * (sat_alpha * loss_sat + chem_alpha * chem_loss + mole_alpha * mole_loss)
    if bulk_alpha != 0:   # Bulk can be numerically unstable, so it stays opt-in.
        total = total + LOSS_SCALE * bulk_alpha * bulk_loss
    hinge_loss = _absent_hinge_loss(model, out, b_batch, hinge_params)
    if hinge_loss is not None:
        total = total + LOSS_SCALE * float(hinge_params['absent_hinge_weight']) * hinge_loss
    return total, loss_sat, chem_loss, mole_loss, bulk_loss


def _weighted_binary_loss_gt_positive_only(
    logits: torch.Tensor,
    targets: torch.Tensor,
    bin_weights: torch.Tensor,
    extra_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply phase weights only where GT phase is present; GT-absent terms stay unweighted.

    `extra_weight`, when given, multiplies in on top of `bin_weights` for every entry
    regardless of GT sign -- unlike `bin_weights` it isn't a static per-phase constant,
    it's per-(sample, phase) (e.g. ContinuousModel's `|tanh(g_phi/T)|` confidence
    weight, see `upper_forward`). The weighted-mean normalisation (divide by the sum of
    weights, not the raw count) means a down-weighted entry shrinks its own
    contribution rather than shrinking the whole batch's effective denominator -- so a
    boundary-adjacent sample that's mostly excluded doesn't silently make the reported
    loss look smaller than the decisive samples actually warrant.
    """
    loss_raw = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    effective_weights = targets * bin_weights + (1.0 - targets)
    if extra_weight is not None:
        effective_weights = effective_weights * extra_weight
    return (loss_raw * effective_weights).sum() / effective_weights.sum().clamp(min=1.0)


def _evaluate_binary_model(model, test_loader, binWeights, device, max_N=np.inf):
    model.eval()
    running_test_loss = 0.0
    N = 0
    tp = torch.zeros(binWeights.shape[1], dtype=torch.float64, device=device)
    fp = torch.zeros(binWeights.shape[1], dtype=torch.float64, device=device)
    fn = torch.zeros(binWeights.shape[1], dtype=torch.float64, device=device)

    with torch.no_grad():
        for output in test_loader:
            xb, yb = output[0], output[1]
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            logits = model.forward_binaries(xb)
            loss = _weighted_binary_loss_gt_positive_only(logits, yb, binWeights)

            preds = torch.sigmoid(logits) > 0.5
            truth = yb > 0.5
            tp += (preds & truth).sum(dim=0).to(torch.float64)
            fp += (preds & (~truth)).sum(dim=0).to(torch.float64)
            fn += ((~preds) & truth).sum(dim=0).to(torch.float64)

            running_test_loss += loss.item() * xb.size(0)
            N += xb.size(0)
            if N >= max_N:
                print(f"Reached max_N={max_N} samples for this epoch. Stopping early.")
                break

    avg_test_loss = running_test_loss / N
    precision = tp / (tp + fp).clamp(min=1.0)
    recall = tp / (tp + fn).clamp(min=1.0)
    return avg_test_loss, precision, recall


def _accumulate_affinity_diagnostics(acc, model, y_hat, y, mask):
    """Per-phase counts for the affinity sign call and scaled error. Everything is in
    units of c, so phases with c spanning orders of magnitude read on one scale."""
    c = model.aff_c.float()
    u, t = y_hat.float() / c, y.float() / c
    mask = mask.to(torch.bool)
    gt_pres, pr_pres = t > 0, u > 0
    new = {
        'present': (mask & gt_pres).sum(0),
        'missed': (mask & gt_pres & ~pr_pres).sum(0),          # present, predicted absent
        'trusted_absent': (mask & ~gt_pres).sum(0),
        'false_pres': (mask & ~gt_pres & pr_pres).sum(0),       # absent, predicted present
        'untrusted': (~mask).sum(0),
        'untrusted_pres': (~mask & pr_pres).sum(0),
        'sq_trusted': torch.where(mask, (u - t) ** 2, torch.zeros_like(u)).sum(0),
    }
    new = {k: v.detach().double().cpu() for k, v in new.items()}
    if acc is None:
        return new
    return {k: acc[k] + new[k] for k in acc}


def _print_affinity_diagnostics(acc, model):
    names = list(model.label_indices.keys())
    tot = {k: float(v.sum()) for k, v in acc.items()}
    n_tr = tot['present'] + tot['trusted_absent']
    print(f"\n[affinity] trusted RMSE(y/c) = {np.sqrt(tot['sq_trusted'] / max(n_tr, 1)):.3f} | "
          f"present->absent {100 * tot['missed'] / max(tot['present'], 1):.2f}% | "
          f"trusted absent->present {100 * tot['false_pres'] / max(tot['trusted_absent'], 1):.2f}% | "
          f"untrusted->present {100 * tot['untrusted_pres'] / max(tot['untrusted'], 1):.2f}%")
    # Precision/recall of the sign call over ALL cells, comparable to the binary model's
    # phasewise printout. Untrusted cells are always GT-absent, so they count toward fp.
    print(f"  {'phase':<20}{'prec':>7}{'recall':>8}{'RMSE y/c':>10}{'miss%':>8}{'falseP%':>9}{'untrP%':>8}{'n_pres':>10}")
    for j, n in enumerate(names):
        ntr = float(acc['present'][j] + acc['trusted_absent'][j])
        if ntr == 0 and float(acc['untrusted'][j]) == 0:
            continue
        rm = np.sqrt(float(acc['sq_trusted'][j]) / max(ntr, 1))
        miss = 100 * float(acc['missed'][j]) / max(float(acc['present'][j]), 1)
        fp = 100 * float(acc['false_pres'][j]) / max(float(acc['trusted_absent'][j]), 1)
        up = 100 * float(acc['untrusted_pres'][j]) / max(float(acc['untrusted'][j]), 1)
        tp = float(acc['present'][j] - acc['missed'][j])
        prec = tp / max(tp + float(acc['false_pres'][j] + acc['untrusted_pres'][j]), 1)
        rec = tp / max(float(acc['present'][j]), 1)
        print(f"  {n:<20}{prec:>7.3f}{rec:>8.3f}{rm:>10.3f}{miss:>8.2f}{fp:>9.2f}{up:>8.2f}{int(acc['present'][j]):>10d}")


def _accumulate_presence_counts(acc, g_phi, b_batch):
    """Per-phase (tp, fp, fn) of the presence call `g_phi > 0` -- the same call
    ContinuousModel makes at inference (`n_phi = clamp(leaky_relu(g_phi), 0) > 0`; leaky_relu
    preserves sign) -- against the ground-truth binaries. Returns a (3, P) float64 tensor."""
    pred, truth = g_phi > 0, b_batch > 0.5
    new = torch.stack([(pred & truth).sum(0), (pred & ~truth).sum(0),
                       (~pred & truth).sum(0)]).double().cpu()
    return new if acc is None else acc + new


def abundance_f1_weights(abundance):
    """Per-phase weight for the pooled presence F1: `0.3 + 0.7*(a - min a)/(max a - min a)`,
    `a` = mean modal wt% (zeros included). Every phase counts; the most abundant counts
    1, the rarest 0.3."""
    a = np.asarray(abundance, dtype=np.float64)
    span = a.max() - a.min()
    return 0.3 + 0.7 * ((a - a.min()) / span if span > 0 else np.zeros_like(a))


def weighted_pooled_f1(tp, fp, fn, w):
    """F1 of the weight-pooled counts: 2*sum(w*TP) / (2*sum(w*TP) + sum(w*FP) + sum(w*FN))."""
    tp, fp, fn = (np.asarray(v, dtype=np.float64) * w for v in (tp, fp, fn))
    return 2 * tp.sum() / max(2 * tp.sum() + fp.sum() + fn.sum(), 1e-12)


def mean_modal_wtpct(bundle_path, n_phases=None):
    """Mean modal wt% per phase over a bundle's mass_labels (zeros included), or None when
    the bundle has none. The input to `abundance_f1_weights`."""
    mass = load_ml_bundle(bundle_path, arrays=['mass_labels']).mass_labels
    if mass is None:
        return None
    mass = np.asarray(mass, dtype=np.float64)
    return mass[:, :n_phases].mean(0) if n_phases else mass.mean(0)


def _print_presence_precision_recall(acc, model):
    tp, fp, fn = (v.numpy() for v in acc)
    names = list(model.label_indices.keys())
    T, F_p, F_n = float(tp.sum()), float(fp.sum()), float(fn.sum())
    # Set by main.py from the validation bundle; absent -> no weighted F1.
    abundance = getattr(getattr(model, 'ml_indexer', None), 'presence_abundance_wtpct', None)
    w = abundance_f1_weights(abundance) if abundance is not None else None
    f1w = f" | abundance-weighted F1 {weighted_pooled_f1(tp, fp, fn, w):.4f}" if w is not None else ''
    print(f"\n[presence g_phi > 0] overall precision {T / max(T + F_p, 1):.4f} | "
          f"recall {T / max(T + F_n, 1):.4f}{f1w}")
    print(f"  {'phase':<20}{'prec':>8}{'recall':>8}{'F1':>8}{'w':>6}{'n_pres':>12}")
    for j, n in enumerate(names):
        prec = float(tp[j]) / max(float(tp[j] + fp[j]), 1)
        rec = float(tp[j]) / max(float(tp[j] + fn[j]), 1)
        f1 = 2 * float(tp[j]) / max(float(2 * tp[j] + fp[j] + fn[j]), 1)
        wj = f"{w[j]:>6.2f}" if w is not None else f"{'-':>6}"
        print(f"  {n:<20}{prec:>8.3f}{rec:>8.3f}{f1:>8.3f}{wj}{int(tp[j] + fn[j]):>12d}")


def _evaluate_upper_model(model, test_loader, feature_offset, criterion_sat, criterion_chem, criterion_mole,
                           criterion_bulk, compWeights, binWeights, sat_alpha, chem_alpha, mole_alpha, bulk_alpha,
                           device, max_N=np.inf, T=None, amp_dtype=torch.float32, use_amp=False,
                           verbose=True, affinity_params=None, act_params=None, hinge_params=None):
    """`verbose=False` skips the per-term prints and the affinity/boundary diagnostics
    (and the work of collecting them) -- used for in-epoch fit checks and the train
    reference, where only the scalar loss is needed."""
    model.eval()
    running_test_loss = 0.0
    running_sat_loss = 0
    running_chem_loss = 0
    running_mole_loss = 0
    running_bulk_loss = 0
    N = 0
    out = None
    g_phi_chunks = []
    residual_chunks = []
    aff_diag = None
    presence = None
    T0 = getattr(getattr(model, 'ml_indexer', None), 'T0', None)
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # testData may carry trailing derivative arrays (dn/dP, dn/dT) when some other
            # episode in the recipe wants them -- main.py loads the dataset once for every
            # episode, so a non-derivative trainer like this one must ignore those extras
            # rather than assume a fixed 4-tuple.
            x_batch, b_batch, y_batch, m_batch = batch[0], batch[1], batch[2], batch[3]
            x_batch, b_batch, y_batch, m_batch = x_batch.to(device, non_blocking=True), b_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True), m_batch.to(device, non_blocking=True)
            mask_batch = _affinity_mask_from_batch(model, batch, device)
            with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                out = _upper_forward(model, x_batch, b_batch, T=T, act_params=act_params)

                loss, loss_sat, chem_loss_masked, mole_loss_masked, bulk_loss_masked = _upper_loss(
                    model, out, x_batch, b_batch, y_batch, m_batch, feature_offset,
                    criterion_sat, criterion_chem, criterion_mole, criterion_bulk,
                    compWeights, binWeights, sat_alpha, chem_alpha, mole_alpha, bulk_alpha,
                    mask_batch=mask_batch, affinity_params=affinity_params,
                    hinge_params=hinge_params,
                )
            if verbose and mask_batch is not None:
                aff_diag = _accumulate_affinity_diagnostics(aff_diag, model, out.mole, m_batch, mask_batch)
            elif verbose and out.g_phi is not None:
                presence = _accumulate_presence_counts(presence, out.g_phi, b_batch)

            batch_size_curr = x_batch.size(0)
            running_sat_loss += loss_sat.item() * batch_size_curr
            running_mole_loss += mole_loss_masked.item() * batch_size_curr
            running_chem_loss += chem_loss_masked.item() * batch_size_curr
            running_bulk_loss += bulk_loss_masked.item() * batch_size_curr

            running_test_loss += loss.item() * batch_size_curr
            N += batch_size_curr

            if verbose and out.g_phi is not None:
                # Normalize by T0 (per phase) before pooling across phases -- T0 spans
                # ~3 orders of magnitude across phases in a typical dataset (e.g. iron
                # polymorphs ~1e-5 vs. a dominant phase ~0.06), so a raw, un-normalized
                # g_phi pooled across all of them answers an ill-posed question: the
                # same absolute g_phi is deeply decisive for a small-T0 phase and barely
                # past the boundary for a large-T0 one. g_phi/T0 puts every phase on the
                # same boundary-relative scale, matching how this is actually read.
                if T0 is not None:
                    T0_norm = torch.as_tensor(T0, dtype=out.g_phi.dtype,
                                              device=out.g_phi.device).reshape(1, -1).clamp(min=1e-30)
                    g_phi_chunks.append((out.g_phi / T0_norm).detach().to('cpu', torch.float32))
                else:
                    g_phi_chunks.append(out.g_phi.detach().to('cpu', torch.float32))
            if verbose and out.g_phi is not None and T0 is not None:
                gt = _mole_targets(model, m_batch)
                T0_t = torch.as_tensor(T0, dtype=gt.dtype, device=gt.device).reshape(1, -1)
                near_boundary = (gt > 0) & (gt <= 2.0 * T0_t)
                if near_boundary.any():
                    resid = (gt - out.mole) / T0_t.clamp(min=1e-30)
                    residual_chunks.append(resid[near_boundary].detach().to('cpu', torch.float32))

            if N > max_N:
                break

    avg_test_loss = running_test_loss / N
    if not verbose:
        return avg_test_loss
    sat_str = 'n/a (no saturation head)' if (out is None or out.logits is None) else f'{running_sat_loss/N:.3e}'
    print(f"[TEST] Running Saturation Loss: {sat_str}\tRunning Chem Loss: {running_chem_loss/N:.3e}")
    print(f"[TEST] Running Molar Loss: {running_mole_loss/N:.3e}\tRunning Bulk Loss: {running_bulk_loss/N:.3e}")

    # Boundary-behavior diagnostics -- always shown when the model exposes g_phi
    # (ContinuousModel does, regardless of whether boundary_temperature annealing is
    # actually enabled this episode), since "is the network confidently separating
    # present/absent" is useful to watch either way.
    if aff_diag is not None:
        _print_affinity_diagnostics(aff_diag, model)
    if presence is not None:
        _print_presence_precision_recall(presence, model)
    if g_phi_chunks and aff_diag is not None:
        _print_histogram(torch.cat(g_phi_chunks).flatten(), "y_hat / c (2nd-98th pct)")
    elif g_phi_chunks:
        title = "g_phi / T0 (2nd-98th pct)" if T0 is not None else "g_phi, RAW -- no T0, not comparable across phases (2nd-98th pct)"
        _print_histogram(torch.cat(g_phi_chunks).flatten(), title)
    if residual_chunks:
        _print_histogram(torch.cat(residual_chunks).flatten(),
                         "(GT - pred) / T0, GT in (0, 2*T0] (2nd-98th pct)")
    elif g_phi_chunks and T0 is None and aff_diag is None:
        print("\n[boundary residual] skipped: ml_indexer.T0 not available "
              "(re-export the bundle or run scripts/compute_T0.py)")

    return avg_test_loss





def train_Lower_MELTS(model, trainData, testData, scheduler, scheduler_kwargs = {},
                      batch_size = 1024, criterion = nn.BCEWithLogitsLoss(), lr = 1e-4,
                      binWeights = None,
                      Epochs = 30, device = 'cuda',
                      max_N = np.inf, early_stopping_patience = 5, DictFilePath = None,
                      dropout_step_up = 0.05, dropout_step_down = 0.02,
                      noise_step_up = 0.002, noise_step_down = 0.001,
                      config_yaml = None, training_yaml = None, processing_yaml = None, stats = None, log_path = None,
                      train_params = None, carry_state = None, history = None):
    
    """    # --- Build (copy) model ---
    scheduler is text: one of ['steplr', 'cosine', 'cosinewarm', 'plateau'] or None for no scheduler. 
    model = NN.MidLevelNetwork(**Model.config)#.to(Model.device) # Copy the old model, so no overwriting. 
    model.load_state_dict(**Model.config)
    model = model.to(device)"""

    print('###### config ######')
    print(model.config)
    print('####################')

    model = model.to(device)
    if binWeights is None:
        nphases = getattr(getattr(model, 'ml_indexer', None), 'nphases', None)
        if nphases is None:
            raise ValueError("train_Lower_MELTS requires binWeights or a model with ml_indexer.nphases")
        binWeights = torch.ones((1, nphases), dtype=torch.float32)
    else:
        binWeights = torch.as_tensor(binWeights, dtype=torch.float32)
        if binWeights.ndim == 1:
            binWeights = binWeights.unsqueeze(0)
    binWeights = binWeights.to(device)
    phase_names = getattr(getattr(model, 'ml_indexer', None), 'all_phases', None)
    if phase_names is None:
        phase_names = [f"phase_{i}" for i in range(binWeights.shape[1])]

    # freeze all but encoder and saturation head
    for p in model.parameters():
        p.requires_grad = False
    for p in model.sat_head.parameters():
        p.requires_grad = True
    for p in model.encoder.parameters():
        p.requires_grad = True

    train_params = _resolve_train_params(model, train_params)
    noise = train_params['noise']
    optimizer = create_optimizer(model, lr=lr, lowWD=train_params['lowWD'])
    wrappedScheduler = create_scheduler(optimizer, scheduler, **scheduler_kwargs) if scheduler else SchedulerWrapper()

    lower_reg = _regularization_spec(model, 'lower', train_params)
    if 'dropout' in lower_reg.lower(): # Only use adaptive dropout if we're not using bulk loss.
        dropout_rate, max_drop = pull_number_range(lower_reg.lower())
        print(f"dropout in {lower_reg}: {dropout_rate} -> {max_drop}")
        _set_adaptive_dropout_rate(model, dropout_rate)
        print(f"Adaptive dropout: {_describe_adaptive_dropout(model)}")
    else:
        dropout_rate = 0
        max_drop = 0 # Max dropout rate for adaptive dropout

    # --- Loaders (same for both "Cr" and "NoCr" if you only want one test here) ---

    train_loader = _make_train_loader(trainData, batch_size, num_workers=12)
    test_loader = DataLoader(testData, batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=True)

    # --- Baseline: evaluate the incoming (pre-training) model first, so a training run that
    # never beats its own starting point cannot overwrite a superior saved checkpoint. ---
    print("Evaluating baseline (pre-training) model on validation set...")
    baseline_test_loss, baseline_precision, baseline_recall = _evaluate_binary_model(
        model, test_loader, binWeights, device, max_N
    )
    print("[BASELINE] Phasewise precision/recall:")
    for i, phase in enumerate(phase_names):
        print(f"  {phase}: precision={baseline_precision[i].item():.3f}, recall={baseline_recall[i].item():.3f}")
    print(f"Baseline Test Loss: {baseline_test_loss:.5f}")

    best_test_loss = baseline_test_loss
    best_epoch = 0  # 0 = the pre-training baseline model
    torch.save(model.state_dict(), str(TEMP_MODELS_DIR / 'temp_binary_train.pt'))
    if history is not None:  # builder.training.history: training-curve CSV
        history.add_val(optimizer.param_groups[0]['lr'], dropout_rate, noise, baseline_test_loss)

    train_losses, test_losses = [], []
    early_stopping_counter = 0

    # --- Train for specified epochs ---
    for epoch in range(Epochs):
        start = time.time()
        model.train()
        running_train_loss = 0.0
        N = 0


        for output in tqdm(train_loader, desc=f"Train Epoch {epoch+1}", leave=False):
            xb, yb = output[0], output[1] # We only need the phase saturation data here
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            #bulk_zero_mask = (x_batch != 0).to(torch.float) # Bulk zero mask different shape for training and testing because this mask is doubling as a filter for the noise
            if noise != 0:
                xb = xb + (xb * torch.randn_like(xb) * noise)  # noise injection

            optimizer.zero_grad()
            logits = model.forward_binaries(xb)
            loss = _weighted_binary_loss_gt_positive_only(logits, yb, binWeights)
            loss.backward()
            optimizer.step()
            batch_lr = optimizer.param_groups[0]['lr']  # the LR this step used (history)
            wrappedScheduler.step_batch() # Step scheduler if it's batch-based (Does nothing if it's epoch-based)

            running_train_loss += loss.item() * xb.size(0)
            N += xb.size(0)
            if history is not None:
                batch_loss = loss.item()
                history.add_batch(epoch, N, xb.size(0), batch_lr, dropout_rate, noise,
                                  {'total': batch_loss, 'sat': batch_loss})
            if N >= max_N:
                print(f"Reached max_N={max_N} samples for this epoch. Stopping early.")
                break
        else:  # the batch loop ran to completion (no max_N break)
            if history is not None:
                history.epoch_done(epoch)

        avg_train_loss = running_train_loss / N
        train_losses.append(avg_train_loss)

        # --- Evaluate ---
        avg_test_loss, precision, recall = _evaluate_binary_model(model, test_loader, binWeights, device, max_N)
        test_losses.append(avg_test_loss)
        if history is not None:
            history.add_val(optimizer.param_groups[0]['lr'], dropout_rate, noise, avg_test_loss)

        print("[TEST] Phasewise precision/recall:")
        for i, phase in enumerate(phase_names):
            print(f"  {phase}: precision={precision[i].item():.3f}, recall={recall[i].item():.3f}")

        print(f"Epoch {epoch+1:02d}: Train {avg_train_loss:.5f} | Test {avg_test_loss:.5f} | time = {time.time()-start:.1f}s")


        wrappedScheduler.step_epoch(avg_test_loss)

        # simple early stopping
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            best_epoch = epoch + 1
            print(f"New best test loss: {best_test_loss:.5f}. Saving model.")
            _record_dropout_state(model, 'lower', dropout_rate, noise, train_params, carry_state)
            torch.save(model.state_dict(), str(TEMP_MODELS_DIR / 'temp_binary_train.pt'))
            if DictFilePath is not None:
                log_text = _read_text_file(log_path)
                model.save(
                    DictFilePath,
                    config_yaml=config_yaml,
                    processing_yaml=processing_yaml,
                    training_yaml=training_yaml,
                    stats=stats,
                    log_text=log_text,
                )
            early_stopping_counter = 0
        elif avg_test_loss > best_test_loss * 1.01:
            early_stopping_counter += 1
            print(f"No improvement. Counter: {early_stopping_counter}/{early_stopping_patience}")
            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break

        #ADAPTIVE DROPOUT / NOISE (noise reacts only when dropout is maxed/floored out and can't respond)
        if avg_test_loss > avg_train_loss * 1.01:
            anyDropout = False
            if min(dropout_rate + dropout_step_up, max_drop) > dropout_rate:
                old_drop = dropout_rate
                dropout_rate = min(dropout_rate + dropout_step_up, max_drop)
                for module in _iter_adaptive_dropout_modules(model):
                    module.p = dropout_rate
                    anyDropout  = True
                if anyDropout:
                    print(f"Overfitting. Increasing Dropout: {old_drop} -> {dropout_rate}")
            else:
                old_noise = noise
                noise = noise + noise_step_up
                print(f"Overfitting, but dropout_rate is at the maximum: {dropout_rate}. Increasing noise instead: {old_noise} -> {noise}")


        elif avg_test_loss < avg_train_loss:
            anyDropout = False
            if max(dropout_rate - dropout_step_down, 0) < dropout_rate:
                old_drop = dropout_rate
                dropout_rate = max(dropout_rate - dropout_step_down, 0)
                for module in _iter_adaptive_dropout_modules(model):
                    module.p = dropout_rate
                    anyDropout  = True
                if anyDropout:
                    print(f"Underfitting. Decreasing Dropout: {old_drop}->{dropout_rate}")

            else:
                old_noise = noise
                noise = max(noise - noise_step_down, 0)
                print(f"Underfitting, but dropout_rate is at the minimum: {dropout_rate}. Decreasing noise instead: {old_noise} -> {noise}")

        gc.collect()



    model.load_state_dict(torch.load(str(TEMP_MODELS_DIR / 'temp_binary_train.pt'), weights_only=False))
    if best_epoch == 0:
        print(f"Best Test Loss: {best_test_loss:.5f} was the pre-training baseline; no epoch improved on it.")
    else:
        print(f"Best Test Loss: {best_test_loss:.5f} at epoch {best_epoch}")
    return best_test_loss



def train_Upper_MELTS(model, trainData, testData, scheduler, scheduler_kwargs = {}, criterion = symmetric_rel_l2, criterion_sat = nn.BCEWithLogitsLoss(),
                      criterion_mole = None, criterion_bulk = None,
                      chem_alpha = 1, mole_alpha = 1, bulk_alpha = 0, sat_alpha = 1, Epochs = 20, batch_size = 1024, lr = 1e-4,
                      binWeights = torch.ones(1), compWeights = torch.ones(1),
                      device = 'cuda', max_N = np.inf, early_stopping_patience = 5, which_heads_to_freeze = ['sat_head', 'encoder'], DictFilePath = None,
                      dropout_step_up = 0.05, dropout_step_down = 0.02,
                      noise_step_up = 0.002, noise_step_down = 0.001,
                      boundary_temperature = None,
                      val_checks_per_epoch = 1, fit_train_mode = 'train', fit_ref_rows = None,
                      overfit_ratio = 1.02, underfit_ratio = 1.0,
                      config_yaml = None, training_yaml = None, processing_yaml = None, stats = None, log_path = None, amsgrad=True, eps = 1E-4,
                      train_params = None, carry_state = None, history = None):
    """`val_checks_per_epoch`: how many fit checks per epoch, evenly spaced, the last at the
    end of the epoch. Each check evaluates the validation set, saves a new best checkpoint,
    and steps adaptive dropout on Test/Train against the dead band [underfit_ratio,
    overfit_ratio]. `fit_train_mode` picks the Train side (see _check_fit_train_mode); in
    'eval' mode it is a fixed random sample of `fit_ref_rows` training rows (default: as
    many as the validation set). The defaults (1 check, 'train', [1.0, 1.02]) reproduce the
    original once-per-epoch behavior exactly. `early_stopping_patience` stays in epochs.

    `boundary_temperature`: None (default, current behavior) or a dict
    `{a_start, a_end}` -- see `_anneal_a`/`_boundary_T`. Only meaningful for a model
    whose `upper_forward` accepts `T` (ContinuousModel); `_upper_forward` no-ops it for
    any other model, so passing this against e.g. MidLevelNetwork is harmless, not an
    error.

    `criterion` is the chemistry criterion; `criterion_mole` / `criterion_bulk` default to
    it when None. All three must be elementwise (see `symmetric_rel_l2`). main.py builds
    them from `loss_config.{chemistry,moles,bulk}.type`.

    `history`: an optional builder.training.history.TrainingHistory whose episode main.py
    has opened; it receives window-mean weighted losses, LR, dropout and noise every
    N batches, plus the validation loss at the baseline and at every fit check."""
    # iF which_heads_to_freeze is [], then this is a full model trainer!
    # Currently does not handle limited VC training!! Need to adjust model to make bulk output optional, then not use it in this loop
    """model = NN.MidLevelNetwork(**Model.config)#.to(Model.device) # Copy the old model, so no overwriting. 
    model.load_state_dict(**Model.config)"""
    print('###### config ######')
    print(model.config)
    print('####################')
    feature_offset = len(model.ml_indexer.featureNames)
    ## Copy Lower Parameters! 
    #model.sat_head.load_state_dict(Model.sat_head.state_dict())
    #model.encoder.load_state_dict(Model.encoder.state_dict())
    model = model.to(device)
    # freeze heads based on which_heads_to_freeze
    for p in model.parameters():
        p.requires_grad = True
    for frozen_head in _resolve_heads_to_freeze(model, which_heads_to_freeze):
        print(f"Freezing head: {frozen_head}")
        for p in getattr(model, frozen_head).parameters():
            p.requires_grad = False

    train_params = _resolve_train_params(model, train_params)
    affinity_params = _affinity_params(train_params)
    act_params = _activation_params(train_params)
    hinge_params = _hinge_params(train_params)
    hinge_on = bool(float(hinge_params['absent_hinge_weight']))
    noise = train_params['noise']
    optimizer = create_optimizer(model, lr=lr, highWD=train_params['highWD'], lowWD=train_params['lowWD'], amsgrad=amsgrad, eps=eps)
    wrappedScheduler = create_scheduler(optimizer, scheduler, **scheduler_kwargs) if scheduler else SchedulerWrapper()
            

    upper_reg = _regularization_spec(model, 'upper', train_params)
    if 'dropout' in upper_reg.lower():
        dropout_rate, configured_max = pull_number_range(upper_reg.lower())
        print(f"dropout in {upper_reg}: {dropout_rate} -> {configured_max}")
        """if bulk_alpha != 0: # Need to limit max dropout to avoid NaNs. 
            max_drop = 0
            print(f'  Bulk loss enabled, disabling dropout (max_drop=0)')
        else:
            max_drop = configured_max"""
        max_drop = configured_max
        
        _set_adaptive_dropout_rate(model, dropout_rate)
        print(f"Adaptive dropout: {_describe_adaptive_dropout(model)}")
    else:
        dropout_rate = 0
        max_drop = 0 # Max dropout rate for adaptive dropout

    # --- Loaders (same for both "Cr" and "NoCr" if you only want one test here) ---

    train_loader = _make_train_loader(trainData, batch_size, num_workers=4)
    test_loader = DataLoader(testData, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    binWeights = binWeights.to(device)
    compWeights = compWeights.to(device)

    criterion_chem = criterion
    criterion_mole = criterion if criterion_mole is None else criterion_mole
    criterion_bulk = criterion if criterion_bulk is None else criterion_bulk

    # --- Boundary-temperature annealing (see NN_continuous.py's upper_forward) ---
    boundary_temperature = _required_boundary_temperature(model, boundary_temperature)
    if boundary_temperature:
        a_start = float(boundary_temperature.get('a_start', BOUNDARY_TEMPERATURE_DEFAULTS['a_start']))
        a_end = float(boundary_temperature.get('a_end', BOUNDARY_TEMPERATURE_DEFAULTS['a_end']))
        print(f"[boundary_temperature] enabled: a {a_start:.3e} -> {a_end:.3e} over {Epochs} epochs")
    else:
        a_start = a_end = None

    def _epoch_T(epoch_idx):
        if a_start is None:
            return None
        a = _anneal_a(epoch_idx, Epochs, a_start, a_end)
        return _boundary_T(model.ml_indexer, a, device)

    # --- Mixed precision (see constants.TRAIN_PRECISION docstring for the tradeoffs) ---
    amp_dtype, use_amp = _resolve_precision(device)
    # GradScaler is only needed for float16 (limited exponent range risks over/underflow
    # in fp16 gradients); bfloat16 shares float32's exponent range so has nothing for a
    # scaler to protect against, and `enabled=False` makes every GradScaler method below
    # a plain passthrough, so this one object covers all three precisions unconditionally.
    scaler = torch.amp.GradScaler(device='cuda', enabled=(use_amp and amp_dtype is torch.float16))
    if use_amp:
        print(f"[precision] autocast dtype={amp_dtype}, GradScaler={'on' if scaler.is_enabled() else 'off'} "
              f"(set constants.TRAIN_PRECISION='float32' in ngibbs/config/constants.py to disable)")

    # --- Baseline: evaluate the incoming (pre-training) model first, so a training run that
    # never beats its own starting point cannot overwrite a superior saved checkpoint. ---
    print("Evaluating baseline (pre-training) model on validation set...")
    baseline_test_loss = _evaluate_upper_model(
        model, test_loader, feature_offset, criterion_sat, criterion_chem, criterion_mole, criterion_bulk,
        compWeights, binWeights, sat_alpha, chem_alpha, mole_alpha, bulk_alpha, device, max_N,
        T=_epoch_T(0), amp_dtype=amp_dtype, use_amp=use_amp, affinity_params=affinity_params,
        act_params=act_params, hinge_params=hinge_params,
    )
    print(f"Baseline Test Loss: {baseline_test_loss:.5f}")

    best_test_loss = baseline_test_loss
    best_epoch = 0  # 0 = the pre-training baseline model
    torch.save(model.state_dict(), str(TEMP_MODELS_DIR / 'temp_upper_train.pt'))
    if history is not None:  # builder.training.history: training-curve CSV
        history.add_val(optimizer.param_groups[0]['lr'], dropout_rate, noise, baseline_test_loss)

    train_losses, test_losses = [], []
    early_stopping_counter = 0
    bench = ThroughputBenchmark()  # one training-speed/inference-speed record, after epoch 5

    # --- Fit checks (see _train_reference_loader / _adapt_regularization) ---
    checks_per_epoch = max(1, int(val_checks_per_epoch))
    check_rows = _fit_check_rows(trainData, max_N, checks_per_epoch)
    fit_train_mode = _check_fit_train_mode(fit_train_mode)
    ref_loader = (_train_reference_loader(trainData, fit_ref_rows or len(testData), batch_size)
                  if fit_train_mode == 'eval' else None)
    patience_checks = early_stopping_patience * checks_per_epoch  # patience is configured in epochs

    def _eval(loader, T, verbose):
        return _evaluate_upper_model(
            model, loader, feature_offset, criterion_sat, criterion_chem, criterion_mole, criterion_bulk,
            compWeights, binWeights, sat_alpha, chem_alpha, mole_alpha, bulk_alpha, device, max_N,
            T=T, amp_dtype=amp_dtype, use_amp=use_amp, verbose=verbose,
            affinity_params=affinity_params, act_params=act_params, hinge_params=hinge_params,
        )

    def _fit_check(label, T, verbose, running_train_loss):
        """Validate, checkpoint, and step adaptive dropout. `running_train_loss` is the
        train-mode loss since the previous check, used when fit_train_mode == 'train'.
        Returns (test_loss, stop)."""
        nonlocal best_test_loss, best_epoch, early_stopping_counter, dropout_rate, noise
        avg_test_loss = _eval(test_loader, T, verbose)
        if fit_train_mode == 'eval':
            ref_loss, ref_desc = _eval(ref_loader, T, False), 'eval mode'
        else:
            ref_loss, ref_desc = running_train_loss, 'running, train mode'
        test_losses.append(avg_test_loss)
        if history is not None:  # dropout/noise as trained with, before this check adapts them
            history.add_val(optimizer.param_groups[0]['lr'], dropout_rate, noise, avg_test_loss)
        print(f"Check {label}: Test {avg_test_loss:.5f} | Train ({ref_desc}) {ref_loss:.5f} | "
              f"Test/Train {avg_test_loss/max(ref_loss, 1e-30):.3f}")

        if avg_test_loss < best_test_loss:
            print(f"New best test loss: {avg_test_loss:.5f} (improvement of {(best_test_loss-avg_test_loss)/best_test_loss*100:.2f}%) Saving model.")
            best_test_loss = avg_test_loss
            best_epoch = label
            _record_dropout_state(model, 'upper', dropout_rate, noise, train_params, carry_state)
            torch.save(model.state_dict(), str(TEMP_MODELS_DIR / 'temp_upper_train.pt'))
            if DictFilePath is not None:
                log_text = _read_text_file(log_path)
                model.save(
                    DictFilePath,
                    config_yaml=config_yaml,
                    processing_yaml=processing_yaml,
                    training_yaml=training_yaml,
                    stats=stats,
                    log_text=log_text,
                )
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            print(f"No improvement. Counter: {early_stopping_counter}/{patience_checks} checks")
            if early_stopping_counter >= patience_checks:
                return avg_test_loss, True

        dropout_rate, noise = _adapt_regularization(
            model, avg_test_loss, ref_loss, dropout_rate, max_drop, noise,
            dropout_step_up, dropout_step_down, noise_step_up, noise_step_down,
            overfit_ratio, underfit_ratio)
        return avg_test_loss, False

    # --- Train for specified epochs ---
    for epoch in range(Epochs):
        start = time.time()
        model.train()
        running_train_loss = 0.0
        running_sat_loss = 0
        running_chem_loss = 0
        running_mole_loss = 0
        running_bulk_loss = 0
        N=0
        out = None
        epoch_T = _epoch_T(epoch)
        next_check = 0
        stop = False
        win_loss0, win_N0 = 0.0, 0  # running totals at the previous check
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training", leave=False)):
            # See _evaluate_upper_model: trainData can likewise carry trailing derivative
            # arrays this non-derivative episode doesn't use.
            x_batch, b_batch, y_batch, m_batch = batch[0], batch[1], batch[2], batch[3]

            # Benchmark: only synchronizes/times during the one target epoch (see
            # benchmark.ThroughputBenchmark) -- torch.cuda.synchronize() forces a GPU
            # pipeline drain, so doing this every batch of every epoch would itself slow
            # training down; guarding it to a single epoch keeps that cost bounded.
            _bench_timing = bench.should_time(epoch)
            if _bench_timing:
                if device == 'cuda' and torch.cuda.is_available():
                    torch.cuda.synchronize()
                _bench_t0 = time.perf_counter()

            optimizer.zero_grad()

            x_batch, b_batch, y_batch, m_batch = x_batch.to(device, non_blocking=True), b_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True), m_batch.to(device, non_blocking=True)
            mask_batch = _affinity_mask_from_batch(model, batch, device)

            # NOTE: the bulk mask is now built inside _upper_loss from the *post-noise*
            # x_batch, identically to the evaluation path. Previously training built it
            # from the pre-noise batch over the full feature vector and evaluation built
            # it post-slice; both select the same entries (noise is multiplicative, so it
            # cannot turn a zero non-zero), so this is a de-duplication, not a change.
            if noise != 0:
                x_batch = x_batch + (x_batch * torch.randn_like(x_batch) * noise)
            with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                out = _upper_forward(model, x_batch, b_batch, T=epoch_T, act_params=act_params)

                loss, loss_sat, chem_loss_masked, mole_loss_masked, bulk_loss_masked = _upper_loss(
                    model, out, x_batch, b_batch, y_batch, m_batch, feature_offset,
                    criterion_sat, criterion_chem, criterion_mole, criterion_bulk,
                    compWeights, binWeights, sat_alpha, chem_alpha, mole_alpha, bulk_alpha,
                    mask_batch=mask_batch, affinity_params=affinity_params,
                    hinge_params=hinge_params,
                )

            batch_size_curr = x_batch.size(0)
            running_sat_loss += loss_sat.item() * batch_size_curr
            running_mole_loss += mole_loss_masked.item() * batch_size_curr
            running_chem_loss += chem_loss_masked.item() * batch_size_curr
            running_bulk_loss += bulk_loss_masked.item() * batch_size_curr

            if not torch.isfinite(loss):
                print("Non-finite loss detected!")
                print(f"Sat loss: {loss_sat.item()}, Chem loss: {chem_loss_masked.item()}, Mole loss: {mole_loss_masked}, Bulk loss: {bulk_loss_masked}")
                print('ENDING EARLY')
                early_stopping_counter = patience_checks+1 # trigger early stopping
                break
                #raise ValueError("Non-finite loss, stopping training.")
                continue

            # scaler.scale/.step/.update are plain passthroughs to loss.backward()/
            # optimizer.step() when the scaler is disabled (float32 and bfloat16 both
            # disable it -- see its construction above), so this one path covers all
            # three precisions without a branch here.
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if _bench_timing:
                if device == 'cuda' and torch.cuda.is_available():
                    torch.cuda.synchronize()
                bench.tick(epoch, batch_size_curr, time.perf_counter() - _bench_t0)
                if bench.ready:
                    bench.finalize(
                        model=model, infer_batches=test_loader, batch_size=batch_size, device=device,
                        episode_label=Path(DictFilePath).stem if DictFilePath else 'unknown',
                        training_config=dict(
                            batch_size=batch_size, lr=lr, chem_alpha=chem_alpha, mole_alpha=mole_alpha,
                            bulk_alpha=bulk_alpha, sat_alpha=sat_alpha, bulk_enabled=bool(bulk_alpha),
                            derivatives_enabled=False,
                            boundary_temperature_enabled=bool(boundary_temperature),
                            which_heads_to_freeze=list(which_heads_to_freeze),
                            precision=str(amp_dtype).replace('torch.', ''),
                        ),
                        amp_dtype=amp_dtype, use_amp=use_amp,
                    )

            running_train_loss += loss.item() * batch_size_curr



            """if np.random.rand() < 0.001: # Occasionally print gradient norms 
                total = 0
                count = 0
                for p in model.parameters():
                    if p.grad is not None:
                        total += (p.grad**2).mean()
                        count += 1

                grad_rms = (total / count).sqrt()
                print(f"Gradient RMS: {grad_rms.item()}")
                print(f"Update Loss: {loss.item():.3e}")"""
                

            #if batch_idx % 200 == 0:
                #percent_done = 100 * batch_idx / len(train_loader)
                #train_losses.append(loss.item())
                #print(f"[{percent_done:>5.1f}%] Batch {batch_idx:>5d} Loss: {loss.item():.4f}")
            N += batch_size_curr
            if history is not None:  # before step_batch, so lr is the one this step used
                history.add_batch(epoch, N, batch_size_curr, optimizer.param_groups[0]['lr'],
                                  dropout_rate, noise,
                                  upper_terms(loss, loss_sat, chem_loss_masked, mole_loss_masked,
                                              bulk_loss_masked, sat_alpha, chem_alpha, mole_alpha,
                                              bulk_alpha, LOSS_SCALE, hinge_on))
            if N > max_N:
                break

            wrappedScheduler.step_batch() # Step scheduler if it's batch-based (Does nothing if it's epoch-based)

            if next_check < len(check_rows) and N >= check_rows[next_check]:
                next_check += 1
                window_loss = (running_train_loss - win_loss0) / max(N - win_N0, 1)
                win_loss0, win_N0 = running_train_loss, N
                _, stop = _fit_check(f"{epoch+1}.{next_check}", epoch_T, verbose=False,
                                     running_train_loss=window_loss)
                if stop:
                    break
                model.train()
        else:  # the batch loop ran to completion (no stop / max_N / non-finite break)
            if history is not None:
                history.epoch_done(epoch)

        if stop:
            break
        avg_train_loss = running_train_loss / N

        sat_str = 'n/a (no saturation head)' if (out is None or out.logits is None) else f'{running_sat_loss/(N):.3e}'
        print(f"[TRAIN] Running Saturation Loss: {sat_str}\tRunning Chem Loss: {running_chem_loss/(N):.3e}")
        print(f"[TRAIN] Running Molar Loss: {running_mole_loss/(N):.3e}\tRunning Bulk Loss: {running_bulk_loss/(N):.3e}")

        print(f"[TRAIN] Running WEIGHTED Saturation Loss: {sat_alpha*running_sat_loss/(N):.3e}\tRunning Weighted Chem Loss: {chem_alpha*running_chem_loss/(N):.3e}")
        print(f"[TRAIN] Running WEIGHTED Molar Loss: {mole_alpha*running_mole_loss/(N):.3e}\tRunning Weighted Bulk Loss: {bulk_alpha*running_bulk_loss/(N):.3e}")

        """lrT = optimizer.param_groups[0]['lr']
        v = state['exp_avg_sq']
        lr_eff = lrT / (v.mean().sqrt() + optimizer.param_groups[0]['eps'])
        print(lr_eff.item())"""

        # ---- Evaluation ----
        # Same T as this epoch's training step - baseline/test loss should reflect the
        # temperature the model was actually just trained at, not a fresh a(0).
        label = f"{epoch+1}.{checks_per_epoch}" if checks_per_epoch > 1 else f"{epoch+1}"
        avg_test_loss, stop = _fit_check(label, epoch_T, verbose=True,
                                         running_train_loss=(running_train_loss - win_loss0) / max(N - win_N0, 1))
        print(f"Epoch {epoch+1:02d}: Train {avg_train_loss:.5f} (running, train mode) | Test {avg_test_loss:.5f} | time = {time.time()-start:.1f}s")
        if stop:
            break

        wrappedScheduler.step_epoch(avg_test_loss) # Step scheduler if it's epoch-based (Does nothing if it's batch-based)

        gc.collect()



    model.load_state_dict(torch.load(str(TEMP_MODELS_DIR / 'temp_upper_train.pt'), weights_only=False))
    if best_epoch == 0:
        print(f"Best Test Loss: {best_test_loss:.5f} was the pre-training baseline; no epoch improved on it.")
    else:
        print(f"Best Test Loss: {best_test_loss:.5f} at {'check (epoch.check)' if checks_per_epoch > 1 else 'epoch'} {best_epoch}")
    return best_test_loss