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

from ngibbs.utils.string_utils import pull_number_range
from builder.training.optimizer_factory import create_optimizer, create_scheduler, SchedulerWrapper
from builder.training.benchmark import ThroughputBenchmark
from ngibbs.config import constants
import ngibbs.engine.NN as NN

# Mixed precision: constants.TRAIN_PRECISION selects the autocast dtype. See that
# constant's docstring in constants.py for the float32/bfloat16/float16 tradeoffs.
_PRECISION_DTYPES = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}


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


def symmetric_rel_l1(pred, target, eps=1e-6):
    denom = torch.clamp(torch.abs(pred) + torch.abs(target), min=eps)
    return torch.mean(torch.abs(pred - target) / denom)

def symmetric_rel_l2(pred, target, eps=1e-6):
    denom = torch.clamp(torch.abs(pred) + torch.abs(target), min=eps)
    return torch.mean((pred - target)**2 / denom)


def _iter_adaptive_dropout_modules(model: nn.Module):
    excluded_ids = set()
    for attr_name in (("mole_head",)):
        if hasattr(model, attr_name):
            head_module = getattr(model, attr_name)
            excluded_ids.update(id(module) for module in head_module.modules())

    for module in model.modules():
        if isinstance(module, nn.Dropout) and id(module) not in excluded_ids:
            yield module


def _set_adaptive_dropout_rate(model: nn.Module, dropout_rate: float) -> None:
    for module in _iter_adaptive_dropout_modules(model):
        module.p = dropout_rate


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


def _upper_forward(model, x_batch, b_batch, T=None) -> UpperBatch:
    fn = getattr(model, 'upper_forward', None)
    if fn is None:
        logits, chem, chem_mask, mole, bulk = model(x_batch, binaries=b_batch, NN_only=True)
        return UpperBatch(logits, chem, chem_mask, mole, bulk, None)

    # T (annealed complementarity-smoothing temperature) is a ContinuousModel-specific
    # concept; only pass it to models that actually declare support for it, so a plain
    # MidLevelNetwork's upper_forward (if one is ever added) isn't forced to accept an
    # argument it has no use for.
    kwargs = {'binaries': b_batch}
    if T is not None and 'T' in inspect.signature(fn).parameters:
        kwargs['T'] = T
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


def _boundary_T(ml_indexer, a, device, dtype=torch.float32, floor_frac=1e-6):
    """Build this epoch's `(1, P)` annealed temperature tensor `T = a*T0`.

    Floors `T0` itself (not just the product) at a tiny fraction of its own nonzero
    median, so a phase this dataset never saw present (`T0=0`, see `compute_T0`)
    doesn't produce `T=0` -> division by zero in `upper_forward`. It just gets an
    extremely sharp, near-hard-clamp `T`, which plays no meaningful smoothing role for
    a phase that's absent everywhere in the data anyway.
    """
    T0 = getattr(ml_indexer, 'T0', None)
    if T0 is None:
        raise ValueError(
            "boundary_temperature is enabled but ml_indexer.T0 is missing -- re-export "
            "this bundle (MLexporter.py computes T0 automatically) or backfill it with "
            "scripts/compute_T0.py.")
    T0 = np.asarray(T0, dtype=np.float64)
    nonzero = T0[T0 > 0]
    floor = float(nonzero.min()) * floor_frac if nonzero.size else 1e-8
    T0_safe = np.maximum(T0, floor)
    T = a * T0_safe
    return torch.as_tensor(T, dtype=dtype, device=device).reshape(1, -1)


def _mole_targets(model, m_batch):
    """Datasets store moles in MidLevelNetwork's output space (log10(n + molar_epsilon)).
    A model that predicts something else declares the conversion instead of the dataset
    being rebuilt per architecture."""
    fn = getattr(model, 'transform_mole_targets', None)
    return m_batch if fn is None else fn(m_batch)


def _regularization_spec(model, which='upper'):
    """Which config key holds this model's dropout/normalisation spec for this half of
    training. MidLevelNetwork says nothing and gets the historical names."""
    key = getattr(model, f'{which}_regularization_config_key', None)
    if key is None:
        key = 'high_regularization' if which == 'upper' else 'low_regularization'
    return str(model.config.get(key, 'none'))


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


def _upper_loss(model, out: UpperBatch, x_batch, b_batch, y_batch, m_batch, feature_offset,
                criterion_sat, criterion_chem, criterion_mole, criterion_bulk,
                compWeights, binWeights, sat_alpha, chem_alpha, mole_alpha, bulk_alpha):
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
    chem_loss_raw = criterion_chem(out.chem, y_batch)
    mole_loss_raw = criterion_mole(out.mole, _mole_targets(model, m_batch))
    bulk_loss_raw = criterion_bulk(out.bulk, bulk_target)

    bulk_zero_mask = (bulk_target != 0).to(torch.float)
    mole_zero_mask = (out.mole_mask if out.mole_mask is not None
                      else (b_batch > 0.5).to(torch.float)).detach()

    chem_loss = (chem_loss_raw * out.chem_mask * compWeights).sum() / (out.chem_mask * compWeights).sum().clamp(min=1)
    mole_loss = (mole_loss_raw * mole_zero_mask * binWeights).sum() / (mole_zero_mask * binWeights).sum().clamp(min=1)
    bulk_loss = (bulk_loss_raw * bulk_zero_mask).sum() / bulk_zero_mask.sum().clamp(min=1)

    # Scale loss to be large wrt Epsilon for optimizer stability.
    total = 1E4 * (sat_alpha * loss_sat + chem_alpha * chem_loss + mole_alpha * mole_loss)
    if bulk_alpha != 0:   # Bulk can be numerically unstable, so it stays opt-in.
        total = total + 1E4 * bulk_alpha * bulk_loss
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


def _evaluate_upper_model(model, test_loader, feature_offset, criterion_sat, criterion_chem, criterion_mole,
                           criterion_bulk, compWeights, binWeights, sat_alpha, chem_alpha, mole_alpha, bulk_alpha,
                           device, max_N=np.inf, T=None, amp_dtype=torch.float32, use_amp=False):
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
    T0 = getattr(getattr(model, 'ml_indexer', None), 'T0', None)
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # testData may carry trailing derivative arrays (dn/dP, dn/dT) when some other
            # episode in the recipe wants them -- main.py loads the dataset once for every
            # episode, so a non-derivative trainer like this one must ignore those extras
            # rather than assume a fixed 4-tuple.
            x_batch, b_batch, y_batch, m_batch = batch[0], batch[1], batch[2], batch[3]
            x_batch, b_batch, y_batch, m_batch = x_batch.to(device, non_blocking=True), b_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True), m_batch.to(device, non_blocking=True)
            with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                out = _upper_forward(model, x_batch, b_batch, T=T)

                loss, loss_sat, chem_loss_masked, mole_loss_masked, bulk_loss_masked = _upper_loss(
                    model, out, x_batch, b_batch, y_batch, m_batch, feature_offset,
                    criterion_sat, criterion_chem, criterion_mole, criterion_bulk,
                    compWeights, binWeights, sat_alpha, chem_alpha, mole_alpha, bulk_alpha,
                )

            batch_size_curr = x_batch.size(0)
            running_sat_loss += loss_sat.item() * batch_size_curr
            running_mole_loss += mole_loss_masked.item() * batch_size_curr
            running_chem_loss += chem_loss_masked.item() * batch_size_curr
            running_bulk_loss += bulk_loss_masked.item() * batch_size_curr

            running_test_loss += loss.item() * batch_size_curr
            N += batch_size_curr

            if out.g_phi is not None:
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
            if out.g_phi is not None and T0 is not None:
                gt = _mole_targets(model, m_batch)
                T0_t = torch.as_tensor(T0, dtype=gt.dtype, device=gt.device).reshape(1, -1)
                near_boundary = (gt > 0) & (gt <= 2.0 * T0_t)
                if near_boundary.any():
                    resid = (gt - out.mole) / T0_t.clamp(min=1e-30)
                    residual_chunks.append(resid[near_boundary].detach().to('cpu', torch.float32))

            if N > max_N:
                break

    avg_test_loss = running_test_loss / N
    sat_str = 'n/a (no saturation head)' if (out is None or out.logits is None) else f'{running_sat_loss/N:.3e}'
    print(f"[TEST] Running Saturation Loss: {sat_str}\tRunning Chem Loss: {running_chem_loss/N:.3e}")
    print(f"[TEST] Running Molar Loss: {running_mole_loss/N:.3e}\tRunning Bulk Loss: {running_bulk_loss/N:.3e}")

    # Boundary-behavior diagnostics -- always shown when the model exposes g_phi
    # (ContinuousModel does, regardless of whether boundary_temperature annealing is
    # actually enabled this episode), since "is the network confidently separating
    # present/absent" is useful to watch either way.
    if g_phi_chunks:
        title = "g_phi / T0 (2nd-98th pct)" if T0 is not None else "g_phi, RAW -- no T0, not comparable across phases (2nd-98th pct)"
        _print_histogram(torch.cat(g_phi_chunks).flatten(), title)
    if residual_chunks:
        _print_histogram(torch.cat(residual_chunks).flatten(),
                         "(GT - pred) / T0, GT in (0, 2*T0] (2nd-98th pct)")
    elif g_phi_chunks and T0 is None:
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
                      config_yaml = None, training_yaml = None, processing_yaml = None, stats = None, log_path = None):
    
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

    noise = model.config['noise']
    optimizer = create_optimizer(model, lr=lr, lowWD=model.config['lowWD'])
    wrappedScheduler = create_scheduler(optimizer, scheduler, **scheduler_kwargs) if scheduler else SchedulerWrapper()

    lower_reg = _regularization_spec(model, 'lower')
    if 'dropout' in lower_reg.lower(): # Only use adaptive dropout if we're not using bulk loss.
        dropout_rate, max_drop = pull_number_range(lower_reg.lower())
        print(f"dropout in {lower_reg}: {dropout_rate} -> {max_drop}")
        _set_adaptive_dropout_rate(model, dropout_rate)
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
            wrappedScheduler.step_batch() # Step scheduler if it's batch-based (Does nothing if it's epoch-based)

            running_train_loss += loss.item() * xb.size(0)
            N += xb.size(0)
            if N >= max_N:
                print(f"Reached max_N={max_N} samples for this epoch. Stopping early.")
                break

        avg_train_loss = running_train_loss / N
        train_losses.append(avg_train_loss)

        # --- Evaluate ---
        avg_test_loss, precision, recall = _evaluate_binary_model(model, test_loader, binWeights, device, max_N)
        test_losses.append(avg_test_loss)

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
                      chem_alpha = 1, mole_alpha = 1, bulk_alpha = 0, sat_alpha = 1, Epochs = 20, batch_size = 1024, lr = 1e-4,
                      binWeights = torch.ones(1), compWeights = torch.ones(1),
                      device = 'cuda', max_N = np.inf, early_stopping_patience = 5, which_heads_to_freeze = ['sat_head', 'encoder'], DictFilePath = None,
                      dropout_step_up = 0.05, dropout_step_down = 0.02,
                      noise_step_up = 0.002, noise_step_down = 0.001,
                      boundary_temperature = None,
                      config_yaml = None, training_yaml = None, processing_yaml = None, stats = None, log_path = None, amsgrad=True, eps = 1E-4):
    """`boundary_temperature`: None (default, current behavior) or a dict
    `{a_start, a_end}` -- see `_anneal_a`/`_boundary_T`. Only meaningful for a model
    whose `upper_forward` accepts `T` (ContinuousModel); `_upper_forward` no-ops it for
    any other model, so passing this against e.g. MidLevelNetwork is harmless, not an
    error."""
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

    noise = model.config['noise']
    optimizer = create_optimizer(model, lr=lr, highWD=model.config['highWD'], lowWD=model.config['lowWD'], amsgrad=amsgrad, eps=eps)
    wrappedScheduler = create_scheduler(optimizer, scheduler, **scheduler_kwargs) if scheduler else SchedulerWrapper()
            

    upper_reg = _regularization_spec(model, 'upper')
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
    else:
        dropout_rate = 0
        max_drop = 0 # Max dropout rate for adaptive dropout

    # --- Loaders (same for both "Cr" and "NoCr" if you only want one test here) ---

    train_loader = _make_train_loader(trainData, batch_size, num_workers=4)
    test_loader = DataLoader(testData, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    binWeights = binWeights.to(device)
    compWeights = compWeights.to(device)

    criterion_chem = criterion
    criterion_mole = criterion
    criterion_bulk = criterion

    # --- Boundary-temperature annealing (see NN_continuous.py's upper_forward) ---
    if boundary_temperature:
        a_start = float(boundary_temperature.get('a_start', 1.0))
        a_end = float(boundary_temperature.get('a_end', 1e-4))
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
        T=_epoch_T(0), amp_dtype=amp_dtype, use_amp=use_amp,
    )
    print(f"Baseline Test Loss: {baseline_test_loss:.5f}")

    best_test_loss = baseline_test_loss
    best_epoch = 0  # 0 = the pre-training baseline model
    torch.save(model.state_dict(), str(TEMP_MODELS_DIR / 'temp_upper_train.pt'))

    train_losses, test_losses = [], []
    early_stopping_counter = 0
    bench = ThroughputBenchmark()  # one training-speed/inference-speed record, after epoch 5

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

            # NOTE: the bulk mask is now built inside _upper_loss from the *post-noise*
            # x_batch, identically to the evaluation path. Previously training built it
            # from the pre-noise batch over the full feature vector and evaluation built
            # it post-slice; both select the same entries (noise is multiplicative, so it
            # cannot turn a zero non-zero), so this is a de-duplication, not a change.
            if noise != 0:
                x_batch = x_batch + (x_batch * torch.randn_like(x_batch) * noise)
            with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                out = _upper_forward(model, x_batch, b_batch, T=epoch_T)

                loss, loss_sat, chem_loss_masked, mole_loss_masked, bulk_loss_masked = _upper_loss(
                    model, out, x_batch, b_batch, y_batch, m_batch, feature_offset,
                    criterion_sat, criterion_chem, criterion_mole, criterion_bulk,
                    compWeights, binWeights, sat_alpha, chem_alpha, mole_alpha, bulk_alpha,
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
                early_stopping_counter = early_stopping_patience+1 # trigger early stopping
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
            if N > max_N:
                break

            wrappedScheduler.step_batch() # Step scheduler if it's batch-based (Does nothing if it's epoch-based)

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
        avg_test_loss = _evaluate_upper_model(
            model, test_loader, feature_offset, criterion_sat, criterion_chem, criterion_mole, criterion_bulk,
            compWeights, binWeights, sat_alpha, chem_alpha, mole_alpha, bulk_alpha, device, max_N,
            T=epoch_T, amp_dtype=amp_dtype, use_amp=use_amp,
        )
        test_losses.append(avg_test_loss)
        print(f"Epoch {epoch+1:02d}: Train {avg_train_loss:.5f} | Test {avg_test_loss:.5f} | time = {time.time()-start:.1f}s")

        if avg_test_loss < best_test_loss:
            print(f"New best test loss: {avg_test_loss:.5f} (improvement of {(best_test_loss-avg_test_loss)/best_test_loss*100:.2f}%) Saving model.")
            best_test_loss = avg_test_loss
            best_epoch = epoch + 1
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
            print(f"No improvement. Counter: {early_stopping_counter}/{early_stopping_patience}")
            if early_stopping_counter >= early_stopping_patience:
                break
            
        wrappedScheduler.step_epoch(avg_test_loss) # Step scheduler if it's epoch-based (Does nothing if it's batch-based)

        #ADAPTIVE DROPOUT / NOISE (noise reacts only when dropout is maxed/floored out and can't respond)
        if avg_test_loss > avg_train_loss * 1.02:
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



    model.load_state_dict(torch.load(str(TEMP_MODELS_DIR / 'temp_upper_train.pt'), weights_only=False))
    if best_epoch == 0:
        print(f"Best Test Loss: {best_test_loss:.5f} was the pre-training baseline; no epoch improved on it.")
    else:
        print(f"Best Test Loss: {best_test_loss:.5f} at epoch {best_epoch}")
    return best_test_loss