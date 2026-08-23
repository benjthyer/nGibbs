"""Base trainer implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
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
import ngibbs.engine.NN as NN

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


def _upper_forward(model, x_batch, b_batch) -> UpperBatch:
    fn = getattr(model, 'upper_forward', None)
    if fn is None:
        logits, chem, chem_mask, mole, bulk = model(x_batch, binaries=b_batch, NN_only=True)
        return UpperBatch(logits, chem, chem_mask, mole, bulk, None)

    out = fn(x_batch, binaries=b_batch)
    missing = {'chem', 'chem_mask', 'mole', 'bulk'} - set(out)
    if missing:
        raise KeyError(f"{type(model).__name__}.upper_forward() omitted {sorted(missing)}")
    return UpperBatch(out.get('logits'), out['chem'], out['chem_mask'],
                      out['mole'], out['bulk'], out.get('mole_mask'))


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
    them on."""
    loss_sat = (criterion_sat(out.logits, b_batch) if out.logits is not None
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
) -> torch.Tensor:
    """Apply phase weights only where GT phase is present; GT-absent terms stay unweighted."""
    loss_raw = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    effective_weights = targets * bin_weights + (1.0 - targets)
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
                           device, max_N=np.inf):
    model.eval()
    running_test_loss = 0.0
    running_sat_loss = 0
    running_chem_loss = 0
    running_mole_loss = 0
    running_bulk_loss = 0
    N = 0
    out = None
    with torch.no_grad():
        for batch_idx, (x_batch, b_batch, y_batch, m_batch) in enumerate(test_loader):
            x_batch, b_batch, y_batch, m_batch = x_batch.to(device, non_blocking=True), b_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True), m_batch.to(device, non_blocking=True)
            out = _upper_forward(model, x_batch, b_batch)

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
            if N > max_N:
                break

    avg_test_loss = running_test_loss / N
    sat_str = 'n/a (no saturation head)' if (out is None or out.logits is None) else f'{running_sat_loss/N:.3e}'
    print(f"[TEST] Running Saturation Loss: {sat_str}\tRunning Chem Loss: {running_chem_loss/N:.3e}")
    print(f"[TEST] Running Molar Loss: {running_mole_loss/N:.3e}\tRunning Bulk Loss: {running_bulk_loss/N:.3e}")
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
                      config_yaml = None, training_yaml = None, processing_yaml = None, stats = None, log_path = None, amsgrad=True, eps = 1E-4):
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

    # --- Baseline: evaluate the incoming (pre-training) model first, so a training run that
    # never beats its own starting point cannot overwrite a superior saved checkpoint. ---
    print("Evaluating baseline (pre-training) model on validation set...")
    baseline_test_loss = _evaluate_upper_model(
        model, test_loader, feature_offset, criterion_sat, criterion_chem, criterion_mole, criterion_bulk,
        compWeights, binWeights, sat_alpha, chem_alpha, mole_alpha, bulk_alpha, device, max_N,
    )
    print(f"Baseline Test Loss: {baseline_test_loss:.5f}")

    best_test_loss = baseline_test_loss
    best_epoch = 0  # 0 = the pre-training baseline model
    torch.save(model.state_dict(), str(TEMP_MODELS_DIR / 'temp_upper_train.pt'))

    train_losses, test_losses = [], []
    early_stopping_counter = 0

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
        for batch_idx, (x_batch, b_batch, y_batch, m_batch) in enumerate(tqdm(train_loader, desc="Training", leave=False)):

            optimizer.zero_grad()

            x_batch, b_batch, y_batch, m_batch = x_batch.to(device, non_blocking=True), b_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True), m_batch.to(device, non_blocking=True)
            
            # NOTE: the bulk mask is now built inside _upper_loss from the *post-noise*
            # x_batch, identically to the evaluation path. Previously training built it
            # from the pre-noise batch over the full feature vector and evaluation built
            # it post-slice; both select the same entries (noise is multiplicative, so it
            # cannot turn a zero non-zero), so this is a de-duplication, not a change.
            if noise != 0:
                x_batch = x_batch + (x_batch * torch.randn_like(x_batch) * noise)
            out = _upper_forward(model, x_batch, b_batch)

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

            loss.backward()


            optimizer.step()

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
        avg_test_loss = _evaluate_upper_model(
            model, test_loader, feature_offset, criterion_sat, criterion_chem, criterion_mole, criterion_bulk,
            compWeights, binWeights, sat_alpha, chem_alpha, mole_alpha, bulk_alpha, device, max_N,
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