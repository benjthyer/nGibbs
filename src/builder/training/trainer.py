"""Base trainer implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional

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
    with torch.no_grad():
        for batch_idx, (x_batch, b_batch, y_batch, m_batch) in enumerate(test_loader):
            x_batch, b_batch, y_batch, m_batch = x_batch.to(device, non_blocking=True), b_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True), m_batch.to(device, non_blocking=True)
            logits, chem_preds, chem_zero_mask, mole_preds, bulk_preds = model(x_batch, binaries=b_batch, NN_only=True)

            loss_sat = criterion_sat(logits, b_batch)

            chem_loss_raw = criterion_chem(chem_preds, y_batch)
            mole_loss_raw = criterion_mole(mole_preds, m_batch)
            bulk_loss_raw = criterion_bulk(bulk_preds, x_batch[:, feature_offset:])
            bulk_zero_mask = (x_batch[:, feature_offset:] != 0).to(torch.float)
            mole_zero_mask = (b_batch > 0.5).to(torch.float)

            chem_loss_masked = (chem_loss_raw * chem_zero_mask * compWeights).sum() / (chem_zero_mask * compWeights).sum().clamp(min=1)
            mole_loss_masked = (mole_loss_raw * mole_zero_mask * binWeights).sum() / (mole_zero_mask * binWeights).sum().clamp(min=1)
            bulk_loss_masked = (bulk_loss_raw * bulk_zero_mask).sum() / bulk_zero_mask.sum().clamp(min=1)

            batch_size_curr = x_batch.size(0)
            running_sat_loss += loss_sat.item() * batch_size_curr
            running_mole_loss += mole_loss_masked.item() * batch_size_curr
            running_chem_loss += chem_loss_masked.item() * batch_size_curr
            running_bulk_loss += bulk_loss_masked.item() * batch_size_curr

            if bulk_alpha == 0:
                loss = 1E4 * (sat_alpha * loss_sat + chem_alpha * chem_loss_masked + mole_alpha * mole_loss_masked)
            else:
                loss = 1E4 * (sat_alpha * loss_sat + chem_alpha * chem_loss_masked + mole_alpha * mole_loss_masked + bulk_alpha * bulk_loss_masked)

            running_test_loss += loss.item() * batch_size_curr
            N += batch_size_curr
            if N > max_N:
                break

    avg_test_loss = running_test_loss / N
    print(f"[TEST] Running Saturation Loss: {running_sat_loss/N:.3e}\tRunning Chem Loss: {running_chem_loss/N:.3e}")
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

    if 'dropout' in model.config['low_regularization'].lower(): # Only use adaptive dropout if we're not using bulk loss.
        dropout_rate, max_drop = pull_number_range(model.config['low_regularization'].lower())
        print(f"dropout in {model.config['low_regularization']}: {dropout_rate} -> {max_drop}")
        _set_adaptive_dropout_rate(model, dropout_rate)
    else:
        dropout_rate = 0
        max_drop = 0 # Max dropout rate for adaptive dropout

    # --- Loaders (same for both "Cr" and "NoCr" if you only want one test here) ---

    train_loader = DataLoader(trainData, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)
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
    for frozen_head in which_heads_to_freeze:
        if hasattr(model, frozen_head):
            print(f"Freezing head: {frozen_head}")
            for p in getattr(model, frozen_head).parameters():
                p.requires_grad = False
        else:
            raise ValueError(f"Warning: Model does not have a head named '{frozen_head}' to freeze.")
    
    noise = model.config['noise']
    optimizer = create_optimizer(model, lr=lr, highWD=model.config['highWD'], lowWD=model.config['lowWD'], amsgrad=amsgrad, eps=eps)
    wrappedScheduler = create_scheduler(optimizer, scheduler, **scheduler_kwargs) if scheduler else SchedulerWrapper()
            

    if 'dropout' in model.config['high_regularization'].lower():
        dropout_rate, configured_max = pull_number_range(model.config['high_regularization'].lower())
        print(f"dropout in {model.config['high_regularization']}: {dropout_rate} -> {configured_max}")
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

    train_loader = DataLoader(trainData, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
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
        for batch_idx, (x_batch, b_batch, y_batch, m_batch) in enumerate(tqdm(train_loader, desc="Training", leave=False)):

            optimizer.zero_grad()

            x_batch, b_batch, y_batch, m_batch = x_batch.to(device, non_blocking=True), b_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True), m_batch.to(device, non_blocking=True)
            
            bulk_zero_mask = (x_batch != 0).to(torch.float) # Bulk zero mask different shape for training and testing because this mask is doubling as a filter for the noise
            #x_batch = x_batch + torch.randn_like(x_batch) * 0.005 * bulk_zero_mask # Add Small Gaussian Noise to avoid overfitting during training Oct 14: Moved 0.002->0.005
            if noise != 0:
                x_batch = x_batch + (x_batch * torch.randn_like(x_batch) * noise)
            logits, chem_preds, chem_zero_mask, mole_preds, bulk_preds = model(x_batch, binaries=b_batch, NN_only = True)

            # Binary saturation loss
            loss_sat = criterion_sat(logits, b_batch) 

            # Chemistry and mass losses: apply masks
            chem_loss_raw = criterion_chem(chem_preds, y_batch)
            mole_loss_raw = criterion_mole(mole_preds, m_batch)
            bulk_loss_raw = criterion_bulk(bulk_preds, x_batch[:,feature_offset:])
            #mole_zero_mask = (torch.sigmoid(logits) > 0.5).to(torch.float).detach() # Only use binary preds for masking when those neurons are free
            mole_zero_mask = (b_batch > 0.5).to(torch.float).detach()

            """if batch_idx == 0:
                print(chem_loss_raw.device)
                print(chem_zero_mask.device)
                print(compWeights.device)"""

            chem_loss_masked = (chem_loss_raw * chem_zero_mask * compWeights).sum() / (chem_zero_mask * compWeights).sum().clamp(min=1)
            mole_loss_masked = (mole_loss_raw * mole_zero_mask * binWeights).sum() / (mole_zero_mask * binWeights).sum().clamp(min=1)
            bulk_loss_masked = (bulk_loss_raw * bulk_zero_mask[:,feature_offset:]).sum() / (bulk_zero_mask[:,feature_offset:]).sum().clamp(min=1)
            
            batch_size_curr = x_batch.size(0)
            running_sat_loss += loss_sat.item() * batch_size_curr
            running_mole_loss += mole_loss_masked.item() * batch_size_curr
            running_chem_loss += chem_loss_masked.item() * batch_size_curr
            running_bulk_loss += bulk_loss_masked.item() * batch_size_curr
            
            
            if bulk_alpha == 0: # Scale loss to be large wrt to Epsilon for optimizer stability. 
                loss = 1E4*(sat_alpha*loss_sat + chem_alpha*chem_loss_masked + mole_alpha*mole_loss_masked) # Bulk can be numerically unstable...
            else:
                loss = 1E4*(sat_alpha*loss_sat + chem_alpha*chem_loss_masked + mole_alpha*mole_loss_masked + bulk_alpha*bulk_loss_masked)

            #print(f"Sat loss: {loss_sat.item()}, Chem loss: {chem_loss_masked.item()}")

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

        print(f"[TRAIN] Running Saturation Loss: {running_sat_loss/(N):.3e}\tRunning Chem Loss: {running_chem_loss/(N):.3e}")
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