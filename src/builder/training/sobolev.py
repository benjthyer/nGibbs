"""Derivative-supervised (Sobolev) training for the upper model.

`train_Upper_Sobolev` is `train_Upper_MELTS` plus two terms: the predicted `dn/dP` and
`dn/dT` compared against HeFESTo's own `fort.42` partials. Everything that is not the
derivative term -- the value loss, head freezing, adaptive dropout/noise, early stopping,
checkpointing -- is imported from `trainer.py` rather than copied, so there is one
implementation of each and a change there reaches both loops.

Why this is worth the compute
-----------------------------
The value loss constrains `n_phase` and, separately, the endmember fractions `x`. It does
not constrain how `x` moves with pressure. But

    n_c = n_phase(c) * x_c    =>    dn_c/dP = (dn_phase/dP) * x_c + n_phase * (dx_c/dP)

and the second term -- Fe partitioning between ringwoodite and bridgmanite+ferropericlase
as the transition proceeds -- is what sets the width of the 660. Supervising on the
species axis is the only way to reach it; contracting to phases annihilates it exactly.

Three design decisions, each measured
-------------------------------------
1. **Forward mode, not reverse.** `dn_c/dP` for all C components in one input direction is
   a Jacobian-*vector* product. At B=128, C=62: forward mode for both directions plus the
   backward is 76 ms/step; the reverse-mode Jacobian alone is ~383 ms, and that gap grows
   linearly in C.

2. **JVP through the network body only.** A JVP through the full `forward`, i.e. including
   `MassBalanceProjector`'s `linalg.solve`, costs 12,807 ms against 29 ms without it. It
   also buys nothing: at the projection's fixed point its Jacobian is exactly the
   orthogonal projector onto `null(A_Omega^T)`, which `ContinuousModel.tangent_project`
   applies analytically. That is implicit differentiation of the projection layer -- not of
   the equilibrium, which would need a Gibbs Hessian the network does not emit.

3. **Project the prediction, not the target.** HeFESTo's own derivatives satisfy
   `A^T dn = 0` to 7e-8 over the full species set, so the target needs no repair. The
   projection is applied to the prediction, where on an untrained net it moves the raw
   tangent by 70% -- structure the network would otherwise have to learn.

Units
-----
Four factors separate raw `fort.42` from what the network's tangent means. Three are
already applied by the exporter (`mol/GPa` needs no conversion since HeFESTo's feature IS
`P(GPa)`; the moles are divided by the row's element total, which is constant along a scan
so no quotient-rule term appears; the model predicts linear moles, not `log10`). The
fourth is applied here: the network sees min-max normalised features, so its tangent is
`d n / d x_norm = range_P * dn/dP`. `_feature_ranges` reads that range off the indexer's
`feature_normalizer`, and refuses to guess if it is absent -- a silently missing factor of
~140 would look like a merely badly-weighted loss rather than a wrong one.

A useful accident: normalised P spans [0, 1], so `dn/dx_norm` lands within an order of
magnitude of `n` itself. The derivative term needs no hand-tuned global weight to be
numerically comparable to the value term.

Which phases are covered
------------------------
The trained model carries only the phases that clear the abundance threshold in the
training data, while the derivative tables carry every species HeFESTo knows. Rows where an
untracked phase is present therefore cannot satisfy the tangent identity on the tracked
subset. `scripts/check_derivatives.py` reports that fraction per simulation; supply
`row_weights` to down-weight those rows if a dataset has many.
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

src_path = str(Path(__file__).parent.parent.parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from ngibbs.utils.string_utils import pull_number_range
from builder.training.optimizer_factory import create_optimizer, create_scheduler, SchedulerWrapper
from builder.training.trainer import (
    TEMP_MODELS_DIR, _make_train_loader, _read_text_file, _regularization_spec,
    _resolve_heads_to_freeze, _set_adaptive_dropout_rate, _iter_adaptive_dropout_modules,
    _upper_forward, _upper_loss, symmetric_rel_l2,
)

# Batch layout. The first four are the historical `TensorDatasetFour`; the derivative
# arrays are appended, so a 4-tuple batch still unpacks and the error, if a bundle without
# them reaches this loop, is raised here by name rather than as an opaque unpack failure.
_N_BASE = 4


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def _feature_ranges(model):
    """`max - min` per input feature, i.e. the factor between a physical derivative and a
    derivative w.r.t. the normalised input the network actually sees.

    Raises rather than defaulting to 1.0. A missing normaliser would make the derivative
    loss quietly wrong by ~140x on P, which reads as a weighting problem and is not one.
    """
    norm = getattr(model.ml_indexer, 'feature_normalizer', None)
    if norm is None or getattr(norm, 'ranger', None) is None:
        raise ValueError(
            "Sobolev training needs ml_indexer.feature_normalizer to convert physical "
            "dn/dP into the normalised-input space the network differentiates in. It is "
            "absent, and defaulting to 1.0 would silently mis-scale the loss. Load the "
            "dataset through load_ML_data/load_ML_data_auto, which fits it.")
    r = torch.as_tensor(np.asarray(norm.ranger, dtype=np.float32)).flatten()
    return r


def _feature_index(model, *prefixes):
    """Column of the first feature whose name starts with any of `prefixes`.

    HeFESTo names them `P(GPa)(System_main)` / `T(K)(System_main)`, MELTS
    `Pressure(System_main)` / `Temperature(System_main)`; matching on a prefix covers both
    without a database switch.
    """
    names = [str(n) for n in model.ml_indexer.featureNames]
    for i, n in enumerate(names):
        if any(n.startswith(p) for p in prefixes):
            return i
    raise KeyError(f"no feature among {names} starts with any of {prefixes}")


def _robust_scale(x, mask, q=95.0, rel_floor=1e-3):
    """Per-component robust scale, so a trace component is neither drowned by forsterite
    nor amplified into the dominant term. The high percentile rather than the max, because
    the max lands on a phase-boundary row where the true derivative is genuinely enormous.

    Components that never move in the sample get a scale of zero from the percentile, which
    would turn any error into a division by ~0. They are floored RELATIVE to the median of
    the components that do move, so the floor tracks the dataset instead of being an
    absolute constant that is wrong the moment the units change.
    """
    if not mask.any():
        return torch.ones(x.shape[1], dtype=torch.float32)
    v = x[mask].abs().float()
    out = torch.quantile(v, q / 100.0, dim=0)
    live = out[out > 0]
    floor = float(live.median()) * rel_floor if len(live) else 1.0
    return out.clamp(min=max(floor, 1e-30))


def _derivative_loss(pred, target, scale, row_mask, support, delta=3.0, row_weights=None):
    """Scaled Huber on the component-axis derivative.

    Huber rather than L2 because the rows that matter most -- phase-in and phase-out -- are
    exactly where the true derivative is large and genuinely discontinuous, and an L2 lets
    them dominate the epoch. They are clipped, not dropped.

    `support` excludes components that are absent in both prediction and target: their
    target is zero, `clamp` already gives the prediction zero derivative and zero gradient,
    so including them only dilutes the mean.
    """
    if not row_mask.any():
        return pred.sum() * 0.0
    r = (pred - target) / scale.to(pred.device).unsqueeze(0)
    per = F.huber_loss(r, torch.zeros_like(r), reduction='none', delta=delta)
    w = (row_mask.unsqueeze(1) & support).to(per.dtype)
    if row_weights is not None:
        w = w * row_weights.to(per.dtype).unsqueeze(1)
    return (per * w).sum() / w.sum().clamp(min=1.0)


def _component_targets(model, y_batch, m_batch):
    """Species-axis moles rebuilt from the phase-axis and endmember-fraction targets.

    `n_c = n_phase(c) * x_c`, which is the model's own `phaseProportions *
    compMultipliers`. Doing it here means the bundle carries no new *value* array -- only
    the two derivative arrays are new.
    """
    fn = getattr(model, 'transform_mole_targets', None)
    moles = m_batch if fn is None else fn(m_batch)
    proportions = y_batch @ model.variedToAllComp + model.fixed_phaseToCompMap
    return proportions * (moles @ model.phaseToCompMap)


# --------------------------------------------------------------------------- #
#  Evaluation
# --------------------------------------------------------------------------- #
def _evaluate_sobolev(model, test_loader, feature_offset, criterion_sat, criterion_chem,
                      criterion_mole, criterion_bulk, compWeights, binWeights,
                      sat_alpha, chem_alpha, mole_alpha, bulk_alpha,
                      dndp_alpha, dndt_alpha, deriv_scale, feat_idx, ranges,
                      device, max_N=np.inf, project_tangent=True, huber_delta=3.0):
    model.eval()
    tot = {k: 0.0 for k in ('loss', 'sat', 'chem', 'mole', 'bulk', 'dndp', 'dndt')}
    N = 0
    out = None
    # Tangents need grad through the inputs, so this cannot run under `no_grad`; the
    # parameter graph is discarded per batch instead.
    for batch in test_loader:
        batch = [t.to(device, non_blocking=True) for t in batch]
        x_batch, b_batch, y_batch, m_batch = batch[:_N_BASE]
        dndp_t, dndt_t = batch[_N_BASE], batch[_N_BASE + 1]

        with torch.no_grad():
            out = _upper_forward(model, x_batch, b_batch)
            loss, l_sat, l_chem, l_mole, l_bulk = _upper_loss(
                model, out, x_batch, b_batch, y_batch, m_batch, feature_offset,
                criterion_sat, criterion_chem, criterion_mole, criterion_bulk,
                compWeights, binWeights, sat_alpha, chem_alpha, mole_alpha, bulk_alpha)

        d_losses = _derivative_terms(
            model, x_batch, y_batch, m_batch, dndp_t, dndt_t, deriv_scale, feat_idx,
            ranges, project_tangent, huber_delta, None)
        loss = loss + 1E4 * (dndp_alpha * d_losses[0] + dndt_alpha * d_losses[1])

        n = x_batch.size(0)
        for k, v in zip(('loss', 'sat', 'chem', 'mole', 'bulk', 'dndp', 'dndt'),
                        (loss, l_sat, l_chem, l_mole, l_bulk, d_losses[0], d_losses[1])):
            tot[k] += float(v.detach()) * n
        N += n
        if N > max_N:
            break

    sat_str = 'n/a (no saturation head)' if (out is None or out.logits is None) else f"{tot['sat']/N:.3e}"
    print(f"[TEST] Sat {sat_str}\tChem {tot['chem']/N:.3e}\tMole {tot['mole']/N:.3e}\t"
          f"Bulk {tot['bulk']/N:.3e}")
    print(f"[TEST] dn/dP {tot['dndp']/N:.3e}\tdn/dT {tot['dndt']/N:.3e}")
    return tot['loss'] / N


def _derivative_terms(model, x_batch, y_batch, m_batch, dndp_t, dndt_t, deriv_scale,
                      feat_idx, ranges, project_tangent, huber_delta, row_weights,
                      subsample_idx=None):
    """The two derivative losses for one batch. Returns `(L_dndp, L_dndt)`."""
    xs = x_batch if subsample_idx is None else x_batch[subsample_idx]
    dp = dndp_t if subsample_idx is None else dndp_t[subsample_idx]
    dt = dndt_t if subsample_idx is None else dndt_t[subsample_idx]
    ys = y_batch if subsample_idx is None else y_batch[subsample_idx]
    ms = m_batch if subsample_idx is None else m_batch[subsample_idx]
    rw = None if row_weights is None else (
        row_weights if subsample_idx is None else row_weights[subsample_idx])

    xs = xs.detach().requires_grad_(False)
    primal, tangents = model.derivative_outputs(xs, feat_idx, project=project_tangent)
    n_pred = primal[0]

    # Physical -> normalised-input space. See the module docstring.
    targets = []
    for k, tgt in enumerate((dp, dt)):
        targets.append(tgt * ranges[feat_idx[k]].to(tgt.device))

    n_tgt = _component_targets(model, ys, ms)
    support = (n_pred.detach() > 0) | (n_tgt > 0)

    losses = []
    for tgt, pred in zip(targets, tangents):
        finite = torch.isfinite(tgt).all(dim=1)
        safe = torch.where(torch.isfinite(tgt), tgt, torch.zeros_like(tgt))
        losses.append(_derivative_loss(pred, safe, deriv_scale, finite, support,
                                       delta=huber_delta, row_weights=rw))
    return losses


# --------------------------------------------------------------------------- #
#  Trainer
# --------------------------------------------------------------------------- #
def train_Upper_Sobolev(model, trainData, testData, scheduler, scheduler_kwargs={},
                        criterion=symmetric_rel_l2, criterion_sat=nn.BCEWithLogitsLoss(),
                        chem_alpha=1, mole_alpha=1, bulk_alpha=0, sat_alpha=1,
                        dndp_alpha=1.0, dndt_alpha=1.0,
                        Epochs=20, batch_size=1024, lr=1e-4,
                        binWeights=torch.ones(1), compWeights=torch.ones(1),
                        device='cuda', max_N=np.inf, early_stopping_patience=5,
                        which_heads_to_freeze=['sat_head', 'encoder'], DictFilePath=None,
                        dropout_step_up=0.05, dropout_step_down=0.02,
                        noise_step_up=0.002, noise_step_down=0.001,
                        config_yaml=None, training_yaml=None, processing_yaml=None,
                        stats=None, log_path=None, amsgrad=True, eps=1E-4,
                        project_tangent=True, deriv_subsample=1.0, huber_delta=3.0,
                        deriv_scale=None, row_weights=None):
    """Upper-model training with `dn/dP` and `dn/dT` supervision.

    Extra arguments beyond `train_Upper_MELTS`:

    dndp_alpha, dndt_alpha : float
        Weights on the two derivative terms, on the same 1E4 scaling as the value terms.
    project_tangent : bool
        Apply the analytic null-space projection to the predicted tangent. On by default;
        turning it off is a diagnostic, not a mode.
    deriv_subsample : float
        Fraction of each batch that gets the JVP. The gradient stays unbiased, so this is
        purely a compute knob; 1.0 costs roughly 3x a plain step.
    huber_delta : float
        Transition point of the Huber, in units of the per-component robust scale.
    deriv_scale : array or None
        Per-component scale. `None` estimates it from the first epoch's targets, which is
        adequate; supplying the exporter's `derivative_stats` value makes runs comparable.
    row_weights : Tensor or None
        Per-row weight, for down-weighting rows where an untracked phase is present.
    """
    if not hasattr(model, 'derivative_outputs'):
        raise TypeError(
            f"{type(model).__name__} has no derivative_outputs(); Sobolev training needs a "
            "double-backward-safe model. Use ContinuousModel -- NN.py's PhaseHead writes "
            "into a softmax output in place and cannot be differentiated twice.")

    print('###### config ######')
    print(model.config)
    print('####################')
    feature_offset = len(model.ml_indexer.featureNames)
    model = model.to(device)

    for p in model.parameters():
        p.requires_grad = True
    for frozen_head in _resolve_heads_to_freeze(model, which_heads_to_freeze):
        print(f"Freezing head: {frozen_head}")
        for p in getattr(model, frozen_head).parameters():
            p.requires_grad = False

    ranges = _feature_ranges(model).to(device)
    p_idx = _feature_index(model, 'P(', 'Pressure')
    t_idx = _feature_index(model, 'T(', 'Temperature')
    feat_idx = [p_idx, t_idx]
    print(f"Derivative directions: feature[{p_idx}] (dn/dP, range {ranges[p_idx]:.4g}), "
          f"feature[{t_idx}] (dn/dT, range {ranges[t_idx]:.4g})")

    upper_reg = _regularization_spec(model, 'upper')
    if 'batchnorm' in upper_reg.lower():
        print("WARNING: batchnorm under forward-mode AD makes each row's tangent depend on "
              "the rest of the batch, which is not a property the physics has. Prefer "
              "layernorm for derivative episodes.")
    if 'dropout' in upper_reg.lower():
        dropout_rate, max_drop = pull_number_range(upper_reg.lower())
        print(f"dropout in {upper_reg}: {dropout_rate} -> {max_drop}. Note each derivative "
              f"direction is a separate forward and so draws its own mask; consider 0.")
        _set_adaptive_dropout_rate(model, dropout_rate)
    else:
        dropout_rate = max_drop = 0

    noise = model.config['noise']
    optimizer = create_optimizer(model, lr=lr, highWD=model.config['highWD'],
                                 lowWD=model.config['lowWD'], amsgrad=amsgrad, eps=eps)
    wrappedScheduler = create_scheduler(optimizer, scheduler, **scheduler_kwargs) \
        if scheduler else SchedulerWrapper()

    train_loader = _make_train_loader(trainData, batch_size, num_workers=4)
    test_loader = DataLoader(testData, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)
    binWeights = binWeights.to(device)
    compWeights = compWeights.to(device)
    criterion_chem = criterion_mole = criterion_bulk = criterion

    # Scale: estimated once from the first batches rather than per batch, so the loss does
    # not drift as the epoch's composition changes.
    if deriv_scale is None:
        acc = []
        for j, batch in enumerate(test_loader):
            tgt = batch[_N_BASE].to(device) * ranges[p_idx]
            acc.append(tgt[torch.isfinite(tgt).all(dim=1)].cpu())
            if j >= 3:
                break
        allt = torch.cat(acc) if acc else torch.zeros(1, 1)
        deriv_scale = _robust_scale(allt, torch.ones(len(allt), dtype=torch.bool))
        print(f"Derivative scale estimated from {len(allt)} rows: "
              f"median {deriv_scale.median():.3e}, max {deriv_scale.max():.3e}")
    else:
        deriv_scale = torch.as_tensor(np.asarray(deriv_scale, dtype=np.float32))

    ev = lambda: _evaluate_sobolev(
        model, test_loader, feature_offset, criterion_sat, criterion_chem, criterion_mole,
        criterion_bulk, compWeights, binWeights, sat_alpha, chem_alpha, mole_alpha,
        bulk_alpha, dndp_alpha, dndt_alpha, deriv_scale, feat_idx, ranges, device, max_N,
        project_tangent, huber_delta)

    print("Evaluating baseline (pre-training) model on validation set...")
    best_test_loss = ev()
    print(f"Baseline Test Loss: {best_test_loss:.5f}")
    best_epoch = 0
    torch.save(model.state_dict(), str(TEMP_MODELS_DIR / 'temp_upper_sobolev.pt'))
    early_stopping_counter = 0

    for epoch in range(Epochs):
        start = time.time()
        model.train()
        run = {k: 0.0 for k in ('train', 'sat', 'chem', 'mole', 'bulk', 'dndp', 'dndt')}
        N = 0
        out = None
        for batch in tqdm(train_loader, desc="Training", leave=False):
            if len(batch) < _N_BASE + 2:
                raise ValueError(
                    f"Sobolev training needs {_N_BASE + 2} tensors per batch "
                    f"(x, binaries, chem, moles, dndp, dndt); this loader yields "
                    f"{len(batch)}. The ML bundle predates derivative export -- re-export "
                    f"it, or set derivatives.enabled=false to use train_Upper_MELTS.")
            optimizer.zero_grad()
            batch = [t.to(device, non_blocking=True) for t in batch]
            x_batch, b_batch, y_batch, m_batch = batch[:_N_BASE]
            dndp_t, dndt_t = batch[_N_BASE], batch[_N_BASE + 1]

            if noise != 0:
                # Noise goes only into the value path. Perturbing the input of a derivative
                # target would be asking the network to reproduce dn/dP at a P it was not
                # given, which is a different and wrong question.
                x_noisy = x_batch + (x_batch * torch.randn_like(x_batch) * noise)
            else:
                x_noisy = x_batch

            out = _upper_forward(model, x_noisy, b_batch)
            loss, l_sat, l_chem, l_mole, l_bulk = _upper_loss(
                model, out, x_noisy, b_batch, y_batch, m_batch, feature_offset,
                criterion_sat, criterion_chem, criterion_mole, criterion_bulk,
                compWeights, binWeights, sat_alpha, chem_alpha, mole_alpha, bulk_alpha)

            sub = None
            if deriv_subsample < 1.0:
                k = max(1, int(deriv_subsample * x_batch.size(0)))
                sub = torch.randperm(x_batch.size(0), device=x_batch.device)[:k]
            l_dp, l_dt = _derivative_terms(
                model, x_batch, y_batch, m_batch, dndp_t, dndt_t, deriv_scale, feat_idx,
                ranges, project_tangent, huber_delta, row_weights, sub)
            loss = loss + 1E4 * (dndp_alpha * l_dp + dndt_alpha * l_dt)

            n = x_batch.size(0)
            for k, v in zip(('sat', 'chem', 'mole', 'bulk', 'dndp', 'dndt'),
                            (l_sat, l_chem, l_mole, l_bulk, l_dp, l_dt)):
                run[k] += float(v.detach()) * n

            if not torch.isfinite(loss):
                print("Non-finite loss detected!")
                print(f"sat {float(l_sat):.3e} chem {float(l_chem):.3e} mole {float(l_mole):.3e} "
                      f"bulk {float(l_bulk):.3e} dndp {float(l_dp):.3e} dndt {float(l_dt):.3e}")
                print('ENDING EARLY')
                early_stopping_counter = early_stopping_patience + 1
                break

            loss.backward()
            optimizer.step()
            run['train'] += float(loss.detach()) * n
            N += n
            if N > max_N:
                break
            wrappedScheduler.step_batch()

        avg_train_loss = run['train'] / max(N, 1)
        sat_str = 'n/a (no saturation head)' if (out is None or out.logits is None) \
            else f"{run['sat']/N:.3e}"
        print(f"[TRAIN] Sat {sat_str}\tChem {run['chem']/N:.3e}\tMole {run['mole']/N:.3e}\t"
              f"Bulk {run['bulk']/N:.3e}")
        print(f"[TRAIN] dn/dP {run['dndp']/N:.3e}\tdn/dT {run['dndt']/N:.3e}\t"
              f"(weighted {dndp_alpha*run['dndp']/N:.3e} / {dndt_alpha*run['dndt']/N:.3e})")

        avg_test_loss = ev()
        print(f"Epoch {epoch+1:02d}: Train {avg_train_loss:.5f} | Test {avg_test_loss:.5f} "
              f"| time = {time.time()-start:.1f}s")

        if avg_test_loss < best_test_loss:
            print(f"New best test loss: {avg_test_loss:.5f} "
                  f"(improvement of {(best_test_loss-avg_test_loss)/best_test_loss*100:.2f}%) Saving model.")
            best_test_loss = avg_test_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), str(TEMP_MODELS_DIR / 'temp_upper_sobolev.pt'))
            if DictFilePath is not None:
                model.save(DictFilePath, config_yaml=config_yaml,
                           processing_yaml=processing_yaml, training_yaml=training_yaml,
                           stats=stats, log_text=_read_text_file(log_path))
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            print(f"No improvement. Counter: {early_stopping_counter}/{early_stopping_patience}")
            if early_stopping_counter >= early_stopping_patience:
                break

        wrappedScheduler.step_epoch(avg_test_loss)

        # Adaptive dropout / noise, identical in policy to train_Upper_MELTS.
        if avg_test_loss > avg_train_loss * 1.02:
            if min(dropout_rate + dropout_step_up, max_drop) > dropout_rate:
                old = dropout_rate
                dropout_rate = min(dropout_rate + dropout_step_up, max_drop)
                _set_adaptive_dropout_rate(model, dropout_rate)
                print(f"Overfitting. Increasing Dropout: {old} -> {dropout_rate}")
            else:
                old = noise
                noise = noise + noise_step_up
                print(f"Overfitting, dropout at max {dropout_rate}. Noise: {old} -> {noise}")
        elif avg_test_loss < avg_train_loss:
            if max(dropout_rate - dropout_step_down, 0) < dropout_rate:
                old = dropout_rate
                dropout_rate = max(dropout_rate - dropout_step_down, 0)
                _set_adaptive_dropout_rate(model, dropout_rate)
                print(f"Underfitting. Decreasing Dropout: {old} -> {dropout_rate}")
            else:
                old = noise
                noise = max(noise - noise_step_down, 0)
                print(f"Underfitting, dropout at min {dropout_rate}. Noise: {old} -> {noise}")
        gc.collect()

    model.load_state_dict(torch.load(str(TEMP_MODELS_DIR / 'temp_upper_sobolev.pt'),
                                     weights_only=False))
    if best_epoch == 0:
        print(f"Best Test Loss: {best_test_loss:.5f} was the pre-training baseline; "
              f"no epoch improved on it.")
    else:
        print(f"Best Test Loss: {best_test_loss:.5f} at epoch {best_epoch}")
    return best_test_loss
