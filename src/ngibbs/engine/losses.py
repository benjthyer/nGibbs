"""
Losses for the continuous saturation model.

Three changes from the binary-gate losses:

1. **Abundance loss is active for absent phases.** `n = 0` is real ground truth and is
   the only signal that teaches where zero lives. But it is a *hinge*, not an MSE: for
   an absent phase we only penalise `m > 0`, leaving the signed saturation free to sit
   anywhere below zero. That is the complementarity condition written as a loss, and it
   keeps gradient alive in the off region where a plain ReLU would give none.

2. **Composition loss is masked by presence** (no ground truth for an absent phase's
   chemistry) and weighted by `sqrt(n)`, so capacity is not spent on chemistry that
   cannot move an aggregate property.

3. **Derivative (Sobolev) terms** on `dn/dP` and `dn/dT`, both supervised directly from
   `fort.42`. This is what makes transition width a trained quantity rather than an
   emergent accident, and it is the same quantity the metamorphic phase-change terms
   need.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def abundance_loss(m_pred, n_true, present=None, hinge_weight=1.0):
    """Signed-saturation abundance loss.

    m_pred : (B, P) signed saturation (pre-clamp), from forward_phase_moles
    n_true : (B, P) target moles, zero where absent
    present: (B, P) bool; defaults to n_true > 0
    """
    if present is None:
        present = n_true > 0
    absent = ~present
    l_present = F.mse_loss(m_pred[present], n_true[present]) if present.any() \
        else m_pred.sum() * 0.0
    # only penalise positive saturation for absent phases
    l_absent = (F.relu(m_pred[absent]) ** 2).mean() if absent.any() \
        else m_pred.sum() * 0.0
    return l_present + hinge_weight * l_absent


def composition_loss(chem_pred, chem_true, comp_present, comp_moles=None, eps=1e-8):
    """Presence-masked, abundance-weighted endmember-composition loss."""
    if not comp_present.any():
        return chem_pred.sum() * 0.0
    err = (chem_pred - chem_true) ** 2
    w = torch.ones_like(err) if comp_moles is None else torch.sqrt(comp_moles.clamp(min=0) + eps)
    w = w * comp_present.to(err.dtype)
    return (err * w).sum() / w.sum().clamp(min=eps)


def derivative_loss(n_pred, inputs, dn_target, col, has_label=None, eps=1e-3):
    """Sobolev term. `dn_pred/dinput[:, col]` by autograd against a `fort.42` target.

    n_pred    : (B, C) predicted component moles (must be part of the autograd graph)
    inputs    : (B, F) the model input tensor, requires_grad_(True) before the forward
    dn_target : (B, C) target derivative, already normalised by the run's element total
    col       : index of P or T in `inputs`
    has_label : (B,) bool. Normalisation is by the count of *labelled rows*, not batch
                size, so the effective weight does not drift with batch composition.
    """
    grad = torch.autograd.grad(
        n_pred, inputs, grad_outputs=torch.ones_like(n_pred),
        create_graph=True, retain_graph=True)[0][:, col]           # (B,)
    # per-component derivative via a vector-Jacobian trick is expensive; supervise the
    # contracted derivative plus per-component terms only where labels exist.
    if has_label is None:
        has_label = torch.ones(n_pred.shape[0], dtype=torch.bool, device=n_pred.device)
    if not has_label.any():
        return n_pred.sum() * 0.0
    tgt = dn_target[has_label].sum(dim=1)
    return F.smooth_l1_loss(grad[has_label], tgt, beta=eps)


def per_component_derivative_loss(n_pred, inputs, dn_target, col, has_label=None,
                                  scale=None, eps=1e-3):
    """Full per-component Sobolev term. Costs C backward passes; use `vmap`/jacrev
    where available, or subsample components per step."""
    B, C = n_pred.shape
    if has_label is None:
        has_label = torch.ones(B, dtype=torch.bool, device=n_pred.device)
    if not has_label.any():
        return n_pred.sum() * 0.0
    rows = torch.zeros(B, C, device=n_pred.device, dtype=n_pred.dtype)
    for c in range(C):
        g = torch.autograd.grad(n_pred[:, c].sum(), inputs,
                                create_graph=True, retain_graph=True)[0][:, col]
        rows[:, c] = g
    w = torch.ones_like(rows) if scale is None else scale
    return F.smooth_l1_loss(rows[has_label] * w[has_label],
                            dn_target[has_label] * w[has_label], beta=eps)


def mass_balance_loss(recon_bulk, b_target):
    return F.mse_loss(recon_bulk, b_target)


def prior_saturation_loss(prior_sat_logits, present):
    """Auxiliary deep supervision only. Never multiplies the output."""
    return F.binary_cross_entropy_with_logits(prior_sat_logits, present.to(prior_sat_logits.dtype))
