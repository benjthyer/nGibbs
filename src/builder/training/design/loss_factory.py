"""Loss computation helpers."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .constraints import symmetric_rel_l2


_LOSS_FNS = {
    "mse": F.mse_loss,
    "l1": F.l1_loss,
    "symmetric_rel_l2": symmetric_rel_l2,
}


def _reduce_loss(loss: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    return loss.mean()


def _masked_mean(loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = loss * mask
    denom = mask.sum().clamp(min=1.0)
    return masked.sum() / denom


def _apply_weights(loss: torch.Tensor, weights: Optional[torch.Tensor]) -> torch.Tensor:
    if weights is None:
        return loss
    return loss * weights


def _select_loss_fn(name: str):
    if name not in _LOSS_FNS:
        raise ValueError(f"Unknown loss function: {name}")
    return _LOSS_FNS[name]


def compute_losses(
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    ml_indexer,
    loss_config: Dict,
) -> Dict[str, torch.Tensor]:
    losses: Dict[str, torch.Tensor] = {}
    total = torch.tensor(0.0, device=next(iter(predictions.values())).device)

    for key, cfg in loss_config.items():
        if key == "binary" and "binary_logits" in predictions:
            loss = F.binary_cross_entropy_with_logits(predictions["binary_logits"], targets["binaries"])
            weight = cfg.get("weight", 1.0)
            loss = loss * weight
            losses["binary_loss"] = loss
            total = total + loss
            continue

        if key == "chemistry" and "chemistry" in predictions:
            loss_fn = _select_loss_fn(cfg.get("type", "symmetric_rel_l2"))
            raw = loss_fn(predictions["chemistry"], targets["chemistries"])
            raw = _apply_weights(raw, cfg.get("weights"))
            mask = predictions.get("chem_mask")
            loss = _masked_mean(raw, mask) if mask is not None else raw.mean()
            loss = loss * cfg.get("weight", 1.0)
            losses["chemistry_loss"] = loss
            total = total + loss
            continue

        if key == "moles" and "moles" in predictions:
            loss_fn = _select_loss_fn(cfg.get("type", "symmetric_rel_l2"))
            raw = loss_fn(predictions["moles"], targets["moles"])
            raw = _apply_weights(raw, cfg.get("weights"))
            mask = predictions.get("mole_mask")
            loss = _masked_mean(raw, mask) if mask is not None else raw.mean()
            loss = loss * cfg.get("weight", 1.0)
            losses["mole_loss"] = loss
            total = total + loss
            continue

        if key == "bulk" and "bulk" in predictions:
            loss_fn = _select_loss_fn(cfg.get("type", "symmetric_rel_l2"))
            raw = loss_fn(predictions["bulk"], targets["bulk"])
            mask = predictions.get("bulk_mask")
            loss = _masked_mean(raw, mask) if mask is not None else raw.mean()
            loss = loss * cfg.get("weight", 1.0)
            losses["bulk_loss"] = loss
            total = total + loss
            continue

    losses["total_loss"] = total
    return losses
