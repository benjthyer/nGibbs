"""Loss constraint utilities."""

from __future__ import annotations

import torch


def symmetric_rel_l2(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Elementwise symmetric relative L2 error."""
    denom = pred.pow(2) + target.pow(2) + eps
    return (pred - target).pow(2) / denom
