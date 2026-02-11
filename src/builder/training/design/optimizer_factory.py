"""Optimizer and scheduler factories."""

from __future__ import annotations

from typing import Dict, Optional

import torch


_OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
}


def create_optimizer(model: torch.nn.Module, config: Dict) -> torch.optim.Optimizer:
    name = config.get("name", "adamw").lower()
    if name not in _OPTIMIZERS:
        raise ValueError(f"Unknown optimizer: {name}")
    lr = float(config.get("lr", 1e-3))
    weight_decay = float(config.get("weight_decay", 0.0))
    return _OPTIMIZERS[name](model.parameters(), lr=lr, weight_decay=weight_decay)


def create_scheduler(optimizer: torch.optim.Optimizer, config: Optional[Dict]) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    if not config:
        return None
    name = config.get("name", "").lower()
    if name == "steplr":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get("step_size", 10),
            gamma=config.get("gamma", 0.1),
        )
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.get("t_max", 10),
        )
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=config.get("factor", 0.5),
            patience=config.get("patience", 2),
        )
    return None
