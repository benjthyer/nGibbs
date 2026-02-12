"""Optimizer and scheduler factories."""

from __future__ import annotations

from typing import Dict, Optional

import torch


_SCHEDULERS = {
    "steplr": [torch.optim.lr_scheduler.StepLR, 'epoch', False],
    "cosine": [torch.optim.lr_scheduler.CosineAnnealingLR, 'epoch', False],
    "cosinewarm": [torch.optim.lr_scheduler.CosineAnnealingWarmRestarts, 'epoch', False],
    "plateau": [torch.optim.lr_scheduler.ReduceLROnPlateau, 'epoch', True],
}

class SchedulerWrapper:
    def __init__(self, scheduler=None, step_on='epoch', needs_metric=False):
        """
        step_on: "batch" or "epoch"
        needs_metric: whether scheduler.step() requires validation metric

        By default, this is a dummy object that does nothing, so you can call step_batch and step_epoch without checking if scheduler exists, using the same training functions. 
        Steps on epochs when scheduler is provided
        """
        if scheduler is None:
            self.scheduler = None
            self.step_on = None
            self.needs_metric = False
        else:
            self.scheduler = scheduler
            self.step_on = step_on
            self.needs_metric = needs_metric

    def step_batch(self):
        if self.step_on == "batch":
            self.scheduler.step()
        else:
            pass # No action needed if stepping on epoch or not at all

    def step_epoch(self, val_loss=None):
        if self.step_on == "epoch":
            if self.needs_metric:
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
        else: 
            pass # No action needed if stepping on batch or not at all

def create_optimizer(model: torch.nn.Module, lr: float = 1e-3, weight_decay: float = 0.0) -> torch.optim.Optimizer:
    """Create optimizer based on weight_decay: Adam if wd=0, else AdamW."""
    if weight_decay == 0:
        return torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    else:
        return torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)


def create_scheduler(optimizer: torch.optim.Optimizer, scheduler_name: str, **kwargs) -> Optional[SchedulerWrapper]:
    """Create scheduler from name and kwargs, wrapping it with step_on and needs_metric metadata."""
    if not scheduler_name:
        return None
    
    name = scheduler_name.lower()
    if name not in _SCHEDULERS:
        raise ValueError(f"Unknown scheduler: {name}. Available: {list(_SCHEDULERS.keys())}")
    
    scheduler_class, step_on, needs_metric = _SCHEDULERS[name]
    scheduler = scheduler_class(optimizer, **kwargs)
    return SchedulerWrapper(scheduler, step_on=step_on, needs_metric=needs_metric)
