"""Optimizer and scheduler factories."""

from __future__ import annotations

from typing import Dict, Optional, Any
from xml.parsers.expat import model

import torch


_SCHEDULERS = {
    "steplr": [torch.optim.lr_scheduler.StepLR, 'epoch', False],
    "cosine": [torch.optim.lr_scheduler.CosineAnnealingLR, 'epoch', False],
    "cosinewarm": [torch.optim.lr_scheduler.CosineAnnealingWarmRestarts, 'epoch', False],
    "plateau": [torch.optim.lr_scheduler.ReduceLROnPlateau, 'epoch', True],
    "reducelronplateau": [torch.optim.lr_scheduler.ReduceLROnPlateau, 'epoch', True],
    "cosineannealinglr": [torch.optim.lr_scheduler.CosineAnnealingLR, 'epoch', False],
    "cosineannealingwarmrestarts": [torch.optim.lr_scheduler.CosineAnnealingWarmRestarts, 'epoch', False]
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
            print(f"Stepping scheduler on batch. Current LR: {self.scheduler.get_last_lr()}")
        else:
            pass # No action needed if stepping on epoch or not at all

    def step_epoch(self, val_loss=None):
        if self.step_on == "epoch":
            if self.needs_metric:
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
            print(f"Stepping scheduler on epoch. Current LR: {self.scheduler.get_last_lr()}")
        else: 
            pass # No action needed if stepping on batch or not at all

def _coerce_numeric_types(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Coerce string representations of numbers to int/float.
    
    This ensures robust type handling when args come from YAML, env vars, 
    or config merging that might stringify numeric values.
    
    Parameters
    ----------
    kwargs : Dict[str, Any]
        Dictionary of keyword arguments
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with numeric strings converted to int/float
    """
    coerced = {}
    for key, value in kwargs.items():
        if isinstance(value, str):
            # Try to convert string to numeric type
            try:
                # Check if it contains a decimal point or scientific notation
                if '.' in value or 'e' in value.lower():
                    coerced[key] = float(value)
                else:
                    # Try int first, fall back to float if needed
                    try:
                        coerced[key] = int(value)
                    except ValueError:
                        coerced[key] = float(value)
            except (ValueError, AttributeError):
                # Not a numeric string, keep as-is
                coerced[key] = value
        elif isinstance(value, dict):
            # Recursively handle nested dictionaries
            coerced[key] = _coerce_numeric_types(value)
        else:
            # Already correct type or non-string type
            coerced[key] = value
    return coerced

# Gather layers to exclude from weight decay (biases and norm layers)
norm_classes = (
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
    torch.nn.LayerNorm,
    torch.nn.GroupNorm,
    torch.nn.InstanceNorm1d,
    torch.nn.InstanceNorm2d,
    torch.nn.InstanceNorm3d,
)

def create_optimizer(model: torch.nn.Module, lr: float = 1e-3, lowWD: float = 0.0, highWD: float = 0.0, amsgrad: bool = False, eps: float = 1e-8) -> torch.optim.Optimizer:
    """Create optimizer based on weight_decay: Adam if wd=0, else AdamW."""
    if float(lowWD) + float(highWD) == 0:
        return torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, amsgrad=amsgrad, eps=eps)
    else:
        # Separate low and high decay and protect some heads
        no_decay = set()
        low_wd = set()
        high_wd = set()
        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):

                if not param.requires_grad:
                    continue

                full_name = f"{module_name}.{param_name}" if module_name else param_name

                # --- Global exclusions ---
                if isinstance(module, norm_classes) or param_name.endswith("bias"):
                    no_decay.add(full_name)
                    continue

                # --- Route by top-level owner ---
                top = module_name.split('.')[0] if module_name else ""

                if top == "mole_head":
                    no_decay.add(full_name)

                elif top in ("encoder", "sat_head"):
                    low_wd.add(full_name)

                elif top in ("middleBrain", "chem_heads"):
                    high_wd.add(full_name)

                else:
                    raise ValueError(f"Unassigned parameter: {full_name}")



        param_dict = dict(model.named_parameters())

        trainable = {
            name for name, p in param_dict.items()
            if p.requires_grad
        }

        low_wd_params  = [param_dict[n] for n in sorted(low_wd)]
        high_wd_params = [param_dict[n] for n in sorted(high_wd)]
        no_decay_params = [param_dict[n] for n in sorted(no_decay)]

        all_grouped = low_wd | high_wd | no_decay
        all_params  = set(param_dict.keys())

        # Make sure groups are mutually exclusive and exhaustive
        assert len(low_wd & high_wd) == 0
        assert len(low_wd & no_decay) == 0
        assert len(high_wd & no_decay) == 0

        total_grouped = len(low_wd) + len(high_wd) + len(no_decay)
        assert total_grouped == len(trainable)

        return torch.optim.AdamW([
            {"params": low_wd_params,  "weight_decay": lowWD},
            {"params": high_wd_params, "weight_decay": highWD},
            {"params": no_decay_params,"weight_decay": 0.0},
        ], lr=lr, amsgrad=amsgrad, eps=eps)


def create_scheduler(optimizer: torch.optim.Optimizer, scheduler_name: str, **kwargs) -> Optional[SchedulerWrapper]:
    """
    Create scheduler from name and kwargs, wrapping it with step_on and needs_metric metadata.
    
    Automatically coerces string representations of numeric values to int/float.
    """
    if not scheduler_name:
        return None
    
    name = scheduler_name.lower()
    if name not in _SCHEDULERS:
        raise ValueError(f"Unknown scheduler: {name}. Available: {list(_SCHEDULERS.keys())}")
    
    # Coerce any string numeric values to proper types
    coerced_kwargs = _coerce_numeric_types(kwargs)
    
    scheduler_class, step_on, needs_metric = _SCHEDULERS[name]
    scheduler = scheduler_class(optimizer, **coerced_kwargs)
    return SchedulerWrapper(scheduler, step_on=step_on, needs_metric=needs_metric)
