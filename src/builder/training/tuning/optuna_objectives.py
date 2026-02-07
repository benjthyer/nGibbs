"""Optuna objective functions."""

from __future__ import annotations

from typing import Callable, Dict

from .search_space import apply_trial


def objective_lower(
    trial,
    base_config: Dict,
    train_loader,
    val_loader,
    trainer_factory: Callable,
) -> float:
    config = apply_trial(base_config, trial, stage="lower")
    trainer = trainer_factory(config)
    history = trainer.fit(train_loader, val_loader)
    return history["best_val_loss"]


def objective_upper(
    trial,
    base_config: Dict,
    train_loader,
    val_loader,
    trainer_factory: Callable,
) -> float:
    config = apply_trial(base_config, trial, stage="upper")
    trainer = trainer_factory(config)
    history = trainer.fit(train_loader, val_loader)
    return history["best_val_loss"]
