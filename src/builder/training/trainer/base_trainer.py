"""Base trainer implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional

import torch


class BaseTrainer(ABC):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda",
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        max_epochs: int = 1,
        early_stopping_patience: int = 0,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience

    @abstractmethod
    def train_epoch(self, train_loader) -> float:
        raise NotImplementedError

    @abstractmethod
    def validate(self, val_loader) -> float:
        raise NotImplementedError

    def fit(self, train_loader, val_loader) -> Dict[str, list]:
        history = {"train_loss": [], "val_loss": []}
        best_val = float("inf")
        patience = 0

        for _ in range(self.max_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                patience = 0
            else:
                patience += 1
                if self.early_stopping_patience and patience > self.early_stopping_patience:
                    break

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

        history["best_val_loss"] = best_val
        return history
