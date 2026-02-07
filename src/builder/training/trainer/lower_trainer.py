"""Trainer for binary saturation heads."""

from __future__ import annotations

from typing import Optional

import torch

from .base_trainer import BaseTrainer


class LowerTrainer(BaseTrainer):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda",
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        max_epochs: int = 1,
        early_stopping_patience: int = 0,
        noise: float = 0.0,
    ) -> None:
        super().__init__(model, optimizer, device, scheduler, max_epochs, early_stopping_patience)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.noise = noise

    def train_epoch(self, train_loader) -> float:
        self.model.train()
        running = 0.0
        count = 0
        for features, binaries, *_ in train_loader:
            features = features.to(self.device, non_blocking=True)
            binaries = binaries.to(self.device, non_blocking=True)
            if self.noise:
                features = features + (features * torch.randn_like(features) * self.noise)

            self.optimizer.zero_grad()
            logits = self.model.forward_binaries(features)
            loss = self.criterion(logits, binaries)
            loss.backward()
            self.optimizer.step()
            running += loss.item() * features.shape[0]
            count += features.shape[0]
        return running / max(count, 1)

    def validate(self, val_loader) -> float:
        self.model.eval()
        running = 0.0
        count = 0
        with torch.no_grad():
            for features, binaries, *_ in val_loader:
                features = features.to(self.device, non_blocking=True)
                binaries = binaries.to(self.device, non_blocking=True)
                logits = self.model.forward_binaries(features)
                loss = self.criterion(logits, binaries)
                running += loss.item() * features.shape[0]
                count += features.shape[0]
        return running / max(count, 1)
