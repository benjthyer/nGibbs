"""Trainer for chemistry and molar outputs."""

from __future__ import annotations

from typing import Dict, Optional

import torch

from ..design.loss_factory import compute_losses
from .base_trainer import BaseTrainer


class UpperTrainer(BaseTrainer):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda",
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        max_epochs: int = 1,
        early_stopping_patience: int = 0,
        noise: float = 0.0,
        loss_config: Optional[Dict] = None,
    ) -> None:
        super().__init__(model, optimizer, device, scheduler, max_epochs, early_stopping_patience)
        self.noise = noise
        self.loss_config = loss_config or {}

    def train_epoch(self, train_loader) -> float:
        self.model.train()
        running = 0.0
        count = 0
        for features, binaries, chemistries, moles, *_ in train_loader:
            features = features.to(self.device, non_blocking=True)
            binaries = binaries.to(self.device, non_blocking=True)
            chemistries = chemistries.to(self.device, non_blocking=True)
            moles = moles.to(self.device, non_blocking=True)

            if self.noise:
                features = features + (features * torch.randn_like(features) * self.noise)

            self.optimizer.zero_grad()
            logits, chem_preds, chem_mask, mole_preds, bulk_preds = self.model(
                features, binaries=binaries, NN_only=True
            )

            bulk_mask = (features != 0).to(torch.float32)[:, 3:]
            mole_mask = (binaries > 0.5).to(torch.float32)

            predictions = {
                "binary_logits": logits,
                "chemistry": chem_preds,
                "moles": mole_preds,
                "bulk": bulk_preds,
                "chem_mask": chem_mask,
                "mole_mask": mole_mask,
                "bulk_mask": bulk_mask,
            }
            targets = {
                "binaries": binaries,
                "chemistries": chemistries,
                "moles": moles,
                "bulk": features[:, 3:],
            }
            losses = compute_losses(predictions, targets, None, self.loss_config)
            loss = losses["total_loss"]
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
            for features, binaries, chemistries, moles, *_ in val_loader:
                features = features.to(self.device, non_blocking=True)
                binaries = binaries.to(self.device, non_blocking=True)
                chemistries = chemistries.to(self.device, non_blocking=True)
                moles = moles.to(self.device, non_blocking=True)

                logits, chem_preds, chem_mask, mole_preds, bulk_preds = self.model(
                    features, binaries=binaries, NN_only=True
                )
                bulk_mask = (features != 0).to(torch.float32)[:, 3:]
                mole_mask = (binaries > 0.5).to(torch.float32)
                predictions = {
                    "binary_logits": logits,
                    "chemistry": chem_preds,
                    "moles": mole_preds,
                    "bulk": bulk_preds,
                    "chem_mask": chem_mask,
                    "mole_mask": mole_mask,
                    "bulk_mask": bulk_mask,
                }
                targets = {
                    "binaries": binaries,
                    "chemistries": chemistries,
                    "moles": moles,
                    "bulk": features[:, 3:],
                }
                losses = compute_losses(predictions, targets, None, self.loss_config)
                loss = losses["total_loss"]
                running += loss.item() * features.shape[0]
                count += features.shape[0]
        return running / max(count, 1)
