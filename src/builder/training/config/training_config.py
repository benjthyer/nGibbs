"""Training config loading utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


@dataclass
class TrainingConfig:
    tarname: Optional[str]
    lower_model: Dict[str, Any]
    upper_model: Dict[str, Any]
    data: Dict[str, Any]
    training: Dict[str, Any]
    optuna: Dict[str, Any]
    training_strategy: Dict[str, Any]

    @classmethod
    def from_yaml(cls, path: str, defaults_path: str) -> "TrainingConfig":
        with open(defaults_path, "r", encoding="utf-8") as handle:
            defaults = yaml.safe_load(handle) or {}
        with open(path, "r", encoding="utf-8") as handle:
            overrides = yaml.safe_load(handle) or {}
        merged = _deep_update(defaults, overrides)
        return cls(
            tarname=merged.get("tarname"),
            lower_model=merged.get("lower_model", {}),
            upper_model=merged.get("upper_model", {}),
            data=merged.get("data", {}),
            training=merged.get("training", {}),
            optuna=merged.get("optuna", {}),
            training_strategy=merged.get("training_strategy", {}),
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "tarname": self.tarname,
            "lower_model": self.lower_model,
            "upper_model": self.upper_model,
            "data": self.data,
            "training": self.training,
            "optuna": self.optuna,
            "training_strategy": self.training_strategy,
        }
