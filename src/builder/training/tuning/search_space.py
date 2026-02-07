"""Optuna search space helpers."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict


def _suggest(trial, name: str, spec: Any):
    if isinstance(spec, list):
        return trial.suggest_categorical(name, spec)
    if isinstance(spec, dict):
        spec_type = spec.get("type", "categorical")
        if spec_type == "int":
            return trial.suggest_int(name, spec["low"], spec["high"], step=spec.get("step", 1))
        if spec_type == "float":
            return trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
        if spec_type == "categorical":
            return trial.suggest_categorical(name, spec.get("choices", []))
    raise ValueError(f"Invalid search space spec for {name}")


def apply_trial(base_config: Dict[str, Any], trial, stage: str) -> Dict[str, Any]:
    config = deepcopy(base_config)
    search_space = config.get("optuna", {}).get("search_space", {}).get(stage, {})
    stage_key = f"{stage}_model"
    stage_config = config.get(stage_key, {})

    for param, spec in search_space.items():
        stage_config[param] = _suggest(trial, param, spec)

    config[stage_key] = stage_config
    return config
