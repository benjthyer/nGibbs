"""CLI entry point for training and tuning."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import yaml
from torch.utils.data import DataLoader

import sys # Perpare the path

src_path = str(Path(__file__).parent.parent.parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)
file_path = str(Path(__file__).parent)
if file_path not in sys.path:
    sys.path.insert(0, file_path)

from nMELTS.config import settings

from builder.training.config.training_config import TrainingConfig
from builder.training.datasets.loadTrainData import load_train_data
from builder.training.design.model_factory import create_model
from builder.training.optimizer_factory import create_optimizer, create_scheduler
from builder.training.trainer.lower_trainer import LowerTrainer
from builder.training.trainer.upper_trainer import UpperTrainer
from builder.training.tuning.optuna_objectives import objective_lower, objective_upper

try:
    import optuna
except ImportError:  # pragma: no cover - optional dependency
    optuna = None



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="nMELTS training CLI")
    parser.add_argument("command", choices=["train", "tune"], help="Action to perform")
    parser.add_argument("--config", required=True, help="Path to training.yaml")
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all", "lower", "upper", "finetune"],
        help="Training stage name",
    )
    parser.add_argument("--n-trials", type=int, default=None, help="Optuna trials")
    return parser


def _resolve_train_roots() -> List[Path]:
    roots: List[Path] = []
    internal_root = settings.INT_DATA_DIR / settings.TRAIN_PATH
    roots.append(internal_root)
    if settings.external_base:
        roots.append(Path(settings.external_base) / settings.TRAIN_PATH)
    return roots


def _resolve_bundle_path(bundle_name: str, search_roots: Iterable[Path]) -> str:
    bundle_path = Path(bundle_name)
    if bundle_path.is_absolute() or str(bundle_path.parent) not in {".", ""}:
        return str(bundle_path)

    for root in search_roots:
        if root is None:
            continue
        direct = root / bundle_name
        if direct.exists():
            return str(direct)
        direct_tar = root / f"{bundle_name}.tar.gz"
        if direct_tar.exists():
            return str(direct_tar)
        for candidate in root.rglob(f"{bundle_name}.tar.gz"):
            return str(candidate)

    if search_roots:
        return str(Path(next(iter(search_roots))) / bundle_name)
    return bundle_name


def _resolve_bundle_names(config_path: str, config: TrainingConfig) -> Tuple[str, str, Optional[str]]:
    with open(config_path, "r", encoding="utf-8") as handle:
        raw_cfg = yaml.safe_load(handle) or {}

    tarname = config.tarname or raw_cfg.get("tarname")
    if not tarname:
        raise ValueError("training.yaml must define tarname for bundle resolution")

    val_bundle = (config.data or {}).get("validation_bundle") or raw_cfg.get("validation_bundle")
    train_name = f"{tarname}_Train"
    test_name = f"{tarname}_Test"
    val_name = val_bundle
    return train_name, test_name, val_name


def _normalize_stage_config(stage_config: Dict) -> Dict:
    resolved: Dict = {}
    for key, value in stage_config.items():
        if isinstance(value, list):
            resolved[key] = value[0] if value else None
        else:
            resolved[key] = value
    return resolved


def _infer_search_space(stage_config: Dict) -> Dict:
    search_space: Dict = {}
    for key, value in stage_config.items():
        if isinstance(value, list) and value:
            search_space[key] = value
    return search_space


def _optimizer_config(training_cfg: Dict, stage_cfg: Dict, stage: str) -> Dict:
    cfg = deepcopy(training_cfg.get("optimizer", {})) if isinstance(training_cfg.get("optimizer", {}), dict) else {}
    cfg.setdefault("name", "adamw")
    cfg.setdefault("lr", training_cfg.get("learning_rate", 1e-3))
    if stage == "lower":
        cfg.setdefault("weight_decay", stage_cfg.get("lowWD", 0.0))
    else:
        cfg.setdefault("weight_decay", stage_cfg.get("highWD", 0.0))
    return cfg

def _stage_settings(training_strategy: Dict, stage: str) -> Dict:
    for entry in training_strategy.get("stages", []):
        if entry.get("name") == stage:
            return entry
    return {}

def _update_stage_config(stage_cfg: Dict, overrides: Dict) -> Dict:
    merged = deepcopy(stage_cfg)
    merged.update(overrides)
    return merged


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    defaults_path = Path(__file__).parent / "config" / "defaults.yaml"
    config = TrainingConfig.from_yaml(args.config, str(defaults_path))

    train_name, test_name, val_name = _resolve_bundle_names(args.config, config)
    roots = _resolve_train_roots()
    train_bundle = _resolve_bundle_path(train_name, roots)
    test_bundle = _resolve_bundle_path(test_name, roots)
    val_bundle = _resolve_bundle_path(val_name, roots) if val_name else test_bundle

    train_set, val_set, ml_indexer = load_train_data(
        train_bundle_path=train_bundle,
        test_bundle_path=val_bundle,
        only_vp=None,
    )

    device = "cuda" #if torch.cuda.is_available() else "cpu"

    # 
    binWeights = torch.ones(ml_indexer.nphases, dtype = torch.float32).reshape(1,-1)
    compWeights = torch.ones(ml_indexer.ncompsVaried, dtype = torch.float32).reshape(1,-1)
    for phase, W in weight_dict.items(): # Weights removed for 110 due to bad overfitting... Take a look at phase abundances??? 
        binWeights[:,ml_indexer.mass_phasedict[phase]] = W
        if phase in ml_indexer.compositionally_variable_phases:
            compWeights[:,ml_indexer.comp_phasedict[phase]] = W

    lower_model = None
    upper_model = None

    if "lower" in stage_order:
        tuned_params = {}
        if args.command == "tune" or (args.command == "train" and n_trials > 0):
            def trainer_factory(cfg):
                model = create_model(cfg["lower_model"], ml_indexer=ml_indexer, device=device)
                _freeze_for_lower(model)
                optimizer = create_optimizer(model, _optimizer_config(training_cfg, cfg["lower_model"], "lower"))
                scheduler = create_scheduler(optimizer, training_cfg.get("scheduler"))
                return LowerTrainer(
                    model=model,
                    optimizer=optimizer,
                    device=device,
                    scheduler=scheduler,
                    max_epochs=int(training_cfg.get("epochs", 1)),
                    early_stopping_patience=int(training_cfg.get("early_stopping_patience", 0)),
                    noise=float(cfg["lower_model"].get("noise", 0.0)),
                )

            tuned_params = _run_tuning("lower", base_config, train_loader, val_loader, trainer_factory, n_trials)

        if args.command == "train":
            stage_cfg = _update_stage_config(lower_config, tuned_params)
            lower_model = create_model(stage_cfg, ml_indexer=ml_indexer, device=device)
            _freeze_for_lower(lower_model)
            optimizer = create_optimizer(lower_model, _optimizer_config(training_cfg, stage_cfg, "lower"))
            scheduler = create_scheduler(optimizer, training_cfg.get("scheduler"))
            trainer = LowerTrainer(
                model=lower_model,
                optimizer=optimizer,
                device=device,
                scheduler=scheduler,
                max_epochs=int(training_cfg.get("epochs", 1)),
                early_stopping_patience=int(training_cfg.get("early_stopping_patience", 0)),
                noise=float(stage_cfg.get("noise", 0.0)),
            )
            trainer.fit(train_loader, val_loader)

    if "upper" in stage_order:
        tuned_params = {}
        if args.command == "tune" or (args.command == "train" and n_trials > 0):
            def trainer_factory(cfg):
                model = create_model(cfg["upper_model"], ml_indexer=ml_indexer, device=device)
                if lower_model is not None:
                    _copy_lower_weights(lower_model, model)
                _freeze_for_upper(model, freeze_encoder=bool(upper_stage_settings.get("freeze_encoder", True)))
                optimizer = create_optimizer(model, _optimizer_config(training_cfg, cfg["upper_model"], "upper"))
                scheduler = create_scheduler(optimizer, training_cfg.get("scheduler"))
                return UpperTrainer(
                    model=model,
                    optimizer=optimizer,
                    device=device,
                    scheduler=scheduler,
                    max_epochs=int(training_cfg.get("epochs", 1)),
                    early_stopping_patience=int(training_cfg.get("early_stopping_patience", 0)),
                    noise=float(cfg["upper_model"].get("noise", 0.0)),
                    loss_config=training_cfg.get("loss_config", {}),
                )

            tuned_params = _run_tuning("upper", base_config, train_loader, val_loader, trainer_factory, n_trials)

        if args.command == "train":
            stage_cfg = _update_stage_config(upper_config, tuned_params)
            upper_model = create_model(stage_cfg, ml_indexer=ml_indexer, device=device)
            if lower_model is not None:
                _copy_lower_weights(lower_model, upper_model)
            _freeze_for_upper(upper_model, freeze_encoder=bool(upper_stage_settings.get("freeze_encoder", True)))
            optimizer = create_optimizer(upper_model, _optimizer_config(training_cfg, stage_cfg, "upper"))
            scheduler = create_scheduler(optimizer, training_cfg.get("scheduler"))
            trainer = UpperTrainer(
                model=upper_model,
                optimizer=optimizer,
                device=device,
                scheduler=scheduler,
                max_epochs=int(training_cfg.get("epochs", 1)),
                early_stopping_patience=int(training_cfg.get("early_stopping_patience", 0)),
                noise=float(stage_cfg.get("noise", 0.0)),
                loss_config=training_cfg.get("loss_config", {}),
            )
            trainer.fit(train_loader, val_loader)

    if "finetune" in stage_order and args.command == "train":
        if upper_model is None:
            stage_cfg = upper_config
            upper_model = create_model(stage_cfg, ml_indexer=ml_indexer, device=device)
            if lower_model is not None:
                _copy_lower_weights(lower_model, upper_model)
        _unfreeze_all(upper_model)
        optimizer = create_optimizer(upper_model, _optimizer_config(training_cfg, upper_config, "upper"))
        scheduler = create_scheduler(optimizer, training_cfg.get("scheduler"))
        trainer = UpperTrainer(
            model=upper_model,
            optimizer=optimizer,
            device=device,
            scheduler=scheduler,
            max_epochs=int(training_cfg.get("epochs", 1)),
            early_stopping_patience=int(training_cfg.get("early_stopping_patience", 0)),
            noise=float(upper_config.get("noise", 0.0)),
            loss_config=training_cfg.get("loss_config", {}),
        )
        trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    main()
