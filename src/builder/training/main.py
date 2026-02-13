"""CLI entry point for training and tuning."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import sys

import numpy as np
import torch
import yaml

src_path = str(Path(__file__).parent.parent.parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)
file_path = str(Path(__file__).parent)
if file_path not in sys.path:
    sys.path.insert(0, file_path)

from nMELTS.config import settings
import nMELTS.engine.NN as NN

from builder.training.loadTrainData import load_ML_data
from builder.training.trainer import train_Lower_MELTS, train_Upper_MELTS, symmetric_rel_l1, symmetric_rel_l2
from builder.training.tuners import tune_Lower_MELTS, tune_Upper_MELTS
from builder.training.logger import setup_training_logger, redirect_output, restore_output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="nMELTS training CLI")
    parser.add_argument("command", nargs="?", default="train", choices=["train", "tune"], help="Action to perform")
    parser.add_argument("--config", required=True, help="Path to training.yaml")
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all", "lower", "upper", "finetune"],
        help="Training stage name",
    )
    return parser


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _resolve_train_roots() -> List[Path]:
    roots: List[Path] = []
    internal_root = settings.INT_DATA_DIR / settings.TRAIN_PATH
    roots.append(internal_root)
    if settings.external_base:
        roots.append(Path(settings.external_base) / settings.TRAIN_PATH)
    return roots


def _strip_tar_suffix(path_str: str) -> str:
    if path_str.endswith(".tar.gz"):
        return path_str[:-7]
    return path_str


def _resolve_bundle_path(bundle_name: str, search_roots: Iterable[Path]) -> str:
    bundle_path = Path(bundle_name)
    if bundle_path.is_absolute() or str(bundle_path.parent) not in {".", ""}:
        return _strip_tar_suffix(str(bundle_path))

    for root in search_roots:
        if root is None:
            continue
        direct = root / bundle_name
        direct_tar = root / f"{bundle_name}.tar.gz"
        if direct_tar.exists():
            return _strip_tar_suffix(str(direct_tar))
        if direct.exists():
            return _strip_tar_suffix(str(direct))
        for candidate in root.rglob(f"{bundle_name}.tar.gz"):
            return _strip_tar_suffix(str(candidate))

    if search_roots:
        return _strip_tar_suffix(str(Path(next(iter(search_roots))) / bundle_name))
    return _strip_tar_suffix(bundle_name)


def _normalize_scheduler_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    return name.strip().lower()


def _build_base_config(defaults: Dict[str, Any]) -> Dict[str, Any]:
    lower_defaults = deepcopy(defaults.get("lower_model", {}))
    upper_defaults = defaults.get("upper_model", {})

    for key in ["middleLayerUp", "middleLayerDown", "high_regularization", "highWD"]:
        if key in upper_defaults:
            lower_defaults[key] = upper_defaults[key]

    for key, value in upper_defaults.items():
        lower_defaults.setdefault(key, value)

    return lower_defaults


def _load_warm_start(path: str, ml_indexer) -> NN.MidLevelNetwork:
    ckpt = torch.load(path, map_location="cpu")
    config = ckpt.get("config", {})
    model = NN.MidLevelNetwork(**config, ml_indexer=ml_indexer)
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    return model


def _loss_fn_from_type(type_name: Optional[str]):
    if not type_name:
        return symmetric_rel_l2
    name = type_name.strip().lower()
    if name == "symmetric_rel_l1":
        return symmetric_rel_l1
    if name == "symmetric_rel_l2":
        return symmetric_rel_l2
    raise ValueError(f"Unknown loss type: {type_name}")


def _build_phase_weights(ml_indexer, weight_dict: Optional[Dict[str, float]]):
    binWeights = torch.ones(ml_indexer.nphases, dtype=torch.float32).reshape(1, -1)
    compWeights = torch.ones(ml_indexer.ncompsVaried, dtype=torch.float32).reshape(1, -1)
    if weight_dict:
        for phase, weight in weight_dict.items():
            if phase in ml_indexer.mass_phasedict:
                binWeights[:, ml_indexer.mass_phasedict[phase]] = weight
            if phase in ml_indexer.compositionally_variable_phases:
                compWeights[:, ml_indexer.comp_phasedict[phase]] = weight
    return binWeights, compWeights


def _stage_allowed(selected_stage: str, stage_name: str) -> bool:
    if selected_stage == "all":
        return True
    return selected_stage == stage_name


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    defaults_path = Path(__file__).parent / "config" / "defaults.yaml"
    with open(defaults_path, "r", encoding="utf-8") as handle:
        defaults = yaml.safe_load(handle) or {}
    with open(args.config, "r", encoding="utf-8") as handle:
        overrides = yaml.safe_load(handle) or {}

    config = _deep_update(deepcopy(defaults), overrides)

    tarname = config.get("tarname")
    if not tarname:
        raise ValueError("training.yaml must define tarname")

    # Set up log directory
    log_dir = config.get("training", {}).get("log_dir", "src/builder/training/logs")
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    only_vp = config.get("data", {}).get("only_vp")
    roots = _resolve_train_roots()
    train_bundle = _resolve_bundle_path(f"{tarname}_Train", roots)
    test_bundle = _resolve_bundle_path(f"{tarname}_Test", roots)

    train_set, ml_indexer = load_ML_data(train_bundle, only_VP=only_vp)
    test_set, _ = load_ML_data(test_bundle, only_VP=only_vp)

    device = config.get("training", {}).get("device", "cuda")

    base_config = _build_base_config(defaults)
    warm_start = config.get("warm_start", 'None')
    if warm_start.lower() != 'none':
        model = _load_warm_start(warm_start, ml_indexer)
    else:
        model = NN.MidLevelNetwork(**base_config, ml_indexer=ml_indexer)

    loss_config = config.get("training", {}).get("loss_config", {})
    chem_cfg = loss_config.get("chemistry", {})
    mole_cfg = loss_config.get("moles", {})
    bulk_cfg = loss_config.get("bulk", {})
    chem_alpha = float(chem_cfg.get("weight", 1.0))
    mole_alpha = float(mole_cfg.get("weight", 1.0))
    bulk_alpha = float(bulk_cfg.get("weight", 0.0))
    criterion = _loss_fn_from_type(chem_cfg.get("type"))

    checkpoints = config.get("training", {}).get("checkpoints", {})
    lower_ckpt = checkpoints.get("lower_path")
    upper_ckpt = checkpoints.get("upper_path")
    finetune_ckpt = checkpoints.get("finetune_path")

    tune_cfg = config.get("tune")
    if tune_cfg and args.command in {"train", "tune"}:
        if _stage_allowed(args.stage, "lower") and "lower_model" in tune_cfg:
            lower_tune = tune_cfg.get("lower_model", {})
            tune_params = lower_tune.get("tune_params", {})
            model_params = lower_tune.get("model_params", lower_tune)
            
            # Set up logging for lower tuning
            logger = setup_training_logger(str(log_dir), "tune_lower", args.command)
            original_stdout = sys.stdout
            redirect_output(logger)
            
            try:
                model, _ = tune_Lower_MELTS(
                    Model=model,
                    trainData=train_set,
                    testData=test_set,
                    lr=float(tune_params.get("learning_rate", config.get("training", {}).get("learning_rate", 1e-3))),
                    Epochs=int(tune_params.get("epochs", config.get("training", {}).get("epochs", 1))),
                    batch_size=int(tune_params.get("batch_size", config.get("data", {}).get("batch_size", 1024))),
                    early_stopping_patience=int(tune_params.get("early_stopping_patience", config.get("training", {}).get("early_stopping_patience", 0))),
                    Param_Dict=model_params,
                    max_N=float(tune_params.get("max_N", config.get("training", {}).get("max_N_tune", 1e6))),
                )
            finally:
                restore_output(original_stdout)
                logger.close()

        if _stage_allowed(args.stage, "upper") and "upper_model" in tune_cfg:
            upper_tune = tune_cfg.get("upper_model", {})
            tune_params = upper_tune.get("tune_params", {})
            model_params = upper_tune.get("model_params", upper_tune)
            binWeights, compWeights = _build_phase_weights(ml_indexer, None)
            
            # Set up logging for upper tuning
            logger = setup_training_logger(str(log_dir), "tune_upper", args.command)
            original_stdout = sys.stdout
            redirect_output(logger)
            
            try:
                model, _ = tune_Upper_MELTS(
                    Model=model,
                    trainData=train_set,
                    testData=test_set,
                    lr=float(tune_params.get("learning_rate", config.get("training", {}).get("learning_rate", 1e-3))),
                    Epochs=int(tune_params.get("epochs", config.get("training", {}).get("epochs", 1))),
                    batch_size=int(tune_params.get("batch_size", config.get("data", {}).get("batch_size", 1024))),
                    early_stopping_patience=int(tune_params.get("early_stopping_patience", config.get("training", {}).get("early_stopping_patience", 0))),
                    Param_Dict=model_params,
                    binWeights=binWeights,
                    compWeights=compWeights,
                    max_N=float(tune_params.get("max_N", config.get("training", {}).get("max_N_tune", 1e6))),
                )
            finally:
                restore_output(original_stdout)
                logger.close()

        if args.command == "tune":
            return

    stage_keys = [key for key in config.keys() if key.startswith("train")]
    stage_keys.sort(key=lambda name: int("".join(filter(str.isdigit, name)) or 0))

    for stage_key in stage_keys:
        stage_cfg = config.get(stage_key, {}) or {}
        strategy = stage_cfg.get("strategy")
        if strategy not in {"lower", "upper", "finetune"}:
            continue
        if not _stage_allowed(args.stage, strategy):
            continue

        scheduler_cfg = stage_cfg.get("scheduler", {}) or {}
        scheduler_name = _normalize_scheduler_name(scheduler_cfg.get("type"))
        scheduler_kwargs = scheduler_cfg.get("args", {}) or {}

        max_n = stage_cfg.get("max_N", config.get("training", {}).get("max_N"))
        max_n = np.inf if max_n in {None, "", "None"} else max_n

        batch_size = int(stage_cfg.get("batch_size", config.get("data", {}).get("batch_size", 1024)))
        lr = float(stage_cfg.get("learning_rate", config.get("training", {}).get("learning_rate", 1e-3)))
        epochs = int(stage_cfg.get("epochs", config.get("training", {}).get("epochs", 1)))
        patience = int(stage_cfg.get("early_stopping_patience", config.get("training", {}).get("early_stopping_patience", 3)))

        if strategy == "lower":
            # Set up logging for lower training
            logger = setup_training_logger(str(log_dir), f"{stage_key}_lower", args.command)
            original_stdout = sys.stdout
            redirect_output(logger)
            
            try:
                train_Lower_MELTS(
                    model,
                    train_set,
                    test_set,
                    scheduler=scheduler_name,
                    scheduler_kwargs=scheduler_kwargs,
                    batch_size=batch_size,
                    lr=lr,
                    Epochs=epochs,
                    device=device,
                    max_N=max_n,
                    early_stopping_patience=patience,
                    DictFilePath=lower_ckpt,
                )
            finally:
                restore_output(original_stdout)
                logger.close()
            continue

        weight_dict = stage_cfg.get("weight_dict")
        if weight_dict is None and isinstance(scheduler_cfg, dict):
            weight_dict = scheduler_cfg.get("weight_dict")

        binWeights, compWeights = _build_phase_weights(ml_indexer, weight_dict)
        which_heads_to_freeze = [] if strategy == "finetune" else ["sat_head", "encoder"]
        DictFilePath = finetune_ckpt if strategy == "finetune" else upper_ckpt

        # Set up logging for upper/finetune training
        logger = setup_training_logger(str(log_dir), f"{stage_key}_{strategy}", args.command)
        original_stdout = sys.stdout
        redirect_output(logger)
        
        try:
            train_Upper_MELTS(
                model,
                train_set,
                test_set,
                scheduler=scheduler_name,
                scheduler_kwargs=scheduler_kwargs,
                criterion=criterion,
                chem_alpha=chem_alpha,
                mole_alpha=mole_alpha,
                bulk_alpha=bulk_alpha,
                Epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                binWeights=binWeights,
                compWeights=compWeights,
                full_test_set=test_set,
                device=device,
                max_N=max_n,
                early_stopping_patience=patience,
                which_heads_to_freeze=which_heads_to_freeze,
                DictFilePath=DictFilePath,
            )
        finally:
            restore_output(original_stdout)
            logger.close()


if __name__ == "__main__":
    main()
