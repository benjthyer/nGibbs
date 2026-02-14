"""CLI entry point for training and tuning with sequential episode orchestration."""

from __future__ import annotations

import argparse
import tarfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import sys
import re

import numpy as np
import torch
import yaml

src_path = str(Path(__file__).parent.parent.parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)
file_path = str(Path(__file__).parent)
if file_path not in sys.path:
    sys.path.insert(0, file_path)
base_path = str(Path(__file__).parent.parent.parent.parent)
if base_path not in sys.path:
    sys.path.insert(0, base_path)

from config import settings
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
    """Recursive Config updates!"""
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


def _read_text_file(path: Optional[Path]) -> Optional[str]:
    if not path:
        return None
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError:
        return None


def _read_bundle_metadata(bundle_path: str) -> Tuple[Optional[str], Optional[str]]:
    tar_path = Path(bundle_path)
    if not str(tar_path).endswith(".tar.gz"):
        tar_path = Path(f"{bundle_path}.tar.gz")

    if not tar_path.exists():
        return None, None

    bundle_yaml = None
    stats_text = None
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            for member in tar.getmembers():
                name = Path(member.name).name
                if stats_text is None and name == "stats.txt":
                    handle = tar.extractfile(member)
                    if handle:
                        stats_text = handle.read().decode("utf-8", errors="ignore")
                if bundle_yaml is None and name.lower().endswith(".yaml"):
                    handle = tar.extractfile(member)
                    if handle:
                        bundle_yaml = handle.read().decode("utf-8", errors="ignore")
                if stats_text is not None and bundle_yaml is not None:
                    break
    except tarfile.TarError:
        return None, None

    if stats_text is None:
        stats_path = Path(bundle_path).with_name(f"{Path(bundle_path).name}_stats.txt")
        stats_text = _read_text_file(stats_path)

    return bundle_yaml, stats_text


def _normalize_scheduler_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    return name.strip().lower()


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
    if name == 'huber':
        return torch.nn.HuberLoss(reduction='mean', delta=0.3)
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


def _discover_episodes(config: Dict[str, Any]) -> List[Tuple[str, str, Dict[str, Any]]]:
    """
    Discover and order all numbered episodes (tune1, train1, tune2, train2, etc.).
    
    Returns list of (episode_key, episode_type, episode_config) tuples in execution order.
    Episode type is 'tune' or 'train'.
    """
    episodes = []
    
    # Find all episode keys
    tune_keys = {k: v for k, v in config.items() if re.match(r'^tune\d+$', k)}
    train_keys = {k: v for k, v in config.items() if re.match(r'^train\d+$', k)}
    
    # Extract episode numbers
    def get_episode_num(key: str) -> int:
        match = re.search(r'\d+', key)
        return int(match.group()) if match else 999
    
    # Interleave tune and train in episode number order
    all_episodes = {}
    for key, cfg in tune_keys.items():
        num = get_episode_num(key)
        all_episodes.setdefault(num, []).append(('tune', key, cfg))
    
    for key, cfg in train_keys.items():
        num = get_episode_num(key)
        all_episodes.setdefault(num, []).append(('train', key, cfg))
    
    # Sort by episode number, then tune before train within same episode
    for num in sorted(all_episodes.keys()):
        # Sort so 'tune' comes before 'train' for same episode number
        for ep_type, ep_key, ep_cfg in sorted(all_episodes[num], key=lambda x: (x[0] != 'tune',)):
            episodes.append((ep_key, ep_type, ep_cfg))
    
    return episodes


def _resolve_dict_filepath(
    episode_key: str,
    config: Dict[str, Any],
    tarname: str,
    default_base: Optional[Path] = None
) -> Path:
    """
    Resolve or generate DictFilePath for model checkpoint/final save.
    
    If episode_config has DictFilePath, use it.
    Otherwise, use default base directory with auto-enumeration by stage.
    
    Parameters
    ----------
    episode_key : str
        Episode key (e.g., 'train1', 'tune2')
    config : Dict
        Episode config from YAML
    tarname : str
        Base tarname for naming
    default_base : Path, optional
        Default base directory. If None, uses nMELTS/TrainedModel
    
    Returns
    -------
    Path
        Full path where model should be saved
    """

    # auto-generate in default directory
    if default_base is None:
        default_base = Path(__file__).parent.parent.parent / 'nMELTS' / 'engine' / 'TrainedModel'
    
    default_base.mkdir(parents=True, exist_ok=True)

    # If explicitly provided in episode config, use it
    if isinstance(config, dict) and config.get('DictFilePath'):
        path = Path(config['DictFilePath'])
        if path.suffix == "":
            path = path.with_suffix(".tar")
        return default_base / path.name
    
    # Generate name: {tarname}_{episode_key}.tar
    return default_base / f"{tarname}_{episode_key}.tar"


def _extract_best_loss(tune_result: Tuple) -> Optional[float]:
    """Extract best_loss from tune result tuple."""
    # tune returns (model, best_loss)
    if isinstance(tune_result, tuple) and len(tune_result) >= 2:
        return tune_result[1]
    return None


def _merge_locked_config(
    previous_best_config: Dict[str, Any],
    current_tune_params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Merge previous episode's best config with current tune parameters.
    
    Locked config: use all items from previous_best_config, but allow
    current_tune_params to override or add new keys.
    
    This implements config locking where previous optimizations are preserved.
    """
    merged = deepcopy(previous_best_config)
    
    # Add/override with new parameters from current episode
    for key, value in current_tune_params.items():
        merged[key] = value
    
    return merged


def main() -> None:
    """
    Main orchestration for sequential train/tune episodes from YAML.
    
    Supported episode types: tune1, tune2, ..., train1, train2, ...
    Episodes are executed in order, with state propagation between them:
    - Warm-start: Previous episode's best model initializes next episode
    - Config locking: Previous tune's best config carries to next tune
    - Baseline propagation: Previous tune's best_loss passed to next tune
    """
    parser = build_parser()
    args = parser.parse_args()

    # Load configuration
    defaults_path = Path(__file__).parent / "config" / "defaults.yaml"
    with open(defaults_path, "r", encoding="utf-8") as handle:
        defaults = yaml.safe_load(handle) or {}
    with open(args.config, "r", encoding="utf-8") as handle:
        overrides = yaml.safe_load(handle) or {}

    config = _deep_update(deepcopy(defaults), overrides)
    config_yaml_text = yaml.safe_dump(config, sort_keys=False)
    training_yaml_text = _read_text_file(Path(args.config))

    tarname = config.get("tarname")
    if not tarname:
        raise ValueError("training.yaml must define tarname")

    # Set up logging
    log_dir = config.get("training", {}).get("log_dir", "src/builder/training/logs")
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    only_vp = config.get("data", {}).get("only_vp")
    roots = _resolve_train_roots()
    train_bundle = _resolve_bundle_path(f"{tarname}_Train", roots)
    test_bundle = _resolve_bundle_path(f"{tarname}_Test", roots)
    bundle_yaml_text, stats_text = _read_bundle_metadata(train_bundle)
    if bundle_yaml_text:
        processing_yaml_text = bundle_yaml_text
    train_set, ml_indexer = load_ML_data(train_bundle, only_VP=only_vp)
    test_set, _ = load_ML_data(test_bundle, only_VP=only_vp)

    device = config.get("training", {}).get("device", "cuda")
    
    # Loss configuration
    loss_config = config.get("training", {}).get("loss_config", {})
    chem_cfg = loss_config.get("chemistry", {})
    mole_cfg = loss_config.get("moles", {})
    bulk_cfg = loss_config.get("bulk", {})
    chem_alpha = float(chem_cfg.get("weight", 1.0))
    mole_alpha = float(mole_cfg.get("weight", 1.0))
    bulk_alpha = float(bulk_cfg.get("weight", 0.0))
    criterion = _loss_fn_from_type(chem_cfg.get("type"))

    # Initialize model
    model_config = deepcopy(config.get("model", {}))
    warm_start = config.get("warm_start", 'None')
    if warm_start.lower() != 'none':
        model = NN.rebuild_MELTS_model(warm_start, ml_indexer=ml_indexer)
    else:
        model = NN.MidLevelNetwork(**model_config, ml_indexer=ml_indexer)

    # State for sequential episodes
    best_model = model
    best_loss = None
    best_config = {}

    # Discover and execute episodes in order
    episodes = _discover_episodes(config)
    
    for episode_key, episode_type, episode_cfg in episodes:
        if episode_cfg is None:
            continue
        
        if not _stage_allowed(args.stage, episode_type):
            continue

        print(f"\n{'='*60}")
        print(f"Episode: {episode_key} ({episode_type})")
        print(f"{'='*60}")

        # Determine DictFilePath for saving
        dict_filepath = _resolve_dict_filepath(episode_key, episode_cfg, tarname)
        print(f"Model save path: {dict_filepath}")

        if episode_type == 'tune':
            # ===== TUNING EPISODE =====
            tune_params = episode_cfg.get("tune_params", {})
            model_params = episode_cfg.get("model_params", {})
            
            # Implement config locking: merge previous best_config with new params
            if best_config:
                model_params = _merge_locked_config(best_config, model_params)
            
            # Set up logging
            logger = setup_training_logger(str(log_dir), episode_key, args.command)
            original_stdout = sys.stdout
            redirect_output(logger)
            log_path = getattr(logger, "log_path", None)
            
            try:
                if 'lower' in episode_key:
                    model, best_loss = tune_Lower_MELTS(
                        Model=best_model,
                        trainData=train_set,
                        testData=test_set,
                        lr=float(tune_params.get("learning_rate", config.get("training", {}).get("learning_rate", 1e-3))),
                        Epochs=int(tune_params.get("epochs", config.get("training", {}).get("epochs", 10))),
                        batch_size=int(tune_params.get("batch_size", config.get("data", {}).get("batch_size", 1024))),
                        early_stopping_patience=int(tune_params.get("early_stopping_patience", config.get("training", {}).get("early_stopping_patience", 5))),
                        Param_Dict=model_params,
                        max_N=float(tune_params.get("max_N", config.get("training", {}).get("max_N_tune", 1e6))),
                        baseline_loss=best_loss  # Pass previous best_loss
                    )
                else:
                    binWeights, compWeights = _build_phase_weights(ml_indexer, None)
                    model, best_loss = tune_Upper_MELTS(
                        Model=best_model,
                        trainData=train_set,
                        testData=test_set,
                        lr=float(tune_params.get("learning_rate", config.get("training", {}).get("learning_rate", 1e-3))),
                        Epochs=int(tune_params.get("epochs", config.get("training", {}).get("epochs", 10))),
                        batch_size=int(tune_params.get("batch_size", config.get("data", {}).get("batch_size", 1024))),
                        early_stopping_patience=int(tune_params.get("early_stopping_patience", config.get("training", {}).get("early_stopping_patience", 5))),
                        Param_Dict=model_params,
                        binWeights=binWeights,
                        compWeights=compWeights,
                        max_N=float(tune_params.get("max_N", config.get("training", {}).get("max_N_tune", 1e6))),
                        baseline_loss=best_loss # Pass previous best_loss
                    )
                
                # Update best_model and best_config for next episode
                best_model = model
                best_config = model_params
                
                # Save model to DictFilePath
                log_text = _read_text_file(Path(log_path)) if log_path else None
                model.save(
                    str(dict_filepath),
                    config_yaml=config_yaml_text,
                    training_yaml=training_yaml_text,
                    stats=stats_text,
                    log_text=log_text,
                )
                print(f"Saved tuned model to {dict_filepath}")
                
            finally:
                restore_output(original_stdout)
                logger.close()

        else:
            # ===== TRAINING EPISODE =====
            strategy = episode_cfg.get("strategy", "lower")
            scheduler_cfg = episode_cfg.get("scheduler", {}) or {}
            scheduler_name = _normalize_scheduler_name(scheduler_cfg.get("type"))
            scheduler_kwargs = scheduler_cfg.get("args", {}) or {}
            
            max_n = episode_cfg.get("max_N", config.get("training", {}).get("max_N"))
            max_n = np.inf if max_n in {None, "",  "None"} else max_n
            batch_size = int(episode_cfg.get("batch_size", config.get("data", {}).get("batch_size", 1024)))
            lr = float(episode_cfg.get("learning_rate", config.get("training", {}).get("learning_rate", 1e-3)))
            epochs = int(episode_cfg.get("epochs", config.get("training", {}).get("epochs", 50)))
            patience = int(episode_cfg.get("early_stopping_patience", config.get("training", {}).get("early_stopping_patience", 5)))

            # Set up logging
            logger = setup_training_logger(str(log_dir), episode_key, args.command)
            original_stdout = sys.stdout
            redirect_output(logger)
            log_path = getattr(logger, "log_path", None)
            
            try:
                if strategy == "lower":
                    train_Lower_MELTS(
                        best_model,
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
                        DictFilePath=str(dict_filepath),
                        config_yaml=config_yaml_text,
                        training_yaml=training_yaml_text,
                        processing_yaml=processing_yaml_text,
                        stats=stats_text,
                        log_path=log_path,
                    )
                else:
                    weight_dict = episode_cfg.get("weight_dict")
                    binWeights, compWeights = _build_phase_weights(ml_indexer, weight_dict)
                    which_heads_to_freeze = [] if strategy == "finetune" else ["sat_head", "encoder"]
                    #with torch.autograd.set_detect_anomaly(True):
                    train_Upper_MELTS(
                        best_model,
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
                        DictFilePath=str(dict_filepath),
                        config_yaml=config_yaml_text,
                        training_yaml=training_yaml_text,
                        processing_yaml=processing_yaml_text,
                        stats=stats_text,
                        log_path=log_path,
                    )
                
                # Reload and update best_model for next episode (warm-start)
                best_model = NN.rebuild_MELTS_model(str(dict_filepath))
                print(f"Saved trained model to {dict_filepath}")
                
            finally:
                restore_output(original_stdout)
                logger.close()

    print(f"\n{'='*60}")
    print("All episodes completed successfully!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
