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
from nMELTS.utils.string_utils import apply_type_conversions
from nMELTS.config.constants import TYPE_CONVERSION_MAP

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


def _best_loss_from_tune_results(results: Any) -> Optional[float]:
    if not isinstance(results, list) or not results:
        return None
    losses: List[float] = []
    for entry in results:
        if isinstance(entry, dict) and "loss" in entry:
            try:
                losses.append(float(entry["loss"]))
            except (TypeError, ValueError):
                continue
    if not losses:
        return None
    return min(losses)


def _discover_episodes(config: Dict[str, Any]) -> List[Tuple[str, str, Dict[str, Any]]]:
    """
    Discover and order all numbered episodes (tune1, train1, tune2, train2, etc.).
    
    Returns list of (episode_key, episode_type, episode_config) tuples in execution order.
    Episode type is 'tune' or 'train'.
    """
    episodes = []
    
    # Extract episode numbers
    def get_episode_num(key: str) -> int:
        match = re.search(r'\d+', key)
        if not match:
            return int(0)
        return int(match.group())
    
    # Tune and train in order of definition in YAML
    for key, cfg in config.items():
        if re.match(r'^tune\d+$', key):
            EpType = 'tune'
        elif re.match(r'^train\d+$', key):
            EpType = 'train'
        else:
            continue
        num = get_episode_num(key)
        episodes.append((key, EpType, cfg))

    print(episodes)
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
        Default base directory. If None, uses temp_models
    
    Returns
    -------
    Path
        Full path where model should be saved
    """

    # auto-generate in default directory
    if default_base is None:
        default_base = Path(__file__).parent / 'temp_models'
    
    default_base.mkdir(parents=True, exist_ok=True)

    # If explicitly provided in episode config, use it
    if isinstance(config, dict) and config.get('DictFilePath'):
        path = Path(config['DictFilePath'])
        if path.suffix == "":
            path = path.with_suffix(".tar")
        return default_base / path.name
    
    # Generate name: {tarname}_{episode_key}.tar
    return default_base / f"{tarname}_{episode_key}.tar"

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
    #print(config)
    config_yaml_text = yaml.safe_dump(config, sort_keys=False)
    training_yaml_text = _read_text_file(Path(args.config))

    tarname = config.get("tarname")
    if not tarname:
        raise ValueError("training.yaml must define tarname")

    # Set up logging
    log_dir = config["log_dir"]
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    only_vp = config["only_vp"]
    roots = _resolve_train_roots()
    train_bundle = _resolve_bundle_path(f"{tarname}_Train", roots)
    test_bundle = _resolve_bundle_path(f"{tarname}_Test", roots)
    bundle_yaml_text, stats_text = _read_bundle_metadata(train_bundle)
    if bundle_yaml_text:
        processing_yaml_text = bundle_yaml_text
    train_set, ml_indexer = load_ML_data(train_bundle, only_VP=only_vp)
    test_set, _ = load_ML_data(test_bundle, only_VP=only_vp)

    print(f"MOLAR EPSILON FROM LOADED DATA: {ml_indexer.molar_epsilon}")

    warm_start = config["warm_start"]

    # Apply type conversions to config using centralized mapping
    config = apply_type_conversions(config, TYPE_CONVERSION_MAP)

    if warm_start.lower() != 'none':
        best_model = NN.rebuild_MELTS_model(Path(__file__).parent / config['checkpoints']['load_dir'] / (warm_start + '.tar'), 
                                            epsilon=ml_indexer.molar_epsilon) # Override default epsilon with training data value so model initializes appropriately
    else:
        # Initialize model, pull relevant subset of configs to initialize model
        model_config = {}
        modelArgs = ['encoderLayerUp', 'encoderLayerDown',
                 'middleLayerUp', 'middleLayerDown',
                 'low_regularization', 'high_regularization', 
                 'activation_leak', 'lowWD', 'highWD', 'noise',
                 'description'
                    ]
               
        for key, val in deepcopy(config).items():
            if key in modelArgs:
                model_config[key] = val
        best_model = NN.MidLevelNetwork(**model_config, ml_indexer=ml_indexer)

    # State for sequential episodes
    best_loss = None

    # Discover and execute episodes in order
    episodes = _discover_episodes(config)
    
    for episode_key, episode_type, episode_cfgI in episodes:

        if episode_cfgI is None:
            print(f'No config passed for {episode_type}{episode_key}! Skippping.')
            continue
        
        episode_cfg = _deep_update(deepcopy(config), episode_cfgI) # Inherit Globals, overprint episode configurration

        if best_loss is None:
            best_loss = episode_cfg.get('best_loss', None) # Initialize best_loss from config if not set by previous tune episode

        # Apply type conversions to episode config
        episode_cfg = apply_type_conversions(episode_cfg, TYPE_CONVERSION_MAP)
    
        # === Apply episode-specific model configuration ===
        modelArgs = ['encoderLayerUp', 'encoderLayerDown',
                     'middleLayerUp', 'middleLayerDown',
                     'low_regularization', 'high_regularization', 
                     'activation_leak', 'lowWD', 'highWD', 'noise',
                     'description']
        
        # Check if any model parameters have changed in episode config
        model_config_changed = False
        new_model_config = {}
        for param in modelArgs:
            episode_val = episode_cfg.get(param)
            current_val = best_model.config.get(param)
            new_model_config[param] = episode_val if episode_val is not None else current_val
            if episode_val is not None and episode_val != current_val:
                model_config_changed = True
                print(f"Model param '{param}' changing: {current_val} -> {episode_val}")
        
        # Rebuild model if configuration changed
        if model_config_changed:
            print(f"Rebuilding model with episode-specific configuration for {episode_key}")
            full_model_config = deepcopy(best_model.config)
            full_model_config.update(new_model_config)
            new_model = NN.MidLevelNetwork(**full_model_config, ml_indexer=ml_indexer)
            # Try to load weights from previous model (strict=False allows for architecture changes)
            try:
                new_model.load_state_dict(best_model.state_dict(), strict=False)
                print(f"Loaded compatible weights from previous episode")
            except Exception as e:
                print(f"Could not load weights from previous episode: {e}")
            best_model = new_model

        if not _stage_allowed(args.stage, episode_type):
            continue

        print(f"\n{'='*60}")
        print(f"Episode: {episode_key} ({episode_type})")
        print(f"{'='*60}")

        # Determine DictFilePath for saving
        dict_filepath = _resolve_dict_filepath(episode_key, episode_cfg, tarname, default_base=Path(__file__).parent / config['checkpoints']['save_dir']) 
        print(f"Model save path: {dict_filepath}")

        strategy = episode_cfg.get('strategy', '')
        Imax_N = episode_cfg["max_N"]
        if Imax_N in ['null', 'None', 'inf', '0', None]:
            max_N = np.inf
        else:
            max_N = float(Imax_N) or np.inf
        ESP = int(episode_cfg["early_stopping_patience"])
        batch_size=int(episode_cfg["batch_size"])
        Epochs = int(episode_cfg["epochs"])
        lr = float(episode_cfg["learning_rate"])
        eps = float(episode_cfg.get("eps", 1E-8))
        amsgrad = bool(episode_cfg.get("amsgrad", True))

        binWeights, compWeights = _build_phase_weights(ml_indexer, episode_cfg['weight_dict'])
        which_heads_to_freeze = episode_cfg.get('which_heads_to_freeze', None)
        if which_heads_to_freeze is None: # what portion of model is frozen can be manually overridden, or is inferred from the strategy.
            if strategy == 'upper':
                which_heads_to_freeze = ['sat_head', 'encoder', 'mole_head'] # Lower model and abundances turned off
            elif strategy == 'isolate':
                which_heads_to_freeze = ['encoder', 'middleBrain'] # Shared cores of model turned off
            else:
                which_heads_to_freeze = []

        device = episode_cfg["device"]
    
        # Loss configuration
        loss_config = episode_cfg["loss_config"]
        chem_cfg = loss_config["chemistry"]
        sat_cfg = loss_config["binary"]
        mole_cfg = loss_config["moles"]
        bulk_cfg = loss_config["bulk"]
        chem_alpha = float(chem_cfg["weight"])
        mole_alpha = float(mole_cfg["weight"])
        bulk_alpha = float(bulk_cfg["weight"])
        sat_alpha = float(sat_cfg["weight"])
        criterion = _loss_fn_from_type(chem_cfg["type"])

        if episode_type == 'tune':
            # ===== TUNING EPISODE =====
            
            # Apply type conversions to tune_params
            episode_cfg['tune_params'] = apply_type_conversions(episode_cfg['tune_params'], TYPE_CONVERSION_MAP)

            # Set up logging
            logger = setup_training_logger(str(log_dir), episode_key, args.command)
            original_stdout = sys.stdout
            redirect_output(logger)
            log_path = getattr(logger, "log_path", None)

            #Strategy determines which heads are frozen, which part of model is trained
            try:
                if strategy == 'lower': # No phase weighting implemented for lower model training...
                    model, tune_results = tune_Lower_MELTS(
                        Model=best_model,
                        trainData=train_set,
                        testData=test_set,
                        lr=lr,
                        Epochs=Epochs,
                        batch_size=batch_size,
                        early_stopping_patience=ESP,
                        max_N=max_N,
                        Param_Dict=episode_cfg['tune_params'],
                        best_loss=best_loss  # Pass previous best_loss
                    )
                    tuned_best_loss = _best_loss_from_tune_results(tune_results)
                    if tuned_best_loss is not None:
                        best_loss = tuned_best_loss
                else:
                    model, tune_results = tune_Upper_MELTS(
                        Model=best_model,
                        trainData=train_set,
                        testData=test_set,
                        lr=lr,
                        Epochs=Epochs,
                        batch_size=batch_size,
                        early_stopping_patience=ESP,
                        max_N=max_N,
                        Param_Dict=episode_cfg['tune_params'],
                        binWeights=binWeights,
                        compWeights=compWeights,
                        best_loss=best_loss, # Pass previous best_loss,
                        which_heads_to_freeze=which_heads_to_freeze,
                        chem_alpha=chem_alpha,
                        mole_alpha=mole_alpha,
                        bulk_alpha=bulk_alpha, 
                        sat_alpha=sat_alpha,
                        eps=eps,
                        amsgrad=amsgrad,
                    )
                    tuned_best_loss = _best_loss_from_tune_results(tune_results)
                    if tuned_best_loss is not None:
                        best_loss = tuned_best_loss
                
                # Update best_model and best_config for next episode
                #best_model = NN.rebuild_MELTS_model(str(dict_filepath))
                best_model = model # No need to reload from disk, we have the tuned model in memory

                #Save tuned configs to global!
                config = _deep_update(config, best_model.config) # Update global config with episode-specific config (including tuned params) for next episode to inherit

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
            scheduler_cfg = episode_cfg["scheduler"]
            scheduler_name = _normalize_scheduler_name(scheduler_cfg["type"])
            scheduler_kwargs = scheduler_cfg["args"]

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
                        lr=lr,
                        Epochs=Epochs,
                        batch_size=batch_size,
                        early_stopping_patience=ESP,
                        max_N=max_N,
                        device=device,
                        DictFilePath=str(dict_filepath),
                        config_yaml=config_yaml_text,
                        training_yaml=training_yaml_text,
                        processing_yaml=processing_yaml_text,
                        stats=stats_text,
                        log_path=log_path,
                    )
                else:
                    #raise ValueError('This temporary config works on lower model only')
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
                        sat_alpha=sat_alpha,
                        Epochs=Epochs,
                        batch_size=batch_size,
                        lr=lr,
                        binWeights=binWeights,
                        compWeights=compWeights,
                        device=device,
                        max_N=max_N,
                        early_stopping_patience=ESP,
                        which_heads_to_freeze=which_heads_to_freeze,
                        DictFilePath=str(dict_filepath),
                        config_yaml=config_yaml_text,
                        training_yaml=training_yaml_text,
                        processing_yaml=processing_yaml_text,
                        stats=stats_text,
                        log_path=log_path,
                        eps=eps,
                        amsgrad=amsgrad
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
    #print('Waiting')
    #import time
    #time.sleep(2700)
    main()
