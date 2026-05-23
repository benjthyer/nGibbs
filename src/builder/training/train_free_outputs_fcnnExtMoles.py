"""
Train configurable FCNN regressors for free outputs in MLready bundles.

Two models are trained:
1) [features, ground-truth extensive component moles] -> free outputs
2) [features, emulator-predicted extensive component moles] -> free outputs

Bundle convention:
- test bundle: in-training validation (used each epoch)
- valid bundle: optional post-training holdout evaluation

The CLI accepts one bundle stem (without _Train/_Test/_Valid); all three
bundles are loaded from that stem.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Ensure repo root and src are importable when called from CLI.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from module.engine.NN import VariableGeometryFCNNRegressor, rebuild_MELTS_model
from module.engine.emulator import NN_MELTS
from module.utils.file_utils import load_ml_bundle
from module.utils.math_utils import Normalizer

ADAPTIVE_DROPOUT_MAX = 0.50
ADAPTIVE_DROPOUT_UP = 0.01
ADAPTIVE_DROPOUT_DOWN = 0.005
OVERFIT_RATIO = 1.02
DEFAULT_EMULATOR_BATCH_SIZE = 2**16


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_limit_outputs(raw_values: Optional[Sequence[str]]) -> Optional[List[str]]:
    if not raw_values:
        return None

    output_names: List[str] = []
    for value in raw_values:
        pieces = [p.strip() for p in value.split(",")]
        output_names.extend([p for p in pieces if p])

    deduped: List[str] = []
    for name in output_names:
        if name not in deduped:
            deduped.append(name)
    return deduped


def _parse_hidden_dims(raw_values: Sequence[str]) -> List[int]:
    hidden_dims: List[int] = []
    for value in raw_values:
        parts = [p.strip() for p in value.split(",")]
        for part in parts:
            if not part:
                continue
            width = int(part)
            if width <= 0:
                raise ValueError("All hidden layer widths must be > 0")
            hidden_dims.append(width)

    if not hidden_dims:
        raise ValueError("hidden-dims must provide at least one positive integer")
    return hidden_dims


def _fit_minmax(y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y_min = np.min(y_train, axis=0)
    y_max = np.max(y_train, axis=0)
    y_range = y_max - y_min
    y_range = np.where(y_range < 1e-7, 1.0, y_range)
    return y_min.astype(np.float32), y_range.astype(np.float32)


def _build_bundle_paths(bundle_stem: Path) -> Dict[str, Path]:
    stem = str(bundle_stem)
    if stem.endswith(".tar.gz"):
        stem = stem[:-7]

    for suffix in ("_Train", "_Test", "_Valid"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break

    paths = {
        "train": Path(f"{stem}_Train.tar.gz"),
        "test": Path(f"{stem}_Test.tar.gz"),
        "valid": Path(f"{stem}_Valid.tar.gz"),
    }

    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        details = ", ".join(f"{name}: {paths[name]}" for name in missing)
        raise FileNotFoundError(
            "Could not resolve required bundle files from --bundle-stem. "
            f"Missing -> {details}"
        )

    return paths


def _ensure_feature_normalizer(ml_indexer, train_features: np.ndarray) -> Normalizer:
    existing = getattr(ml_indexer, "feature_normalizer", None)
    n_features_total = train_features.shape[1]
    n_named_features = len(getattr(ml_indexer, "featureNames", []) or [])

    if existing is not None:
        min_tensor = existing.miner.detach().cpu().to(torch.float32).clone()
        range_tensor = existing.ranger.detach().cpu().to(torch.float32).clone()
        if min_tensor.numel() != n_features_total or range_tensor.numel() != n_features_total:
            raise ValueError(
                "Existing feature_normalizer dimension mismatch: "
                f"expected {n_features_total}, got min={min_tensor.numel()} range={range_tensor.numel()}"
            )
        # Enforce identity mapping for non-featureName dimensions.
        if n_named_features < n_features_total:
            min_tensor[n_named_features:] = 0.0
            range_tensor[n_named_features:] = 1.0
        normalizer = Normalizer(min_tensor=min_tensor, range_tensor=range_tensor)
        ml_indexer.feature_normalizer = normalizer
        return normalizer

    min_tensor = torch.zeros(n_features_total, dtype=torch.float32)
    range_tensor = torch.ones(n_features_total, dtype=torch.float32)

    if n_named_features > 0:
        x_named = torch.tensor(train_features[:, :n_named_features], dtype=torch.float32)
        mins = torch.min(x_named, dim=0).values
        ranges = torch.max(x_named, dim=0).values - mins
        ranges = torch.clamp(ranges, min=1e-7)
        min_tensor[:n_named_features] = mins
        range_tensor[:n_named_features] = ranges

    normalizer = Normalizer(min_tensor=min_tensor, range_tensor=range_tensor)
    ml_indexer.feature_normalizer = normalizer
    return normalizer


def _build_output_normalizer(y_train: np.ndarray) -> Normalizer:
    y_min, y_range = _fit_minmax(y_train)
    return Normalizer(
        min_tensor=torch.tensor(y_min, dtype=torch.float32),
        range_tensor=torch.tensor(y_range, dtype=torch.float32),
    )


def _apply_normalizer(x: np.ndarray, normalizer: Normalizer) -> np.ndarray:
    x_t = torch.tensor(x, dtype=torch.float32)
    return normalizer.norm(x_t).cpu().numpy().astype(np.float32)


def _normalizer_arrays(normalizer: Normalizer) -> Tuple[np.ndarray, np.ndarray]:
    mins = normalizer.miner.detach().cpu().numpy().astype(np.float32)
    ranges = normalizer.ranger.detach().cpu().numpy().astype(np.float32)
    return mins, ranges


def _normalizer_pairs(mins: np.ndarray, ranges: np.ndarray) -> np.ndarray:
    mins = np.asarray(mins, dtype=np.float32).reshape(-1)
    ranges = np.asarray(ranges, dtype=np.float32).reshape(-1)
    return np.stack([mins, ranges], axis=1).astype(np.float32)


def _build_extended_input_normalizer_arrays(
    base_feature_mins: np.ndarray,
    base_feature_ranges: np.ndarray,
    extended_dim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if extended_dim < len(base_feature_mins):
        raise ValueError(
            f"extended_dim={extended_dim} is smaller than base feature dim={len(base_feature_mins)}"
        )
    mins = np.zeros(extended_dim, dtype=np.float32)
    ranges = np.ones(extended_dim, dtype=np.float32)
    mins[: len(base_feature_mins)] = base_feature_mins
    ranges[: len(base_feature_ranges)] = base_feature_ranges
    return mins, ranges


def _build_component_moles(
    emulator: NN_MELTS,
    labels: np.ndarray,
    molar_labels: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    if labels.shape[0] != molar_labels.shape[0]:
        raise ValueError(
            "labels and molar_labels row counts must match for component mole tabulation"
        )

    batches: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, labels.shape[0], batch_size):
            stop = min(start + batch_size, labels.shape[0])
            labels_t = torch.tensor(labels[start:stop], dtype=torch.float32, device=device)
            molar_labels_t = torch.tensor(molar_labels[start:stop], dtype=torch.float32, device=device)
            component_moles = emulator.getExtensiveComps(
                intensiveLabels=labels_t,
                molarLabels=molar_labels_t,
            )
            batches.append(component_moles.detach().cpu())

    return torch.cat(batches, dim=0)


def _build_predicted_component_moles(
    emulator: NN_MELTS,
    features_raw: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")

    batches: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, features_raw.shape[0], batch_size):
            stop = min(start + batch_size, features_raw.shape[0])
            features_t = torch.tensor(features_raw[start:stop], dtype=torch.float32, device=device)
            predicted = emulator.forwardMB(
                features_t,
                Normalize=True,
                WtPercent=False,
                outputs=["component_moles"],
            )
            batches.append(predicted["component_moles"].detach().cpu())

    return torch.cat(batches, dim=0)


def _select_outputs(
    free_outputs: np.ndarray,
    available_names: Sequence[str],
    requested_names: Optional[Sequence[str]],
) -> Tuple[np.ndarray, List[str], List[int]]:
    if len(available_names) != free_outputs.shape[1]:
        raise ValueError(
            "Mismatch between ml_indexer.freeOutputs names and free_outputs columns: "
            f"{len(available_names)} names vs {free_outputs.shape[1]} columns."
        )

    if not requested_names:
        selected_indices = list(range(free_outputs.shape[1]))
        selected_names = list(available_names)
    else:
        index_map = {name: idx for idx, name in enumerate(available_names)}
        missing = [name for name in requested_names if name not in index_map]
        if missing:
            raise KeyError(
                "Requested output labels were not found: "
                f"{missing}. Available labels: {list(available_names)}"
            )
        selected_indices = [index_map[name] for name in requested_names]
        selected_names = list(requested_names)

    selected = free_outputs[:, selected_indices]
    return selected, selected_names, selected_indices


def _make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    x_t = torch.tensor(x, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    ds = TensorDataset(x_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _evaluate_denorm(
    model: torch.nn.Module,
    x: np.ndarray,
    y_true_denorm: np.ndarray,
    y_min: np.ndarray,
    y_range: np.ndarray,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(x, dtype=torch.float32, device=device)
        pred_norm = model(x_t).cpu().numpy()

    pred_denorm = pred_norm * y_range + y_min
    mse = float(np.mean((pred_denorm - y_true_denorm) ** 2))
    mae = float(np.mean(np.abs(pred_denorm - y_true_denorm)))
    return {"mse": mse, "mae": mae}


def _train_model(
    model: VariableGeometryFCNNRegressor,
    train_loader: DataLoader,
    valid_loader: Optional[DataLoader],
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    patience: int = 10,
    best_checkpoint_path: Optional[Path] = None,
    model_config_for_checkpoint: Optional[Dict] = None,
) -> Dict[str, any]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.MSELoss()

    history: Dict[str, List[float]] = {"train_loss": [], "valid_loss": [], "dropout": []}
    model.to(device)

    dropout_rate = 0.0
    model.set_dropout_rate(dropout_rate)

    best_valid_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in tqdm(range(epochs), desc="Training", unit="epoch"):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch + 1} train", leave=False, unit="batch"):
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            batch_size = xb.shape[0]
            train_loss_sum += float(loss.item()) * batch_size
            train_count += batch_size

        mean_train_loss = train_loss_sum / max(1, train_count)
        history["train_loss"].append(mean_train_loss)

        if valid_loader is not None:
            model.eval()
            valid_loss_sum = 0.0
            valid_count = 0
            with torch.no_grad():
                for xb, yb in tqdm(valid_loader, desc=f"Epoch {epoch + 1} valid", leave=False, unit="batch"):
                    xb = xb.to(device)
                    yb = yb.to(device)
                    pred = model(xb)
                    loss = criterion(pred, yb)
                    batch_size = xb.shape[0]
                    valid_loss_sum += float(loss.item()) * batch_size
                    valid_count += batch_size
            mean_valid_loss = valid_loss_sum / max(1, valid_count)
        else:
            mean_valid_loss = float("nan")

        history["valid_loss"].append(mean_valid_loss)

        # Best checkpoint saving on validation improvement
        if valid_loader is not None and np.isfinite(mean_valid_loss):
            if mean_valid_loss < best_valid_loss:
                best_valid_loss = mean_valid_loss
                epochs_without_improvement = 0
                if best_checkpoint_path is not None:
                    best_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), best_checkpoint_path)
                    print(f"Epoch {epoch + 1}: improved to {mean_valid_loss:.6f}, saved best checkpoint")
            else:
                epochs_without_improvement += 1

        # Adaptive dropout
        if valid_loader is not None and np.isfinite(mean_valid_loss):
            if mean_valid_loss > mean_train_loss * OVERFIT_RATIO:
                new_dropout = min(dropout_rate + ADAPTIVE_DROPOUT_UP, ADAPTIVE_DROPOUT_MAX)
                if new_dropout > dropout_rate:
                    print(f"Overfitting: increasing dropout {dropout_rate:.2f} -> {new_dropout:.2f}")
                dropout_rate = new_dropout
            elif mean_valid_loss < mean_train_loss:
                new_dropout = max(dropout_rate - ADAPTIVE_DROPOUT_DOWN, 0.0)
                if new_dropout < dropout_rate:
                    print(f"Underfitting: decreasing dropout {dropout_rate:.2f} -> {new_dropout:.2f}")
                dropout_rate = new_dropout

            model.set_dropout_rate(dropout_rate)

        history["dropout"].append(dropout_rate)
        print(
            f"Epoch {epoch + 1:4d}/{epochs}: "
            f"train_mse_norm={mean_train_loss:.6f}, "
            f"valid_mse_norm={mean_valid_loss:.6f}, "
            f"dropout={dropout_rate:.2f}, "
            f"patience={epochs_without_improvement}/{patience}"
        )

        # Early stopping
        if valid_loader is not None and epochs_without_improvement >= patience:
            print(f"\nEarly stopping: no improvement for {patience} epochs")
            break

    history["best_valid_loss"] = best_valid_loss
    history["best_checkpoint_path"] = str(best_checkpoint_path) if best_checkpoint_path else None
    return history


def _save_checkpoint(
    out_path: Path,
    model: VariableGeometryFCNNRegressor,
    selected_output_names: Sequence[str],
    selected_output_indices: Sequence[int],
    x_min: np.ndarray,
    x_range: np.ndarray,
    y_min: np.ndarray,
    y_range: np.ndarray,
    input_kind: str,
    metrics: Dict[str, Dict[str, float]],
    history: Dict[str, List[float]],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "model_config": model.config,
        "selected_output_names": list(selected_output_names),
        "selected_output_indices": list(selected_output_indices),
        "input_min": x_min.astype(np.float32),
        "input_range": x_range.astype(np.float32),
        "input_min_range": _normalizer_pairs(x_min, x_range),
        "target_min": y_min,
        "target_range": y_range,
        "target_min_range": _normalizer_pairs(y_min, y_range),
        "input_kind": input_kind,
        "metrics": metrics,
        "history": history,
    }
    torch.save(payload, out_path)


def _load_and_validate_bundle(bundle_path: Path, label: str):
    print(f"Loading {label} bundle: {bundle_path}")
    bundle = load_ml_bundle(bundle_path)
    if bundle.free_outputs is None:
        raise ValueError(f"Bundle has no free_outputs.npy: {bundle_path}")
    if bundle.features is None or bundle.molar_labels is None or bundle.labels is None:
        raise ValueError(
            "Bundle is missing required arrays for this workflow "
            f"(features/molar_labels/labels): {bundle_path}"
        )
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train two configurable FCNN regressors for ML bundle free outputs: "
            "(1) features-only and (2) features+moles+labels. "
            "Uses test as in-training validation and valid as holdout. "
            "Loads all three bundles from one bundle stem."
        )
    )
    parser.add_argument(
        "--bundle-stem",
        required=True,
        type=Path,
        help=(
            "Bundle stem path without _Train/_Test/_Valid suffix; script loads "
            "all three split bundles"
        ),
    )
    parser.add_argument(
        "--emulator-model",
        required=True,
        type=Path,
        help="Path to a trained emulator checkpoint used to tabulate predicted component moles",
    )
    parser.add_argument(
        "--limit-outputs",
        nargs="+",
        default=None,
        help=(
            "Optional list of free output labels to predict; accepts space-separated "
            "and/or comma-separated values"
        ),
    )
    parser.add_argument(
        "--hidden-dims",
        nargs="+",
        default=["256", "128", "64"],
        help="Hidden-layer widths, e.g. --hidden-dims 512 256 128 or 512,256,128",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--activation-leak", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=20, help="Epochs without improvement before early stopping")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--skip-1",
        action="store_true",
        help="Skip training model 2 ([features, predicted component moles] -> free outputs)",
    )
    parser.add_argument(
        "--emulator-batch-size",
        type=int,
        default=DEFAULT_EMULATOR_BATCH_SIZE,
        help="Batch size for emulator calls when tabulating component moles",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("src") / "builder" / "training" / "temp_models",
        help="Directory for model checkpoints and metrics",
    )

    args = parser.parse_args()
    if args.emulator_batch_size <= 0:
        raise ValueError(f"--emulator-batch-size must be > 0, got {args.emulator_batch_size}")

    set_seed(args.seed)

    hidden_dims = _parse_hidden_dims(args.hidden_dims)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    emulator = NN_MELTS(
        rebuild_MELTS_model(args.emulator_model),
        cuda=(device.type == "cuda"),
    )

    limit_outputs = _parse_limit_outputs(args.limit_outputs)

    bundle_paths = _build_bundle_paths(args.bundle_stem)
    train_bundle = _load_and_validate_bundle(bundle_paths["train"], "train")
    test_bundle = _load_and_validate_bundle(bundle_paths["test"], "test")
    valid_bundle = _load_and_validate_bundle(bundle_paths["valid"], "valid")

    available_output_names = list(getattr(train_bundle.ml_indexer, "freeOutputs", []) or [])
    if not available_output_names:
        raise ValueError("ml_indexer.freeOutputs is missing or empty in train bundle")

    y_train_denorm, selected_output_names, selected_output_indices = _select_outputs(
        train_bundle.free_outputs,
        available_output_names,
        limit_outputs,
    )

    feature_normalizer = _ensure_feature_normalizer(train_bundle.ml_indexer, np.asarray(train_bundle.features, dtype=np.float32))
    output_normalizer = _build_output_normalizer(y_train_denorm)
    x1_min, x1_range = _normalizer_arrays(feature_normalizer)
    y_min, y_range = _normalizer_arrays(output_normalizer)

    # Share fitted normalizers across split indexers without mutating bundle arrays.
    train_bundle.ml_indexer.feature_normalizer = feature_normalizer
    test_bundle.ml_indexer.feature_normalizer = feature_normalizer
    valid_bundle.ml_indexer.feature_normalizer = feature_normalizer
    train_bundle.ml_indexer.output_normalizer = output_normalizer
    test_bundle.ml_indexer.output_normalizer = output_normalizer
    valid_bundle.ml_indexer.output_normalizer = output_normalizer

    y_train_norm = _apply_normalizer(y_train_denorm, output_normalizer)

    x1_train = _apply_normalizer(np.asarray(train_bundle.features, dtype=np.float32), feature_normalizer)
    gt_component_moles_train = _build_component_moles(
        emulator=emulator,
        labels=np.asarray(train_bundle.labels, dtype=np.float32),
        molar_labels=np.asarray(train_bundle.molar_labels, dtype=np.float32),
        device=device,
        batch_size=args.emulator_batch_size,
    ).detach().cpu().numpy().astype(np.float32)

    x1_train = np.concatenate([x1_train, gt_component_moles_train], axis=1).astype(np.float32)
    x1_min_ext, x1_range_ext = _build_extended_input_normalizer_arrays(x1_min, x1_range, x1_train.shape[1])

    y_test_denorm, _, _ = _select_outputs(
        test_bundle.free_outputs,
        available_output_names,
        selected_output_names,
    )
    y_test_norm = _apply_normalizer(y_test_denorm, output_normalizer)

    x1_test = _apply_normalizer(np.asarray(test_bundle.features, dtype=np.float32), feature_normalizer)
    gt_component_moles_test = _build_component_moles(
        emulator=emulator,
        labels=np.asarray(test_bundle.labels, dtype=np.float32),
        molar_labels=np.asarray(test_bundle.molar_labels, dtype=np.float32),
        device=device,
        batch_size=args.emulator_batch_size,
    ).detach().cpu().numpy().astype(np.float32)

    x1_test = np.concatenate([x1_test, gt_component_moles_test], axis=1).astype(np.float32)
    train_loader_x1 = _make_loader(x1_train, y_train_norm, batch_size=args.batch_size, shuffle=True)
    valid_loader_x1 = _make_loader(x1_test, y_test_norm, batch_size=args.batch_size, shuffle=False)
    model_features_only = VariableGeometryFCNNRegressor(
        input_dim=x1_train.shape[1],
        output_dim=y_train_norm.shape[1],
        hidden_dims=hidden_dims,
        activation_leak=args.activation_leak,
        dropout=0.0,
    )
    print("\nTraining model 1/2: [features, GT component moles] -> free outputs")
    best_model1_path = args.out_dir / f"best_features_gt_component_moles_{args.bundle_stem.name}.pt"
    history_features = _train_model(
        model=model_features_only,
        train_loader=train_loader_x1,
        valid_loader=valid_loader_x1,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        patience=args.patience,
        best_checkpoint_path=best_model1_path,
    )

    model_augmented = None
    history_augmented = None
    best_model2_path = args.out_dir / f"best_features_pred_component_moles_{args.bundle_stem.name}.pt"
    x2_train = None
    x2_test = None
    x2_min = None
    x2_range = None

    if not args.skip_1:
        pred_component_moles_train = _build_predicted_component_moles(
            emulator=emulator,
            features_raw=np.asarray(train_bundle.features, dtype=np.float32),
            device=device,
            batch_size=args.emulator_batch_size,
        ).detach().cpu().numpy().astype(np.float32)
        x2_train = np.concatenate(
            [
                x1_train[:, : x1_train.shape[1] - gt_component_moles_train.shape[1]],
                pred_component_moles_train,
            ],
            axis=1,
        ).astype(np.float32)
        x2_min, x2_range = _build_extended_input_normalizer_arrays(x1_min, x1_range, x2_train.shape[1])

        pred_component_moles_test = _build_predicted_component_moles(
            emulator=emulator,
            features_raw=np.asarray(test_bundle.features, dtype=np.float32),
            device=device,
            batch_size=args.emulator_batch_size,
        ).detach().cpu().numpy().astype(np.float32)
        x2_test = np.concatenate(
            [
                _apply_normalizer(np.asarray(test_bundle.features, dtype=np.float32), feature_normalizer),
                pred_component_moles_test,
            ],
            axis=1,
        ).astype(np.float32)

        train_loader_x2 = _make_loader(x2_train, y_train_norm, batch_size=args.batch_size, shuffle=True)
        valid_loader_x2 = _make_loader(x2_test, y_test_norm, batch_size=args.batch_size, shuffle=False)

        model_augmented = VariableGeometryFCNNRegressor(
            input_dim=x2_train.shape[1],
            output_dim=y_train_norm.shape[1],
            hidden_dims=hidden_dims,
            activation_leak=args.activation_leak,
            dropout=0.0,
        )

        print("\nTraining model 2/2: [features, predicted component moles] -> free outputs")
        history_augmented = _train_model(
            model=model_augmented,
            train_loader=train_loader_x2,
            valid_loader=valid_loader_x2,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=device,
            patience=args.patience,
            best_checkpoint_path=best_model2_path,
        )
    else:
        print("\nSkipping model 2/2 due to --skip-1")

    metrics: Dict[str, Dict[str, Dict[str, float]]] = {
        "features_gt_component_moles": {},
        "features_pred_component_moles": {},
    }

    # Load best checkpoints for evaluation if available.
    if best_model1_path.exists():
        print(f"\nLoading best checkpoint for model 1: {best_model1_path}")
        model_features_only.load_state_dict(torch.load(best_model1_path, map_location=device, weights_only=False))

    metrics["features_gt_component_moles"]["train"] = _evaluate_denorm(
        model=model_features_only,
        x=x1_train,
        y_true_denorm=y_train_denorm,
        y_min=y_min,
        y_range=y_range,
        device=device,
    )
    metrics["features_gt_component_moles"]["test"] = _evaluate_denorm(
        model=model_features_only,
        x=x1_test,
        y_true_denorm=y_test_denorm,
        y_min=y_min,
        y_range=y_range,
        device=device,
    )

    if model_augmented is not None and x2_train is not None and x2_test is not None:
        if best_model2_path.exists():
            print(f"Loading best checkpoint for model 2: {best_model2_path}")
            model_augmented.load_state_dict(torch.load(best_model2_path, map_location=device, weights_only=False))

        metrics["features_pred_component_moles"]["train"] = _evaluate_denorm(
            model=model_augmented,
            x=x2_train,
            y_true_denorm=y_train_denorm,
            y_min=y_min,
            y_range=y_range,
            device=device,
        )

        metrics["features_pred_component_moles"]["test"] = _evaluate_denorm(
            model=model_augmented,
            x=x2_test,
            y_true_denorm=y_test_denorm,
            y_min=y_min,
            y_range=y_range,
            device=device,
        )

    y_valid_denorm, _, _ = _select_outputs(
        valid_bundle.free_outputs,
        available_output_names,
        selected_output_names,
    )
    x1_valid = _apply_normalizer(np.asarray(valid_bundle.features, dtype=np.float32), feature_normalizer)
    gt_component_moles_valid = _build_component_moles(
        emulator=emulator,
        labels=np.asarray(valid_bundle.labels, dtype=np.float32),
        molar_labels=np.asarray(valid_bundle.molar_labels, dtype=np.float32),
        device=device,
        batch_size=args.emulator_batch_size,
    ).detach().cpu().numpy().astype(np.float32)

    x1_valid = np.concatenate([x1_valid, gt_component_moles_valid], axis=1).astype(np.float32)
    metrics["features_gt_component_moles"]["valid"] = _evaluate_denorm(
        model=model_features_only,
        x=x1_valid,
        y_true_denorm=y_valid_denorm,
        y_min=y_min,
        y_range=y_range,
        device=device,
    )

    if model_augmented is not None:
        pred_component_moles_valid = _build_predicted_component_moles(
            emulator=emulator,
            features_raw=np.asarray(valid_bundle.features, dtype=np.float32),
            device=device,
            batch_size=args.emulator_batch_size,
        ).detach().cpu().numpy().astype(np.float32)
        x2_valid = np.concatenate(
            [
                _apply_normalizer(np.asarray(valid_bundle.features, dtype=np.float32), feature_normalizer),
                pred_component_moles_valid,
            ],
            axis=1,
        ).astype(np.float32)
        metrics["features_pred_component_moles"]["valid"] = _evaluate_denorm(
            model=model_augmented,
            x=x2_valid,
            y_true_denorm=y_valid_denorm,
            y_min=y_min,
            y_range=y_range,
            device=device,
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    bundle_basename = args.bundle_stem.name
    metrics_path = args.out_dir / f"free_outputs_training_metrics_{bundle_basename}.json"

    model1_path = args.out_dir / f"free_outputs_features_gt_component_moles_{bundle_basename}.pt"
    _save_checkpoint(
        out_path=model1_path,
        model=model_features_only,
        selected_output_names=selected_output_names,
        selected_output_indices=selected_output_indices,
        x_min=x1_min_ext,
        x_range=x1_range_ext,
        y_min=y_min,
        y_range=y_range,
        input_kind="features_gt_component_moles",
        metrics=metrics["features_gt_component_moles"],
        history=history_features,
    )

    model2_path: Path | str
    if model_augmented is not None and x2_min is not None and x2_range is not None and history_augmented is not None:
        model2_path = args.out_dir / f"free_outputs_features_pred_component_moles_{bundle_basename}.pt"
        _save_checkpoint(
            out_path=model2_path,
            model=model_augmented,
            selected_output_names=selected_output_names,
            selected_output_indices=selected_output_indices,
            x_min=x2_min,
            x_range=x2_range,
            y_min=y_min,
            y_range=y_range,
            input_kind="features_pred_component_moles",
            metrics=metrics["features_pred_component_moles"],
            history=history_augmented,
        )
    else:
        model2_path = "Not Trained!"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "selected_output_names": selected_output_names,
                "selected_output_indices": selected_output_indices,
                "hidden_dims": hidden_dims,
                "bundle_paths": {k: str(v) for k, v in bundle_paths.items()},
                "normalizers": {
                    "features_gt_component_moles_input_min_range": _normalizer_pairs(x1_min_ext, x1_range_ext).tolist(),
                    "features_pred_component_moles_input_min_range": _normalizer_pairs(x2_min, x2_range).tolist(),
                    "output_min_range": _normalizer_pairs(y_min, y_range).tolist(),
                },
                "metrics": metrics,
                "model_paths": {
                    "features_gt_component_moles": str(model1_path),
                    "features_pred_component_moles": str(model2_path),
                },
                "emulator_model": str(args.emulator_model),
                "emulator_batch_size": args.emulator_batch_size,
            },
            f,
            indent=2,
        )

    print("\nTraining complete.")
    print(f"Saved model: {model1_path}")
    print(f"Saved model: {model2_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Selected outputs ({len(selected_output_names)}): {selected_output_names}")
    print(f"Hidden dims: {hidden_dims}")
    print("Evaluation metrics (denormalized units):")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
