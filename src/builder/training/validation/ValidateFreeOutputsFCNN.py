"""
Validate FCNN free-output models with three input modes.

Runs three prediction modes on a validation bundle:
1) FCNN1: normalized features only
2) FCNN2 with GT extras: [normalized features, GT phase moles, GT labels]
3) FCNN2 with emulator extras: [normalized features, emulator phase moles, emulator labels]

For each mode and output label, saves a parity scatter plot that includes
average absolute residual (MAE in output units).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure repo root and src are importable when called from CLI.
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from module.engine.NN import VariableGeometryFCNNRegressor, rebuild_MELTS_model
from module.engine.emulator import NN_MELTS
from module.utils.file_utils import load_ml_bundle


def _sanitize_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name).strip("_")


def _resolve_bundle_path(bundle_path: Path) -> Path:
    if bundle_path.suffixes[-2:] == [".tar", ".gz"]:
        return bundle_path
    if bundle_path.suffix == ".tar":
        return bundle_path.with_suffix(".tar.gz")
    if bundle_path.suffix == ".gz" and bundle_path.name.endswith(".tar.gz"):
        return bundle_path
    return (
        bundle_path.with_suffix(bundle_path.suffix + ".tar.gz")
        if bundle_path.suffix
        else bundle_path.with_suffix(".tar.gz")
    )


def _get_target_min_range(payload: Dict) -> Tuple[np.ndarray, np.ndarray]:
    if "target_min" in payload and "target_range" in payload:
        y_min = np.asarray(payload["target_min"], dtype=np.float32).reshape(-1)
        y_range = np.asarray(payload["target_range"], dtype=np.float32).reshape(-1)
        return y_min, y_range

    if "target_min_range" in payload:
        min_range = np.asarray(payload["target_min_range"], dtype=np.float32)
        if min_range.ndim != 2 or min_range.shape[1] != 2:
            raise ValueError("target_min_range must be Nx2")
        return min_range[:, 0], min_range[:, 1]

    raise KeyError("Checkpoint missing target normalizer arrays")


def _get_input_min_range(payload: Dict) -> Tuple[np.ndarray, np.ndarray]:
    if "input_min" in payload and "input_range" in payload:
        x_min = np.asarray(payload["input_min"], dtype=np.float32).reshape(-1)
        x_range = np.asarray(payload["input_range"], dtype=np.float32).reshape(-1)
        return x_min, x_range

    if "input_min_range" in payload:
        min_range = np.asarray(payload["input_min_range"], dtype=np.float32)
        if min_range.ndim != 2 or min_range.shape[1] != 2:
            raise ValueError("input_min_range must be Nx2")
        return min_range[:, 0], min_range[:, 1]

    raise KeyError("Checkpoint missing input normalizer arrays")


def _load_fcnn_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[VariableGeometryFCNNRegressor, Dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_config = payload.get("model_config")
    if not model_config:
        raise KeyError(f"Checkpoint missing model_config: {checkpoint_path}")

    model = VariableGeometryFCNNRegressor(
        input_dim=int(model_config["input_dim"]),
        output_dim=int(model_config["output_dim"]),
        hidden_dims=list(model_config["hidden_dims"]),
        activation_leak=float(model_config.get("activation_leak", 0.05)),
        dropout=float(model_config.get("dropout", 0.0)),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()

    x_min, x_range = _get_input_min_range(payload)
    y_min, y_range = _get_target_min_range(payload)

    return model, payload, x_min, x_range, y_min, y_range


def _normalize_features(x: np.ndarray, x_min: np.ndarray, x_range: np.ndarray) -> np.ndarray:
    if x.shape[1] != x_min.shape[0] or x.shape[1] != x_range.shape[0]:
        raise ValueError(
            f"Feature shape mismatch: x has {x.shape[1]} cols, "
            f"x_min={x_min.shape[0]}, x_range={x_range.shape[0]}"
        )
    safe_range = np.where(np.abs(x_range) < 1e-7, 1.0, x_range)
    return ((x - x_min) / safe_range).astype(np.float32)


def _denormalize_targets(y_norm: np.ndarray, y_min: np.ndarray, y_range: np.ndarray) -> np.ndarray:
    return (y_norm * y_range + y_min).astype(np.float32)


def _select_outputs(
    free_outputs: np.ndarray,
    available_names: Sequence[str],
    selected_names: Sequence[str],
) -> np.ndarray:
    index_map = {name: idx for idx, name in enumerate(available_names)}
    missing = [name for name in selected_names if name not in index_map]
    if missing:
        raise KeyError(
            f"Validation bundle is missing selected output names: {missing}. "
            f"Available: {list(available_names)}"
        )
    indices = [index_map[name] for name in selected_names]
    return free_outputs[:, indices].astype(np.float32)


def _predict(
    model: VariableGeometryFCNNRegressor,
    x: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    chunks: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            stop = min(start + batch_size, x.shape[0])
            xb = torch.tensor(x[start:stop], dtype=torch.float32, device=device)
            chunks.append(model(xb).detach().cpu().numpy())
    return np.vstack(chunks).astype(np.float32)


def _plot_mode(
    output_dir: Path,
    mode_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_names: Sequence[str],
) -> Dict[str, float]:
    mode_dir = output_dir / _sanitize_name(mode_name)
    mode_dir.mkdir(parents=True, exist_ok=True)

    metrics: Dict[str, float] = {}
    for j, output_name in enumerate(output_names):
        gt = y_true[:, j]
        pred = y_pred[:, j]
        mae = float(np.mean(np.abs(pred - gt)))
        metrics[output_name] = mae

        lo = float(min(np.min(gt), np.min(pred)))
        hi = float(max(np.max(gt), np.max(pred)))
        if np.isclose(lo, hi):
            lo -= 1.0
            hi += 1.0

        fig = plt.figure(figsize=(6, 6))
        plt.scatter(gt, pred, s=6, alpha=0.35, color="tab:blue")
        plt.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1)
        plt.xlabel("Ground truth")
        plt.ylabel("Prediction")
        plt.title(f"{output_name}\nMean absolute residual = {mae:.6g}")
        plt.xlim(lo, hi)
        plt.ylim(lo, hi)
        plt.tight_layout()

        out_path = mode_dir / f"{_sanitize_name(output_name)}_parity.png"
        fig.savefig(out_path, dpi=220)
        plt.close(fig)

    return metrics


def _build_emulator_augmented_input(
    emulator: NN_MELTS,
    features_raw: np.ndarray,
    use_cuda: bool,
    normalize_features_for_emulator: bool,
) -> np.ndarray:
    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    features_tensor = torch.tensor(features_raw, device=device, dtype=torch.float32)
    if normalize_features_for_emulator:
        features_tensor = emulator.norm_features.norm(features_tensor)

    ml_indexer = emulator.model.ml_indexer
    px_sp_transform = ml_indexer.PxSpTransform
    comp_subset = ml_indexer.compositional_component_subset
    px_sp_sub = px_sp_transform[np.ix_(comp_subset, comp_subset)]

    #with torch.no_grad():
    #    likelihoods, transcomponent_hat, log_moles, recon_bulk, component_moles, phase_proportions, phase_moles = (
    #        emulator.model.forward(features_tensor, detailed=True)
    #    )

    del likelihoods, log_moles, recon_bulk, component_moles, phase_proportions

    transcomponent_hat_np = transcomponent_hat.detach().cpu().numpy().astype(np.float32)
    phase_moles_np = phase_moles.detach().cpu().numpy().astype(np.float32)
    component_hat_np = transcomponent_hat_np @ np.linalg.inv(px_sp_sub).astype(np.float32)

    return np.concatenate([features_raw.astype(np.float32), phase_moles_np, component_hat_np], axis=1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate free-output FCNN models with direct and augmented inputs."
        )
    )
    parser.add_argument("--bundle", required=True, type=Path, help="Validation bundle path")
    parser.add_argument(
        "--fcnn-features-model",
        required=True,
        type=Path,
        help="Path to features-only FCNN checkpoint",
    )
    parser.add_argument(
        "--fcnn-augmented-model",
        required=True,
        type=Path,
        help="Path to augmented FCNN checkpoint",
    )
    parser.add_argument(
        "--emulator-model",
        required=True,
        type=Path,
        help="Path to trained emulator checkpoint used to generate extra inputs",
    )
    parser.add_argument("--batch-size", type=int, default=4096, help="Prediction batch size")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument("--max-samples", type=int, default=2**16, help="Max samples to evaluate")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument(
        "--no-normalize-emulator-features",
        action="store_true",
        help="Disable emulator feature normalization before forward pass",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for plots and metrics JSON",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    bundle_path = _resolve_bundle_path(args.bundle)
    bundle = load_ml_bundle(bundle_path)

    if bundle.free_outputs is None:
        raise ValueError(f"Bundle missing free_outputs: {bundle_path}")
    if bundle.features is None or bundle.molar_labels is None or bundle.labels is None:
        raise ValueError(
            "Bundle missing required inputs (features, molar_labels, labels) for validation"
        )

    features = np.asarray(bundle.features, dtype=np.float32)
    gt_moles = np.asarray(bundle.molar_labels, dtype=np.float32)
    gt_labels = np.asarray(bundle.labels, dtype=np.float32)

    n_total = features.shape[0]
    if args.max_samples < n_total:
        subset = np.random.default_rng(args.seed).choice(n_total, size=args.max_samples, replace=False)
        features = features[subset]
        gt_moles = gt_moles[subset]
        gt_labels = gt_labels[subset]
        free_outputs = np.asarray(bundle.free_outputs, dtype=np.float32)[subset]
    else:
        free_outputs = np.asarray(bundle.free_outputs, dtype=np.float32)

    fcnn1, ckpt1, x1_min, x1_range, y1_min, y1_range = _load_fcnn_checkpoint(
        args.fcnn_features_model,
        device,
    )
    fcnn2, ckpt2, x2_min, x2_range, y2_min, y2_range = _load_fcnn_checkpoint(
        args.fcnn_augmented_model,
        device,
    )

    selected_output_names = list(ckpt1.get("selected_output_names", []))
    if not selected_output_names:
        raise KeyError("FCNN1 checkpoint missing selected_output_names")

    selected_output_names_2 = list(ckpt2.get("selected_output_names", []))
    if selected_output_names_2 != selected_output_names:
        raise ValueError("FCNN1 and FCNN2 selected_output_names do not match")

    available_output_names = list(getattr(bundle.ml_indexer, "freeOutputs", []) or [])
    if not available_output_names:
        raise ValueError("Bundle ml_indexer.freeOutputs is missing or empty")

    y_true = _select_outputs(free_outputs, available_output_names, selected_output_names)

    x1_norm = _normalize_features(features, x1_min, x1_range)

    x2_gt = np.concatenate([x1_norm, gt_moles, gt_labels], axis=1).astype(np.float32)
    if x2_gt.shape[1] != x2_min.shape[0] or x2_gt.shape[1] != x2_range.shape[0]:
        raise ValueError(
            f"FCNN2 GT-augmented input dimension mismatch: x2={x2_gt.shape[1]}, "
            f"checkpoint={x2_min.shape[0]}"
        )

    emulator_model = rebuild_MELTS_model(str(args.emulator_model))
    emulator = NN_MELTS(emulator_model, cuda=bool(args.cuda))
    x2_emulator_raw = _build_emulator_augmented_input(
        emulator=emulator,
        features_raw=features,
        use_cuda=bool(args.cuda),
        normalize_features_for_emulator=not args.no_normalize_emulator_features,
    )
    x2_emulator = np.concatenate([x1_norm, x2_emulator_raw[:, x1_norm.shape[1]:]], axis=1).astype(np.float32)

    if x2_emulator.shape[1] != x2_gt.shape[1]:
        raise ValueError(
            f"FCNN2 emulator-augmented input dimension mismatch: "
            f"expected {x2_gt.shape[1]}, got {x2_emulator.shape[1]}"
        )

    y1_pred_norm = _predict(fcnn1, x1_norm, device=device, batch_size=args.batch_size)
    y2_gt_pred_norm = _predict(fcnn2, x2_gt, device=device, batch_size=args.batch_size)
    y2_emu_pred_norm = _predict(fcnn2, x2_emulator, device=device, batch_size=args.batch_size)

    y1_pred = _denormalize_targets(y1_pred_norm, y1_min, y1_range)
    y2_gt_pred = _denormalize_targets(y2_gt_pred_norm, y2_min, y2_range)
    y2_emu_pred = _denormalize_targets(y2_emu_pred_norm, y2_min, y2_range)

    if args.out_dir is None:
        model_stem = _sanitize_name(args.fcnn_features_model.stem)
        out_dir = Path(__file__).parent / "plots" / f"free_outputs_validation_{model_stem}"
    else:
        out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    mode_metrics: Dict[str, Dict[str, float]] = {}
    mode_metrics["fcnn1_features_only"] = _plot_mode(
        output_dir=out_dir,
        mode_name="fcnn1_features_only",
        y_true=y_true,
        y_pred=y1_pred,
        output_names=selected_output_names,
    )
    mode_metrics["fcnn2_with_gt_extras"] = _plot_mode(
        output_dir=out_dir,
        mode_name="fcnn2_with_gt_extras",
        y_true=y_true,
        y_pred=y2_gt_pred,
        output_names=selected_output_names,
    )
    mode_metrics["fcnn2_with_emulator_extras"] = _plot_mode(
        output_dir=out_dir,
        mode_name="fcnn2_with_emulator_extras",
        y_true=y_true,
        y_pred=y2_emu_pred,
        output_names=selected_output_names,
    )

    summary = {
        "bundle": str(bundle_path),
        "fcnn_features_model": str(args.fcnn_features_model),
        "fcnn_augmented_model": str(args.fcnn_augmented_model),
        "emulator_model": str(args.emulator_model),
        "n_samples": int(y_true.shape[0]),
        "selected_output_names": selected_output_names,
        "mae_by_mode_and_output": mode_metrics,
    }

    summary_path = out_dir / "free_outputs_validation_metrics.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("Validation complete.")
    print(f"Saved outputs to: {out_dir}")
    print(f"Saved metrics JSON: {summary_path}")


if __name__ == "__main__":
    main()
