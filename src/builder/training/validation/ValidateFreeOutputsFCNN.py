"""
Validate a single free-output FCNN model with emulator-derived inputs.

Use `--geometry extensive` for inputs of the form:
[normalized features, emulator-predicted component moles]

Use `--geometry intensive` for inputs of the form:
[normalized features, emulator-predicted phase moles, emulator-predicted chem_out]

The script plots parity for one checkpoint against the validation bundle and
writes a compact MAE summary.
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

from nMELTS.engine.NN import VariableGeometryFCNNRegressor, rebuild_MELTS_model, _load_temperature_model
from nMELTS.engine.emulator import NN_MELTS
from nMELTS.utils.file_utils import load_ml_bundle


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


def _get_target_min_range(payload: Dict): # -> tuple[np.ndarray, np.ndarray]:
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


def _get_input_min_range(payload: Dict): # -> tuple[np.ndarray, np.ndarray]:
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
): # -> tuple[VariableGeometryFCNNRegressor, Dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    payload = torch.load(checkpoint_path, map_location=device)

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
    geometry: str,
    use_cuda: bool,
    normalize_features_for_emulator: bool,
    batch_size: int,
) -> np.ndarray:
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")

    if geometry == "extensive":
        output_names = ["component_moles"]
    elif geometry == "intensive":
        output_names = ["phase_moles", "chem_out"]
    else:
        raise ValueError(f"Unsupported geometry: {geometry}")

    batches: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, features_raw.shape[0], batch_size):
            stop = min(start + batch_size, features_raw.shape[0])
            features_t = torch.tensor(features_raw[start:stop], dtype=torch.float32, device=device)
            predicted = emulator.forwardMB(
                features_t,
                Normalize=normalize_features_for_emulator,
                WtPercent=False,
                outputs=output_names,
            )
            if geometry == "extensive":
                extras = predicted["component_moles"].detach().cpu().numpy().astype(np.float32)
            else:
                phase_moles = predicted["phase_moles"].detach().cpu().numpy().astype(np.float32)
                print(f"Emulator predicted phase moles range: {phase_moles.min():.3g} to {phase_moles.max():.3g}")
                print(phase_moles[:5])
                chem_out = predicted["chem_out"].detach().cpu().numpy().astype(np.float32)
                print(f"Emulator predicted chem_out range: {chem_out.min():.3g} to {chem_out.max():.3g}")
                print(chem_out[:5])
                extras = np.concatenate([phase_moles, chem_out], axis=1).astype(np.float32)
            batches.append(extras.clip(0,1))

    if not batches:
        return np.zeros((0, 0), dtype=np.float32)

    return np.concatenate(batches, axis=0).astype(np.float32)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate one free-output FCNN checkpoint with emulator-derived inputs."
        )
    )
    parser.add_argument("--bundle", required=True, type=Path, help="Validation bundle path")
    parser.add_argument("--model", required=True, type=Path, help="Path to FCNN checkpoint")
    parser.add_argument(
        "--geometry",
        required=True,
        choices=["extensive", "intensive"],
        help=(
            "Input geometry for the checkpoint: extensive uses component moles, "
            "intensive uses phase moles plus chem_out"
        ),
    )
    parser.add_argument(
        "--emulator-model",
        required=True,
        type=Path,
        help="Path to trained emulator checkpoint used to generate extra inputs",
    )
    parser.add_argument("--batch-size", type=int, default=4096, help="Prediction batch size")
    parser.add_argument("--emulator-batch-size", type=int, default=2 ** 16, help="Batch size for emulator calls")
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

    free_outputs = np.asarray(bundle.free_outputs, dtype=np.float32)

    if args.max_samples < features.shape[0]:
        subset = np.random.default_rng(args.seed).choice(features.shape[0], size=args.max_samples, replace=False)
        features = features[subset]
        free_outputs = free_outputs[subset]

    model, checkpoint, x_min, x_range, y_min, y_range = _load_fcnn_checkpoint(args.model, device)

    selected_output_names = list(checkpoint.get("selected_output_names", []))
    if not selected_output_names:
        raise KeyError("Checkpoint missing selected_output_names")

    available_output_names = list(getattr(bundle.ml_indexer, "freeOutputs", []) or [])
    if not available_output_names:
        raise ValueError("Bundle ml_indexer.freeOutputs is missing or empty")

    y_true = _select_outputs(free_outputs, available_output_names, selected_output_names)

    emulator_model = rebuild_MELTS_model(str(args.emulator_model))
    emulator = NN_MELTS(emulator_model, cuda=bool(args.cuda))
    emulator_extras = _build_emulator_augmented_input(
        emulator=emulator,
        features_raw=features,
        geometry=args.geometry,
        use_cuda=bool(args.cuda),
        normalize_features_for_emulator=not args.no_normalize_emulator_features,
        batch_size=args.emulator_batch_size,
    )
    x_features = np.concatenate([features, emulator_extras], axis=1).astype(np.float32)
    x_model = _normalize_features(x_features, x_min, x_range)


    if x_model.shape[1] != x_min.shape[0]:
        raise ValueError(
            f"Checkpoint input dimension mismatch for geometry '{args.geometry}': "
            f"built {x_model.shape[1]} cols, checkpoint expects {x_min.shape[0]}"
        )

    y_pred_norm = _predict(model, x_model, device=device, batch_size=args.batch_size)
    y_pred = _denormalize_targets(y_pred_norm, y_min, y_range)

    if args.out_dir is None:
        model_stem = _sanitize_name(args.model.stem)
        out_dir = Path(__file__).parent / "plots" / f"free_outputs_validation_{model_stem}_{args.geometry}"
    else:
        out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    mode_metrics = _plot_mode(
        output_dir=out_dir,
        mode_name=args.geometry,
        y_true=y_true,
        y_pred=y_pred,
        output_names=selected_output_names,
    )

    summary = {
        "bundle": str(bundle_path),
        "model": str(args.model),
        "geometry": args.geometry,
        "emulator_model": str(args.emulator_model),
        "n_samples": int(y_true.shape[0]),
        "selected_output_names": selected_output_names,
        "mae_by_output": mode_metrics,
    }

    summary_path = out_dir / "free_outputs_validation_metrics.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("Validation complete.")
    print(f"Geometry: {args.geometry}")
    print(f"Saved outputs to: {out_dir}")
    print(f"Saved metrics JSON: {summary_path}")


if __name__ == "__main__":
    main()
