"""
Validate FCNN temperature-residual models trained by train_temperature_residual_fcnn.py.

For each checkpoint, predicts a normalized T_residual then reconstructs:
    T_pred = T_residual_denorm + reference_adiabat(P, S)

Two input kinds are supported (stored in the checkpoint as ``input_kind``):
  - ``emulator_path``: [norm features | emulator phase_moles | emulator chem_out]
  - ``gt_path``      : [norm features | GT molar_labels      | GT labels        ]

For each checkpoint, saves:
  - A residual parity plot  (T_residual predicted vs true, in K)
  - A temperature parity plot (T_pred vs T_true, in K)
  - A summary metrics JSON
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from ngibbs.engine.NN import VariableGeometryFCNNRegressor, rebuild_MELTS_model
from ngibbs.engine.emulator import NN_MELTS
from ngibbs.utils.file_utils import load_ml_bundle
from ngibbs.utils.math_utils import get_T as reference_adiabat


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name).strip("_")


def _resolve_bundle_path(bundle_path: Path) -> Path:
    if bundle_path.suffixes[-2:] == [".tar", ".gz"]:
        return bundle_path
    if bundle_path.suffix == ".tar":
        return bundle_path.with_suffix(".tar.gz")
    return (
        bundle_path.with_suffix(bundle_path.suffix + ".tar.gz")
        if bundle_path.suffix
        else bundle_path.with_suffix(".tar.gz")
    )


def _load_temperature_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[VariableGeometryFCNNRegressor, Dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)

    for key in ("model_config", "input_kind", "temperature_label", "p_feature_idx", "s_feature_idx"):
        if key not in payload:
            raise KeyError(f"Checkpoint missing required key '{key}': {checkpoint_path}")

    cfg = payload["model_config"]
    model = VariableGeometryFCNNRegressor(
        input_dim=int(cfg["input_dim"]),
        output_dim=int(cfg["output_dim"]),
        hidden_dims=list(cfg["hidden_dims"]),
        activation_leak=float(cfg.get("activation_leak", 0.05)),
        dropout=float(cfg.get("dropout", 0.0)),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, payload


def _extract_min_range(payload: Dict, prefix: str) -> Tuple[np.ndarray, np.ndarray]:
    k_min, k_range = f"{prefix}_min", f"{prefix}_range"
    k_pair = f"{prefix}_min_range"
    if k_min in payload and k_range in payload:
        return (
            np.asarray(payload[k_min], dtype=np.float32).reshape(-1),
            np.asarray(payload[k_range], dtype=np.float32).reshape(-1),
        )
    if k_pair in payload:
        arr = np.asarray(payload[k_pair], dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"{k_pair} must be Nx2")
        return arr[:, 0], arr[:, 1]
    raise KeyError(f"Checkpoint missing normalizer arrays for prefix '{prefix}'")


def _normalize(x: np.ndarray, x_min: np.ndarray, x_range: np.ndarray) -> np.ndarray:
    safe_range = np.where(np.abs(x_range) < 1e-7, 1.0, x_range)
    return ((x - x_min) / safe_range).astype(np.float32)


def _eval_adiabat_poly(
    P: np.ndarray,
    S: np.ndarray,
    coefs: Dict[str, float],
    features: Optional[np.ndarray] = None,
    comp_indices: Optional[Dict[str, int]] = None,
) -> np.ndarray:
    """Evaluate the reference adiabat polynomial (P in GPa, T in K).

    Base: T = b0 + b_S*S + b_S2*S^2 + b_P*P + b_P2*P^2
    Compositional extension (when features and comp_indices provided):
        T += Σ_i  b_Xi*X_i + b_Xi2*X_i^2
    """
    T = (
        coefs["b0"]
        + coefs["b_S"] * S
        + coefs["b_S2"] * S ** 2
        + coefs["b_P"] * P
        + coefs["b_P2"] * P ** 2
    )
    if comp_indices and features is not None:
        for elem, col_idx in comp_indices.items():
            X = features[:, col_idx]
            if f"b_{elem}" in coefs:
                T = T + coefs[f"b_{elem}"] * X
            if f"b_{elem}2" in coefs:
                T = T + coefs[f"b_{elem}2"] * X ** 2
    return np.asarray(T, dtype=np.float32)


def _build_emulator_augmented(
    emulator: NN_MELTS,
    features_raw: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    PM_chunks: List[np.ndarray] = []
    CO_chunks: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, features_raw.shape[0], batch_size):
            stop = min(start + batch_size, features_raw.shape[0])
            feats_t = torch.tensor(features_raw[start:stop], dtype=torch.float32, device=device)
            predicted = emulator.forwardNN(
                feats_t,
                Normalize=True,
                WtPercent=False,
                outputs=["phase_moles", "chem_out"],
            )
            PM_chunks.append(predicted["phase_moles"].detach().cpu().clamp(0, 1).numpy())
            CO_chunks.append(predicted["chem_out"].detach().cpu().clamp(0, 1).numpy())
    return np.concatenate(
        [np.vstack(PM_chunks).astype(np.float32), np.vstack(CO_chunks).astype(np.float32)],
        axis=1,
    )


def _predict_batched(
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
    return np.vstack(chunks).astype(np.float32).reshape(-1)


def _save_parity_plot(
    out_path: Path,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    mae: float,
) -> None:
    lo = float(min(np.min(x_vals), np.min(y_vals)))
    hi = float(max(np.max(x_vals), np.max(y_vals)))
    if np.isclose(lo, hi):
        lo -= 1.0
        hi += 1.0

    fig = plt.figure(figsize=(6, 6))
    plt.scatter(x_vals, y_vals, s=6, alpha=0.35, color="tab:blue")
    plt.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{title}\nMean absolute residual = {mae:.6g}")
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _validate_checkpoint(
    checkpoint_path: Path,
    bundle,
    features_raw: np.ndarray,
    T_true: np.ndarray,
    free_outputs: np.ndarray,
    device: torch.device,
    out_dir: Path,
    emulator: Optional[NN_MELTS],
    batch_size: int,
) -> Dict:
    model, payload = _load_temperature_checkpoint(checkpoint_path, device)
    input_kind: str = payload["input_kind"]
    temperature_label: str = payload["temperature_label"]
    p_idx: int = int(payload["p_feature_idx"])
    s_idx: int = int(payload["s_feature_idx"])

    x_min, x_range = _extract_min_range(payload, "input")
    y_min, y_range = _extract_min_range(payload, "target")

    # Reference adiabat (must mirror train_temperature_residual_fcnn._compute_residuals
    # exactly, including is_melts unit handling, or the reconstructed T_pred will be
    # systematically biased by whatever the stored/default adiabat disagrees on).
    adiabat_coefs: Optional[Dict[str, float]] = payload.get("adiabat_coefs")
    comp_indices: Optional[Dict[str, int]] = payload.get("coef_feature_indices")
    is_melts: bool = bool(payload.get("is_melts", False))
    P = features_raw[:, p_idx].astype(np.float64)
    if is_melts:
        P = P / 10000.0  # bars → GPa
        S = np.full_like(P, 2.5)
    else:
        S = features_raw[:, s_idx].astype(np.float64)
    if adiabat_coefs is not None:
        kind = "compositional" if comp_indices else "S+P only"
        print(f"    Using checkpoint-stored adiabat ({kind}): {list(adiabat_coefs.keys())}")
        T_ref = _eval_adiabat_poly(
            P, S, adiabat_coefs,
            features=features_raw.astype(np.float64),
            comp_indices=comp_indices,
        )
    else:
        T_ref = np.asarray(reference_adiabat(P, S), dtype=np.float32)
    if is_melts:
        T_ref = T_ref - 273.15  # K → Celsius
    T_residual_true = (T_true - T_ref).astype(np.float32)

    # Build model input
    n_base_features = x_min.shape[0]
    x_norm_base = _normalize(features_raw, x_min[:features_raw.shape[1]], x_range[:features_raw.shape[1]])

    if input_kind == "emulator_path":
        if emulator is None:
            raise ValueError(
                f"Checkpoint '{checkpoint_path.name}' has input_kind='emulator_path' "
                "but no --emulator-model was provided."
            )
        aux = _build_emulator_augmented(emulator, features_raw, device, batch_size)
        x_input = np.concatenate([x_norm_base, aux], axis=1).astype(np.float32)
    elif input_kind == "gt_path":
        gt_moles = np.asarray(bundle.molar_labels, dtype=np.float32)
        gt_labels = np.asarray(bundle.labels, dtype=np.float32)
        x_input = np.concatenate([x_norm_base, gt_moles, gt_labels], axis=1).astype(np.float32)
    else:
        raise ValueError(f"Unknown input_kind '{input_kind}' in checkpoint")

    if x_input.shape[1] != n_base_features:
        raise ValueError(
            f"Built input dim {x_input.shape[1]} != checkpoint input dim {n_base_features} "
            f"for '{checkpoint_path.name}'"
        )

    # Predict residuals (normalized), then convert to K
    residual_norm = _predict_batched(model, x_input, device, batch_size)
    T_residual_pred = residual_norm * y_range[0] + y_min[0]
    T_pred = T_residual_pred + T_ref

    # Metrics
    mae_residual = float(np.mean(np.abs(T_residual_pred - T_residual_true)))
    mae_temperature = float(np.mean(np.abs(T_pred - T_true)))
    mse_residual = float(np.mean((T_residual_pred - T_residual_true) ** 2))
    mse_temperature = float(np.mean((T_pred - T_true) ** 2))

    # Plots
    mode_dir = out_dir / _sanitize_name(checkpoint_path.stem)
    mode_dir.mkdir(parents=True, exist_ok=True)

    _save_parity_plot(
        out_path=mode_dir / "temperature_residual_parity.png",
        x_vals=T_residual_true,
        y_vals=T_residual_pred,
        xlabel="Ground truth residual (K)",
        ylabel="Predicted residual (K)",
        title=f"T residual | {input_kind}",
        mae=mae_residual,
    )
    _save_parity_plot(
        out_path=mode_dir / "temperature_parity.png",
        x_vals=T_true,
        y_vals=T_pred,
        xlabel="Ground truth T (K)",
        ylabel="Predicted T (K)",
        title=f"Temperature | {input_kind}",
        mae=mae_temperature,
    )

    print(
        f"  [{input_kind}] MAE residual={mae_residual:.3f} K | MAE temperature={mae_temperature:.3f} K"
    )

    return {
        "checkpoint": str(checkpoint_path),
        "input_kind": input_kind,
        "temperature_label": temperature_label,
        "n_samples": int(T_true.shape[0]),
        "mae_residual_K": mae_residual,
        "mse_residual_K2": mse_residual,
        "mae_temperature_K": mae_temperature,
        "mse_temperature_K2": mse_temperature,
        "T_true_mean": float(np.mean(T_true)),
        "T_residual_true_mean": float(np.mean(T_residual_true)),
        "T_residual_true_std": float(np.std(T_residual_true)),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate temperature-residual FCNN checkpoints produced by "
            "train_temperature_residual_fcnn.py."
        )
    )
    parser.add_argument(
        "--bundle",
        required=True,
        type=Path,
        help="Validation bundle path (any split; _Valid recommended)",
    )
    parser.add_argument(
        "--checkpoints",
        required=True,
        nargs="+",
        type=Path,
        help="One or more checkpoint .pt files to validate",
    )
    parser.add_argument(
        "--emulator-model",
        type=Path,
        default=None,
        help="Emulator checkpoint needed for emulator_path checkpoints",
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-samples", type=int, default=2 ** 16)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for plots and metrics JSON (default: auto-named next to first checkpoint)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    bundle_path = _resolve_bundle_path(args.bundle)
    print(f"Loading bundle: {bundle_path}")
    bundle = load_ml_bundle(bundle_path)

    if bundle.free_outputs is None:
        raise ValueError(f"Bundle missing free_outputs: {bundle_path}")
    if bundle.features is None or bundle.molar_labels is None or bundle.labels is None:
        raise ValueError("Bundle missing required arrays (features / molar_labels / labels)")

    # Identify temperature column from first checkpoint's metadata
    first_payload = torch.load(args.checkpoints[0], map_location="cpu", weights_only=False)
    temperature_label: str = first_payload.get("temperature_label", "")
    available_output_names = list(getattr(bundle.ml_indexer, "freeOutputs", []) or [])
    if not temperature_label or temperature_label not in available_output_names:
        raise ValueError(
            f"Temperature label '{temperature_label}' not found in bundle freeOutputs. "
            f"Available: {available_output_names}"
        )
    temp_col = available_output_names.index(temperature_label)

    # Subset to max_samples
    features_raw = np.asarray(bundle.features, dtype=np.float32)
    free_outputs = np.asarray(bundle.free_outputs, dtype=np.float32)
    n_total = features_raw.shape[0]
    if args.max_samples < n_total:
        rng = np.random.default_rng(args.seed)
        subset = rng.choice(n_total, size=args.max_samples, replace=False)
        features_raw = features_raw[subset]
        free_outputs = free_outputs[subset]
        # We need sliced versions of molar_labels / labels too
        molar_labels = np.asarray(bundle.molar_labels, dtype=np.float32)[subset]
        labels = np.asarray(bundle.labels, dtype=np.float32)[subset]
    else:
        molar_labels = np.asarray(bundle.molar_labels, dtype=np.float32)
        labels = np.asarray(bundle.labels, dtype=np.float32)

    T_true = free_outputs[:, temp_col].astype(np.float32)

    # Attach sliced arrays back so _validate_checkpoint can access them via bundle
    class _SlicedBundle:
        pass

    sliced = _SlicedBundle()
    sliced.molar_labels = molar_labels
    sliced.labels = labels
    sliced.ml_indexer = bundle.ml_indexer

    # Build emulator once if any checkpoint needs it
    emulator: Optional[NN_MELTS] = None
    needs_emulator = False
    for cp in args.checkpoints:
        p = torch.load(cp, map_location="cpu", weights_only=False)
        if p.get("input_kind") == "emulator_path":
            needs_emulator = True
            break
    if needs_emulator:
        if args.emulator_model is None:
            raise ValueError(
                "One or more checkpoints use input_kind='emulator_path'; "
                "please supply --emulator-model."
            )
        print(f"Loading emulator: {args.emulator_model}")
        emulator = NN_MELTS(rebuild_MELTS_model(str(args.emulator_model)), cuda=args.cuda)

    # Output directory
    if args.out_dir is None:
        out_dir = args.checkpoints[0].parent / f"temp_validation_{args.checkpoints[0].stem}"
    else:
        out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Validate each checkpoint
    all_metrics: List[Dict] = []
    for cp in args.checkpoints:
        print(f"\nValidating: {cp.name}")
        metrics = _validate_checkpoint(
            checkpoint_path=cp,
            bundle=sliced,
            features_raw=features_raw,
            T_true=T_true,
            free_outputs=free_outputs,
            device=device,
            out_dir=out_dir,
            emulator=emulator,
            batch_size=args.batch_size,
        )
        all_metrics.append(metrics)

    summary = {
        "bundle": str(bundle_path),
        "n_samples": int(T_true.shape[0]),
        "temperature_label": temperature_label,
        "results": all_metrics,
    }
    summary_path = out_dir / "temperature_residual_validation_metrics.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("\nValidation complete.")
    print(f"Outputs saved to: {out_dir}")
    print(f"Metrics JSON:     {summary_path}")


if __name__ == "__main__":
    main()
