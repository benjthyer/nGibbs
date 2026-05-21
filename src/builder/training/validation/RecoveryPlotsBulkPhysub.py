"""
Generate parity/recovery plots for HeFESTo bulk physub properties.

Ground-truth bulk properties are computed from dataset labels by first mapping
intensive labels + phase moles -> extensive component moles, then applying the
GPU-friendly physub matrix transform.

Predicted bulk properties are computed from NN_MELTS.forwardMB component moles,
with temperature estimated by a separate FCNN regressor checkpoint.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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

from module.engine.NN import VariableGeometryFCNNRegressor, rebuild_MELTS_model, _load_temperature_model
from module.engine.emulator import NN_MELTS
from module.engine.EOS_arithmetic import (
    PHYSUB_BULK_ATTRIBUTE_NAMES,
    compute_physub_bulk_matrix,
    get_hefesto_physub_context,
)
from module.utils.file_utils import load_ml_bundle


PHYSUB_COMPONENT_ATTRIBUTE_NAMES: Tuple[str, ...] = (
    "molar_volume",
    "bulk_modulus",
    "shear_modulus",
    "heat_capacity_p",
    "heat_capacity_v",
    "thermal_expansivity",
    "entropy",
    "enthalpy",
    "gibbs",
)


def _sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def _resolve_bundle_path(bundle_path: str) -> Path:
    path = Path(bundle_path)
    if path.suffixes[-2:] == [".tar", ".gz"]:
        return path
    if path.suffix == ".tar":
        return path.with_suffix(".tar.gz")
    if path.suffix == ".gz" and path.name.endswith(".tar.gz"):
        return path
    return path.with_suffix(path.suffix + ".tar.gz") if path.suffix else path.with_suffix(".tar.gz")


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def _find_name_index(names: Sequence[str], candidates: Sequence[str]) -> Optional[int]:
    lowered = [name.lower() for name in names]
    for candidate in candidates:
        candidate_low = candidate.lower()
        for idx, name_low in enumerate(lowered):
            if candidate_low == name_low:
                return idx
    for idx, name_low in enumerate(lowered):
        if "temperature" in name_low:
            return idx
    return None


def _has_temperature_feature(feature_names: Sequence[str]) -> Tuple[bool, Optional[int], Optional[str]]:
    idx = _find_name_index(
        feature_names,
        ("Temperature(System_main)", "Temperature", "T(K)(System_main)", "T(K)"),
    )
    return idx is not None, idx, (str(feature_names[idx]) if idx is not None else None)


def _resolve_ground_truth_temperature(
    features: np.ndarray,
    feature_names: Sequence[str],
    free_outputs: Optional[np.ndarray],
    free_output_names: Sequence[str],
    source_mode: str,
) -> Tuple[np.ndarray, str, str]:
    feature_idx = _find_name_index(
        feature_names,
        ("Temperature(System_main)", "Temperature", "T(K)(System_main)", "T(K)"),
    )
    free_idx = _find_name_index(
        free_output_names,
        ("Temperature(System_main)", "Temperature", "T(K)(System_main)", "T(K)"),
    )

    source_mode = source_mode.lower()
    if source_mode not in {"auto", "feature", "free_output"}:
        raise ValueError("temperature-source must be one of: auto, feature, free_output")

    if source_mode in {"auto", "feature"} and feature_idx is not None:
        return features[:, feature_idx].astype(np.float32), "feature", str(feature_names[feature_idx])

    if source_mode in {"auto", "free_output"} and free_outputs is not None and free_idx is not None:
        return free_outputs[:, free_idx].astype(np.float32), "free_output", str(free_output_names[free_idx])

    if source_mode == "feature":
        raise ValueError("Requested feature temperature source, but no temperature-like feature was found")
    if source_mode == "free_output":
        raise ValueError("Requested free_output temperature source, but no temperature-like free output was found")

    raise ValueError(
        "Could not locate ground-truth temperature in features or free outputs. "
        "Use a bundle containing temperature columns or pass a different source mode."
    )


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

    raise KeyError("Temperature model checkpoint is missing input normalizer arrays")


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

    raise KeyError("Temperature model checkpoint is missing target normalizer arrays")


def _normalize_inputs(x: np.ndarray, x_min: np.ndarray, x_range: np.ndarray) -> np.ndarray:
    if x.shape[1] != x_min.shape[0] or x.shape[1] != x_range.shape[0]:
        raise ValueError(
            f"Temperature model input dimension mismatch: x={x.shape[1]}, "
            f"x_min={x_min.shape[0]}, x_range={x_range.shape[0]}"
        )
    safe_range = np.where(np.abs(x_range) < 1e-7, 1.0, x_range)
    return ((x - x_min) / safe_range).astype(np.float32)


def _resolve_temperature_output_index(payload: Dict, requested_name: Optional[str]) -> Tuple[int, str]:
    selected_names = list(payload.get("selected_output_names", []) or [])

    if requested_name is not None:
        if requested_name not in selected_names:
            raise KeyError(
                f"Requested --temperature-output-name '{requested_name}' is missing from "
                f"checkpoint selected_output_names: {selected_names}"
            )
        return selected_names.index(requested_name), requested_name

    if not selected_names:
        if int(payload.get("model_config", {}).get("output_dim", 0)) == 1:
            return 0, "output_0"
        raise KeyError(
            "Temperature checkpoint has no selected_output_names and output_dim>1; "
            "pass --temperature-output-name"
        )

    temp_like = [name for name in selected_names if "temperature" in name.lower() or "t(k" in name.lower()]
    if len(temp_like) == 1:
        return selected_names.index(temp_like[0]), temp_like[0]

    if len(selected_names) == 1:
        return 0, selected_names[0]

    raise KeyError(
        "Could not uniquely identify temperature output in checkpoint. "
        f"selected_output_names={selected_names}. Use --temperature-output-name."
    )


def _predict_temperature(
    model: VariableGeometryFCNNRegressor,
    x_norm: np.ndarray,
    y_min: np.ndarray,
    y_range: np.ndarray,
    output_index: int,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    pred_chunks: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, x_norm.shape[0], batch_size):
            stop = min(start + batch_size, x_norm.shape[0])
            xb = torch.tensor(x_norm[start:stop], dtype=torch.float32, device=device)
            pred_chunks.append(model(xb).detach().cpu().numpy())

    y_norm = np.vstack(pred_chunks).astype(np.float32)
    y_denorm = y_norm * y_range.reshape(1, -1) + y_min.reshape(1, -1)
    return y_denorm[:, output_index].astype(np.float32)


def _plot_parity(x_true: np.ndarray, y_pred: np.ndarray, name: str, out_path: Path, verbose: bool = False) -> Dict[str, float]:
    # Data validation with verbose diagnostics
    if verbose:
        print(f"  [DEBUG] {name}:")
        print(f"    - GT shape: {x_true.shape}, dtype: {x_true.dtype}")
        print(f"    - Pred shape: {y_pred.shape}, dtype: {y_pred.dtype}")
        print(f"    - GT NaN count: {np.isnan(x_true).sum()}, Inf count: {np.isinf(x_true).sum()}")
        print(f"    - Pred NaN count: {np.isnan(y_pred).sum()}, Inf count: {np.isinf(y_pred).sum()}")
        print(f"    - GT range: [{np.nanmin(x_true):.6e}, {np.nanmax(x_true):.6e}]")
        print(f"    - Pred range: [{np.nanmin(y_pred):.6e}, {np.nanmax(y_pred):.6e}]")
    
    # Filter out NaN/Inf before metrics
    valid_mask = np.isfinite(x_true) & np.isfinite(y_pred)
    if verbose:
        print(f"    - Valid samples: {valid_mask.sum()}/{len(valid_mask)}")
    
    if valid_mask.sum() == 0:
        if verbose:
            print(f"    [WARNING] No valid samples! Skipping plot.")
        return {"mae": np.nan, "rmse": np.nan, "mae_over_mean_abs_gt_percent": np.nan, "status": "no_valid_data"}
    
    x_true_valid = x_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    
    abs_err = np.abs(y_pred_valid - x_true_valid)
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean((y_pred_valid - x_true_valid) ** 2)))
    mean_true = float(np.mean(np.abs(x_true_valid)))
    mape_like = 100.0 * _safe_div(mae, mean_true)

    lo = float(min(np.min(x_true_valid), np.min(y_pred_valid)))
    hi = float(max(np.max(x_true_valid), np.max(y_pred_valid)))
    
    if verbose:
        print(f"    - MAE: {mae:.6e}, RMSE: {rmse:.6e}, MAE%: {mape_like:.3f}%")
        print(f"    - Plot range: [{lo:.6e}, {hi:.6e}]")
    
    if np.isclose(lo, hi):
        lo -= 1.0
        hi += 1.0

    try:
        fig = plt.figure(figsize=(6, 6))
        plt.scatter(x_true_valid, y_pred_valid, s=6, alpha=0.30, color="tab:blue")
        plt.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1)
        plt.xlabel("Ground truth")
        plt.ylabel("Prediction")
        plt.title(f"{name} | MAE={mae:.5g}, RMSE={rmse:.5g}, MAE/|GT|={mape_like:.3f}%")
        plt.xlim(lo, hi)
        plt.ylim(lo, hi)
        plt.tight_layout()
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        if verbose:
            print(f"    [OK] Plot saved to {out_path.name}")
    except Exception as e:
        if verbose:
            print(f"    [ERROR] Plot generation failed: {e}")
        return {"mae": mae, "rmse": rmse, "mae_over_mean_abs_gt_percent": mape_like, "status": "plot_failed", "error": str(e)}

    return {"mae": mae, "rmse": rmse, "mae_over_mean_abs_gt_percent": mape_like, "status": "ok"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate recovery/parity plots for HeFESTo bulk physub properties."
    )
    parser.add_argument("--bundle", required=True, help="Path to validation ML bundle (.tar.gz)")
    parser.add_argument("--model", required=True, help="Path to trained emulator model (.pt or .zip)")
    parser.add_argument(
        "--temperature-model",
        default=None,
        help="Checkpoint for temperature FCNN regressor (VariableGeometryFCNNRegressor)",
    )
    parser.add_argument(
        "--temperature-output-name",
        default=None,
        help="Optional explicit temperature output label in checkpoint selected_output_names",
    )
    parser.add_argument(
        "--temperature-input-mode",
        choices=["components", "components_molar_labels"],
        default="components",
        help=(
            "Input structure for temperature model. "
            "'components' uses extensive component moles only. "
            "'components_molar_labels' appends phase molar labels + intensive labels."
        ),
    )
    parser.add_argument(
        "--temperature-source",
        choices=["auto", "feature", "free_output"],
        default="auto",
        help="Where to pull ground-truth temperature from for parity (auto tries feature first then free output)",
    )
    parser.add_argument(
        "--bulk-selectors",
        nargs="+",
        default=list(PHYSUB_BULK_ATTRIBUTE_NAMES),
        help="Bulk attribute selectors to plot (subset of PHYSUB_BULK_ATTRIBUTE_NAMES)",
    )
    parser.add_argument("--max-samples", type=int, default=2**16, help="Max samples to plot")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument("--batch-size", type=int, default=4096, help="Batch size for temperature model")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument(
        "--no-normalize-features",
        action="store_true",
        help="Disable emulator feature normalization before inference",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for plots/metrics (default: validation/plots/recovery_physub_<model_stem>)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debugging output for bulk property calculations and plots",
    )
    return parser


def run_recovery_plots_bulk_physub(
    bundle_path: str,
    model_path: str,
    temperature_model_path: str,
    output_dir: str,
    temperature_output_name: Optional[str] = None,
    temperature_input_mode: str = "components",
    temperature_source: str = "auto",
    bulk_selectors: Optional[Sequence[str]] = None,
    max_samples: int = 2**16,
    seed: int = 1337,
    batch_size: int = 4096,
    use_cuda: bool = False,
    normalize_features: bool = True,
    verbose: bool = False,
) -> None:
    if verbose:
        print("[INFO] Starting bulk physub recovery plot generation...")
        print(f"  - Bundle: {bundle_path}")
        print(f"  - Model: {model_path}")
        print(f"  - Output dir: {output_dir}")
    
    rng = np.random.default_rng(seed)
    bundle = load_ml_bundle(_resolve_bundle_path(bundle_path))

    if bundle.features is None or bundle.labels is None or bundle.molar_labels is None:
        raise ValueError("Bundle must include features, labels, and molar_labels")

    model = rebuild_MELTS_model(model_path)
    emulator = NN_MELTS(model, cuda=use_cuda)

    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

    nrows = int(bundle.features.shape[0])
    if nrows == 0:
        raise ValueError("Bundle has zero rows")
    if max_samples < nrows:
        subset = rng.choice(nrows, size=max_samples, replace=False)
    else:
        subset = np.arange(nrows)

    features = np.asarray(bundle.features[subset], dtype=np.float32)
    labels = np.asarray(bundle.labels[subset], dtype=np.float32)
    molar_labels = np.asarray(bundle.molar_labels[subset], dtype=np.float32)
    free_outputs = None if bundle.free_outputs is None else np.asarray(bundle.free_outputs[subset], dtype=np.float32)

    ml_indexer = bundle.ml_indexer
    feature_names = list(getattr(ml_indexer, "featureNames", []) or [])
    free_output_names = list(getattr(ml_indexer, "freeOutputs", []) or [])
    component_names = list(getattr(ml_indexer, "label_names", []) or [])
    has_temperature_feature, temperature_feature_idx, temperature_feature_name = _has_temperature_feature(feature_names)

    if labels.ndim != 2:
        raise ValueError(f"labels must be a 2D intensive component matrix, got shape {labels.shape}")
    if molar_labels.ndim != 2:
        raise ValueError(f"molar_labels must be a 2D phase-mole matrix, got shape {molar_labels.shape}")

    expected_vc = int(getattr(ml_indexer, "ncompsVaried", labels.shape[1]))
    if labels.shape[1] != expected_vc:
        raise ValueError(
            "labels must be the intensive variable-component matrix with shape (B, VC); "
            f"got {labels.shape[1]} columns, expected VC={expected_vc}"
        )

    # make_phase_tables and related mappings use transformed pyroxene/spinel basis.
    px_sp_transform = ml_indexer.PxSpTransform
    comp_subset = ml_indexer.compositional_component_subset
    px_sp_sub = px_sp_transform[np.ix_(comp_subset, comp_subset)]
    labels_model = labels @ px_sp_sub

    features_t = torch.tensor(features, dtype=torch.float32, device=device)
    if normalize_features:
        normed_features_t = emulator.norm_features.norm(features_t)
    else:
        normed_features_t = features_t

    labels_model_t = torch.tensor(labels_model, dtype=torch.float32, device=device)
    molar_labels_t = torch.tensor(molar_labels, dtype=torch.float32, device=device)

    with torch.no_grad():
        gt_component_moles_t = emulator.getExtensiveComps(
            intensiveLabels=labels_model_t,
            molarLabels=molar_labels_t,
        )

    # Predicted extensive component moles from forwardMB selector path.
    pred_outputs = emulator.forwardMB(
        features_t,
        Normalize=normalize_features,
        WtPercent=False,
        outputs=["component_moles"],
    )
    pred_component_moles_t = pred_outputs["component_moles"].to(device)

    gt_temperature, gt_temp_source, gt_temp_name = _resolve_ground_truth_temperature(
        features=features,
        feature_names=feature_names,
        free_outputs=free_outputs,
        free_output_names=free_output_names,
        source_mode=temperature_source,
    )

    # Build temperature inputs only when the emulator does not already carry temperature as a feature.
    gt_component_moles_np = gt_component_moles_t.detach().cpu().numpy().astype(np.float32)
    pred_component_moles_np = pred_component_moles_t.detach().cpu().numpy().astype(np.float32)

    temp_model = None
    temp_payload = None
    temp_out_name = temperature_feature_name if has_temperature_feature else None
    if has_temperature_feature:
        if temperature_feature_idx is None:
            raise RuntimeError("Temperature feature detection failed unexpectedly")
        temperature_hat_gt = features[:, temperature_feature_idx].astype(np.float32)
        temperature_hat_pred = temperature_hat_gt.copy()
        temperature_mode = "feature"
    else:
        if temperature_model_path is None:
            raise ValueError(
                "temperature-model is required when the emulator feature vector does not include temperature"
            )

        if temperature_input_mode == "components":
            x_temp_gt = gt_component_moles_np
            x_temp_pred = pred_component_moles_np
        else:
            with torch.no_grad():
                _, transcomponent_hat_t, _, _, _, _, phase_moles_t = emulator.model.forward(
                    normed_features_t,
                    detailed=True,
                )
            pred_labels_np = (
                transcomponent_hat_t.detach().cpu().numpy().astype(np.float32)
                @ np.linalg.inv(px_sp_sub).astype(np.float32)
            )
            pred_moles_np = phase_moles_t.detach().cpu().numpy().astype(np.float32)

            x_temp_gt = np.concatenate([gt_component_moles_np, molar_labels, labels], axis=1).astype(np.float32)
            x_temp_pred = np.concatenate([pred_component_moles_np, pred_moles_np, pred_labels_np], axis=1).astype(np.float32)

        temp_model, temp_payload, x_min, x_range, y_min, y_range = _load_temperature_model(
            Path(temperature_model_path),
            device,
        )
        temp_out_idx, temp_out_name = _resolve_temperature_output_index(temp_payload, temperature_output_name)

        x_temp_gt_norm = _normalize_inputs(x_temp_gt, x_min, x_range)
        x_temp_pred_norm = _normalize_inputs(x_temp_pred, x_min, x_range)

        temperature_hat_gt = _predict_temperature(
            model=temp_model,
            x_norm=x_temp_gt_norm,
            y_min=y_min,
            y_range=y_range,
            output_index=temp_out_idx,
            device=device,
            batch_size=batch_size,
        )
        temperature_hat_pred = _predict_temperature(
            model=temp_model,
            x_norm=x_temp_pred_norm,
            y_min=y_min,
            y_range=y_range,
            output_index=temp_out_idx,
            device=device,
            batch_size=batch_size,
        )
        temperature_mode = "submodel"

    # Physub bulk properties from aligned component moles.
    context = get_hefesto_physub_context()
    gt_component_aligned_t = context.align_component_tensor(gt_component_moles_t, component_names=component_names)
    pred_component_aligned_t = context.align_component_tensor(pred_component_moles_t, component_names=component_names)

    if verbose:
        print("[DEBUG] Component mole alignment:")
        print(f"  - GT aligned shape: {gt_component_aligned_t.shape}")
        print(f"  - Pred aligned shape: {pred_component_aligned_t.shape}")
        print(f"  - GT aligned NaN: {torch.isnan(gt_component_aligned_t).sum()}, Inf: {torch.isinf(gt_component_aligned_t).sum()}")
        print(f"  - Pred aligned NaN: {torch.isnan(pred_component_aligned_t).sum()}, Inf: {torch.isinf(pred_component_aligned_t).sum()}")

    molar_mass_t = context.formula_mass_g_mol.to(device)

    # Compute bulk properties using ground-truth temperature
    if verbose:
        print(f"[DEBUG] Computing bulk properties at GT temperature: {temperature_hat_gt.mean():.2f} K")
    
    gt_bulk_t, resolved_names = compute_physub_bulk_matrix(
        component_moles=gt_component_aligned_t,
        molar_mass=molar_mass_t,
        component_attributes=None,  # Will be computed from temperature-dependent models
        selectors=bulk_selectors,
        temperature_k=float(np.mean(temperature_hat_gt)),  # Use mean GT temperature
        hefesto_context=context,
    )
    
    # Compute bulk properties using predicted temperature
    if verbose:
        print(f"[DEBUG] Computing bulk properties at predicted temperature: {temperature_hat_pred.mean():.2f} K")
    
    pred_bulk_t, _ = compute_physub_bulk_matrix(
        component_moles=pred_component_aligned_t,
        molar_mass=molar_mass_t,
        component_attributes=None,
        selectors=bulk_selectors,
        temperature_k=float(np.mean(temperature_hat_pred)),  # Use mean predicted temperature
        hefesto_context=context,
    )

    gt_bulk = gt_bulk_t.detach().cpu().numpy().astype(np.float32)
    pred_bulk = pred_bulk_t.detach().cpu().numpy().astype(np.float32)

    if verbose:
        print(f"[DEBUG] Bulk property computation ({len(resolved_names)} properties):")
        for j, name in enumerate(resolved_names):
            gt_col = gt_bulk[:, j]
            pred_col = pred_bulk[:, j]
            print(f"  {name}:")
            print(f"    - GT: shape={gt_col.shape}, NaN={np.isnan(gt_col).sum()}, Inf={np.isinf(gt_col).sum()}")
            print(f"    - GT range: [{np.nanmin(gt_col):.6e}, {np.nanmax(gt_col):.6e}]")
            print(f"    - Pred: shape={pred_col.shape}, NaN={np.isnan(pred_col).sum()}, Inf={np.isinf(pred_col).sum()}")
            print(f"    - Pred range: [{np.nanmin(pred_col):.6e}, {np.nanmax(pred_col):.6e}]")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metrics: Dict[str, Dict[str, float]] = {}

    # Temperature parity diagnostics.
    temp_gt_plot = output_path / "temperature_pred_from_gt_inputs.png"
    temp_pred_plot = output_path / "temperature_pred_from_pred_inputs.png"
    metrics["temperature_pred_from_gt_inputs"] = _plot_parity(
        gt_temperature,
        temperature_hat_gt,
        f"Temperature ({temp_out_name}) from GT model inputs",
        temp_gt_plot,
        verbose=verbose,
    )
    metrics["temperature_pred_from_pred_inputs"] = _plot_parity(
        gt_temperature,
        temperature_hat_pred,
        f"Temperature ({temp_out_name}) from predicted model inputs",
        temp_pred_plot,
        verbose=verbose,
    )

    # Bulk property parity plots.
    if verbose:
        print("[INFO] Generating bulk property parity plots...")
    for j, name in enumerate(resolved_names):
        out_path = output_path / f"bulk_{_sanitize_name(name)}_parity.png"
        metrics[name] = _plot_parity(gt_bulk[:, j], pred_bulk[:, j], f"Bulk {name}", out_path, verbose=verbose)

    summary = {
        "bundle": str(bundle_path),
        "model": str(model_path),
        "temperature_model": None if has_temperature_feature else str(temperature_model_path),
        "samples": int(len(subset)),
        "normalize_features": bool(normalize_features),
        "temperature_input_mode": temperature_input_mode,
        "temperature_mode": temperature_mode,
        "temperature_output_name": temp_out_name,
        "gt_temperature_source": gt_temp_source,
        "gt_temperature_column": gt_temp_name,
        "bulk_selectors": list(resolved_names),
        "note": (
            "Ground-truth temperature is read from features/free_outputs when needed; "
            "predicted temperature uses the bundle feature directly when present, or the "
            "FCNN temperature model otherwise."
        ),
        "metrics": metrics,
    }

    with (output_path / "bulk_physub_recovery_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    np.savez_compressed(
        output_path / "bulk_physub_recovery_arrays.npz",
        gt_bulk=gt_bulk,
        pred_bulk=pred_bulk,
        bulk_names=np.asarray(resolved_names),
        gt_temperature=gt_temperature,
        temperature_hat_gt_inputs=temperature_hat_gt,
        temperature_hat_pred_inputs=temperature_hat_pred,
    )

    print(f"Saved bulk physub recovery plots to: {output_path}")
    print(f"Resolved GT temperature source: {gt_temp_source} ({gt_temp_name})")
    if has_temperature_feature:
        print(f"Temperature feature used directly: {temperature_feature_name}")
    else:
        print(f"Temperature model output used: {temp_out_name}")
    print("Bulk properties plotted:")
    for name in resolved_names:
        print(f"  - {name}")
    
    if verbose:
        print("\n[INFO] Summary of metric results:")
        print("[TEMPERATURE METRICS]")
        for temp_key in ["temperature_pred_from_gt_inputs", "temperature_pred_from_pred_inputs"]:
            if temp_key in metrics:
                m = metrics[temp_key]
                status = m.get("status", "unknown")
                print(f"  {temp_key}: {status}")
                if status == "ok":
                    print(f"    MAE={m.get('mae', 'N/A'):.6e}, RMSE={m.get('rmse', 'N/A'):.6e}")
        print("[BULK PROPERTY METRICS]")
        ok_count = 0
        failed_count = 0
        for name in resolved_names:
            if name in metrics:
                m = metrics[name]
                status = m.get("status", "unknown")
                if status == "ok":
                    ok_count += 1
                    print(f"  {name}: OK (MAE={m.get('mae', 'N/A'):.6e})")
                else:
                    failed_count += 1
                    error = m.get("error", "")
                    error_str = f" - {error}" if error else ""
                    print(f"  {name}: {status}{error_str}")
        print(f"\n[SUMMARY] {ok_count} properties succeeded, {failed_count} failed")


def main() -> None:
    args = build_parser().parse_args()

    if args.out_dir is None:
        model_stem = Path(args.model).stem
        out_dir = Path(__file__).parent / "plots" / f"recovery_physub_{_sanitize_name(model_stem)}"
    else:
        out_dir = Path(args.out_dir)

    run_recovery_plots_bulk_physub(
        bundle_path=args.bundle,
        model_path=args.model,
        temperature_model_path=args.temperature_model,
        output_dir=str(out_dir),
        temperature_output_name=args.temperature_output_name,
        temperature_input_mode=args.temperature_input_mode,
        temperature_source=args.temperature_source,
        bulk_selectors=args.bulk_selectors,
        max_samples=args.max_samples,
        seed=args.seed,
        batch_size=args.batch_size,
        use_cuda=args.cuda,
        normalize_features=not args.no_normalize_features,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
