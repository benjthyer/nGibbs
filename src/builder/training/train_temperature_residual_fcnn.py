"""
Train FCNN regressors to predict temperature residuals against a reference adiabat.

Two models are trained:
1) [features, emulator-predicted molar labels, emulator-predicted intensive labels] -> T_residual
2) [features, GT molar labels, GT intensive labels] -> T_residual (GT path)

The residual target is T_true - reference_adiabat(P, S) where reference_adiabat is get_T
from math_utils. The NN learns the correction to the polynomial reference.

Bundle convention:
- test bundle: in-training validation (used each epoch)
- valid bundle: post-training holdout evaluation

Only one of the test/valid bundles is strictly required. If just one of the two
is present on disk, it is used for both roles (in-training validation and
post-training evaluation). If both are present, each keeps its own role as
described above.

Memory: per-bundle training arrays (normalized features, emulator predictions,
filtered labels) are built once in chunked passes over memmapped source
arrays and cached to disk under a workspace directory (see
builder.training.residual_workspace) rather than materialized fully in RAM up
front - see --ram-threshold-gb. A split whose arrays fit under that threshold
is still loaded fully into RAM for fast iteration (today's behavior); above
it, training streams from the on-disk workspace in chunks instead.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from ngibbs.engine.NN import VariableGeometryFCNNRegressor, rebuild_MELTS_model
from ngibbs.engine.emulator import NN_MELTS

from builder.training.adiabat_utils import (
    build_extended_input_normalizer_arrays,
    is_compositional,
    normalizer_arrays,
    normalizer_pairs,
    resolve_comp_feature_indices,
)
from builder.training.residual_workspace import (
    ResidualChunkedLoader,
    ResidualWorkspaceHandle,
    WORKSPACE_DIR_DEFAULT,
    get_or_build_residual_workspace,
)

ADAPTIVE_DROPOUT_MAX = 0.50
ADAPTIVE_DROPOUT_UP = 0.01
ADAPTIVE_DROPOUT_DOWN = 0.005
OVERFIT_RATIO = 1.02
DEFAULT_EMULATOR_BATCH_SIZE = 2 ** 16
MAX_HISTOGRAM_SAMPLE = 5_000_000

TEMPERATURE_LABEL_DEFAULT = "T(K)(System_main)"
TEMPERATURE_LABEL_ALIASES = (TEMPERATURE_LABEL_DEFAULT, "Temperature(System_main)")
P_FEATURE_NAME = "P(GPa)(System_main)"
S_FEATURE_NAME = "S(J/g/K)(System_main)"
P_FEATURE_ALIASES = (P_FEATURE_NAME, "Pressure(System_main)")
S_FEATURE_ALIASES = (S_FEATURE_NAME, "S(System_main) / mass(System_main)")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_hidden_dims(raw_values: Sequence[str]) -> List[int]:
    hidden_dims: List[int] = []
    for value in raw_values:
        for part in (p.strip() for p in value.split(",")):
            if not part:
                continue
            width = int(part)
            if width <= 0:
                raise ValueError("All hidden layer widths must be > 0")
            hidden_dims.append(width)
    if not hidden_dims:
        raise ValueError("hidden-dims must provide at least one positive integer")
    return hidden_dims


def _resolve_temperature_label(
    requested_label: Optional[str], available_output_names: Sequence[str]
) -> str:
    """Resolve the HeFESTo or MELTS temperature output name."""
    if requested_label is not None:
        if requested_label not in available_output_names:
            raise ValueError(
                f"Temperature label '{requested_label}' not found in free_outputs. "
                f"Available: {list(available_output_names)}"
            )
        return requested_label

    for label in TEMPERATURE_LABEL_ALIASES:
        if label in available_output_names:
            return label
    raise ValueError(
        "Could not find a supported temperature label in free_outputs. "
        f"Expected one of {list(TEMPERATURE_LABEL_ALIASES)}; "
        f"available: {list(available_output_names)}"
    )


def _resolve_feature_index(
    feature_aliases: Sequence[str], feature_names: Sequence[str], description: str
) -> int:
    """Resolve a canonical HeFESTo or MELTS feature name."""
    for name in feature_aliases:
        if name in feature_names:
            return feature_names.index(name)
    raise ValueError(
        f"Could not find a supported {description} feature in featureNames. "
        f"Expected one of {list(feature_aliases)}; available: {list(feature_names)}"
    )


def _build_bundle_paths(bundle_stem: Path) -> Dict[str, Path]:
    """Resolve train/test/valid bundle paths from a shared stem.

    Train is always required. Of test/valid, only one is strictly required —
    if the other is missing on disk, its path is aliased to the one that is
    present so it gets used for both the in-training validation role and the
    post-training holdout evaluation role (see module docstring).
    """
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
    if not paths["train"].exists():
        raise FileNotFoundError(f"Could not resolve required train bundle: {paths['train']}")

    have_test = paths["test"].exists()
    have_valid = paths["valid"].exists()
    if not have_test and not have_valid:
        raise FileNotFoundError(
            "Could not resolve either test or valid bundle; at least one is required -> "
            f"test: {paths['test']}, valid: {paths['valid']}"
        )
    if not have_test:
        print(f"Test bundle not found ({paths['test']}); using valid bundle for both roles.")
        paths["test"] = paths["valid"]
    elif not have_valid:
        print(f"Valid bundle not found ({paths['valid']}); using test bundle for both roles.")
        paths["valid"] = paths["test"]
    return paths


def _peek_bundle_metadata(bundle_path: Path) -> "object":
    """Load just the bundle's ml_indexer (feature/output names, Elkeys, any
    existing feature_normalizer) without extracting or reading any of the big
    per-row arrays - cheap, used to resolve names/indices once up front before
    building the (expensive) per-split residual workspaces."""
    tmp_base = REPO_ROOT / "data" / "tmp"
    tmp_base.mkdir(parents=True, exist_ok=True)
    extract_dir = Path(tempfile.mkdtemp(dir=tmp_base))
    try:
        with tarfile.open(bundle_path, "r:gz") as tar:
            members = [
                m for m in tar.getmembers()
                if m.name.lstrip("./").startswith("ml_indexer/")
            ]
            tar.extractall(path=extract_dir, members=members)
        from ngibbs.config.ml_indexer import load_ml_indexer_from_state
        return load_ml_indexer_from_state(str(extract_dir / "ml_indexer"))
    finally:
        shutil.rmtree(extract_dir, ignore_errors=True)


def _parse_adiabat_coefs(path: Path) -> Dict[str, float]:
    """Parse polynomial coefficients from a text file (original case preserved).

    Handles both the 5-parameter S+P form and the extended compositional form that
    adds linear+quadratic terms for each bulk-composition element:

        T(K) = b0 + b_S*S + b_S2*S^2 + b_P*P + b_P2*P^2
             [+ b_Si*Si + b_Si2*Si^2 + b_Mg*Mg + b_Mg2*Mg^2 + ...]

    Key normalisation: ``^2`` suffix is replaced with ``2`` so "b_S^2" becomes "b_S2",
    "b_Si^2" becomes "b_Si2", etc.  Original case is preserved so "b_Si" stays "b_Si"
    (not "b_si"), which is required for later name-matching against featureNames.
    At minimum b0, b_S, b_S2, b_P, b_P2 must be present.
    """
    coefs: Dict[str, float] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if "=" not in line:
                continue
            left, _, right = line.partition("=")
            token = left.strip().split()[0]          # first word, original case
            key = token[:-2] + "2" if token.endswith("^2") else token
            try:
                coefs[key] = float(right.strip())
            except ValueError:
                continue
    missing = {"b0", "b_S", "b_S2", "b_P", "b_P2"} - set(coefs)
    if missing:
        raise ValueError(f"Adiabat coefficients file '{path}' is missing keys: {missing}")
    return coefs


def _make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _materialize_full(workspace: ResidualWorkspaceHandle, kind: str) -> Tuple[np.ndarray, np.ndarray]:
    feats = np.array(workspace.features_norm)
    a, b = workspace.secondary_arrays(kind)
    x = np.concatenate([feats, np.array(a), np.array(b)], axis=1).astype(np.float32)
    y = np.array(workspace.y_norm, dtype=np.float32)
    return x, y


def _make_loader_for_split(
    workspace: ResidualWorkspaceHandle,
    kind: str,
    batch_size: int,
    ram_threshold_bytes: float,
    chunk_rows: int,
):
    """Full in-RAM DataLoader when this split fits under the RAM threshold
    (today's fast path); an async chunked memmap loader above it, so training
    scales to datasets much larger than available RAM.

    Gated on `workspace.total_bytes()` (both "emulator" and "gt" arrays
    combined), not just this `kind`'s own arrays: both loaders for a split are
    built before training starts, so their materialized tensors coexist in
    RAM - checking `kind` alone lets each pass an 8GB threshold independently
    while the pair together exceeds it.
    """
    total_bytes = workspace.total_bytes()
    if total_bytes <= ram_threshold_bytes:
        x, y = _materialize_full(workspace, kind)
        return _make_loader(x, y, batch_size, shuffle=True)
    print(
        f"[{kind}] split total (emulator+gt combined) {total_bytes / 1024**3:.2f} GiB > --ram-threshold-gb "
        f"({ram_threshold_bytes / 1024**3:.2f} GiB); using chunked memmap loader"
    )
    return ResidualChunkedLoader(workspace, kind, batch_size, chunk_rows=chunk_rows)


def _sample_residuals(workspace: ResidualWorkspaceHandle, max_sample: int = MAX_HISTOGRAM_SAMPLE) -> np.ndarray:
    """Bounded, memmap-friendly sample of the raw residual array for the
    diagnostic histogram - a strided read instead of `np.array(residual)`,
    which at large scale would itself materialize a full-size copy just to
    plot it."""
    n = workspace.residual.shape[0]
    if n <= max_sample:
        return np.array(workspace.residual)
    step = max(1, n // max_sample)
    return np.array(workspace.residual[::step])


def _evaluate_temperature_from_workspace(
    model: torch.nn.Module,
    workspace: ResidualWorkspaceHandle,
    kind: str,
    y_min: np.ndarray,
    y_range: np.ndarray,
    device: torch.device,
    batch_size: int = 8192,
) -> Dict[str, float]:
    """Evaluate in temperature space (residual prediction + reference adiabat),
    chunked over the workspace so peak RAM stays O(batch_size) regardless of
    split size."""
    model.eval()
    sq_err_sum = 0.0
    abs_err_sum = 0.0
    n_total = 0
    with torch.no_grad():
        for start in range(0, workspace.n_rows, batch_size):
            end = min(start + batch_size, workspace.n_rows)
            feats = np.array(workspace.features_norm[start:end])
            a, b = workspace.secondary_arrays(kind)
            x = np.concatenate([feats, np.array(a[start:end]), np.array(b[start:end])], axis=1).astype(np.float32)
            residual_norm = model(torch.tensor(x, dtype=torch.float32, device=device)).cpu().numpy()
            T_ref_chunk = np.array(workspace.T_ref[start:end])
            T_true_chunk = np.array(workspace.T_true[start:end])
            T_pred = residual_norm.reshape(-1) * y_range[0] + y_min[0] + T_ref_chunk
            diff = T_pred - T_true_chunk.reshape(-1)
            sq_err_sum += float(np.sum(diff ** 2))
            abs_err_sum += float(np.sum(np.abs(diff)))
            n_total += end - start
    return {"mse": sq_err_sum / n_total, "mae": abs_err_sum / n_total}


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
    checkpoint_saver: Optional[Callable] = None,
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
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * xb.shape[0]
            train_count += xb.shape[0]
        mean_train_loss = train_loss_sum / train_count
        history["train_loss"].append(mean_train_loss)

        if valid_loader is not None:
            model.eval()
            valid_loss_sum = 0.0
            valid_count = 0
            with torch.no_grad():
                for xb, yb in tqdm(valid_loader, desc=f"Epoch {epoch + 1} valid", leave=False, unit="batch"):
                    xb, yb = xb.to(device), yb.to(device)
                    loss = criterion(model(xb), yb)
                    valid_loss_sum += float(loss.item()) * xb.shape[0]
                    valid_count += xb.shape[0]
            mean_valid_loss = valid_loss_sum / valid_count
        else:
            mean_valid_loss = float("nan")
        history["valid_loss"].append(mean_valid_loss)

        if valid_loader is not None and np.isfinite(mean_valid_loss):
            if mean_valid_loss < best_valid_loss:
                best_valid_loss = mean_valid_loss
                epochs_without_improvement = 0
                if best_checkpoint_path is not None:
                    best_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    if checkpoint_saver is not None:
                        checkpoint_saver(model, history)
                    else:
                        torch.save(model.state_dict(), best_checkpoint_path)
                    print(f"Epoch {epoch + 1}: improved to {mean_valid_loss:.6f}, saved checkpoint")
            else:
                epochs_without_improvement += 1

            if mean_valid_loss > mean_train_loss * OVERFIT_RATIO:
                new_dropout = min(dropout_rate + ADAPTIVE_DROPOUT_UP, ADAPTIVE_DROPOUT_MAX)
                if new_dropout > dropout_rate:
                    print(f"Overfitting: dropout {dropout_rate:.2f} -> {new_dropout:.2f}")
                dropout_rate = new_dropout
            elif mean_valid_loss < mean_train_loss:
                new_dropout = max(dropout_rate - ADAPTIVE_DROPOUT_DOWN, 0.0)
                if new_dropout < dropout_rate:
                    print(f"Underfitting: dropout {dropout_rate:.2f} -> {new_dropout:.2f}")
                dropout_rate = new_dropout
            model.set_dropout_rate(dropout_rate)

        history["dropout"].append(dropout_rate)
        print(
            f"Epoch {epoch + 1:4d}/{epochs}: "
            f"train={mean_train_loss:.6f}, valid={mean_valid_loss:.6f}, "
            f"dropout={dropout_rate:.2f}, patience={epochs_without_improvement}/{patience}"
        )
        if valid_loader is not None and epochs_without_improvement >= patience:
            print(f"\nEarly stopping: no improvement for {patience} epochs")
            break

    history["best_valid_loss"] = best_valid_loss
    history["best_checkpoint_path"] = str(best_checkpoint_path) if best_checkpoint_path else None
    return history


def _save_checkpoint(
    out_path: Path,
    model: VariableGeometryFCNNRegressor,
    temperature_label: str,
    p_feature_idx: int,
    s_feature_idx: int,
    x_min: np.ndarray,
    x_range: np.ndarray,
    y_min: np.ndarray,
    y_range: np.ndarray,
    input_kind: str,
    metrics: Dict,
    history: Dict,
    is_melts: bool = False,
    adiabat_coefs: Optional[Dict[str, float]] = None,
    comp_indices: Optional[Dict[str, object]] = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": model.config,
            "temperature_label": temperature_label,
            "p_feature_idx": int(p_feature_idx),
            "s_feature_idx": int(s_feature_idx),
            "input_min": x_min.astype(np.float32),
            "input_range": x_range.astype(np.float32),
            "input_min_range": normalizer_pairs(x_min, x_range),
            "target_min": y_min,
            "target_range": y_range,
            "target_min_range": normalizer_pairs(y_min, y_range),
            "input_kind": input_kind,
            "is_melts": is_melts,
            "adiabat_coefs": adiabat_coefs,
            "coef_feature_indices": comp_indices,
            "metrics": metrics,
            "history": history,
        },
        out_path,
    )


def _load_checkpoint_state_dict(checkpoint_path: Path, device: torch.device) -> Dict:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError(f"Unsupported checkpoint format: {checkpoint_path}")


def _save_residual_histogram(
    residuals: np.ndarray,
    out_path: Path,
    max_abs: Optional[float] = None,
) -> None:
    """Best-effort residual histogram (log-y full range + linear-y central zoom).

    Never raises: if matplotlib is unavailable (e.g. headless training box), it
    just prints a note and returns. Draws the max_abs cut lines when provided so
    the effect of --max-abs-residual is visible.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"Skipping residual histogram (matplotlib unavailable: {exc})")
        return
    r = np.asarray(residuals, dtype=np.float64).reshape(-1)
    if r.size == 0:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(r, bins=300, color="steelblue")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("residual = T_true - T_ref (K)")
    axes[0].set_ylabel("count (log)")
    axes[0].set_title("Full range (log-y)")
    lim = float(max_abs) if max_abs is not None else float(np.percentile(np.abs(r), 99.9))
    if max_abs is not None:
        for x in (max_abs, -max_abs):
            axes[0].axvline(x, color="red", ls="--", lw=0.9)
    central = r[np.abs(r) <= lim]
    axes[1].hist(central, bins=200, color="seagreen")
    axes[1].set_xlabel(f"residual (K), |r| <= {lim:.0f}")
    axes[1].set_ylabel("count")
    axes[1].set_title("Central region (linear-y)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"Saved residual histogram -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train two FCNN regressors predicting temperature residuals against reference_adiabat(P,S): "
            "(1) features+emulator-predicted moles+intensive, (2) features+GT moles+intensive."
        )
    )
    parser.add_argument("--bundle-stem", required=True, type=Path)
    parser.add_argument("--emulator-model", required=True, type=Path)
    parser.add_argument(
        "--temperature-label",
        default=None,
        help=(
            "Name of temperature output in bundle free_outputs. "
            f"If omitted, auto-detects {TEMPERATURE_LABEL_DEFAULT} or "
            "Temperature(System_main)."
        ),
    )
    parser.add_argument("--hidden-dims", nargs="+", default=["256", "128", "64"])
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--activation-leak", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--skip-1", action="store_true", help="Skip model 1 (emulator-predicted path)")
    parser.add_argument("--isMELTS", action="store_true", help="Bundle uses MELTS units (P in bars, T in Celsius). Converts P bars→GPa before the reference adiabat call and converts the reference result K→Celsius so residuals and targets are in Celsius.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("src") / "builder" / "training" / "temp_models",
    )
    parser.add_argument("--emulator-batch-size", type=int, default=DEFAULT_EMULATOR_BATCH_SIZE)
    parser.add_argument(
        "--adiabat-coefs",
        type=Path,
        default=None,
        help=(
            "Path to a text file containing 5 polynomial coefficients for the reference "
            "adiabat T(K) = b0 + b_S*S + b_S2*S^2 + b_P*P + b_P2*P^2 (P in GPa). "
            "When provided, these coefficients are used instead of the built-in get_T "
            "function and are stored in the output .pt checkpoint. "
            "Example: data/MELTStables/HeFESTo/T_from_S_P_coefs_R2_0p98602.txt"
        ),
    )
    parser.add_argument(
        "--s-min",
        type=float,
        default=None,
        help=(
            "Minimum entropy S (J/g/K), exclusive. Rows with S <= s-min are dropped from "
            "all bundle splits before training, mirroring the KEEP_RANGE convention in "
            "scripts/fit_T_from_S_and_P.py. Default: no lower bound."
        ),
    )
    parser.add_argument(
        "--s-max",
        type=float,
        default=None,
        help=(
            "Maximum entropy S (J/g/K), exclusive. Rows with S >= s-max are dropped. "
            "Default: no upper bound."
        ),
    )
    parser.add_argument(
        "--max-abs-residual",
        type=float,
        default=None,
        help=(
            "Drop rows whose |T_true - reference_adiabat| exceeds this many K, after "
            "residuals are computed, across all splits. These extreme residuals are "
            "reference-adiabat extrapolation blowups at compositions outside the fit "
            "domain (e.g. very high Cr) and otherwise dominate the target normalizer's "
            "dynamic range. Typical values 100-500. Default: no residual filtering."
        ),
    )
    parser.add_argument(
        "--adiabat-elem-norm",
        type=float,
        default=24.0,
        help=(
            "Target sum of (cations + oxygen) atoms in the formula unit the "
            "compositional reference adiabat was fit against. ML bundles store cation "
            "atom fractions (all Elkeys sum to 1) with iron split across signed Fe/Fe3 "
            "columns and no oxygen column; at eval time oxygen is derived and the whole "
            "composition is rescaled per row so cations+O sum to this value (default: "
            "24.0, matching scripts/fit_T_from_S_and_bulk.py). Only used when "
            "--adiabat-coefs is a compositional file."
        ),
    )
    parser.add_argument(
        "--ram-threshold-gb",
        type=float,
        default=8.0,
        help=(
            "If a split's training arrays total at or below this many GiB, materialize "
            "them fully in RAM (fast, today's behavior for small/medium datasets). Above "
            "it, stream from an on-disk memory-mapped workspace in chunks instead, so "
            "training scales to datasets far larger than available RAM. Measured against "
            "the uncompressed workspace size, not the .tar.gz bundle size."
        ),
    )
    parser.add_argument(
        "--workspace-dir",
        type=Path,
        default=None,
        help=(
            "Where to cache the per-bundle memory-mapped training workspace (normalized "
            "features, emulator predictions, filtered labels/residuals). Default: "
            f"{WORKSPACE_DIR_DEFAULT}. Reused across runs against the same bundle + "
            "emulator + filtering params; rebuilt automatically when any of those change."
        ),
    )
    parser.add_argument(
        "--workspace-chunk-size",
        type=int,
        default=1_000_000,
        help="Row chunk size used while building the on-disk workspace (residual "
             "computation, normalization, filtering). Does not affect training results.",
    )
    parser.add_argument(
        "--loader-chunk-rows",
        type=int,
        default=1_000_000,
        help="Row chunk size for the chunked training loader when a split exceeds "
             "--ram-threshold-gb. Does not affect training results.",
    )

    args = parser.parse_args()
    if args.emulator_batch_size <= 0:
        raise ValueError(f"--emulator-batch-size must be > 0, got {args.emulator_batch_size}")

    set_seed(args.seed)
    hidden_dims = _parse_hidden_dims(args.hidden_dims)

    adiabat_coefs: Optional[Dict[str, float]] = None
    if args.adiabat_coefs is not None:
        adiabat_coefs = _parse_adiabat_coefs(args.adiabat_coefs)
        print(f"Using custom adiabat coefficients from {args.adiabat_coefs}: {adiabat_coefs}")
    else:
        print("Using built-in get_T as reference adiabat (no --adiabat-coefs provided)")

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    emulator = NN_MELTS(rebuild_MELTS_model(args.emulator_model), cuda=(device.type == "cuda"))

    bundle_paths = _build_bundle_paths(args.bundle_stem)

    # Cheap peek (ml_indexer only, no big arrays) to resolve names/indices once,
    # before building the per-split workspaces.
    ml_indexer = _peek_bundle_metadata(bundle_paths["train"])
    available_output_names = list(getattr(ml_indexer, "free_outputs", []) or [])
    if not available_output_names:
        raise ValueError("ml_indexer.free_outputs is missing or empty in train bundle")
    args.temperature_label = _resolve_temperature_label(
        args.temperature_label, available_output_names
    )
    print(f"Using temperature label: {args.temperature_label}")
    temp_output_idx = available_output_names.index(args.temperature_label)

    feature_names = list(getattr(ml_indexer, "featureNames", []) or [])
    p_idx = _resolve_feature_index(P_FEATURE_ALIASES, feature_names, "pressure")
    s_idx = _resolve_feature_index(S_FEATURE_ALIASES, feature_names, "entropy")
    print(f"P feature index: {p_idx} ({feature_names[p_idx]})")
    print(f"S feature index: {s_idx} ({feature_names[s_idx]})")

    el_keys = list(getattr(ml_indexer, "Elkeys", []) or [])
    comp_indices: Optional[Dict[str, object]] = None
    if adiabat_coefs is not None and is_compositional(adiabat_coefs):
        comp_indices = resolve_comp_feature_indices(adiabat_coefs, feature_names, el_keys, norm=args.adiabat_elem_norm)
        print(f"Compositional reference adiabat (norm={args.adiabat_elem_norm}, Elkeys={el_keys}):")
        print(f"  elem→weighted-cols: {comp_indices['elem_weighted']}")
        print(f"  cation cols: {comp_indices['cation_cols']}")

    existing_feature_normalizer = getattr(ml_indexer, "feature_normalizer", None)

    workspace_root = args.workspace_dir if args.workspace_dir is not None else WORKSPACE_DIR_DEFAULT
    ram_threshold_bytes = args.ram_threshold_gb * 1024 ** 3

    common_ws_kwargs = dict(
        n_named_features=len(feature_names), p_idx=p_idx, s_idx=s_idx,
        temperature_label_idx=temp_output_idx, is_melts=args.isMELTS,
        adiabat_coefs=adiabat_coefs, comp_indices=comp_indices,
        adiabat_elem_norm=args.adiabat_elem_norm,
        s_min=args.s_min, s_max=args.s_max, max_abs_residual=args.max_abs_residual,
        emulator=emulator, emulator_model_path=args.emulator_model, device=device,
        emulator_batch_size=args.emulator_batch_size, chunk_size=args.workspace_chunk_size,
    )

    train_ws, feature_normalizer, output_normalizer = get_or_build_residual_workspace(
        bundle_paths["train"], workspace_root, split_label="train",
        feature_normalizer=existing_feature_normalizer, output_normalizer=None,
        fit_normalizers=True, **common_ws_kwargs,
    )
    test_ws, _, _ = get_or_build_residual_workspace(
        bundle_paths["test"], workspace_root, split_label="test",
        feature_normalizer=feature_normalizer, output_normalizer=output_normalizer,
        fit_normalizers=False, **common_ws_kwargs,
    )
    if bundle_paths["valid"] == bundle_paths["test"]:
        valid_ws = test_ws
    else:
        valid_ws, _, _ = get_or_build_residual_workspace(
            bundle_paths["valid"], workspace_root, split_label="valid",
            feature_normalizer=feature_normalizer, output_normalizer=output_normalizer,
            fit_normalizers=False, **common_ws_kwargs,
        )

    # Histogram of the (post-filter) train residuals so the extrapolation tail
    # and the effect of --max-abs-residual are visible.
    _save_residual_histogram(
        _sample_residuals(train_ws),
        args.out_dir / f"residual_histogram_{args.bundle_stem.name}.png",
        max_abs=args.max_abs_residual,
    )
    print(
        f"Train residuals (K): min={float(train_ws.residual.min()):.1f}, "
        f"max={float(train_ws.residual.max()):.1f}, mean={float(np.mean(train_ws.residual)):.1f}"
    )

    y_min, y_range = normalizer_arrays(output_normalizer)
    x1_min, x1_range = normalizer_arrays(feature_normalizer)

    x1_dim = train_ws.x_dim("emulator")
    x2_dim = train_ws.x_dim("gt")
    x1_emul_min, x1_emul_range = build_extended_input_normalizer_arrays(x1_min, x1_range, x1_dim)
    x2_min, x2_range = build_extended_input_normalizer_arrays(x1_min, x1_range, x2_dim)

    train_loader_x1 = _make_loader_for_split(train_ws, "emulator", args.batch_size, ram_threshold_bytes, args.loader_chunk_rows)
    valid_loader_x1 = _make_loader_for_split(test_ws, "emulator", args.batch_size, ram_threshold_bytes, args.loader_chunk_rows)
    train_loader_x2 = _make_loader_for_split(train_ws, "gt", args.batch_size, ram_threshold_bytes, args.loader_chunk_rows)
    valid_loader_x2 = _make_loader_for_split(test_ws, "gt", args.batch_size, ram_threshold_bytes, args.loader_chunk_rows)

    metrics: Dict[str, Dict] = {"emulator_path": {}, "gt_path": {}}
    bundle_basename = args.bundle_stem.name
    best_model1_path = args.out_dir / f"best_temperature_emulator_path_{bundle_basename}.pt"
    best_model2_path = args.out_dir / f"best_temperature_gt_path_{bundle_basename}.pt"

    def _make_saver(out_path, x_min_, x_range_, kind, metrics_ref):
        def saver(m, hist):
            _save_checkpoint(out_path, m, args.temperature_label, p_idx, s_idx, x_min_, x_range_, y_min, y_range, kind, metrics_ref, hist, is_melts=args.isMELTS, adiabat_coefs=adiabat_coefs, comp_indices=comp_indices)
        return saver

    # Train model 1 (emulator-predicted path)
    model_emulator = VariableGeometryFCNNRegressor(
        input_dim=x1_dim, output_dim=1,
        hidden_dims=hidden_dims, activation_leak=args.activation_leak, dropout=0.0,
    )
    history_emulator = None
    if not args.skip_1:
        print("\nTraining model 1/2: [features, emul-predicted moles+intensive] -> T_residual")
        history_emulator = _train_model(
            model_emulator, train_loader_x1, valid_loader_x1,
            args.epochs, args.lr, args.weight_decay, device, args.patience,
            best_model1_path, _make_saver(best_model1_path, x1_emul_min, x1_emul_range, "emulator_path", metrics["emulator_path"]),
        )
    else:
        print("\nSkipping model 1/2 (--skip-1)")

    # Train model 2 (GT path)
    model_gt = VariableGeometryFCNNRegressor(
        input_dim=x2_dim, output_dim=1,
        hidden_dims=hidden_dims, activation_leak=args.activation_leak, dropout=0.0,
    )
    print("\nTraining model 2/2: [features, GT moles+intensive] -> T_residual")
    history_gt = _train_model(
        model_gt, train_loader_x2, valid_loader_x2,
        args.epochs, args.lr, args.weight_decay, device, args.patience,
        best_model2_path, _make_saver(best_model2_path, x2_min, x2_range, "gt_path", metrics["gt_path"]),
    )

    # Reload best checkpoints for final evaluation
    if best_model1_path.exists():
        model_emulator.load_state_dict(_load_checkpoint_state_dict(best_model1_path, device))
    if best_model2_path.exists():
        model_gt.load_state_dict(_load_checkpoint_state_dict(best_model2_path, device))

    # Evaluate in temperature space (K)
    metrics["emulator_path"]["train"] = _evaluate_temperature_from_workspace(model_emulator, train_ws, "emulator", y_min, y_range, device)
    metrics["emulator_path"]["test"] = _evaluate_temperature_from_workspace(model_emulator, test_ws, "emulator", y_min, y_range, device)
    metrics["emulator_path"]["valid"] = _evaluate_temperature_from_workspace(model_emulator, valid_ws, "emulator", y_min, y_range, device)
    metrics["gt_path"]["train"] = _evaluate_temperature_from_workspace(model_gt, train_ws, "gt", y_min, y_range, device)
    metrics["gt_path"]["test"] = _evaluate_temperature_from_workspace(model_gt, test_ws, "gt", y_min, y_range, device)
    metrics["gt_path"]["valid"] = _evaluate_temperature_from_workspace(model_gt, valid_ws, "gt", y_min, y_range, device)

    # Save final checkpoints
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model1_out_str = "Not Trained!"
    if not args.skip_1 and history_emulator is not None:
        model1_out = args.out_dir / f"temperature_residual_emulator_path_{bundle_basename}.pt"
        _save_checkpoint(model1_out, model_emulator, args.temperature_label, p_idx, s_idx, x1_emul_min, x1_emul_range, y_min, y_range, "emulator_path", metrics["emulator_path"], history_emulator, is_melts=args.isMELTS, adiabat_coefs=adiabat_coefs, comp_indices=comp_indices)
        model1_out_str = str(model1_out)

    model2_out = args.out_dir / f"temperature_residual_gt_path_{bundle_basename}.pt"
    _save_checkpoint(model2_out, model_gt, args.temperature_label, p_idx, s_idx, x2_min, x2_range, y_min, y_range, "gt_path", metrics["gt_path"], history_gt, is_melts=args.isMELTS, adiabat_coefs=adiabat_coefs, comp_indices=comp_indices)

    metrics_path = args.out_dir / f"temperature_residual_metrics_{bundle_basename}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "temperature_label": args.temperature_label,
                "p_feature_idx": p_idx,
                "s_feature_idx": s_idx,
                "p_feature_name": P_FEATURE_NAME,
                "s_feature_name": S_FEATURE_NAME,
                "is_melts": args.isMELTS,
                "s_min": args.s_min,
                "s_max": args.s_max,
                "hidden_dims": hidden_dims,
                "bundle_paths": {k: str(v) for k, v in bundle_paths.items()},
                "normalizers": {
                    "emulator_path_input_min_range": normalizer_pairs(x1_emul_min, x1_emul_range).tolist(),
                    "gt_path_input_min_range": normalizer_pairs(x2_min, x2_range).tolist(),
                    "residual_min_range": normalizer_pairs(y_min, y_range).tolist(),
                },
                "metrics": metrics,
                "model_paths": {"emulator_path": model1_out_str, "gt_path": str(model2_out)},
                "emulator_model": str(args.emulator_model),
            },
            f,
            indent=2,
        )

    print("\nTraining complete.")
    print(f"Model 1 (emulator path): {model1_out_str}")
    print(f"Model 2 (GT path):       {model2_out}")
    print(f"Metrics: {metrics_path}")
    print("Evaluation (K):")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
