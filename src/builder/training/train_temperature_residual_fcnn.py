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
"""

from __future__ import annotations

import argparse
import json
import random
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
from ngibbs.utils.file_utils import load_ml_bundle
from ngibbs.utils.math_utils import Normalizer, get_T as reference_adiabat

ADAPTIVE_DROPOUT_MAX = 0.50
ADAPTIVE_DROPOUT_UP = 0.01
ADAPTIVE_DROPOUT_DOWN = 0.005
OVERFIT_RATIO = 1.02
DEFAULT_EMULATOR_BATCH_SIZE = 2 ** 16

TEMPERATURE_LABEL_DEFAULT = "T(K)(System_main)"
P_FEATURE_NAME = "P(GPa)(System_main)"
S_FEATURE_NAME = "S(J/g/K)(System_main)"

# Oxygen atoms contributed per cation atom in each end-member oxide.
# Used to compute O when "O" appears in a compositional reference adiabat.
_OXIDE_O_PER_CATION: Dict[str, float] = {
    "Si": 2.0, "Ti": 2.0, "Al": 1.5,
    "Fe": 1.0, "Fe3": 1.5,   # FeO, Fe2O3
    "Mg": 1.0, "Ca": 1.0, "Mn": 1.0, "Ni": 1.0,
    "Na": 0.5, "K": 0.5,     # Na2O, K2O
    "Cr": 1.5,                # Cr2O3
    "P": 2.5,                 # P2O5
    "H": 0.5,                 # H2O
    "C": 2.0,                 # CO2
}


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


def _fit_minmax(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y_min = np.min(y, axis=0)
    y_max = np.max(y, axis=0)
    y_range = np.where((y_max - y_min) < 1e-7, 1.0, y_max - y_min)
    return y_min.astype(np.float32), y_range.astype(np.float32)


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


def _ensure_feature_normalizer(ml_indexer, train_features: np.ndarray) -> Normalizer:
    existing = getattr(ml_indexer, "feature_normalizer", None)
    n_features_total = train_features.shape[1]
    n_named_features = len(getattr(ml_indexer, "featureNames", []) or [])

    if existing is not None:
        min_tensor = existing.miner.detach().cpu().to(torch.float32).clone()
        range_tensor = existing.ranger.detach().cpu().to(torch.float32).clone()
        if min_tensor.numel() != n_features_total or range_tensor.numel() != n_features_total:
            raise ValueError(
                f"Existing feature_normalizer dimension mismatch: "
                f"expected {n_features_total}, got min={min_tensor.numel()} range={range_tensor.numel()}"
            )
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
        ranges = torch.clamp(torch.max(x_named, dim=0).values - mins, min=1e-7)
        min_tensor[:n_named_features] = mins
        range_tensor[:n_named_features] = ranges
    normalizer = Normalizer(min_tensor=min_tensor, range_tensor=range_tensor)
    ml_indexer.feature_normalizer = normalizer
    return normalizer


def _build_output_normalizer(y: np.ndarray) -> Normalizer:
    y_min, y_range = _fit_minmax(y)
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
    base_mins: np.ndarray,
    base_ranges: np.ndarray,
    extended_dim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if extended_dim < len(base_mins):
        raise ValueError(f"extended_dim={extended_dim} < base dim={len(base_mins)}")
    mins = np.zeros(extended_dim, dtype=np.float32)
    ranges = np.ones(extended_dim, dtype=np.float32)
    mins[: len(base_mins)] = base_mins
    ranges[: len(base_ranges)] = base_ranges
    return mins, ranges


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


def _is_compositional(coefs: Dict[str, float]) -> bool:
    """Return True if coefs contains element terms beyond the basic S+P polynomial."""
    basic = {"b0", "b_S", "b_S2", "b_P", "b_P2"}
    return any(k not in basic for k in coefs)


def _resolve_comp_feature_indices(
    coefs: Dict[str, float],
    feature_names: List[str],
    el_keys: List[str],
    norm: float = 24.0,
) -> Dict[str, object]:
    """Build the structure needed to evaluate a compositional reference adiabat.

    The regression (see scripts/fit_T_from_S_and_bulk.py) was fit on a bulk
    composition expressed as atoms in a formula unit whose cations + oxygen sum
    to `norm` (default 24), with total iron as a single "Fe" column and oxygen
    "O" derived from oxide stoichiometry.

    ML-bundle features instead store per-element atom FRACTIONS (all Elkeys sum
    to 1) with iron split across signed "Fe"/"Fe3" columns and no oxygen column.
    The two do not translate by a fixed per-element scale: the fit's total=norm
    convention makes the effective scale composition-dependent. So rather than
    baking a constant into each weight, we store the raw oxide stoichiometry and
    recompute the scale per row in _eval_adiabat_poly:

      1. total Fe   = Fe + Fe3
      2. O          = Σ_i stoich_i · frac_i     (oxide O per cation; Fe→1, Fe3→1.5, …)
      3. grand_total = Σ cation fractions + O
      4. scale       = norm / grand_total
      5. X_i = scale · (raw element),  X_O = scale · O

    Feature layout is [featureNames | Elkeys], so element columns start at
    offset = len(feature_names).

    Returns a dict with:
      elem_weighted: {elem: [(col_idx, stoich_weight), ...]}  raw weights (no norm)
                     "Fe" → Fe + Fe3 columns; "O" → every Elkeys col × oxide stoich
      cation_cols:   [col_idx, ...] every Elkeys column (used for the cation sum)
      norm_total:    float target sum of (cations + O)
    """
    basic_keys = {"b0", "b_S", "b_S2", "b_P", "b_P2"}
    element_names: List[str] = []
    for key in coefs:
        if key in basic_keys or key.endswith("2") or not key.startswith("b_"):
            continue
        elem = key[2:]  # strip "b_" prefix, e.g. "b_Mg" → "Mg"
        if elem not in element_names:
            element_names.append(elem)

    offset = len(feature_names)
    cation_cols = [offset + i for i in range(len(el_keys))]
    elem_weighted: Dict[str, List[Tuple[int, float]]] = {}

    for elem in element_names:
        if elem == "Fe":
            weighted: List[Tuple[int, float]] = []
            for fe_key in ("Fe", "Fe3"):
                if fe_key in el_keys:
                    weighted.append((offset + el_keys.index(fe_key), 1.0))
            if not weighted:
                raise ValueError(f"No Fe/Fe3 keys found in el_keys for 'Fe': {el_keys}")
            elem_weighted["Fe"] = weighted
        elif elem == "O":
            weighted = []
            for el in el_keys:
                if el not in _OXIDE_O_PER_CATION:
                    raise ValueError(
                        f"No oxide oxygen stoichiometry known for element '{el}'; "
                        f"cannot compute O for adiabat. Known: {sorted(_OXIDE_O_PER_CATION)}"
                    )
                weighted.append((offset + el_keys.index(el), _OXIDE_O_PER_CATION[el]))
            elem_weighted["O"] = weighted
        else:
            if elem not in el_keys:
                raise ValueError(
                    f"Element '{elem}' from adiabat coefs not found in ml_indexer.Elkeys: {el_keys}"
                )
            elem_weighted[elem] = [(offset + el_keys.index(elem), 1.0)]

    return {
        "elem_weighted": elem_weighted,
        "cation_cols": cation_cols,
        "norm_total": float(norm),
    }


def _eval_compositional_terms(
    T,
    coefs: Dict[str, float],
    features: np.ndarray,
    comp_indices: Dict[str, object],
):
    """Add compositional adiabat terms to base T. See _resolve_comp_feature_indices.

    Reconstructs the fit's total=norm formula-unit composition from bundle atom
    fractions (per row) before applying b_Xi*X_i + b_Xi2*X_i^2 for each element.
    """
    elem_weighted: Dict[str, List[Tuple[int, float]]] = comp_indices["elem_weighted"]
    cation_cols: List[int] = comp_indices["cation_cols"]
    norm_total: float = comp_indices["norm_total"]
    n = features.shape[0]

    # Raw (fraction-scale) quantity per polynomial element term.
    raw: Dict[str, np.ndarray] = {}
    for elem, weighted_cols in elem_weighted.items():
        v = np.zeros(n, dtype=np.float64)
        for col_idx, weight in weighted_cols:
            v += weight * features[:, col_idx]
        raw[elem] = v

    # grand_total = Σ cation fractions (+ O when oxygen is a fitted term).
    cation_sum = np.zeros(n, dtype=np.float64)
    for col_idx in cation_cols:
        cation_sum += features[:, col_idx]
    grand_total = cation_sum + raw["O"] if "O" in raw else cation_sum
    scale = np.where(np.abs(grand_total) < 1e-12, 0.0, norm_total / grand_total)

    for elem, v in raw.items():
        X = scale * v
        if f"b_{elem}" in coefs:
            T = T + coefs[f"b_{elem}"] * X
        if f"b_{elem}2" in coefs:
            T = T + coefs[f"b_{elem}2"] * X ** 2
    return T


def _eval_adiabat_poly(
    P: np.ndarray,
    S,
    coefs: Dict[str, float],
    features: Optional[np.ndarray] = None,
    comp_indices: Optional[Dict[str, object]] = None,
) -> np.ndarray:
    """Evaluate the reference adiabat polynomial (P in GPa, T in K).

    Base form (always applied):
        T = b0 + b_S*S + b_S2*S^2 + b_P*P + b_P2*P^2

    Compositional extension (when features and comp_indices are provided):
        T += Σ_i  b_Xi*X_i + b_Xi2*X_i^2   for each element X_i, where X_i is the
        per-row reconstruction of the fit's total=norm composition (see
        _resolve_comp_feature_indices / _eval_compositional_terms).
    """
    T = (
        coefs["b0"]
        + coefs["b_S"] * S
        + coefs["b_S2"] * S ** 2
        + coefs["b_P"] * P
        + coefs["b_P2"] * P ** 2
    )
    if comp_indices and features is not None:
        T = _eval_compositional_terms(T, coefs, features, comp_indices)
    return np.asarray(T, dtype=np.float32)


def _compute_residuals(
    T_true: np.ndarray,
    features_raw: np.ndarray,
    p_idx: int,
    s_idx: int,
    is_melts: bool = False,
    adiabat_coefs: Optional[Dict[str, float]] = None,
    comp_indices: Optional[Dict[str, object]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute T_residual = T_true - reference_adiabat(P, S[, X...]). Returns (residuals, T_ref).

    When is_melts=True, P is expected in bars and T in Celsius. P is converted to GPa
    before the adiabat call; the returned T_ref is in Celsius (K - 273.15) so the
    residual and all downstream targets remain in Celsius.

    If adiabat_coefs is provided, the reference adiabat is the stored polynomial.
    If comp_indices is also provided, element columns are extracted from features_raw
    and fed into the compositional extension of the polynomial.
    """
    P = features_raw[:, p_idx].astype(np.float64)
    if is_melts:
        P = P / 10000.0  # bars → GPa
        S = 2.5
    else:
        S = features_raw[:, s_idx].astype(np.float64)
    if adiabat_coefs is not None:
        T_ref_K = _eval_adiabat_poly(
            P, S, adiabat_coefs,
            features=features_raw.astype(np.float64),
            comp_indices=comp_indices,
        )
    else:
        T_ref_K = np.asarray(reference_adiabat(P, S), dtype=np.float32)
    T_ref = (T_ref_K - 273.15) if is_melts else T_ref_K
    residuals = (T_true.reshape(-1) - T_ref).astype(np.float32)
    return residuals, T_ref


def _build_predicted_labels(
    emulator: NN_MELTS,
    features_raw: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    PM_batches: List[torch.Tensor] = []
    CO_batches: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, features_raw.shape[0], batch_size):
            stop = min(start + batch_size, features_raw.shape[0])
            features_t = torch.tensor(features_raw[start:stop], dtype=torch.float32, device=device)
            predicted = emulator.forwardMB(
                features_t,
                Normalize=True,
                WtPercent=False,
                outputs=["phase_moles", "chem_out"],
            )
            PM_batches.append(predicted["phase_moles"].detach().cpu().clamp(0, 1))
            CO_batches.append(predicted["chem_out"].detach().cpu().clamp(0, 1))
    return torch.cat(PM_batches, dim=0), torch.cat(CO_batches, dim=0)


def _make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _evaluate_temperature(
    model: torch.nn.Module,
    x: np.ndarray,
    T_true: np.ndarray,
    T_ref: np.ndarray,
    y_min: np.ndarray,
    y_range: np.ndarray,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate in temperature space (residual prediction + reference adiabat)."""
    model.eval()
    with torch.no_grad():
        residual_norm = model(torch.tensor(x, dtype=torch.float32, device=device)).cpu().numpy()
    T_pred = residual_norm.reshape(-1) * y_range[0] + y_min[0] + T_ref
    mse = float(np.mean((T_pred - T_true.reshape(-1)) ** 2))
    mae = float(np.mean(np.abs(T_pred - T_true.reshape(-1))))
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
            "input_min_range": _normalizer_pairs(x_min, x_range),
            "target_min": y_min,
            "target_range": y_range,
            "target_min_range": _normalizer_pairs(y_min, y_range),
            "input_kind": input_kind,
            "is_melts": is_melts,
            "adiabat_coefs": adiabat_coefs,
            "coef_feature_indices": comp_indices,
            "metrics": metrics,
            "history": history,
        },
        out_path,
    )


def _load_and_validate_bundle(bundle_path: Path, label: str):
    print(f"Loading {label} bundle: {bundle_path}")
    bundle = load_ml_bundle(bundle_path)
    if bundle.free_outputs is None:
        raise ValueError(f"Bundle has no free_outputs.npy: {bundle_path}")
    if bundle.features is None or bundle.molar_labels is None or bundle.labels is None:
        raise ValueError(f"Bundle missing required arrays (features/molar_labels/labels): {bundle_path}")
    return bundle


def _load_checkpoint_state_dict(checkpoint_path: Path, device: torch.device) -> Dict:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError(f"Unsupported checkpoint format: {checkpoint_path}")


def _filter_by_entropy_range(
    bundle,
    features: np.ndarray,
    T_true: np.ndarray,
    s_idx: int,
    s_min: Optional[float],
    s_max: Optional[float],
    label: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Keep only rows with S strictly inside (s_min, s_max).

    Mirrors the KEEP_RANGE convention in scripts/fit_T_from_S_and_P.py (strict
    inequalities, either bound optional). Filters features/T_true and, to keep all
    per-row arrays aligned, also overwrites bundle.molar_labels and bundle.labels
    in place. No-op when both bounds are None.
    """
    if s_min is None and s_max is None:
        return features, T_true
    S = features[:, s_idx]
    mask = np.ones(S.shape[0], dtype=bool)
    if s_min is not None:
        mask &= S > s_min
    if s_max is not None:
        mask &= S < s_max
    print(f"  {label}: keeping {int(mask.sum())}/{mask.shape[0]} rows with S in ({s_min}, {s_max})")
    bundle.molar_labels = np.asarray(bundle.molar_labels, dtype=np.float32)[mask]
    bundle.labels = np.asarray(bundle.labels, dtype=np.float32)[mask]
    return features[mask], T_true[mask]


def _filter_by_residual_magnitude(
    bundle,
    features: np.ndarray,
    T_true: np.ndarray,
    residuals: np.ndarray,
    T_ref: np.ndarray,
    max_abs: Optional[float],
    label: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Drop rows whose |residual| exceeds max_abs (K).

    These extreme residuals are almost always reference-adiabat extrapolation
    blowups at compositions far outside the polynomial's fit domain (e.g. very
    high Cr, where the large b_Cr2 quadratic term dominates). A handful of such
    rows otherwise stretch the output normalizer's range by an order of
    magnitude, squashing the dynamic range of all the well-behaved targets.

    Keeps features/T_true/residuals/T_ref aligned and mutates
    bundle.molar_labels/labels in place. Returns the filtered arrays plus the
    number of rows removed. No-op (removed=0) when max_abs is None.
    """
    if max_abs is None:
        return features, T_true, residuals, T_ref, 0
    mask = np.abs(residuals) <= max_abs
    removed = int((~mask).sum())
    print(f"  {label}: removing {removed}/{mask.shape[0]} rows with |residual| > {max_abs} K")
    bundle.molar_labels = np.asarray(bundle.molar_labels, dtype=np.float32)[mask]
    bundle.labels = np.asarray(bundle.labels, dtype=np.float32)[mask]
    return features[mask], T_true[mask], residuals[mask], T_ref[mask], removed


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
        default=TEMPERATURE_LABEL_DEFAULT,
        help=f"Name of temperature output in bundle free_outputs (default: {TEMPERATURE_LABEL_DEFAULT})",
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
    train_bundle = _load_and_validate_bundle(bundle_paths["train"], "train")
    test_bundle = _load_and_validate_bundle(bundle_paths["test"], "test")
    valid_bundle = _load_and_validate_bundle(bundle_paths["valid"], "valid")

    # Locate temperature output column
    available_output_names = list(getattr(train_bundle.ml_indexer, "free_outputs", []) or [])
    if not available_output_names:
        raise ValueError("ml_indexer.free_outputs is missing or empty in train bundle")
    if args.temperature_label not in available_output_names:
        raise ValueError(
            f"Temperature label '{args.temperature_label}' not found in free_outputs. "
            f"Available: {available_output_names}"
        )
    temp_output_idx = available_output_names.index(args.temperature_label)

    # Locate P and S feature columns
    feature_names = list(getattr(train_bundle.ml_indexer, "featureNames", []) or [])
    for name in (P_FEATURE_NAME, S_FEATURE_NAME):
        if name not in feature_names:
            raise ValueError(f"Feature '{name}' not found in featureNames: {feature_names}")
    p_idx = feature_names.index(P_FEATURE_NAME)
    s_idx = feature_names.index(S_FEATURE_NAME)
    print(f"P feature index: {p_idx} ({P_FEATURE_NAME})")
    print(f"S feature index: {s_idx} ({S_FEATURE_NAME})")

    el_keys = list(getattr(train_bundle.ml_indexer, "Elkeys", []) or [])
    comp_indices: Optional[Dict[str, object]] = None
    if adiabat_coefs is not None and _is_compositional(adiabat_coefs):
        comp_indices = _resolve_comp_feature_indices(adiabat_coefs, feature_names, el_keys, norm=args.adiabat_elem_norm)
        print(f"Compositional reference adiabat (norm={args.adiabat_elem_norm}, Elkeys={el_keys}):")
        print(f"  elem→weighted-cols: {comp_indices['elem_weighted']}")
        print(f"  cation cols: {comp_indices['cation_cols']}")

    # Extract temperature targets and compute residuals against reference adiabat
    train_feats = np.asarray(train_bundle.features, dtype=np.float32)
    test_feats = np.asarray(test_bundle.features, dtype=np.float32)
    valid_feats = np.asarray(valid_bundle.features, dtype=np.float32)

    T_train_true = np.asarray(train_bundle.free_outputs[:, temp_output_idx], dtype=np.float32)
    T_test_true = np.asarray(test_bundle.free_outputs[:, temp_output_idx], dtype=np.float32)
    T_valid_true = np.asarray(valid_bundle.free_outputs[:, temp_output_idx], dtype=np.float32)

    if args.s_min is not None or args.s_max is not None:
        print(f"Filtering bundles to S in ({args.s_min}, {args.s_max}):")
        train_feats, T_train_true = _filter_by_entropy_range(train_bundle, train_feats, T_train_true, s_idx, args.s_min, args.s_max, "train")
        test_feats, T_test_true = _filter_by_entropy_range(test_bundle, test_feats, T_test_true, s_idx, args.s_min, args.s_max, "test")
        valid_feats, T_valid_true = _filter_by_entropy_range(valid_bundle, valid_feats, T_valid_true, s_idx, args.s_min, args.s_max, "valid")

    T_train_residuals, T_train_ref = _compute_residuals(T_train_true, train_feats, p_idx, s_idx, is_melts=args.isMELTS, adiabat_coefs=adiabat_coefs, comp_indices=comp_indices)
    T_test_residuals, T_test_ref = _compute_residuals(T_test_true, test_feats, p_idx, s_idx, is_melts=args.isMELTS, adiabat_coefs=adiabat_coefs, comp_indices=comp_indices)
    T_valid_residuals, T_valid_ref = _compute_residuals(T_valid_true, valid_feats, p_idx, s_idx, is_melts=args.isMELTS, adiabat_coefs=adiabat_coefs, comp_indices=comp_indices)

    # Histogram of the raw (pre-filter) train residuals so the extrapolation tail
    # and the effect of --max-abs-residual are visible.
    _save_residual_histogram(
        T_train_residuals,
        args.out_dir / f"residual_histogram_{args.bundle_stem.name}.png",
        max_abs=args.max_abs_residual,
    )

    if args.max_abs_residual is not None:
        print(f"Filtering rows with |residual| > {args.max_abs_residual} K:")
        train_feats, T_train_true, T_train_residuals, T_train_ref, n_tr = _filter_by_residual_magnitude(
            train_bundle, train_feats, T_train_true, T_train_residuals, T_train_ref, args.max_abs_residual, "train")
        test_feats, T_test_true, T_test_residuals, T_test_ref, n_te = _filter_by_residual_magnitude(
            test_bundle, test_feats, T_test_true, T_test_residuals, T_test_ref, args.max_abs_residual, "test")
        valid_feats, T_valid_true, T_valid_residuals, T_valid_ref, n_va = _filter_by_residual_magnitude(
            valid_bundle, valid_feats, T_valid_true, T_valid_residuals, T_valid_ref, args.max_abs_residual, "valid")
        print(f"  total removed: {n_tr + n_te + n_va}")

    print(
        f"Train residuals (K): min={T_train_residuals.min():.1f}, "
        f"max={T_train_residuals.max():.1f}, mean={T_train_residuals.mean():.1f}"
    )

    # Normalizers
    feature_normalizer = _ensure_feature_normalizer(train_bundle.ml_indexer, train_feats)
    for bundle in (train_bundle, test_bundle, valid_bundle):
        bundle.ml_indexer.feature_normalizer = feature_normalizer

    output_normalizer = _build_output_normalizer(T_train_residuals.reshape(-1, 1))
    y_min, y_range = _normalizer_arrays(output_normalizer)
    x1_min, x1_range = _normalizer_arrays(feature_normalizer)

    y_train_norm = _apply_normalizer(T_train_residuals.reshape(-1, 1), output_normalizer)
    y_test_norm = _apply_normalizer(T_test_residuals.reshape(-1, 1), output_normalizer)

    # Build model 1 inputs: normalized features + emulator-predicted moles + emulator-predicted intensive
    x1_train_base = _apply_normalizer(train_feats, feature_normalizer)
    x1_test_base = _apply_normalizer(test_feats, feature_normalizer)
    x1_valid_base = _apply_normalizer(valid_feats, feature_normalizer)

    pred_molar_train, pred_intensive_train = _build_predicted_labels(emulator, train_feats, device, args.emulator_batch_size)
    pred_molar_test, pred_intensive_test = _build_predicted_labels(emulator, test_feats, device, args.emulator_batch_size)
    pred_molar_valid, pred_intensive_valid = _build_predicted_labels(emulator, valid_feats, device, args.emulator_batch_size)

    x1_train = np.concatenate([x1_train_base, pred_molar_train.numpy(), pred_intensive_train.numpy()], axis=1).astype(np.float32)
    x1_emul_test = np.concatenate([x1_test_base, pred_molar_test.numpy(), pred_intensive_test.numpy()], axis=1).astype(np.float32)
    x1_emul_valid = np.concatenate([x1_valid_base, pred_molar_valid.numpy(), pred_intensive_valid.numpy()], axis=1).astype(np.float32)
    x1_emul_min, x1_emul_range = _build_extended_input_normalizer_arrays(x1_min, x1_range, x1_train.shape[1])

    # Build model 2 inputs: normalized features + GT moles + GT intensive
    x2_train = np.concatenate([x1_train_base, np.asarray(train_bundle.molar_labels, dtype=np.float32), np.asarray(train_bundle.labels, dtype=np.float32)], axis=1).astype(np.float32)
    x2_test = np.concatenate([x1_test_base, np.asarray(test_bundle.molar_labels, dtype=np.float32), np.asarray(test_bundle.labels, dtype=np.float32)], axis=1).astype(np.float32)
    x2_valid = np.concatenate([x1_valid_base, np.asarray(valid_bundle.molar_labels, dtype=np.float32), np.asarray(valid_bundle.labels, dtype=np.float32)], axis=1).astype(np.float32)
    x2_min, x2_range = _build_extended_input_normalizer_arrays(x1_min, x1_range, x2_train.shape[1])

    train_loader_x1 = _make_loader(x1_train, y_train_norm, args.batch_size, shuffle=True)
    valid_loader_x1 = _make_loader(x1_emul_test, y_test_norm, args.batch_size, shuffle=False)
    train_loader_x2 = _make_loader(x2_train, y_train_norm, args.batch_size, shuffle=True)
    valid_loader_x2 = _make_loader(x2_test, y_test_norm, args.batch_size, shuffle=False)

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
        input_dim=x1_train.shape[1], output_dim=1,
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
        input_dim=x2_train.shape[1], output_dim=1,
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
    metrics["emulator_path"]["train"] = _evaluate_temperature(model_emulator, x1_train, T_train_true, T_train_ref, y_min, y_range, device)
    metrics["emulator_path"]["test"] = _evaluate_temperature(model_emulator, x1_emul_test, T_test_true, T_test_ref, y_min, y_range, device)
    metrics["emulator_path"]["valid"] = _evaluate_temperature(model_emulator, x1_emul_valid, T_valid_true, T_valid_ref, y_min, y_range, device)
    metrics["gt_path"]["train"] = _evaluate_temperature(model_gt, x2_train, T_train_true, T_train_ref, y_min, y_range, device)
    metrics["gt_path"]["test"] = _evaluate_temperature(model_gt, x2_test, T_test_true, T_test_ref, y_min, y_range, device)
    metrics["gt_path"]["valid"] = _evaluate_temperature(model_gt, x2_valid, T_valid_true, T_valid_ref, y_min, y_range, device)

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
                    "emulator_path_input_min_range": _normalizer_pairs(x1_emul_min, x1_emul_range).tolist(),
                    "gt_path_input_min_range": _normalizer_pairs(x2_min, x2_range).tolist(),
                    "residual_min_range": _normalizer_pairs(y_min, y_range).tolist(),
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
