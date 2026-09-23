"""Reference-adiabat polynomial evaluation and residual/normalizer helpers.

Split out of train_temperature_residual_fcnn.py so both that script and
residual_workspace.py (which computes residuals chunk-wise while building the
on-disk training workspace) can share one implementation without a circular
import between the two.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ngibbs.utils.math_utils import Normalizer, get_T as reference_adiabat

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


def is_compositional(coefs: Dict[str, float]) -> bool:
    """Return True if coefs contains element terms beyond the basic S+P polynomial."""
    basic = {"b0", "b_S", "b_S2", "b_P", "b_P2"}
    return any(k not in basic for k in coefs)


def resolve_comp_feature_indices(
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
    recompute the scale per row in eval_adiabat_poly:

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
    """Add compositional adiabat terms to base T. See resolve_comp_feature_indices.

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


def eval_adiabat_poly(
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
        resolve_comp_feature_indices / _eval_compositional_terms).
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


def compute_residuals(
    T_true: np.ndarray,
    features_raw: np.ndarray,
    p_idx: int,
    s_idx: int,
    adiabat_coefs: Optional[Dict[str, float]] = None,
    comp_indices: Optional[Dict[str, object]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute T_residual = T_true - reference_adiabat(P, S[, X...]). Returns (residuals, T_ref).

    P, S and T are used in the bundle's own native units (e.g. bar / J/g/K / Celsius
    for MELTS, GPa / J/g/K / K for HeFESTo) -- no conversion -- so the adiabat
    coefficients must be fit in those same units. API.get_T mirrors this exactly.

    If adiabat_coefs is provided, the reference adiabat is the stored polynomial.
    If comp_indices is also provided, element columns are extracted from features_raw
    and fed into the compositional extension of the polynomial.
    """
    P = features_raw[:, p_idx].astype(np.float64)
    S = features_raw[:, s_idx].astype(np.float64)
    if adiabat_coefs is not None:
        T_ref = eval_adiabat_poly(
            P, S, adiabat_coefs,
            features=features_raw.astype(np.float64),
            comp_indices=comp_indices,
        )
    else:
        T_ref = np.asarray(reference_adiabat(P, S), dtype=np.float32)
    residuals = (T_true.reshape(-1) - T_ref).astype(np.float32)
    return residuals, T_ref


def apply_normalizer(x: np.ndarray, normalizer: Normalizer) -> np.ndarray:
    x_t = torch.tensor(x, dtype=torch.float32)
    return normalizer.norm(x_t).cpu().numpy().astype(np.float32)


def normalizer_arrays(normalizer: Normalizer) -> Tuple[np.ndarray, np.ndarray]:
    mins = normalizer.miner.detach().cpu().numpy().astype(np.float32)
    ranges = normalizer.ranger.detach().cpu().numpy().astype(np.float32)
    return mins, ranges


def normalizer_pairs(mins: np.ndarray, ranges: np.ndarray) -> np.ndarray:
    mins = np.asarray(mins, dtype=np.float32).reshape(-1)
    ranges = np.asarray(ranges, dtype=np.float32).reshape(-1)
    return np.stack([mins, ranges], axis=1).astype(np.float32)


def build_extended_input_normalizer_arrays(
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
