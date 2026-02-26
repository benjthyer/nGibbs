"""
Generate phase/component pseudo-section diagrams from a trained model and MELTS bundle.

This script is CLI-driven and saves model-vs-ground-truth phase diagrams.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
from matplotlib import colors as mcolors

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure repo root and src are on path
repo_root = str(Path(__file__).resolve().parents[3])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
src_path = str(Path(__file__).resolve().parents[2])
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from nMELTS.engine.NN import rebuild_MELTS_model
from nMELTS.engine.emulator import NN_MELTS
from nMELTS.utils.file_utils import load_ml_bundle
from nMELTS.utils.math_utils import grid_sample


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


def fill_internal_gaps(matrix: np.ndarray) -> np.ndarray:
    """
    Fill internal zero/NaN gaps in each column with nearest value above.

    Keeps leading/trailing gaps unchanged.
    """
    output = matrix.copy()
    nrows, ncols = output.shape

    for col in range(ncols):
        col_data = output[:, col]
        nz = np.flatnonzero(np.isfinite(col_data) & (col_data != 0))
        if len(nz) < 2:
            continue
        top = nz[0]
        bottom = nz[-1]
        for row in range(top + 1, bottom + 1):
            if col_data[row] == 0 or np.isnan(col_data[row]):
                col_data[row] = col_data[row - 1]
        output[:, col] = col_data

    return output


def _check_constant_bulk(features: np.ndarray, feature_offset: int, tol: float) -> np.ndarray:
    bulk = features[:, feature_offset:]
    reference = bulk[0]
    max_abs_diff = np.max(np.abs(bulk - reference))
    if max_abs_diff > tol:
        raise ValueError(
            f"Bundle does not have constant bulk composition. max|delta|={max_abs_diff:.3e}, tol={tol:.3e}"
        )
    return reference


def _condition_grid_params(features: np.ndarray, feature_offset: int) -> tuple[list[list[float]], list[np.ndarray]]:
    condition_cols = features[:, :feature_offset]
    params = []
    unique_values = []
    for col in range(condition_cols.shape[1]):
        vals = np.unique(condition_cols[:, col])
        vals = np.sort(vals)
        unique_values.append(vals)
        params.append([float(vals.min()), float(vals.max()), int(len(vals))])
    return params, unique_values


def _select_phase_components(
    ml_indexer,
    requested_phases: list[str] | None,
    requested_components: list[str] | None,
) -> tuple[list[str], list[str | None], list[int | None]]:
    detail = ml_indexer.detail_label_indices
    available_phases = list(ml_indexer.label_indices.keys())

    if requested_phases is None:
        phases = available_phases
    else:
        phases = requested_phases

    components = []
    component_indices = []

    for i, phase in enumerate(phases):
        if phase not in available_phases:
            raise ValueError(f"Requested phase '{phase}' not found in ml_indexer.label_indices")

        detail_map = detail.get(phase, {})
        if requested_components is not None:
            comp_name = requested_components[i]
            if comp_name is None:
                components.append(None)
                component_indices.append(None)
                continue
            if comp_name not in detail_map:
                raise ValueError(
                    f"Requested component '{comp_name}' not found for phase '{phase}'. "
                    f"Available: {list(detail_map.keys())}"
                )
            components.append(comp_name)
            component_indices.append(detail_map[comp_name])
            continue

        if not detail_map:
            components.append(None)
            component_indices.append(None)
            continue

        if phase == "melts-liquid" and "Si" in detail_map:
            comp_name = "Si"
        else:
            comp_name = next(iter(detail_map.keys()))

        components.append(comp_name)
        component_indices.append(detail_map[comp_name])

    return phases, components, component_indices


def _feature_lookup(features_2d: np.ndarray) -> dict[tuple[float, float], int]:
    return {(float(p), float(t)): i for i, (p, t) in enumerate(features_2d[:, :2])}


def _plot_phase_components(
    features_2d: np.ndarray,
    binary_labels: np.ndarray,
    components: np.ndarray,
    phases: list[str],
    phase_components: list[str | None],
    component_indices: list[int | None],
    phase_to_index: dict[str, int],
    filename: Path,
    title_prefix: str,
    fill_liquid_gaps: bool,
) -> None:
    pressures = np.unique(features_2d[:, 0])
    temperatures = np.unique(features_2d[:, 1])
    n_phase = len(phases)

    p_grid, t_grid = np.meshgrid(pressures, temperatures)
    fig, axes = plt.subplots(
        nrows=int(np.ceil(n_phase / 3)),
        ncols=min(3, n_phase),
        figsize=(5 / 1.5 * min(3, n_phase), 4 / 1.5 * int(np.ceil(n_phase / 3))),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes).flatten()

    lookup = _feature_lookup(features_2d)
    cmap = plt.get_cmap("viridis")
    norm = mcolors.Normalize(vmin=0, vmax=1)

    for i, phase in enumerate(phases):
        ax = axes[i]
        phase_idx = phase_to_index[phase]
        comp_idx = component_indices[i]

        z = np.full((len(temperatures), len(pressures)), np.nan)
        c = np.full((len(temperatures), len(pressures)), 1.0)

        for ti, temperature in enumerate(temperatures):
            for pi, pressure in enumerate(pressures):
                idx = lookup.get((float(pressure), float(temperature)))
                if idx is None:
                    continue
                z[ti, pi] = binary_labels[idx, phase_idx]
                if comp_idx is not None:
                    c[ti, pi] = components[idx, comp_idx]

        if fill_liquid_gaps and phase == "melts-liquid":
            z = fill_internal_gaps(z)
            c = fill_internal_gaps(c)

        rgba = np.zeros((len(temperatures), len(pressures), 4))

        mask_nan = np.isnan(z)
        rgba[mask_nan] = np.array([0, 0, 0, 1])

        mask_zero = z == 0
        rgba[mask_zero] = np.array([0.7, 0.7, 0.7, 1])

        mask_one = z == 1
        if np.any(mask_one):
            rgba[mask_one] = cmap(norm(c[mask_one]))

        ax.imshow(
            rgba,
            origin="lower",
            extent=[pressures.min(), pressures.max(), temperatures.min(), temperatures.max()],
            aspect="auto",
        )

        comp_name = phase_components[i]
        if comp_name is None:
            ax.set_title(phase)
        else:
            ax.set_title(f"{phase}: {comp_name}")

        if np.any(mask_one):
            try:
                ax.contour(p_grid, t_grid, np.nan_to_num(z, nan=0.0), levels=[0.5], colors="white", linewidths=1.2)
            except Exception:
                pass

    for ax in axes:
        if ax.get_subplotspec().is_last_row():
            ax.set_xlabel("Pressure")
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel("Temperature")

    for k in range(n_phase, len(axes)):
        fig.delaxes(axes[k])

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=axes[:n_phase], label="Component Fraction (F)", shrink=0.85)
    fig.suptitle(title_prefix)
    fig.savefig(filename, dpi=256)
    plt.close(fig)


def _predict_for_grid(
    emulator: NN_MELTS,
    grid_features: np.ndarray,
    normalize_features: bool,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():        
        features_tensor = torch.tensor(grid_features, device=device, dtype=torch.float32)
        if normalize_features:
            input_tensor = emulator.norm_features.norm(features_tensor)

        likelihoods, transcomponents, _, _, _, _, _ = emulator.model.forward(input_tensor if normalize_features else features_tensor, detailed=True)

    binaries = (likelihoods > 0.5).float().detach().cpu().numpy()
    transcomponents = transcomponents.detach().cpu().numpy()

    ml_indexer = emulator.model.ml_indexer
    comp_subset = ml_indexer.compositional_component_subset
    px_sp_sub = ml_indexer.PxSpTransform[np.ix_(comp_subset, comp_subset)]
    component_space = transcomponents @ np.linalg.inv(px_sp_sub)
    return binaries, component_space


def _validate_bundle_feature_reordering(
    emulator: NN_MELTS,
    gt_features: np.ndarray,
    bundle_feature_columns: list[str],
) -> None:
    """
    Validate reorder_input_table on ground-truth bundle features.

    Checks:
    1) Reordering already-canonical columns leaves values unchanged.
    2) Shuffled columns are correctly recovered to canonical ordering.
    """
    canonical = emulator.reorder_input_table(
        gt_features,
        headers=bundle_feature_columns,
        composition_space="elements",
        strict=True,
        return_type="numpy",
    )

    if not np.allclose(canonical, gt_features, rtol=1e-6, atol=1e-8):
        raise ValueError(
            "Ground-truth feature reordering changed values for canonical input order."
        )

    rng = np.random.default_rng(seed=0)
    perm = rng.permutation(len(bundle_feature_columns))
    shuffled_features = gt_features[:, perm]
    shuffled_headers = [bundle_feature_columns[i] for i in perm]

    recovered = emulator.reorder_input_table(
        shuffled_features,
        headers=shuffled_headers,
        composition_space="elements",
        strict=True,
        return_type="numpy",
    )

    if not np.allclose(recovered, canonical, rtol=1e-6, atol=1e-8):
        raise ValueError(
            "Ground-truth shuffled-column recovery failed in reorder_input_table."
        )


def run_phase_diagrams(
    model_path: str,
    bundle_path: str,
    output_dir: str,
    use_cuda: bool = False,
    normalize_features: bool = True,
    phases: list[str] | None = None,
    components: list[str | None] | None = None,
    bulk_tolerance: float = 3e-5,
    fill_liquid_gaps: bool = True,
) -> None:
    bundle_path = _resolve_bundle_path(bundle_path)
    bundle = load_ml_bundle(bundle_path)

    model = rebuild_MELTS_model(model_path)
    emulator = NN_MELTS(model, cuda=use_cuda)

    ml_indexer = model.ml_indexer
    feature_offset = len(ml_indexer.featureNames)

    if phases is not None and components is not None and len(phases) != len(components):
        raise ValueError("--components length must match --phases length")

    selected_phases, selected_components, selected_component_indices = _select_phase_components(
        ml_indexer, phases, components
    )

    gt_features = bundle.features
    gt_binaries = bundle.binary_labels
    gt_components = bundle.labels
    bundle_feature_columns = list(bundle.ml_indexer.featureNames) + list(bundle.ml_indexer.Elkeys)
    print(f"Bundle features columns: {bundle_feature_columns}. len={len(bundle_feature_columns)}")
    """_validate_bundle_feature_reordering(
        emulator=emulator,
        gt_features=gt_features,
        bundle_feature_columns=bundle_feature_columns,
    )"""

    bulk_comp = _check_constant_bulk(gt_features, feature_offset=feature_offset, tol=bulk_tolerance)
    params, _ = _condition_grid_params(gt_features, feature_offset=feature_offset) 
    grid_conditions = grid_sample(params) # These three look alright
    grid_features = np.concatenate(
        [grid_conditions, np.repeat(bulk_comp.reshape(1, -1), grid_conditions.shape[0], axis=0)],
        axis=1,
    )

    reordered_grid = emulator.reorder_input_table(
        grid_features,
        headers=bundle_feature_columns,
        composition_space="elements",
        strict=True,
        return_type="numpy")
    print(f"Grid features shape: {grid_features.shape}, Ground truth features shape: {gt_features.shape}")
    print(f"reordered_grid features shape: {reordered_grid.shape}")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    phase_to_index = ml_indexer.mass_phasedict
    fO2_idx = None
    for i, name in enumerate(ml_indexer.featureNames):
        low = name.lower()
        if "logfo2" in low or "fugacity" in low:
            fO2_idx = i
            break

    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    pred_binaries, pred_components = _predict_for_grid(
        emulator=emulator,
        grid_features=reordered_grid,        
        normalize_features=normalize_features,
        device=device,
    )

    if fO2_idx is None:
        model_name = _sanitize_name(Path(model_path).stem)
        _plot_phase_components(
            features_2d=grid_features[:, :2],
            binary_labels=pred_binaries,
            components=pred_components,
            phases=selected_phases,
            phase_components=selected_components,
            component_indices=selected_component_indices,
            phase_to_index=phase_to_index,
            filename=output_path / f"{model_name}_model.png",
            title_prefix="Model",
            fill_liquid_gaps=fill_liquid_gaps,
        )

        _plot_phase_components(
            features_2d=gt_features[:, :2],
            binary_labels=gt_binaries,
            components=gt_components,
            phases=selected_phases,
            phase_components=selected_components,
            component_indices=selected_component_indices,
            phase_to_index=phase_to_index,
            filename=output_path / f"{model_name}_ground_truth.png",
            title_prefix="Ground Truth",
            fill_liquid_gaps=fill_liquid_gaps,
        )
        return

    unique_fO2 = np.unique(grid_features[:, fO2_idx])
    model_name = _sanitize_name(Path(model_path).stem)

    for value in unique_fO2:
        f_label = f"QFM_{value:+.3f}".replace("+", "plus").replace("-", "minus")

        pred_mask = np.isclose(grid_features[:, fO2_idx], value)
        gt_mask = np.isclose(gt_features[:, fO2_idx], value)

        if np.sum(pred_mask) == 0 or np.sum(gt_mask) == 0:
            continue

        _plot_phase_components(
            features_2d=grid_features[pred_mask][:, :2],
            binary_labels=pred_binaries[pred_mask],
            components=pred_components[pred_mask],
            phases=selected_phases,
            phase_components=selected_components,
            component_indices=selected_component_indices,
            phase_to_index=phase_to_index,
            filename=output_path / f"{model_name}_model_{f_label}.png",
            title_prefix=f"Model | logfO2 {value:+.3f}",
            fill_liquid_gaps=fill_liquid_gaps,
        )

        _plot_phase_components(
            features_2d=gt_features[gt_mask][:, :2],
            binary_labels=gt_binaries[gt_mask],
            components=gt_components[gt_mask],
            phases=selected_phases,
            phase_components=selected_components,
            component_indices=selected_component_indices,
            phase_to_index=phase_to_index,
            filename=output_path / f"{model_name}_ground_truth_{f_label}.png",
            title_prefix=f"Ground Truth | logfO2 {value:+.3f}",
            fill_liquid_gaps=fill_liquid_gaps,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate phase/component pseudo-section diagrams")
    parser.add_argument("--model", required=True, help="Path to trained model (.pt/.tar/.zip)")
    parser.add_argument("--bundle", required=True, help="Path to MELTS ML bundle (.tar.gz)")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: validation/plots/<model>_phase_diagrams)")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--no-normalize", action="store_true", help="Skip feature normalization")
    parser.add_argument("--phases", default=None, help="Comma-separated phase names to plot")
    parser.add_argument(
        "--components",
        default=None,
        help="Comma-separated component names matching --phases. Use 'none' for phase-only panel.",
    )
    parser.add_argument(
        "--bulk-tolerance",
        type=float,
        default=3E-5,
        help="Absolute tolerance for constant bulk-composition check",
    )
    parser.add_argument(
        "--no-fill-liquid-gaps",
        action="store_true",
        help="Disable internal gap filling for melts-liquid panel",
    )
    return parser


def _parse_csv_arg(csv_text: str | None) -> list[str] | None:
    if csv_text is None:
        return None
    vals = [v.strip() for v in csv_text.split(",") if v.strip()]
    return vals if vals else None


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    phases = _parse_csv_arg(args.phases)
    components_raw = _parse_csv_arg(args.components)
    components = None
    if components_raw is not None:
        components = [None if c.lower() == "none" else c for c in components_raw]

    if args.output_dir is None:
        script_dir = Path(__file__).parent
        output_dir = script_dir / "plots" / f"{Path(args.model).stem}_phase_diagrams"
    else:
        output_dir = Path(args.output_dir)

    run_phase_diagrams(
        model_path=args.model,
        bundle_path=args.bundle,
        output_dir=str(output_dir),
        use_cuda=args.cuda,
        normalize_features=not args.no_normalize,
        phases=phases,
        components=components,
        bulk_tolerance=args.bulk_tolerance,
        fill_liquid_gaps=not args.no_fill_liquid_gaps,
    )


if __name__ == "__main__":
    main()
