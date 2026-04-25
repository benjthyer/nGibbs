"""
Generate recovery plots from a validation bundle and trained model.

This script runs non-interactively and saves plots to a subdirectory.
"""

from __future__ import annotations

import argparse
import gc
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib

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


def _sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _write_phase_metrics(
    output_path: Path,
    phase_names: list[str],
    real_pos: dict[str, np.ndarray],
    pred_pos: dict[str, np.ndarray],
) -> None:
    metrics_path = output_path / "phase_binary_metrics.txt"
    header = (
        f"{'phase':<30}"
        f"{'precision':>12}"
        f"{'recall':>12}"
        f"{'gt_abund_%':>14}"
        f"{'pred_abund_%':>14}\n"
    )
    divider = f"{'-' * 30}{'-' * 12}{'-' * 12}{'-' * 14}{'-' * 14}\n"
    with metrics_path.open("w", encoding="utf-8") as handle:
        handle.write(header)
        handle.write(divider)
        for phase in phase_names:
            phase_real = real_pos[phase]
            phase_pred = pred_pos[phase]
            tp = np.sum(phase_real & phase_pred)
            fp = np.sum(~phase_real & phase_pred)
            fn = np.sum(phase_real & ~phase_pred)
            precision = _safe_div(tp, tp + fp)
            recall = _safe_div(tp, tp + fn)
            gt_abundance = 100.0 * _safe_div(np.sum(phase_real), phase_real.size)
            pred_abundance = 100.0 * _safe_div(np.sum(phase_pred), phase_pred.size)
            row = (
                f"{phase:<30}"
                f"{precision:>12.6f}"
                f"{recall:>12.6f}"
                f"{gt_abundance:>14.3f}"
                f"{pred_abundance:>14.3f}\n"
            )
            handle.write(row)


def _resolve_bundle_path(bundle_path: str) -> Path:
    path = Path(bundle_path)
    if path.suffixes[-2:] == [".tar", ".gz"]:
        return path
    if path.suffix == ".tar":
        return path.with_suffix(".tar.gz")
    if path.suffix == ".gz" and path.name.endswith(".tar.gz"):
        return path
    return path.with_suffix(path.suffix + ".tar.gz") if path.suffix else path.with_suffix(".tar.gz")


def _check_indexer_compat(bundle_indexer, model_indexer) -> None:
    if model_indexer is None:
        return
    mismatch = []
    if ~np.all(np.array(bundle_indexer.featureNames) == np.array(model_indexer.featureNames)):
        print("Feature name mismatch!")
        print(f"Bundle features: {bundle_indexer.featureNames}")
        print(f"Model features: {model_indexer.featureNames}")
        mismatch.append("featureNames")
    if ~np.all(np.array(bundle_indexer.WRkeys) == np.array(model_indexer.WRkeys)):
        print("WRkeys mismatch!")
        print(f"Bundle WRkeys: {bundle_indexer.WRkeys}")
        print(f"Model WRkeys: {model_indexer.WRkeys}")
        mismatch.append("WRkeys")
    if ~np.all(np.array(bundle_indexer.all_phases) == np.array(model_indexer.all_phases)):
        print("Phase name mismatch!")
        print(f"Bundle phases: {bundle_indexer.all_phases}")
        print(f"Model phases: {model_indexer.all_phases}")
        mismatch.append("all_phases")
    if ~np.all(np.array(bundle_indexer.label_names) == np.array(model_indexer.label_names)):
        print("Component name mismatch!")
        print(f"Bundle component names: {bundle_indexer.label_names}")
        print(f"Model component names: {model_indexer.label_names}")
        mismatch.append("label_names")
    if bundle_indexer.ncompsVaried != model_indexer.ncompsVaried:
        mismatch.append("ncompsVaried")
    if bundle_indexer.ncomps != model_indexer.ncomps:
        mismatch.append("ncomps")
    if mismatch:
        joined = ", ".join(mismatch)
        raise ValueError(f"Model ml_indexer mismatch on: {joined}")


def _compute_active_oxides(comp_to_ox: np.ndarray, phase_indices: np.ndarray) -> np.ndarray:
    active = np.any(comp_to_ox[phase_indices] != 0, axis=0)
    return np.where(active)[0]


def run_recovery_plots(
    bundle_path: str,
    model_path: str,
    output_dir: str,
    max_samples: int = 2**16,
    seed: int = 1337,
    use_cuda: bool = False,
    normalize_features: bool = True,
    min_liquid_mass: float = 0.0,
) -> None:
    bundle_path = _resolve_bundle_path(bundle_path)
    bundle = load_ml_bundle(bundle_path)

    model = rebuild_MELTS_model(model_path)
    _check_indexer_compat(bundle.ml_indexer, getattr(model, "ml_indexer", None))
    emulator = NN_MELTS(model, cuda=use_cuda)

    ml_indexer = model.ml_indexer
    label_indices = ml_indexer.label_indices
    label_indices_comp = ml_indexer.label_indices_comp
    label_names = ml_indexer.label_names
    comp_phasedict = ml_indexer.comp_phasedict
    mass_phasedict = ml_indexer.mass_phasedict
    Oxides = ml_indexer.Oxides
    comp_to_ox = ml_indexer.compToOx
    mm = ml_indexer.MM
    px_sp_transform = ml_indexer.PxSpTransform
    comp_subset = ml_indexer.compositional_component_subset
    px_sp_sub = px_sp_transform[np.ix_(comp_subset, comp_subset)]

    validation_features = bundle.features
    validation_binaries = bundle.binary_labels
    validation_moles = bundle.molar_labels
    validation_labels = bundle.labels
    # make_phase_tables uses transformed component basis for pyroxenes/spinel;
    # transform VC labels to that same basis before constructing extensive C-space.
    validation_labels_model = validation_labels @ px_sp_sub

    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    validation_labels_tensor = torch.tensor(
        validation_labels_model, device=device, dtype=torch.float32
    )
    validation_moles_tensor = torch.tensor(
        validation_moles, device=device, dtype=torch.float32
    )
    validation_features_tensor = torch.tensor(
        validation_features, device=device, dtype=torch.float32
    )
    if normalize_features:
        validation_features_tensor = emulator.norm_features.norm(validation_features_tensor)

    with torch.no_grad():
        validation_newcomps_tensor = emulator.getExtensiveComps(
            intensiveLabels=validation_labels_tensor,
            molarLabels=validation_moles_tensor,
        )
        validation_masses_tensor = emulator.make_phase_tables(
            validation_newcomps_tensor,
            emulator.compToOx,
            emulator.MM,
            emulator.phaseToCompMap.T,
            validation_features_tensor,
            out=None,
        )
    validation_masses = validation_masses_tensor.detach().cpu().numpy()
    
    #validation_masses = bundle.mass_labels

    liquid_idx = mass_phasedict.get("melts-liquid", None)
    if min_liquid_mass > 0 and liquid_idx is not None:
        eligible = np.where(validation_masses[:, liquid_idx] > min_liquid_mass)[0]
    else:
        eligible = np.arange(validation_features.shape[0])

    rng = np.random.default_rng(seed)
    if len(eligible) > max_samples:
        subset = rng.choice(eligible, size=max_samples, replace=False)
    else:
        subset = eligible

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    features_tensor = torch.tensor(validation_features[subset], device=device, dtype=torch.float32)
    if normalize_features:
        features_tensor = emulator.norm_features.norm(features_tensor)

    print(f"emulator.molar_epsilon = {emulator.model.molar_epsilon}")

    with torch.no_grad():
        start_time = time.time()
        likelihoods, transcomponent_hat, log_moles, recon_bulk, component_moles, phase_proportions, phase_moles = (
            emulator.model.forward(features_tensor, detailed=True)
        )
        nn_time = time.time()
        (comp_tens, mass_tens), _, _ = emulator.polish_masses(
            phase_moles,
            recon_bulk,
            component_moles,
            phase_proportions,
            features=features_tensor,
            optimize_masses=False,
            output_componentMoles=True,
            protect_opx=False,
        )
        la_time = time.time()

    print(f"NN Time: {nn_time - start_time:.2f} sec")
    print(f"Linear Algebra Time: {la_time - nn_time:.2f} sec")
    print(f"Total Time: {la_time - start_time:.2f} sec")

    binary_hat = (likelihoods > 0.5).float().detach().cpu().numpy()
    likelihoods = likelihoods.detach().cpu().numpy()
    transcomponent_hat = transcomponent_hat.detach().cpu().numpy()
    phase_moles = phase_moles.detach().cpu().numpy()
    comp_tens = comp_tens.detach().cpu().numpy()
    mass_tens = mass_tens.detach().cpu().numpy()

    component_hat = transcomponent_hat @ np.linalg.inv(px_sp_sub)
    validation_labels_trans = validation_labels @ px_sp_sub

    correct_phases = np.all((validation_binaries[subset] > 0.5).astype(float) == binary_hat, axis=1)
    datalen = len(component_hat)

    real_pos_map: dict[str, np.ndarray] = {}
    pred_pos_map: dict[str, np.ndarray] = {}

    for phase in label_indices:
        phase_idx = mass_phasedict[phase]
        real_pos = validation_binaries[subset, phase_idx] > 0.5
        pred_pos = binary_hat[:, phase_idx] > 0.5
        real_pos_map[phase] = real_pos
        pred_pos_map[phase] = pred_pos
        plotable = np.logical_or(real_pos, pred_pos)

        fp = _safe_div(np.sum(real_pos[plotable] == 0), np.sum(plotable))
        fn = _safe_div(np.sum(pred_pos[plotable] == 0), np.sum(plotable))
        prop_correct_plotable = 100 * _safe_div(np.sum(plotable * correct_phases), np.sum(plotable))

        hist_path = output_path / f"{_sanitize_name(phase)}_saturation_hist.png"
        plt.figure()
        plt.hist(
            likelihoods[real_pos, phase_idx],
            bins=30,
            alpha=0.5,
            color="blue",
            label=f"{phase} Present",
            density=True,
            log=True,
        )
        plt.hist(
            likelihoods[~real_pos, phase_idx],
            bins=30,
            alpha=0.5,
            color="red",
            label=f"{phase} Absent",
            density=True,
            log=True,
        )
        plt.legend()
        plt.xlabel("Probability")
        plt.ylabel("Normalized Frequency (Log Scale)")
        plt.title(
            f"NN {phase} Saturation Probabilities\nPresent in {round(100 * real_pos.sum() / max(datalen, 1), 2)}%"
        )
        plt.tight_layout()
        plt.savefig(hist_path, dpi=256)
        plt.close()

        size_scale = max(np.sum(real_pos), 1)

        mass_name = f"{phase}_system_mass_consistent"
        mass_path = output_path / f"{_sanitize_name(mass_name)}.png"
        plt.figure()
        plt.title(f"{phase} System Mass (consistent)")
        plt.scatter(
            validation_masses[subset[plotable & ~correct_phases], phase_idx],
            mass_tens[plotable & ~correct_phases, phase_idx],
            color="red",
            s=0.1 * (datalen / size_scale),
            alpha=0.2,
            label=f"Assemblage Miss ({100 - prop_correct_plotable:.2f}%)",
        )
        plt.scatter(
            validation_masses[subset[plotable & correct_phases], phase_idx],
            mass_tens[plotable & correct_phases, phase_idx],
            color="blue",
            s=0.1 * (datalen / size_scale),
            alpha=0.2,
            label=f"Complete Assemblage Recovered ({prop_correct_plotable:.2f}%)",
        )
        plt.xlabel(f"GT Wt% {phase}, FN = {round(100 * fn, 2)}%")
        plt.ylabel(f"Predicted Wt% {phase}, FP = {round(100 * fp, 2)}%")
        xlim = plt.xlim()
        plt.plot([xlim[0], xlim[1]], [xlim[0], xlim[1]], linestyle="--", color="black")
        plt.legend()
        plt.tight_layout()
        plt.savefig(mass_path, dpi=256)
        plt.close()

        mole_name = f"{phase}_moles_unconstrained"
        mole_path = output_path / f"{_sanitize_name(mole_name)}.png"
        plt.figure()
        plt.title(f"{phase} Moles, Unconstrained NN Output")
        plt.scatter(
            validation_moles[subset[plotable & ~correct_phases], phase_idx],
            phase_moles[plotable & ~correct_phases, phase_idx],
            color="red",
            s=0.3 * (datalen / size_scale),
            alpha=0.2,
            label=f"Assemblage Miss ({100 - prop_correct_plotable:.2f}%)",
        )
        plt.scatter(
            validation_moles[subset[plotable & correct_phases], phase_idx],
            phase_moles[plotable & correct_phases, phase_idx],
            color="blue",
            s=0.3 * (datalen / size_scale),
            alpha=0.2,
            label=f"Complete Assemblage Recovered ({prop_correct_plotable:.2f}%)",
        )
        plt.xlabel(f"GT Moles {phase}, FN = {round(100 * fn, 2)}%")
        plt.ylabel(f"Predicted Moles {phase}, FP = {round(100 * fp, 2)}%")
        xlim = plt.xlim()
        plt.plot([xlim[0], xlim[1]], [xlim[0], xlim[1]], linestyle="--", color="black")
        plt.legend()
        plt.tight_layout()
        plt.savefig(mole_path, dpi=256)
        plt.close()

        if phase not in label_indices_comp:
            continue

        indices = label_indices[phase]
        comp_indices = label_indices_comp[phase]
        active_oxides = _compute_active_oxides(comp_to_ox, indices)

        oxides_hat = transcomponent_hat[:, comp_indices] @ comp_to_ox[indices]
        oxides_gt = validation_labels_trans[np.ix_(subset, comp_indices)] @ comp_to_ox[indices]

        if phase == "melts-liquid" and 'Fe3' not in ml_indexer.Elkeys:
            oxides_gt = emulator.Iron_Speciator(
                torch.tensor(oxides_gt, device=device, dtype=torch.float32),
                features_tensor,
            ).detach().cpu().numpy()
            oxides_hat = emulator.Iron_Speciator(
                torch.tensor(oxides_hat, device=device, dtype=torch.float32),
                features_tensor,
            ).detach().cpu().numpy()

        print(f"Phase: {phase}, Active Oxides: {[Oxides[i] for i in active_oxides]}")
        nonzero_gt = np.unique(np.where(np.sum(oxides_gt[:, active_oxides], axis=1) > 1e-6)[0])
        print(f"Oxides GT (first 5 rows):\n{oxides_gt[np.ix_(nonzero_gt[:5], active_oxides)]}"
              f"\nOxides Hat (first 5 rows):\n{oxides_hat[np.ix_(nonzero_gt[:5], active_oxides)]}")
        oxides_hat = oxides_hat @ mm
        oxides_gt = oxides_gt @ mm
        oxides_gt = oxides_gt * (100 / (1E-6 + np.sum(oxides_gt, axis=1)).reshape(-1, 1)) # safe Division
        oxides_hat = oxides_hat * (100 / (1E-6 + np.sum(oxides_hat, axis=1)).reshape(-1, 1))

        for ox_ind in active_oxides:
            oxide_name = Oxides[ox_ind]

            name = f"{phase}_unconstrained_{oxide_name}"
            path = output_path / f"{_sanitize_name(name)}.png"
            plt.figure()
            plt.title(f"{phase} Unconstrained NN {oxide_name}")
            plt.scatter(
                oxides_gt[plotable & ~correct_phases, ox_ind],
                oxides_hat[plotable & ~correct_phases, ox_ind],
                color="red",
                s=0.3 * (datalen / size_scale),
                alpha=0.2,
                label=f"Assemblage Miss ({100 - prop_correct_plotable:.2f}%)",
            )
            plt.scatter(
                oxides_gt[plotable & correct_phases, ox_ind],
                oxides_hat[plotable & correct_phases, ox_ind],
                color="blue",
                s=0.3 * (datalen / size_scale),
                alpha=0.2,
                label=f"Complete Assemblage Recovered ({prop_correct_plotable:.2f}%)",
            )
            plt.xlabel(f"True {oxide_name} wt%, FN = {round(100 * fn, 2)}%")
            plt.ylabel(f"Predicted {oxide_name} wt%, FP = {round(100 * fp, 2)}%")
            xlim = plt.xlim()
            plt.plot([xlim[0], xlim[1]], [xlim[0], xlim[1]], linestyle="--", color="black")
            plt.legend()
            plt.tight_layout()
            plt.savefig(path, dpi=256)
            plt.close()

            xdata = oxides_gt[plotable, ox_ind]
            ydata = comp_tens[plotable, comp_phasedict[phase], ox_ind]
            xmin = max(0, np.nanmin(xdata))
            xmax = min(100, np.nanmax(xdata))
            ymin = max(0, float(np.nanmin(ydata)))
            ymax = min(100, float(np.nanmax(ydata)))

            xpad = 0.05 * (xmax - xmin) if xmax > xmin else 1
            ypad = 0.05 * (ymax - ymin) if ymax > ymin else 1
            xmin, xmax = xmin - xpad, xmax + xpad
            ymin, ymax = ymin - ypad, ymax + ypad

            offplot = (
                ((xdata < xmin) | (xdata > xmax)) | ((ydata < ymin) | (ydata > ymax))
            ).sum()

            name = f"{phase}_consistent_{oxide_name}"
            path = output_path / f"{_sanitize_name(name)}.png"
            plt.figure()
            plt.title(f"{phase} Consistent {oxide_name}")
            plt.scatter(
                oxides_gt[plotable & ~correct_phases, ox_ind],
                comp_tens[plotable & ~correct_phases, comp_phasedict[phase], ox_ind],
                color="red",
                s=0.3 * (datalen / size_scale),
                alpha=0.2,
                label=f"Assemblage Miss ({100 - prop_correct_plotable:.2f}%)",
            )
            plt.scatter(
                oxides_gt[plotable & correct_phases, ox_ind],
                comp_tens[plotable & correct_phases, comp_phasedict[phase], ox_ind],
                color="blue",
                s=0.3 * (datalen / size_scale),
                alpha=0.2,
                label=f"Complete Assemblage Recovered ({prop_correct_plotable:.2f}%)",
            )
            plt.xlabel(f"True {oxide_name} wt%, FN = {round(100 * fn, 2)}%")
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
            plt.ylabel(
                f"Predicted {oxide_name} wt%, FP = {round(100 * fp, 2)}%, "
                f"Off-plot = {round(100 * _safe_div(offplot, np.sum(plotable)), 2)}%"
            )
            plt.plot([xlim[0], xlim[1]], [xlim[0], xlim[1]], linestyle="--", color="black")
            plt.legend()
            plt.tight_layout()
            plt.savefig(path, dpi=256)
            plt.close()

        for k, ind in enumerate(comp_indices):
            comp_name = label_names[indices[k]]

            name = f"{phase}_{comp_name}_component"
            path = output_path / f"{_sanitize_name(name)}.png"
            plt.figure()
            plt.title(f"{phase} {comp_name}")
            plt.scatter(
                validation_labels[subset[plotable & ~correct_phases], ind],
                component_hat[plotable & ~correct_phases, ind],
                color="red",
                s=0.3 * (datalen / size_scale),
                alpha=0.2,
                label=f"Assemblage Miss ({100 - prop_correct_plotable:.2f}%)",
            )
            plt.scatter(
                validation_labels[subset[plotable & correct_phases], ind],
                component_hat[plotable & correct_phases, ind],
                color="blue",
                s=0.3 * (datalen / size_scale),
                alpha=0.2,
                label=f"Complete Assemblage Recovered ({prop_correct_plotable:.2f}%)",
            )
            
            plt.xlabel(f"True molar {comp_name}, FN = {round(100 * fn, 2)}%")
            plt.ylabel(f"Predicted molar {comp_name}, FP = {round(100 * fp, 2)}%")
            xlim = plt.xlim()
            plt.plot([xlim[0], xlim[1]], [xlim[0], xlim[1]], linestyle="--", color="black")
            plt.legend()
            plt.tight_layout()
            plt.savefig(path, dpi=256)
            plt.close()

            if (phase in {"orthopyroxene", "clinopyroxene", "spinel"}) and ('melts-liquid' in label_indices):
                name = f"{phase}_g{k - 1}_component"
                path = output_path / f"{_sanitize_name(name)}.png"
                plt.figure()
                plt.title(f"{phase} g{k - 1} component")
                plt.scatter(
                    validation_labels_trans[subset[plotable & ~correct_phases], ind],
                    transcomponent_hat[plotable & ~correct_phases, ind],
                    color="red",
                    s=0.3 * (datalen / size_scale),
                    alpha=0.2,
                    label=f"Assemblage Miss ({100 - prop_correct_plotable:.2f}%)",
                )
                plt.scatter(
                    validation_labels_trans[subset[plotable & correct_phases], ind],
                    transcomponent_hat[plotable & correct_phases, ind],
                    color="blue",
                    s=0.3 * (datalen / size_scale),
                    alpha=0.2,
                    label=f"Complete Assemblage Recovered ({prop_correct_plotable:.2f}%)",
                )
                plt.xlabel(f"True molar g{k - 1}, FN = {round(100 * fn, 2)}%")
                plt.ylabel(f"Predicted molar g{k - 1}, FP = {round(100 * fp, 2)}%")
                xlim = plt.xlim()
                plt.plot([xlim[0], xlim[1]], [xlim[0], xlim[1]], linestyle="--", color="black")
                plt.legend()
                plt.tight_layout()
                plt.savefig(path, dpi=256)
                plt.close()

        gc.collect()

    _write_phase_metrics(output_path, list(label_indices.keys()), real_pos_map, pred_pos_map)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate recovery plots from a validation bundle.")
    parser.add_argument("--bundle", required=True, help="Path to validation bundle (.tar.gz)")
    parser.add_argument("--model", required=True, help="Path to trained model (.pt or .zip)")
    parser.add_argument("--max-samples", type=int, default=2**16, help="Max samples to plot")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for sampling")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--no-normalize", action="store_true", help="Skip feature normalization")
    parser.add_argument(
        "--min-liquid-mass",
        type=float,
        default=0.0,
        help="Minimum melts-liquid mass to include (wt%)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Auto-generate output directory within plots subdirectory
    script_dir = Path(__file__).parent
    plots_dir = script_dir / "plots"
    model_name = Path(args.model).stem
    output_dir = str(plots_dir / model_name)

    run_recovery_plots(
        bundle_path=args.bundle,
        model_path=args.model,
        output_dir=output_dir,
        max_samples=args.max_samples,
        seed=args.seed,
        use_cuda=args.cuda,
        normalize_features=not args.no_normalize,
        min_liquid_mass=args.min_liquid_mass,
    )


if __name__ == "__main__":
    main()