"""
Compare multiple trained models on a single validation dataset.

Finds all model files in a directory whose names share a common prefix,
evaluates each on the same sample subset, and writes per-phase precision/recall
and wt%-abundance / oxide-composition error tables.

Usage:
  python ModelComparison.py \\
      --bundle path/to/validation.tar.gz \\
      --model-dir path/to/models/ \\
      --prefix run_v3 \\
      [--output path/to/output/] \\
      [--max-samples 65536] [--seed 1337] [--cuda] [--no-normalize] \\
      [--min-liquid-mass 0.0]
"""

from __future__ import annotations

import argparse
import csv
import gc
import sys
import time
from pathlib import Path

import numpy as np
import torch
import tqdm

repo_root = str(Path(__file__).resolve().parents[3])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
src_path = str(Path(__file__).resolve().parents[2])
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from ngibbs.engine.NN import rebuild_MELTS_model
from ngibbs.engine.emulator import NN_MELTS
from ngibbs.utils.file_utils import load_ml_bundle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_bundle_path(bundle_path: str) -> Path:
    path = Path(bundle_path)
    if path.suffixes[-2:] == [".tar", ".gz"]:
        return path
    if path.suffix == ".tar":
        return path.with_suffix(".tar.gz")
    if path.suffix == ".gz" and path.name.endswith(".tar.gz"):
        return path
    return path.with_suffix(path.suffix + ".tar.gz") if path.suffix else path.with_suffix(".tar.gz")


def _find_models(model_dir: str, prefix: str) -> list[Path]:
    dir_path = Path(model_dir)
    return sorted(
        p for p in dir_path.iterdir()
        if p.name.startswith(prefix) and p.suffix in {".pt", ".zip", ".tar"}
    )


def _compute_active_oxides(comp_to_ox: np.ndarray, phase_indices: np.ndarray) -> np.ndarray:
    return np.where(np.any(comp_to_ox[phase_indices] != 0, axis=0))[0]


def _safe_mean_abs_rel(residuals: np.ndarray, reference: np.ndarray, threshold: float = 0.1) -> float:
    """Mean |residual / reference|, ignoring reference values below threshold."""
    denom = np.where(reference > threshold, reference, np.nan)
    return float(np.nanmean(np.abs(residuals / denom)))


def _staged_forward(func, input_tensor: torch.Tensor, batch_size: int, **kwargs):
    """
    Batch-execute func(input_tensor[start:end], **kwargs), merging results.
    Mirrors EmulatorAPI._staged_forward but as a standalone function.
    """
    def _merge(a, b):
        if isinstance(a, dict) and isinstance(b, dict):
            return {k: _merge(a[k], b[k]) for k in a}
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            merged = [_merge(a[i], b[i]) for i in range(len(a))]
            return type(a)(merged)
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            return torch.cat([a.detach().cpu(), b.detach().cpu()], dim=0)
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            return np.concatenate([a, b], axis=0)
        return b

    n = input_tensor.size(0)
    if n <= batch_size:
        return func(input_tensor, **kwargs)

    n_batches = (n + batch_size - 1) // batch_size
    with torch.no_grad():
        out = func(input_tensor[:batch_size], **kwargs)
        for i in tqdm.tqdm(range(1, n_batches), desc=f"  batching (size={batch_size})"):
            start = i * batch_size
            end = min(start + batch_size, n)
            batch_out = func(input_tensor[start:end], **kwargs)
            out = _merge(out, batch_out)
            del batch_out
            gc.collect()
    return out


# ---------------------------------------------------------------------------
# Per-model evaluation
# ---------------------------------------------------------------------------

def _evaluate_model(
    model_path: Path,
    bundle,
    subset: np.ndarray,
    use_cuda: bool,
    normalize_features: bool,
    batch_size: int = 2**14,
) -> tuple[list[dict], list[dict]]:
    """
    Returns (phase_records, oxide_records) for this model.

    phase_records — one dict per phase with binary and abundance metrics.
    oxide_records — one dict per (phase, oxide) with composition metrics.
    """
    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"

    model = rebuild_MELTS_model(str(model_path))
    emulator = NN_MELTS(model, cuda=use_cuda)
    ml_indexer = model.ml_indexer

    px_sp_transform = ml_indexer.PxSpTransform
    comp_subset = ml_indexer.compositional_component_subset
    px_sp_sub = px_sp_transform[np.ix_(comp_subset, comp_subset)]

    validation_features = bundle.features
    validation_binaries = bundle.binary_labels
    validation_moles = bundle.molar_labels
    validation_labels = bundle.labels
    validation_labels_trans = validation_labels @ px_sp_sub

    # Raw (un-normalized) subset tensors — forwardMB and gt_batch_fn handle normalization.
    labels_sub_tensor = torch.tensor(
        (validation_labels @ px_sp_sub)[subset], device=device, dtype=torch.float32
    )
    moles_sub_tensor = torch.tensor(validation_moles[subset], device=device, dtype=torch.float32)
    features_sub_raw = torch.tensor(validation_features[subset], device=device, dtype=torch.float32)
    # Normalized features needed only for Iron_Speciator in the composition-error loop below.
    features_sub_normed = emulator.norm_features.norm(features_sub_raw) if normalize_features else features_sub_raw

    # --- Ground-truth masses: batch getExtensiveComps + make_phase_tables ---
    # Pack all three inputs into one tensor so _staged_forward can slice them together.
    n_vc = labels_sub_tensor.shape[1]
    n_ph = moles_sub_tensor.shape[1]
    combined_gt = torch.cat([labels_sub_tensor, moles_sub_tensor, features_sub_raw], dim=1)

    def gt_batch_fn(chunk: torch.Tensor) -> torch.Tensor:
        labels_c = chunk[:, :n_vc]
        moles_c = chunk[:, n_vc:n_vc + n_ph]
        feats_c = chunk[:, n_vc + n_ph:]
        if normalize_features:
            feats_c = emulator.norm_features.norm(feats_c)
        newcomps = emulator.getExtensiveComps(intensiveLabels=labels_c, molarLabels=moles_c)
        return emulator.make_phase_tables(
            newcomps, emulator.compToOx, emulator.MM, emulator.phaseToCompMap.T, feats_c, out=None
        )

    gt_masses = _staged_forward(gt_batch_fn, combined_gt, batch_size).detach().cpu().numpy()

    # --- Model inference: batched forwardMB ---
    infer_out = _staged_forward(
        emulator.forwardMB,
        features_sub_raw,
        batch_size,
        Normalize=normalize_features,
        optimize_masses=True,
        protect_opx=False,
        outputs=["likelihoods", "phase_tables"],
    )

    binary_hat = (infer_out["likelihoods"] > 0.5).float().numpy()  # already on CPU via _merge
    comp_tens, mass_tens = infer_out["phase_tables"]
    #transcomponent_hat = infer_out["transcomponent_hat"]

    def _to_np(t):
        return t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else t

    comp_tens = _to_np(comp_tens)        # (n_subset, n_comp_phases, n_oxides)  wt% within phase
    mass_tens = _to_np(mass_tens)        # (n_subset, n_phases)                 wt% of system
    #transcomponent_hat = _to_np(transcomponent_hat)

    label_indices = ml_indexer.label_indices
    label_indices_comp = ml_indexer.label_indices_comp
    comp_phasedict = ml_indexer.comp_phasedict
    mass_phasedict = ml_indexer.mass_phasedict
    Oxides = ml_indexer.Oxides
    comp_to_ox = ml_indexer.compToOx
    mm = ml_indexer.MM
    n = len(subset)

    phase_records: list[dict] = []
    oxide_records: list[dict] = []

    for phase in label_indices:
        phase_idx = mass_phasedict[phase]
        real_pos = validation_binaries[subset, phase_idx] > 0.5
        pred_pos = binary_hat[:, phase_idx] > 0.5

        tp = int(np.sum(real_pos & pred_pos))
        fp = int(np.sum(~real_pos & pred_pos))
        fn = int(np.sum(real_pos & ~pred_pos))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Abundance errors (wt% of system) where GT says phase is present
        if real_pos.any():
            gt_mass = gt_masses[real_pos, phase_idx]
            pred_mass = mass_tens[real_pos, phase_idx]
            abs_mass_err = float(np.mean(np.abs(gt_mass - pred_mass)))
            rel_mass_err = _safe_mean_abs_rel(gt_mass - pred_mass, gt_mass, threshold=0.01)
        else:
            abs_mass_err = float("nan")
            rel_mass_err = float("nan")

        rec: dict = {
            "model": model_path.stem,
            "phase": phase,
            "precision": precision,
            "recall": recall,
            "gt_abundance_pct": 100.0 * int(real_pos.sum()) / n,
            "pred_abundance_pct": 100.0 * int(pred_pos.sum()) / n,
            "abs_mass_err_wt": abs_mass_err,
            "rel_mass_err": rel_mass_err,
        }

        # Composition errors (oxide wt% within phase) for compositionally variable phases
        if phase in label_indices_comp:
            indices = label_indices[phase]
            comp_indices = label_indices_comp[phase]
            active_oxides = _compute_active_oxides(comp_to_ox, indices)

            oxides_gt = validation_labels_trans[np.ix_(subset, comp_indices)] @ comp_to_ox[indices]

            if phase == "melts-liquid" and "Fe3" not in ml_indexer.Elkeys:
                oxides_gt = emulator.Iron_Speciator(
                    torch.tensor(oxides_gt, device=device, dtype=torch.float32),
                    features_sub_normed,
                ).detach().cpu().numpy()

            oxides_gt = oxides_gt @ mm
            oxides_gt_wt = oxides_gt * (100.0 / (1e-6 + np.sum(oxides_gt, axis=1, keepdims=True)))

            both_present = real_pos & pred_pos
            if both_present.any():
                both_idx = np.where(both_present)[0]
                n_both = int(both_present.sum())

                for ox_i in active_oxides:
                    oxide_name = Oxides[ox_i]
                    gt_ox = oxides_gt_wt[both_idx, ox_i]
                    pred_ox = comp_tens[both_idx, comp_phasedict[phase], ox_i]
                    abs_ox = float(np.nanmean(np.abs(gt_ox - pred_ox)))
                    rel_ox = _safe_mean_abs_rel(gt_ox - pred_ox, gt_ox, threshold=0.1)
                    rec[f"abs_{oxide_name}_wt"] = abs_ox
                    rec[f"rel_{oxide_name}"] = rel_ox
                    oxide_records.append({
                        "model": model_path.stem,
                        "phase": phase,
                        "oxide": oxide_name,
                        "abs_err_wt": abs_ox,
                        "rel_err": rel_ox,
                        "n_samples": n_both,
                    })

        phase_records.append(rec)
        gc.collect()

    del model, emulator
    gc.collect()
    return phase_records, oxide_records


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

def _build_phase_cols(rows: list[dict]) -> list[tuple[str, str, int, str]]:
    """Return (key, label, width, fmt) tuples for a phase table, adding per-oxide columns."""
    fixed: list[tuple[str, str, int, str]] = [
        ("model",              "Model",           38, "s"),
        ("precision",          "Precision",        11, ".4f"),
        ("recall",             "Recall",           10, ".4f"),
        ("gt_abundance_pct",   "GT%",               8, ".2f"),
        ("pred_abundance_pct", "Pred%",              8, ".2f"),
        ("abs_mass_err_wt",    "AbsMassErr(wt%)",  17, ".4f"),
        ("rel_mass_err",       "RelMassErr",        12, ".4f"),
    ]
    # Collect oxide names in insertion order across all rows
    seen_oxides: list[str] = []
    seen_set: set[str] = set()
    for row in rows:
        for k in row:
            if k.startswith("abs_") and k.endswith("_wt") and k != "abs_mass_err_wt":
                ox = k[4:-3]
                if ox not in seen_set:
                    seen_oxides.append(ox)
                    seen_set.add(ox)
    oxide_cols: list[tuple[str, str, int, str]] = []
    for ox in seen_oxides:
        w = max(len(ox) + 5, 12)   # enough for header + value
        oxide_cols += [
            (f"abs_{ox}_wt", f"{ox}_abs", w, ".4f"),
            (f"rel_{ox}",    f"{ox}_rel", w, ".4f"),
        ]
    return fixed + oxide_cols


def _render_table(rows: list[dict], cols: list[tuple[str, str, int, str]]) -> str:
    header = "".join(f"{label:<{w}}" for _, label, w, _ in cols)
    div = "-" * len(header)
    lines = [header, div]
    for row in rows:
        line = ""
        for key, _, w, fmt in cols:
            val = row.get(key, float("nan"))
            if isinstance(val, float):
                if np.isnan(val):
                    line += f"{'—':<{w}}"
                else:
                    line += f"{val:{w}{fmt}}"
            else:
                line += f"{str(val):<{w}}"
        lines.append(line)
    return "\n".join(lines)




# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_comparison(
    bundle_path: str,
    model_dir: str,
    prefix: str,
    output_dir: str,
    max_samples: int = 2**18,
    seed: int = 1337,
    use_cuda: bool = False,
    normalize_features: bool = True,
    min_liquid_mass: float = 0.0,
    batch_size: int = 2**14,
) -> None:
    bundle_path = _resolve_bundle_path(bundle_path)
    bundle = load_ml_bundle(bundle_path)

    model_paths = _find_models(model_dir, prefix)
    if not model_paths:
        print(f"No models found in '{model_dir}' with prefix '{prefix}'.")
        return
    print(f"Found {len(model_paths)} model(s):")
    for p in model_paths:
        print(f"  {p.name}")

    # Sample selection
    n_total = bundle.features.shape[0]
    rng = np.random.default_rng(seed)

    eligible: np.ndarray
    if min_liquid_mass > 0 and hasattr(bundle, "mass_labels") and bundle.mass_labels is not None:
        # Use a temporary model to find the liquid column index
        tmp_model = rebuild_MELTS_model(str(model_paths[0]))
        liquid_idx = tmp_model.ml_indexer.mass_phasedict.get("melts-liquid", None)
        del tmp_model
        gc.collect()
        if liquid_idx is not None:
            eligible = np.where(bundle.mass_labels[:, liquid_idx] > min_liquid_mass)[0]
        else:
            eligible = np.arange(n_total)
    else:
        eligible = np.arange(n_total)

    subset = rng.choice(eligible, size=min(max_samples, len(eligible)), replace=False)
    print(f"\nEvaluating on {len(subset):,} samples (from {len(eligible):,} eligible, total {n_total:,}).")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_phase_records: list[dict] = []
    all_oxide_records: list[dict] = []

    for model_path in model_paths:
        print(f"\n[{model_path.name}] evaluating ...", flush=True)
        t0 = time.time()
        try:
            phase_recs, oxide_recs = _evaluate_model(
                model_path, bundle, subset, use_cuda, normalize_features, batch_size
            )
            elapsed = time.time() - t0
            all_phase_records.extend(phase_recs)
            all_oxide_records.extend(oxide_recs)
            print(f"  finished in {elapsed:.1f}s")
        except Exception as e:
            print(f"  ERROR evaluating {model_path.name}: {e}. Skipping.") # Singular matrix lstsq fail for early models
            continue

    # ---- Save CSVs ----
    # phase_metrics.csv: union of all keys (different phases have different oxide columns)
    phase_csv = output_path / "phase_metrics.csv"
    if all_phase_records:
        all_keys: list[str] = []
        seen_keys: set[str] = set()
        for rec in all_phase_records:
            for k in rec:
                if k not in seen_keys:
                    all_keys.append(k)
                    seen_keys.add(k)
        with phase_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore", restval="")
            writer.writeheader()
            writer.writerows(all_phase_records)
        print(f"\nPhase metrics → {phase_csv}")

    # oxide_metrics.csv: long-format (model, phase, oxide) for programmatic use
    oxide_csv = output_path / "oxide_metrics.csv"
    if all_oxide_records:
        with oxide_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_oxide_records[0].keys()))
            writer.writeheader()
            writer.writerows(all_oxide_records)
        print(f"Oxide metrics  → {oxide_csv}")

    # ---- Print summary grouped by phase ----
    phases = list(dict.fromkeys(r["phase"] for r in all_phase_records))
    summary_path = output_path / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as fout:
        title = f"Model comparison  |  prefix='{prefix}'  |  n={len(subset):,}"
        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)
        fout.write(title + "\n")

        for phase in phases:
            rows = [r for r in all_phase_records if r["phase"] == phase]
            block = f"\n--- {phase} ---"
            print(block)
            fout.write(block + "\n")
            cols = _build_phase_cols(rows)
            text = _render_table(rows, cols)
            print(text)
            fout.write(text + "\n")

    print(f"\nSummary written to {summary_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare multiple trained models on a validation dataset."
    )
    parser.add_argument("--bundle", required=True, help="Path to validation bundle (.tar.gz)")
    parser.add_argument("--model-dir", required=True, help="Directory containing model .pt/.zip files")
    parser.add_argument("--prefix", required=True, help="Model filename prefix to match (e.g. 'run_v3')")
    parser.add_argument(
        "--output", default=None,
        help="Output directory (default: <model-dir>/comparison/<prefix>)"
    )
    parser.add_argument("--max-samples", type=int, default=2**18, help="Max samples to evaluate")
    parser.add_argument("--batch-size", type=int, default=2**14, help="Batch size for inference")
    parser.add_argument("--seed", type=int, default=1337, help="RNG seed for sample selection")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--no-normalize", action="store_true", help="Skip feature normalization")
    parser.add_argument(
        "--min-liquid-mass", type=float, default=0.0,
        help="Only include samples where melts-liquid wt%% > this value"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output or str(
        Path(args.model_dir) / "comparison" / args.prefix.rstrip("_-")
    )
    run_comparison(
        bundle_path=args.bundle,
        model_dir=args.model_dir,
        prefix=args.prefix,
        output_dir=output_dir,
        max_samples=args.max_samples,
        seed=args.seed,
        use_cuda=args.cuda,
        normalize_features=not args.no_normalize,
        min_liquid_mass=args.min_liquid_mass,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
