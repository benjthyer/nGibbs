"""
Parity figure for an isentropic emulator's temperature model.

Isentropic emulators take (P, S, bulk composition) as inputs, so temperature is
recovered by an auxiliary model: a reference regression (quadratic in P, S and
each bulk element) plus a neural-network residual correction that also sees the
emulator's predicted phase abundances and compositions. This script compares
both stages with the thermodynamic model's own temperatures on the isentropic
test subset that ships with the installed nGibbs package:

  (a) predicted vs. true T -- regression alone (orange), regression + NN (blue)
  (b) histogram of prediction errors (log scale)

Only the installed ``ngibbs`` package is used: the model, its temperature
checkpoint and its test bundle are all read from the model directory shipped
inside the package (``ngibbs/engine/TrainedModels/<model-dir>``), loaded with
``HeFESToAPI`` exactly as ``ngibbs.engine.models`` does, and the prediction
mirrors ``EmulatorAPI.get_T``. (The model is constructed directly rather than
imported from ``ngibbs.engine.models`` so that only this one model is loaded.)

Usage:
    python plot_temperature_model_parity.py                       # HeFESTo Mars, S > 1.75
    python plot_temperature_model_parity.py --s-min 1.75 --out figs/temperature_model_parity
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
# matplotlib.use("Agg")  # disabled: a global backend switch breaks interactive plotting (e.g. PetThermoTools)
import matplotlib.pyplot as plt
import numpy as np
import torch

import ngibbs.engine
from ngibbs.engine.API import HeFESToAPI, _eval_adiabat_poly
from ngibbs.utils.file_utils import load_ml_bundle

ORANGE, BLUE = "#E8730C", "#1F6FB4"


def predict_temperatures(api, features: np.ndarray, batch_size: int = 2 ** 14):
    """Return (T_regression, T_regression_plus_NN) for raw bundle features."""
    payload = api.temperature_payload
    T_reg = _eval_adiabat_poly(
        features[:, payload["p_feature_idx"]].astype(np.float64),
        features[:, payload["s_feature_idx"]].astype(np.float64),
        payload["adiabat_coefs"],
        features=features.astype(np.float64),
        comp_indices=payload.get("coef_feature_indices"),
    )

    T_full = []
    emulator = api.isentropic_emulator
    for start in range(0, len(features), batch_size):
        x = torch.as_tensor(features[start:start + batch_size], dtype=torch.float32, device=api.device)
        with torch.no_grad():
            out = emulator.forwardMB(x, Normalize=True, WtPercent=False,
                                     outputs=["phase_moles", "chem_out"])
        temp_in = torch.cat([x, out["phase_moles"].to(api.device).float(),
                             out["chem_out"].to(api.device).float()], dim=1)
        T_full.append(api.get_T(temp_in).detach().cpu().numpy())
    return np.asarray(T_reg, dtype=np.float64), np.concatenate(T_full).astype(np.float64)


def error_stats(pred: np.ndarray, true: np.ndarray) -> dict:
    e = pred - true
    return {
        "rmse_K": float(np.sqrt(np.mean(e ** 2))),
        "mae_K": float(np.mean(np.abs(e))),
        "bias_K": float(np.mean(e)),
        "p99_abs_K": float(np.percentile(np.abs(e), 99)),
        "max_abs_K": float(np.max(np.abs(e))),
        "r2": float(1 - np.sum(e ** 2) / np.sum((true - true.mean()) ** 2)),
    }


def plot_parity(T, T_reg, T_full, out_stem: Path, model_label: str) -> None:
    rmse = lambda p: np.sqrt(np.mean((p - T) ** 2))
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 2, figsize=(6.5, 3.1), gridspec_kw=dict(width_ratios=[1, 1.05]))

    a = ax[0]
    lo = 100 * np.floor(min(T.min(), T_reg.min(), T_full.min()) / 100)
    hi = 100 * np.ceil(max(T.max(), T_reg.max(), T_full.max()) / 100)
    a.plot([lo, hi], [lo, hi], color="0.3", lw=0.8, ls="--", zorder=1)
    a.scatter(T, T_reg, s=2, color=ORANGE, alpha=0.35, lw=0, rasterized=True, zorder=2,
              label=f"Regression only (RMSE {rmse(T_reg):.1f} K)")
    a.scatter(T, T_full, s=2, color=BLUE, alpha=0.5, lw=0, rasterized=True, zorder=3,
              label=f"Regression + NN (RMSE {rmse(T_full):.1f} K)")
    a.set_xlim(lo, hi); a.set_ylim(lo, hi); a.set_aspect("equal")
    a.set_xlabel(f"{model_label} temperature (K)")
    a.set_ylabel("Predicted temperature (K)")
    leg = a.legend(loc="upper left", fontsize=7, frameon=False, markerscale=4)
    for h in leg.legend_handles:
        h.set_alpha(1)
    a.text(-0.3, 1.04, "(a)", transform=a.transAxes, fontweight="bold")

    b = ax[1]
    lim = 10 * np.ceil(max(np.abs(T_reg - T).max(), np.abs(T_full - T).max()) / 10)
    bins = np.linspace(-lim, lim, int(lim) + 1)  # 2 K bins
    b.hist(T_reg - T, bins=bins, color=ORANGE, alpha=0.85, label="Regression only")
    b.hist(T_full - T, bins=bins, color=BLUE, alpha=0.85, label="Regression + NN")
    b.set_yscale("log")
    b.set_xlabel(f"Predicted − {model_label} temperature (K)")
    b.set_ylabel("Count")
    b.legend(fontsize=7, frameon=False, loc="upper left")
    b.text(-0.25, 1.04, "(b)", transform=b.transAxes, fontweight="bold")

    for x in ax:
        x.grid(color="0.9", lw=0.5)
        x.set_axisbelow(True)
    fig.tight_layout()
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_stem.with_suffix(f".{ext}"), dpi=300)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", default="HeFESToMars",
                   help="HeFESTo model directory under ngibbs/engine/TrainedModels "
                        "(default: HeFESToMars).")
    p.add_argument("--label", default="HeFESTo",
                   help="Name of the reference thermodynamic model, used in axis labels.")
    p.add_argument("--s-min", type=float, default=1.75,
                   help="Keep only test states with S > s-min (J/g/K); the temperature model "
                        "was trained on adiabats only. Default 1.75.")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--out", type=Path, default=Path("temperature_model_parity"),
                   help="Output path stem; writes <out>.png, <out>.pdf and <out>_metrics.json.")
    args = p.parse_args()

    model_dir = Path(ngibbs.engine.__file__).parent / "TrainedModels" / args.model_dir
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is not available")
    api = HeFESToAPI(model_dir=str(model_dir), device=args.device)
    if getattr(api, "temperature_model", None) is None or api.isentropic_emulator is None:
        raise SystemExit(f"{model_dir} has no isentropic emulator + temperature model.")
    bundle_path = api._test_bundles.get("isentropic")
    if bundle_path is None:
        raise SystemExit(f"{model_dir} ships no isentropic test bundle.")

    bundle = load_ml_bundle(bundle_path, arrays=("features", "free_outputs"))
    idx = bundle.ml_indexer
    t_col = list(idx.free_outputs).index(api.temperature_payload["temperature_label"])
    s_col = api.temperature_payload["s_feature_idx"]

    keep = bundle.features[:, s_col] > args.s_min
    features = np.asarray(bundle.features[keep], dtype=np.float32)
    T = np.asarray(bundle.free_outputs[keep, t_col], dtype=np.float64)
    print(f"Test bundle: {bundle_path}")
    print(f"States with S > {args.s_min}: {keep.sum()} of {keep.size}")

    T_reg, T_full = predict_temperatures(api, features)
    metrics = {
        "model_dir": str(model_dir),
        "test_bundle": str(bundle_path),
        "s_min": args.s_min,
        "n_states": int(keep.sum()),
        "T_range_K": [float(T.min()), float(T.max())],
        "regression_only": error_stats(T_reg, T),
        "regression_plus_nn": error_stats(T_full, T),
    }
    print(json.dumps(metrics, indent=2))

    plot_parity(T, T_reg, T_full, args.out, args.label)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{args.out}_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Wrote {args.out}.png, {args.out}.pdf, {args.out}_metrics.json")


if __name__ == "__main__":
    main()
