"""
Run the .assert_quality() deployable self-check for every model in CPU_MODELS
and GPU_MODELS (ngibbs.engine.models). GPU_MODELS is empty when CUDA isn't
available, so this naturally skips GPU testing on CPU-only machines.

Each model's metrics are checked against the fixed tolerances in
ngibbs.deployment_tests.quality_thresholds (precision/recall by modal
abundance, major-phase abundance/oxide error, and -- for HeFESTo -- the
MELTStable EOS property error). Exits non-zero if any model fails or errors.

Usage:
    python scripts/run_model_tests.py
"""

import sys

from ngibbs.deployment_tests import EmulatorQualityError
from ngibbs.engine.models import CPU_MODELS, GPU_MODELS


def run_all_tests(models: dict, device_label: str) -> dict:
    results = {}
    for name, model in models.items():
        print(f"\n=== [{device_label}] {name} ===")
        try:
            results[name] = model.assert_quality()
        except EmulatorQualityError as exc:
            print(f"[{device_label}] {name} FAILED quality gate:\n{exc}")
            results[name] = exc
        except Exception as exc:
            print(f"[{device_label}] {name} ERRORED: {exc}")
            results[name] = exc
    return results


def main():
    all_results = {
        "cpu": run_all_tests(CPU_MODELS, "cpu"),
        "gpu": run_all_tests(GPU_MODELS, "gpu") if GPU_MODELS else {},
    }

    if not GPU_MODELS:
        print("\n[gpu] no GPU models available (CUDA not detected); skipped.")

    print("\n=== Summary ===")
    any_failed = False
    for device_label, results in all_results.items():
        for name, result in results.items():
            failed = isinstance(result, Exception)
            any_failed = any_failed or failed
            status = "FAILED" if failed else "ok"
            print(f"[{device_label}] {name}: {status}")

    if any_failed:
        sys.exit(1)
    return all_results


if __name__ == "__main__":
    main()
