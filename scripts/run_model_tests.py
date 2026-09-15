"""
Run the .test() deployable self-check for every model in CPU_MODELS and
GPU_MODELS (ngibbs.engine.models). GPU_MODELS is empty when CUDA isn't
available, so this naturally skips GPU testing on CPU-only machines.

Usage:
    python scripts/run_model_tests.py
"""

from ngibbs.engine.models import CPU_MODELS, GPU_MODELS


def run_all_tests(models: dict, device_label: str) -> dict:
    results = {}
    for name, model in models.items():
        print(f"\n=== [{device_label}] {name} ===")
        try:
            results[name] = model.test()
        except Exception as exc:
            print(f"[{device_label}] {name} FAILED: {exc}")
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
    for device_label, results in all_results.items():
        for name, result in results.items():
            status = "FAILED" if isinstance(result, Exception) else "ok"
            print(f"[{device_label}] {name}: {status}")

    return all_results


if __name__ == "__main__":
    main()
