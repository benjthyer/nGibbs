"""
Model loading for nGibbs thermodynamic emulators.

Instantiates the ready-to-use EmulatorAPI singletons (HeFESTo, HeFESTo Mars,
MELTS102, MELTS120) on CPU, and on GPU when available. Import the model
singletons you need from here rather than from API.py, which is reserved
for the emulator classes and real computation, not for loading/dispatch.
"""

import torch
from pathlib import Path

from .API import HeFESToAPI, MELTSAPI

# Model paths - resolved relative to the engine package location. Each
# directory self-contains its checkpoints (*.tar/*.pt) and, alongside them
# or under the sibling deployment_tests/, its *_Test_subset*.tar.gz quality
# bundle -- both auto-discovered by EmulatorAPI.__init__ (see its docstring).
_this_file_dir = Path(__file__).parent

# Model name -> (API class, constructor kwargs minus `device`). Each entry is
# written once here; the CPU and GPU singleton dicts below both build from
# this same table so paths never have to be repeated per device.
#
# Mass balance: every emulator defaults to the 'iterative' (MassBalanceProjector)
# correction -- generalizable across architectures and GPU-safe. Add
# `mass_balance='pinv'` (one-shot pseudo-inverse) or `'none'` to an entry's kwargs
# to change it for that model; this is the single place a deployment default lives.
_MODEL_SPECS = {
    "HeFESToEmulator": (
        HeFESToAPI,
        dict(model_dir=str(_this_file_dir / "TrainedModels" / "HeFESToEarth")),
    ),
    "MELTS120Emulator": (
        MELTSAPI,
        dict(model_dir=str(_this_file_dir / "TrainedModels" / "120")),
    ),
    "MELTS102Emulator": (
            MELTSAPI,
            dict(model_dir=str(_this_file_dir / "TrainedModels" / "102")),
        ),
    "HeFESToMarsEmulator": (
        HeFESToAPI,
        dict(model_dir=str(_this_file_dir / "TrainedModels" / "HeFESToMars")),
    ),
}

# CPU_MODELS is always fully populated; GPU_MODELS only gains entries when
# CUDA is available. Both are keyed by the same model names as _MODEL_SPECS.
CPU_MODELS = {
    name: cls(**kwargs, device='cpu')
    for name, (cls, kwargs) in _MODEL_SPECS.items()
}

GPU_MODELS = {
    name: cls(**kwargs, device='cuda')
    for name, (cls, kwargs) in _MODEL_SPECS.items()
} if torch.cuda.is_available() else {}

# Deployment tests (see ngibbs.deployment_tests). Each model's standard
# workspace is the set of raw simulator runs shipped with the package for it;
# test(), evaluate_workspace() with no path, and the benchmark all score it.
# Benchmark baselines live with each model's checkpoints, under
# TrainedModels/<family>/benchmark_baseline/<cpu|cuda>/, and are packaged so an
# installed copy can check itself; regenerate with
# `python scripts/run_benchmarks.py --generate [--device cuda]`. They are kept
# per device because CPU and GPU float32 kernels round differently, and minor
# phases' compositions amplify that past any tolerance tight enough to be useful.
_DEPLOYMENT_TESTS = _this_file_dir.parent / "deployment_tests"
BENCHMARK_BASELINE_SUBDIR = "benchmark_baseline"
STANDARD_WORKSPACES = {
    "HeFESToEmulator": _DEPLOYMENT_TESTS / "HeFESToAdiabatStandards" / "Earth",
    "HeFESToMarsEmulator": _DEPLOYMENT_TESTS / "HeFESToAdiabatStandards" / "Mars",
    "MELTS120Emulator": _DEPLOYMENT_TESTS / "MELTSIsobaricStandards231" / "120",
    "MELTS102Emulator": _DEPLOYMENT_TESTS / "MELTSIsobaricStandards231" / "102",
}
for _name in _MODEL_SPECS:
    for _model, _device in ((CPU_MODELS[_name], "cpu"), (GPU_MODELS.get(_name), "cuda")):
        if _model is not None:
            _model.benchmark_name = _name
            _model.benchmark_baseline_dir = (
                Path(_MODEL_SPECS[_name][1]['model_dir']) / BENCHMARK_BASELINE_SUBDIR / _device)
            if _name in STANDARD_WORKSPACES:
                _model.standard_workspace_dir = STANDARD_WORKSPACES[_name]

# Public module-level singletons, kept under their original names so
# existing `from ngibbs.engine.models import ...` call sites don't change.
# GPU names default to None (rather than being absent) when CUDA isn't
# available, matching the original module's behavior.
for _name in _MODEL_SPECS:
    globals()[f"{_name}CPU"] = CPU_MODELS[_name]
    globals()[f"{_name}GPU"] = GPU_MODELS.get(_name)
