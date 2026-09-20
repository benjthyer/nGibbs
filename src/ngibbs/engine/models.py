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
        dict(model_dir=str(_this_file_dir / "TrainedModels" / "HeFESToEarthAdiabat")),
    ),
    "MELTS120Emulator": (
        MELTSAPI,
        dict(model_dir=str(_this_file_dir / "TrainedModels" / "120")),
    ),
    "HeFESToMarsEmulator": (
        HeFESToAPI,
        dict(model_dir=str(_this_file_dir / "TrainedModels" / "HeFESToMars")),
    ),
    "HeFESToMarsGatedEmulator": (
        HeFESToAPI,
        dict(model_dir=str(_this_file_dir / "TrainedModels" / "HeFESToMarsGated")),
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

# Public module-level singletons, kept under their original names so
# existing `from ngibbs.engine.models import ...` call sites don't change.
# GPU names default to None (rather than being absent) when CUDA isn't
# available, matching the original module's behavior.
for _name in _MODEL_SPECS:
    globals()[f"{_name}CPU"] = CPU_MODELS[_name]
    globals()[f"{_name}GPU"] = GPU_MODELS.get(_name)
