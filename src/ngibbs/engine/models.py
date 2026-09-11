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

# Model paths - resolved relative to the engine package location
_this_file_dir = Path(__file__).parent
_HeFESTo_dir = _this_file_dir / "TrainedModels" / "HeFESTo_Adiabats_Light"
_HeFESTo_heavy_dir = _this_file_dir / "TrainedModels" / "HeFESTo_EarthAdiabats_Heavy"
_HeFESTo_Mars_dir = _this_file_dir / "TrainedModels" / "HeFESTo_Mars"
_MELTS102_dir = _this_file_dir / "TrainedModels" / "MELTS102"
_MELTS120_dir = _this_file_dir / "TrainedModels" / "MELTS120"

# Ground-truth data shipped for the deployable .test() self-checks.
_deploy_tests_dir = _this_file_dir.parent / "deployment_tests"
_HeFESTo_adiabat_standards = _deploy_tests_dir / "HeFESToAdiabatStandards"


def _bundle(name: str) -> str:
    """Absolute path to a test bundle in the deployment_tests directory. The
    file need not exist yet -- .test() skips a configured-but-absent bundle with
    a note, so every emulator's slot is wired here as a template."""
    return str(_deploy_tests_dir / f"{name}_Test_subset50000.tar.gz")


def _hefesto_test_bundles(variant: str) -> dict:
    """{'isothermal': NPT bundle, 'isentropic': NPS bundle} for a HeFESTo model.
    `variant` is the model tag, e.g. 'light', 'heavy', 'Mars'."""
    stem = "HeFESTo_Mars" if variant == "Mars" else "HeFESTo_Earth_Adiabat"
    suffix = "" if variant == "Mars" else f"_{variant}"
    return {
        "isothermal": _bundle(f"{stem}_NPT{suffix}"),
        "isentropic": _bundle(f"{stem}_NPS{suffix}"),
    }


def _melts_test_bundles(gen: str, cr: bool) -> dict:
    """The three bundles for one Cr / NoCr MELTS sub-API: NPT (closed), NPS,
    and NPT (open oxygen). `gen` is '102' or '120'."""
    tag = "Cr" if cr else "NoCr"
    return {
        "isothermal": _bundle(f"{gen}SedIgClosed_{tag}_NPT"),
        "isentropic": _bundle(f"{gen}SedIgClosed_{tag}_NPS"),
        "openox":     _bundle(f"{gen}SedIgOpen_{tag}_NPT"),
    }

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
        dict(
            isothermal_model_path=str(_HeFESTo_dir / "HeFESTo_Earth_Adiabat_NPT_light.tar"),
            isentropic_model_path=str(_HeFESTo_dir / "HeFESTo_Earth_Adiabat_NPS_light.tar"),
            temperature_model_path=str(_HeFESTo_dir / "Residual_T_from_S_NN_light.pt"),
            test_bundles=_hefesto_test_bundles("light"),
            test_meltstable_dir=str(_HeFESTo_adiabat_standards),
        ),
    ),
    "HeFESToHeavyEmulator": (
            HeFESToAPI,
            dict(
                isothermal_model_path=str(_HeFESTo_heavy_dir / "HeFESTo_Earth_Adiabat_NPT_heavy.tar"),
                isentropic_model_path=str(_HeFESTo_heavy_dir / "HeFESTo_Earth_Adiabat_NPS_heavy.tar"),
                temperature_model_path=str(_HeFESTo_heavy_dir / "HeFESTo_Earth_Temp_heavy.pt"),
                test_bundles=_hefesto_test_bundles("heavy"),
                test_meltstable_dir=str(_HeFESTo_adiabat_standards),
            ),
        ),
    "HeFESToMarsEmulator": (
        HeFESToAPI,
        dict(
            isothermal_model_path=str(_HeFESTo_Mars_dir / "HeFESTo_Mars_Isothermal.tar"),
            isentropic_model_path=str(_HeFESTo_Mars_dir / "HeFESTo_Mars_Isentropic.tar"),
            temperature_model_path=str(_HeFESTo_Mars_dir / "HeFESTo_Mars_Temp.pt"),
            test_bundles=_hefesto_test_bundles("Mars"),
        ),
    ),
    "MELTS102Emulator": (
        MELTSAPI,
        dict(
            isothermal_NoCr_model_path=str(_MELTS102_dir / "102Isothermal_NoCr.tar"),
            isothermal_Cr_model_path=str(_MELTS102_dir / "102Isothermal_Cr.tar"),
            isentropic_NoCr_model_path=str(_MELTS102_dir / "102Isentropic_NoCr.tar"),
            isentropic_Cr_model_path=str(_MELTS102_dir / "102Isentropic_Cr.tar"),
            openox_NoCr_model_path=str(_MELTS102_dir / "102OpenOx_NoCr.tar"),
            openox_Cr_model_path=str(_MELTS102_dir / "102OpenOx_Cr.tar"),
            test_NoCr_bundles=_melts_test_bundles("102", cr=False),
            test_Cr_bundles=_melts_test_bundles("102", cr=True),
        ),
    ),
    "MELTS120Emulator": (
        MELTSAPI,
        dict(
            isothermal_NoCr_model_path=str(_MELTS120_dir / "120Isothermal_NoCr.tar"),
            isothermal_Cr_model_path=str(_MELTS120_dir / "120Isothermal_Cr.tar"),
            isentropic_NoCr_model_path=str(_MELTS120_dir / "120Isentropic_NoCr.tar"),
            isentropic_Cr_model_path=str(_MELTS120_dir / "120Isentropic_Cr.tar"),
            openox_NoCr_model_path=str(_MELTS120_dir / "120OpenOx_NoCr.tar"),
            openox_Cr_model_path=str(_MELTS120_dir / "120OpenOx_Cr.tar"),
            test_NoCr_bundles=_melts_test_bundles("120", cr=False),
            test_Cr_bundles=_melts_test_bundles("120", cr=True),
        ),
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
