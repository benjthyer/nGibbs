"""EOS_arithmetic Python helpers."""

from .ctherm import Ctherm
from .ener import Ener
from .etherm import Etherm
from .ftherm import Ftherm
from .gamset import HeFESToGamsetResult, gamset
from .heat import Heat
from .helm import Helm
from .hefesto_thermal_data import (
    HCOK,
    HeFESToParsetResult,
    HeFESToThermalModeParameters,
    HeFESToThermalModeTensors,
)
from .parset import parset
from .ztherm import Ztherm

# NOTE: The following imports are commented out because they have circular or broken dependencies
# in the current codebase. They will be available via lazy loading through __getattr__.
# from .param_state import (...)
# from .entrop import entrop

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    """Lazy import for modules with circular/broken dependencies."""
    if name == "PHYSUB_BULK_ATTRIBUTE_NAMES":
        from .param_state import PHYSUB_BULK_ATTRIBUTE_NAMES
        return PHYSUB_BULK_ATTRIBUTE_NAMES
    elif name == "HeFESToParameterRecord":
        from .param_state import HeFESToParameterRecord
        return HeFESToParameterRecord
    elif name == "entrop":
        from .entrop import entrop
        return entrop
    elif name == "bserch":
        from .bserch import bserch
        return bserch
    elif name == "thetacal":
        from .thetacal import thetacal
        return thetacal
    elif name == "qr19":
        from .qr19 import qr19
        return qr19
    elif name == "landau":
        from .landau import landau
        return landau
    elif name == "landauqr":
        from .landauqr import landauqr
        return landauqr
    elif name == "hessian":
        from .hessian import hessian
        return hessian
    elif name == "hessfunc":
        from .hessfunc import hessfunc
        return hessfunc
    elif name == "cp":
        from .cp import cp
        return cp
    elif name == "gspec":
        from .gspec import gspec
        return gspec
    elif name == "Ftotsub":
        from .Ftotsub import Ftotsub
        return Ftotsub
    elif name == "volume":
        from .volume import volume
        return volume
    elif name == "volumel":
        from .volumel import volumel
        return volumel
    elif name == "volumew":
        from .volumew import volumew
        return volumew
    elif name == "volumeh":
        from .volumeh import volumeh
        return volumeh
    elif name == "therm":
        from .therm import therm
        return therm
    elif name == "therml":
        from .therml import therml
        return therml
    elif name == "thermg":
        from .thermg import thermg
        return thermg
    elif name == "thermw":
        from .thermw import thermw
        return thermw
    elif name == "thermh":
        from .thermh import thermh
        return thermh
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
