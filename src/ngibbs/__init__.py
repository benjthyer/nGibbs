"""nMELTS - MELTS thermodynamic modeling emulator package."""
import torch


from .engine.API import HeFESToEmulatorCPU,  MELTS102EmulatorCPU

if torch.cuda.is_available():
    from .engine.API import HeFESToEmulatorGPU, MELTS102EmulatorGPU