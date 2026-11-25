"""Deprecated ready-made model objects (``HeFESToEmulatorCPU``, ``MELTS120EmulatorGPU``,
``CPU_MODELS``, ...).

Use ``ngibbs.make_ngibbs`` instead, which also chooses between model versions::

    from ngibbs import make_ngibbs
    api = make_ngibbs('MELTS120', device='cuda')

The old names still work and warn once each: each one is the latest compatible
version of its model, built on first use and then reused.
"""

import warnings
from collections.abc import Mapping

# Old object name prefix -> model name.
DEPRECATED_NAMES = {
    "HeFESToEmulator": "HeFESToEarth",
    "HeFESToMarsEmulator": "HeFESToMars",
    "MELTS102Emulator": "MELTS102",
    "MELTS120Emulator": "MELTS120",
}

_BUILT = {}


def _get(old_name: str, device: str):
    key = (old_name, device)
    if key not in _BUILT:
        from ..registry import make_ngibbs
        _BUILT[key] = make_ngibbs(DEPRECATED_NAMES[old_name], device=device)
    return _BUILT[key]


def _warn(what: str, instead: str) -> None:
    # 1 _warn, 2 deprecated_attr, 3 a module __getattr__, 4 the caller's line
    warnings.warn(f"{what} is deprecated and will be removed; use {instead}.",
                  FutureWarning, stacklevel=4)


class _DeprecatedModels(Mapping):
    """``CPU_MODELS`` / ``GPU_MODELS``: {old name: model}, built on first lookup."""

    def __init__(self, device: str):
        self._device = device

    def _available(self):
        if self._device == "cuda":
            import torch
            if not torch.cuda.is_available():
                return ()
        return tuple(DEPRECATED_NAMES)

    def __getitem__(self, name):
        if name not in self._available():
            raise KeyError(name)
        return _get(name, self._device)

    def __contains__(self, name):  # Mapping's default would build the model
        return name in self._available()

    def __iter__(self):
        return iter(self._available())

    def __len__(self):
        return len(self._available())


_LAZY_ATTRS = {f"{n}{suffix}": (n, suffix) for n in DEPRECATED_NAMES for suffix in ("CPU", "GPU")}


def __getattr__(attr):
    return deprecated_attr(attr)


def deprecated_attr(attr):
    """Resolve a deprecated name (warns). ngibbs.__getattr__ uses it too."""
    if attr in ("CPU_MODELS", "GPU_MODELS"):
        _warn(attr, "ngibbs.make_ngibbs(name, device=...)")
        return _DeprecatedModels("cpu" if attr == "CPU_MODELS" else "cuda")
    if attr in _LAZY_ATTRS:
        old, suffix = _LAZY_ATTRS[attr]
        device = "cpu" if suffix == "CPU" else "cuda"
        _warn(attr, f"ngibbs.make_ngibbs('{DEPRECATED_NAMES[old]}', device='{device}')")
        if device == "cuda":
            import torch
            if not torch.cuda.is_available():
                return None
        return _get(old, device)
    raise AttributeError(f"module {__name__!r} has no attribute {attr!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY_ATTRS) | {"CPU_MODELS", "GPU_MODELS"})
