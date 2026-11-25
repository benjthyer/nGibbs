"""nGibbs - Neural network emulators for MELTS and HeFESTo thermodynamic modeling.

    from ngibbs import make_ngibbs
    api = make_ngibbs('MELTS120', device='cuda')   # newest compatible version
    api = make_ngibbs('MELTS120v1.2')              # a specific version

    import ngibbs
    ngibbs.available_models()                      # what is on Hugging Face

Models download on first use (see ngibbs.registry). Imports are deferred so
`import ngibbs` stays light.
"""


def make_ngibbs(name, device='cpu', **kwargs):
    """Load an nGibbs emulator by name, downloading it on first use.
    See ngibbs.registry.make_ngibbs for all options."""
    from .registry import make_ngibbs as _make
    return _make(name, device=device, **kwargs)


def available_models(name=None, **kwargs):
    """Print the models on Hugging Face: the newest version of each (default),
    'all' versions of every model, or every version of one model by name.
    See ngibbs.registry.available_models."""
    from .registry import available_models as _available
    return _available(name, **kwargs)


# GPU batch-size tuning (see ngibbs.utils.batch_tuning).


def tune_batch_size(api=None, model='HeFESToEarth', **kwargs):
    """Tune and persist VRAM-aware batch sizes for this GPU.

    ``ngibbs.tune_batch_size()`` tunes the packaged HeFESTo GPU model; pass
    ``api=`` for your own EmulatorAPI/HeFESToAPI/MELTSAPI instance, or
    ``model=`` for another packaged one. Results persist across sessions and
    are used automatically whenever ``batch_size`` is left as None.
    Options: see ``ngibbs.utils.batch_tuning.tune_batch_size`` (and
    ``verify_rows=2**18`` to time defaults vs tuned afterwards).
    """
    from .utils import batch_tuning
    return batch_tuning.tune(api, model, **kwargs)


def batch_tuning_summary(device='cuda'):
    """{workload: entry} saved for ``device``'s GPU (empty if untuned)."""
    from .utils import batch_tuning
    return batch_tuning.summary(device)


def show_batch_tuning(device='cuda'):
    """Print the saved profile and the batch size each workload would use now."""
    from .utils import batch_tuning
    batch_tuning.show(device)


def clear_batch_tuning(device='cuda', workloads=None):
    """Forget tuned values for ``device``'s GPU (all, or the given workload keys)."""
    from .utils import batch_tuning
    batch_tuning.clear(device, workloads=workloads)


# Deprecated model objects (HeFESToEmulatorCPU, CPU_MODELS, ...), resolved
# lazily from ngibbs.engine.models, which warns; use make_ngibbs instead.
_MODEL_NAMES = ('HeFESToEmulator', 'HeFESToMarsEmulator', 'MELTS102Emulator', 'MELTS120Emulator')
_LAZY_MODEL_ATTRS = frozenset(
    [f'{n}{d}' for n in _MODEL_NAMES for d in ('CPU', 'GPU')] + ['CPU_MODELS', 'GPU_MODELS'])

__all__ = ['make_ngibbs', 'available_models', 'tune_batch_size', 'batch_tuning_summary',
           'show_batch_tuning', 'clear_batch_tuning']


def __getattr__(name):
    if name in _LAZY_MODEL_ATTRS:
        from .engine.models import deprecated_attr
        return deprecated_attr(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _LAZY_MODEL_ATTRS)
