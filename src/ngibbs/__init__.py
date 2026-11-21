#nGibbs - Neural network emulators for MELTS and HeFESTo thermodynamic modeling.

#Model singletons are loaded in ngibbs.engine.models for easy access.

# GPU batch-size tuning (see ngibbs.utils.batch_tuning). Imports are deferred
# so `import ngibbs` stays light; the packaged models are only loaded when a
# tuning call needs one.


def tune_batch_size(api=None, model='HeFESToEmulator', **kwargs):
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


# Model objects, importable at package level:
#     from ngibbs import MELTS120EmulatorCPU, HeFESToEmulatorGPU
# Resolved lazily (PEP 562) from ngibbs.engine.models, so `import ngibbs` alone
# loads no checkpoints; the first model access loads all of them, once.
_MODEL_NAMES = ('HeFESToEmulator', 'HeFESToMarsEmulator', 'MELTS102Emulator', 'MELTS120Emulator')
_LAZY_MODEL_ATTRS = frozenset(
    [f'{n}{d}' for n in _MODEL_NAMES for d in ('CPU', 'GPU')] + ['CPU_MODELS', 'GPU_MODELS'])

__all__ = ['tune_batch_size', 'batch_tuning_summary', 'show_batch_tuning',
           'clear_batch_tuning', *sorted(_LAZY_MODEL_ATTRS)]


def __getattr__(name):
    if name in _LAZY_MODEL_ATTRS:
        from .engine import models
        value = getattr(models, name)
        globals()[name] = value  # cache: later lookups skip __getattr__
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _LAZY_MODEL_ATTRS)
