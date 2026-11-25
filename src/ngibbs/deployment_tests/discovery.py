"""Enumerate the model working copies in ``ngibbs/engine/TrainedModels`` together
with their Test_subset ground-truth bundles.

Uses the same filename conventions (and the same helpers) as the ``EmulatorAPI``
/ ``MELTSAPI`` / ``HeFESToAPI`` constructors, so a pair listed here is exactly the
pair ``api.test()`` would score - without building a full API (no temperature
FCNN, EOS parameters or standards are loaded).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Sequence


@dataclass(frozen=True)
class ShippedEmulator:
    family: str                 # TrainedModels/<family>
    variant: Optional[str]      # 'nocr' / 'cr' for MELTS, None otherwise
    kind: str                   # 'isothermal' / 'isentropic' / 'openox'
    checkpoint: Path
    bundle: Optional[Path]      # None if no Test_subset bundle matches

    @property
    def label(self) -> str:
        parts = [self.family] + ([self.variant] if self.variant else []) + [self.kind]
        return '_'.join(parts)


def trained_models_root() -> Path:
    """``engine/TrainedModels/``: the model working copies in a source checkout."""
    from ngibbs.registry import CHECKOUT_MODELS_DIR
    return CHECKOUT_MODELS_DIR


def shipped_emulators(models_root=None, *, families: Optional[Sequence[str]] = None,
                      kinds: Optional[Sequence[str]] = None) -> Iterator[ShippedEmulator]:
    """Yield every (checkpoint, bundle) pair of the models under ``models_root``
    (default: the working copies in this source checkout), NoCr before Cr,
    isothermal before isentropic before openox. ``families`` filters by model
    folder name (e.g. 'MELTS120', 'HeFESToEarth')."""
    from ngibbs.engine.API import (
        _classify_checkpoint_variant, _discover_checkpoints, _discover_test_bundle,
    )
    root = Path(models_root) if models_root is not None else trained_models_root()
    dirs = sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
    for d in dirs:
        if families and d.name not in families:
            continue
        variants = {_classify_checkpoint_variant(t) for t in d.glob('*.tar')} - {None}
        for v in (sorted(variants, key=lambda x: x != 'nocr') or [None]):
            found = _discover_checkpoints(d, variant=v)
            for kind in ('isothermal', 'isentropic', 'openox'):
                ckpt = found[kind]
                if ckpt is None or (kinds and kind not in kinds):
                    continue
                bundle = _discover_test_bundle(ckpt, d)
                yield ShippedEmulator(d.name, v, kind, ckpt, bundle)
