"""Model factory helpers."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from nMELTS.engine.NN import MidLevelNetwork


def create_model(config: Dict[str, Any], ml_indexer=None, device: str = "cuda") -> MidLevelNetwork:
    """Create a MidLevelNetwork instance from config and optional ml_indexer."""
    try:
        model = MidLevelNetwork(ml_indexer=ml_indexer, **config)
    except TypeError:
        model = MidLevelNetwork(**config)
    return model.to(device)


def load_model(checkpoint_path: str, device: str = "cuda") -> MidLevelNetwork:
    """Load a checkpoint and reconstruct the model and ml_indexer."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    ml_indexer = checkpoint.get("ml_indexer")
    model = create_model(config=config, ml_indexer=ml_indexer, device=device)
    state_dict = checkpoint.get("state_dict")
    if state_dict is None:
        raise ValueError("Checkpoint missing state_dict")
    model.load_state_dict(state_dict, strict=False)
    return model
