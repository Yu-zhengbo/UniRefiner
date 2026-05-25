"""Model weight and state-dict compatibility helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping

import torch
from torch import nn


StateDict = dict[str, torch.Tensor]


def unwrap_state_dict(checkpoint) -> StateDict:
    """Accept common state-dict checkpoint wrappers."""

    if isinstance(checkpoint, Mapping) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    elif isinstance(checkpoint, Mapping) and "model" in checkpoint:
        checkpoint = checkpoint["model"]

    if not isinstance(checkpoint, Mapping):
        raise TypeError("Expected a state-dict-like checkpoint object.")

    state_dict = {str(key): value for key, value in checkpoint.items()}
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {key[7:]: value for key, value in state_dict.items()}
    return state_dict


def load_state_dict_file(path: str | Path, *, map_location: str | torch.device = "cpu") -> StateDict:
    checkpoint = torch.load(path, map_location=map_location)
    return unwrap_state_dict(checkpoint)


def load_checkpoint_if_needed(model: nn.Module, checkpoint_path: str | Path | None, *, role: str) -> nn.Module:
    if checkpoint_path is None:
        return model

    state_dict = load_state_dict_file(checkpoint_path)
    logging.info("Loading %s checkpoint from %s", role, checkpoint_path)
    load_result = model.load_state_dict(state_dict, strict=False)
    missing = getattr(load_result, "missing_keys", [])
    unexpected = getattr(load_result, "unexpected_keys", [])
    if missing or unexpected:
        logging.info(
            "Loaded %s checkpoint with %d missing and %d unexpected keys.",
            role,
            len(missing),
            len(unexpected),
        )
    return model
