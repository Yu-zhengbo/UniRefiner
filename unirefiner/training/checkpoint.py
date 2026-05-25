"""Training checkpoint save/resume policy."""

from __future__ import annotations

from pathlib import Path

import torch

from unirefiner.models.weights import unwrap_state_dict


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def save_epoch_checkpoint(model, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": unwrap_model(model).state_dict()}, output_path)


def save_final_model(model, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_model = unwrap_model(model)
    if hasattr(raw_model, "merge_and_unload"):
        final_state = raw_model.merge_and_unload().state_dict()
    else:
        final_state = raw_model.state_dict()
    torch.save(final_state, output_path)


def load_resume_checkpoint(model, resume_path: str | Path | None, *, map_location="cpu") -> None:
    if not resume_path:
        return
    checkpoint = torch.load(resume_path, map_location=map_location)
    state_dict = unwrap_state_dict(checkpoint)
    unwrap_model(model).load_state_dict(state_dict, strict=False)
