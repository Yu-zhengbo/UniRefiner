"""LoRA wrapper factory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from torch import nn

from ..wrappers.base import MissingModelDependency
from .target_modules import module_types_from_names, select_lora_target_modules


PRESET_DIR = Path(__file__).resolve().parent / "presets"


def _import_peft():
    try:
        from peft import LoraConfig, PeftModel, get_peft_model
    except Exception as error:  # pragma: no cover - depends on optional install
        raise MissingModelDependency("PEFT is required to attach LoRA adapters.") from error
    return LoraConfig, PeftModel, get_peft_model


def normalize_lora_spec(lora_spec: str | list[str] | tuple[str, ...] | None) -> str | None:
    """Normalize the public LoRA config value.

    First-release UniRefiner supports one LoRA preset per run. Lists and comma
    separated strings are accepted only when they contain a single item, so a
    mistaken multi-LoRA config fails early instead of silently changing the
    training policy.
    """

    if lora_spec is None or lora_spec == "":
        return None
    if isinstance(lora_spec, str):
        specs = [item.strip() for item in lora_spec.split(",") if item.strip()]
    elif isinstance(lora_spec, (list, tuple)):
        specs = [str(item).strip() for item in lora_spec if str(item).strip()]
    else:
        raise TypeError("LoRA spec must be None, a string, or a single-item list/tuple.")

    if not specs:
        return None
    if len(specs) > 1:
        raise ValueError("Only a single LoRA preset is supported in the first release.")
    return specs[0]


def load_lora_preset(name: str) -> dict[str, Any]:
    path = PRESET_DIR / f"{name}.json"
    if not path.is_file():
        raise ValueError(f"LoRA preset `{name}` was not found at {path}.")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def create_lora_config(model: nn.Module, preset_name: str):
    LoraConfig, _PeftModel, _get_peft_model = _import_peft()
    preset = load_lora_preset(preset_name)
    lora_config = LoraConfig(**preset["lora_config"])
    module_types = module_types_from_names(preset["module_type_list"])
    targets = select_lora_target_modules(model, module_types, layer_cfg=preset.get("layer"))
    if not targets:
        raise ValueError(f"LoRA preset `{preset_name}` did not match any target modules.")
    lora_config.target_modules = targets
    return lora_config


def apply_lora(model: nn.Module, preset_name: str | list[str] | tuple[str, ...] | None, *, merge: bool = False) -> nn.Module:
    """Attach at most one LoRA preset."""

    preset_name = normalize_lora_spec(preset_name)
    if preset_name is None:
        return model

    _LoraConfig, PeftModel, get_peft_model = _import_peft()
    peft_model = get_peft_model(model, create_lora_config(model, preset_name))
    if merge:
        if not isinstance(peft_model, PeftModel) or not hasattr(peft_model, "merge_and_unload"):
            raise RuntimeError("PEFT merge_and_unload is required for merge=True.")
        return peft_model.merge_and_unload()
    return peft_model


lora_model_factory = apply_lora
