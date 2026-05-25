"""LoRA target-module selection."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from torch import nn


def _first_layer_index(module_name: str) -> int | None:
    for part in module_name.split("."):
        if part.isdigit():
            return int(part)
    return None


def _in_layer_range(module_name: str, layer_cfg: Mapping[str, object] | None) -> bool:
    if layer_cfg is None:
        return True
    layer_index = _first_layer_index(module_name)
    if layer_index is None:
        return False
    start = int(layer_cfg.get("start", 0))
    end_value = layer_cfg.get("end", "none")
    end = float("inf") if end_value == "none" else int(end_value)
    return start <= layer_index <= end


def module_types_from_names(names: Iterable[str]) -> tuple[type[nn.Module], ...]:
    module_types: list[type[nn.Module]] = []
    for name in names:
        module_type = getattr(nn, str(name), None)
        if module_type is None or not isinstance(module_type, type) or not issubclass(module_type, nn.Module):
            raise ValueError(f"Unknown torch.nn module type for LoRA target selection: {name!r}.")
        module_types.append(module_type)
    return tuple(module_types)


def select_lora_target_modules(
    model: nn.Module,
    module_types: Iterable[type[nn.Module]],
    *,
    layer_cfg: Mapping[str, object] | None = None,
) -> list[str]:
    """Select deterministic PEFT target module names.

    Without a layer range, PEFT can match suffix names such as `q_proj`; with a
    layer range, full module paths are returned so only the chosen layers match.
    Output is sorted to make target selection easy to inspect.
    """

    module_types = tuple(module_types)
    target_names: set[str] = set()
    for name, module in model.named_modules():
        if not name or not isinstance(module, module_types):
            continue
        if not _in_layer_range(name, layer_cfg):
            continue
        target_names.add(name if layer_cfg is not None else name.split(".")[-1])
    return sorted(target_names)
