"""Attention-hook helpers for model wrappers.

The cache stores a `record` flag and layer-indexed keys such as `10_q` and
`10_k` for AH filtering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

import torch
from torch import nn

from .base import AttentionHooksNotSupported


HookTransform = Callable[[torch.Tensor, nn.Module, str, int], torch.Tensor]


class AttentionHookCache(dict):
    """Record-gated dict cache for Q/K/V/state hooks."""

    def __init__(self, *, record: bool = False) -> None:
        super().__init__()
        self["record"] = bool(record)

    @property
    def record(self) -> bool:
        return bool(self.get("record", False))

    @record.setter
    def record(self, value: bool) -> None:
        self["record"] = bool(value)

    def clear_features(self) -> None:
        record = self.record
        super().clear()
        self["record"] = record


@dataclass(slots=True)
class HookHandleGroup:
    """Container that removes all hook handles together."""

    handles: list[torch.utils.hooks.RemovableHandle]

    def remove(self) -> None:
        while self.handles:
            self.handles.pop().remove()

    def __enter__(self) -> "HookHandleGroup":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.remove()


def _as_tensor(output) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"Hook output must contain a tensor, got {type(output)!r}.")


def _slice_prefix(tensor: torch.Tensor, skip_prefix_tokens: int) -> torch.Tensor:
    if skip_prefix_tokens <= 0:
        return tensor
    return tensor[:, skip_prefix_tokens:, :]


def make_projection_hook(
    cache: Mapping[str, object],
    *,
    layer_id: int,
    kind: str,
    skip_prefix_tokens: int = 1,
    detach: bool = True,
    transform: HookTransform | None = None,
) -> Callable[[nn.Module, tuple[object, ...], object], None]:
    """Create a hook for separate Q/K/V projection modules."""

    def hook(module: nn.Module, inputs: tuple[object, ...], output: object) -> None:
        if not bool(cache.get("record", False)):
            return
        tensor = _as_tensor(output)
        if transform is not None:
            tensor = transform(tensor, module, kind, layer_id)
        tensor = _slice_prefix(tensor, skip_prefix_tokens)
        cache[f"{layer_id}_{kind}"] = tensor.detach() if detach else tensor

    return hook


def make_state_hook(
    cache: Mapping[str, object],
    *,
    layer_id: int,
    skip_prefix_tokens: int = 1,
    detach: bool = True,
) -> Callable[[nn.Module, tuple[object, ...], object], None]:
    """Create a hook that records a layer output as `{layer_id}_states`."""

    def hook(module: nn.Module, inputs: tuple[object, ...], output: object) -> None:
        if not bool(cache.get("record", False)):
            return
        tensor = _slice_prefix(_as_tensor(output), skip_prefix_tokens)
        cache[f"{layer_id}_states"] = tensor.detach() if detach else tensor

    return hook


def make_packed_qkv_hook(
    cache: Mapping[str, object],
    *,
    layer_id: int,
    capture: Iterable[str] = ("q", "k"),
    skip_prefix_tokens: int = 1,
    detach: bool = True,
) -> Callable[[nn.Module, tuple[object, ...], object], None]:
    """Create a hook for a packed projection that returns QKV on the last dim."""

    capture = tuple(capture)

    def hook(module: nn.Module, inputs: tuple[object, ...], output: object) -> None:
        if not bool(cache.get("record", False)):
            return
        tensor = _as_tensor(output)
        if tensor.shape[-1] % 3 != 0:
            raise AttentionHooksNotSupported(
                f"Packed QKV output last dimension must be divisible by 3, got {tuple(tensor.shape)}."
            )
        q, k, v = tensor.split(tensor.shape[-1] // 3, dim=-1)
        values = {"q": q, "k": k, "v": v}
        for kind in capture:
            if kind not in values:
                continue
            value = _slice_prefix(values[kind], skip_prefix_tokens)
            cache[f"{layer_id}_{kind}"] = value.detach() if detach else value

    return hook


def get_nested_module(module: nn.Module, dotted_path: str) -> nn.Module:
    current: object = module
    for part in dotted_path.split("."):
        current = getattr(current, part)
    if not isinstance(current, nn.Module):
        raise AttentionHooksNotSupported(f"`{dotted_path}` is not an nn.Module.")
    return current


def register_projection_hooks(
    layers: Iterable[nn.Module],
    cache: Mapping[str, object],
    *,
    q_path: str,
    k_path: str,
    v_path: str | None = None,
    capture: Iterable[str] = ("q", "k"),
    skip_prefix_tokens: int = 1,
    get_states: bool = False,
    transform: HookTransform | None = None,
) -> HookHandleGroup:
    """Register hooks for layers with separate projection modules."""

    capture = set(capture)
    handles: list[torch.utils.hooks.RemovableHandle] = []
    for layer_id, layer in enumerate(layers):
        if "q" in capture:
            handles.append(
                get_nested_module(layer, q_path).register_forward_hook(
                    make_projection_hook(
                        cache,
                        layer_id=layer_id,
                        kind="q",
                        skip_prefix_tokens=skip_prefix_tokens,
                        transform=transform,
                    )
                )
            )
        if "k" in capture:
            handles.append(
                get_nested_module(layer, k_path).register_forward_hook(
                    make_projection_hook(
                        cache,
                        layer_id=layer_id,
                        kind="k",
                        skip_prefix_tokens=skip_prefix_tokens,
                        transform=transform,
                    )
                )
            )
        if "v" in capture and v_path is not None:
            handles.append(
                get_nested_module(layer, v_path).register_forward_hook(
                    make_projection_hook(
                        cache,
                        layer_id=layer_id,
                        kind="v",
                        skip_prefix_tokens=skip_prefix_tokens,
                        transform=transform,
                    )
                )
            )
        if get_states:
            handles.append(
                layer.register_forward_hook(
                    make_state_hook(cache, layer_id=layer_id, skip_prefix_tokens=skip_prefix_tokens)
                )
            )
    return HookHandleGroup(handles)


def register_packed_qkv_hooks(
    layers: Iterable[nn.Module],
    cache: Mapping[str, object],
    *,
    qkv_path: str,
    capture: Iterable[str] = ("q", "k"),
    skip_prefix_tokens: int = 1,
    get_states: bool = False,
) -> HookHandleGroup:
    handles: list[torch.utils.hooks.RemovableHandle] = []
    for layer_id, layer in enumerate(layers):
        handles.append(
            get_nested_module(layer, qkv_path).register_forward_hook(
                make_packed_qkv_hook(
                    cache,
                    layer_id=layer_id,
                    capture=capture,
                    skip_prefix_tokens=skip_prefix_tokens,
                )
            )
        )
        if get_states:
            handles.append(
                layer.register_forward_hook(
                    make_state_hook(cache, layer_id=layer_id, skip_prefix_tokens=skip_prefix_tokens)
                )
            )
    return HookHandleGroup(handles)
