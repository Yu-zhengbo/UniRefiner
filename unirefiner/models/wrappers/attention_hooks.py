"""Attention-hook helpers for model wrappers.

The cache stores a `record` flag and layer-indexed keys such as `10_q` and
`10_k` for AH filtering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

import torch
import torch.nn.functional as F
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


def _extract_eva_rope(inputs: tuple[object, ...], kwargs: Mapping[str, object] | None) -> torch.Tensor | None:
    if kwargs is not None and isinstance(kwargs.get("rope"), torch.Tensor):
        return kwargs["rope"]
    if len(inputs) > 1 and isinstance(inputs[1], torch.Tensor):
        return inputs[1]
    return None


def _apply_eva_rope(
    module: nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    rope: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if rope is None:
        return q, k

    try:
        from timm.models.eva import apply_rot_embed_cat
    except Exception as error:  # pragma: no cover - timm-version compatibility
        raise AttentionHooksNotSupported(
            "EvaAttention RoPE hooks require timm.models.eva.apply_rot_embed_cat."
        ) from error

    num_prefix_tokens = int(getattr(module, "num_prefix_tokens", 1))
    rotate_half = bool(getattr(module, "rotate_half", False))
    q = torch.cat(
        [q[:, :, :num_prefix_tokens, :], apply_rot_embed_cat(q[:, :, num_prefix_tokens:, :], rope, half=rotate_half)],
        dim=2,
    )
    k = torch.cat(
        [k[:, :, :num_prefix_tokens, :], apply_rot_embed_cat(k[:, :, num_prefix_tokens:, :], rope, half=rotate_half)],
        dim=2,
    )
    return q, k


def make_eva_attention_pre_hook(
    cache: Mapping[str, object],
    *,
    layer_id: int,
    capture: Iterable[str] = ("q", "k"),
    skip_prefix_tokens: int | None = None,
    detach: bool = True,
) -> Callable[[nn.Module, tuple[object, ...], Mapping[str, object]], None]:
    """Create a pre-hook for timm EvaAttention, including qkv-bias and RoPE."""

    capture = tuple(capture)

    def hook(module: nn.Module, inputs: tuple[object, ...], kwargs: Mapping[str, object] | None = None) -> None:
        if not bool(cache.get("record", False)):
            return
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            raise AttentionHooksNotSupported("EvaAttention pre-hook expected the first input to be a tensor.")

        x = inputs[0]
        batch_size, token_count, _ = x.shape
        num_heads = int(getattr(module, "num_heads"))

        if getattr(module, "qkv", None) is not None:
            qkv = module.qkv(x)
            q_bias = getattr(module, "q_bias", None)
            if q_bias is not None:
                qkv_bias = torch.cat((module.q_bias, module.k_bias, module.v_bias))
                if bool(getattr(module, "qkv_bias_separate", False)):
                    qkv = qkv + qkv_bias
                else:
                    qkv = F.linear(x, weight=module.qkv.weight, bias=qkv_bias)
            qkv = qkv.reshape(batch_size, token_count, 3, num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
        else:
            q = module.q_proj(x).reshape(batch_size, token_count, num_heads, -1).transpose(1, 2)
            k = module.k_proj(x).reshape(batch_size, token_count, num_heads, -1).transpose(1, 2)
            v = module.v_proj(x).reshape(batch_size, token_count, num_heads, -1).transpose(1, 2)

        q = module.q_norm(q)
        k = module.k_norm(k)
        q, k = _apply_eva_rope(module, q, k, _extract_eva_rope(inputs, kwargs))

        skip = int(getattr(module, "num_prefix_tokens", 1) if skip_prefix_tokens is None else skip_prefix_tokens)
        values = {
            "q": q.transpose(1, 2).reshape(batch_size, token_count, -1),
            "k": k.transpose(1, 2).reshape(batch_size, token_count, -1),
            "v": v.transpose(1, 2).reshape(batch_size, token_count, -1),
        }
        for kind in capture:
            if kind not in values:
                continue
            value = _slice_prefix(values[kind], skip)
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


def register_eva_attention_pre_hooks(
    layers: Iterable[nn.Module],
    cache: Mapping[str, object],
    *,
    attn_path: str = "attn",
    capture: Iterable[str] = ("q", "k"),
    skip_prefix_tokens: int | None = None,
    get_states: bool = False,
) -> HookHandleGroup:
    """Register EvaAttention pre-hooks that record post-bias, post-RoPE Q/K."""

    handles: list[torch.utils.hooks.RemovableHandle] = []
    for layer_id, layer in enumerate(layers):
        handles.append(
            get_nested_module(layer, attn_path).register_forward_pre_hook(
                make_eva_attention_pre_hook(
                    cache,
                    layer_id=layer_id,
                    capture=capture,
                    skip_prefix_tokens=skip_prefix_tokens,
                ),
                with_kwargs=True,
            )
        )
        if get_states:
            handles.append(
                layer.register_forward_hook(
                    make_state_hook(
                        cache,
                        layer_id=layer_id,
                        skip_prefix_tokens=1 if skip_prefix_tokens is None else skip_prefix_tokens,
                    )
                )
            )
    return HookHandleGroup(handles)
