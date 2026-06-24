"""Generic dense-token wrappers for ViT-like vision backbones."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import torch
from torch import nn

from .attention_hooks import (
    AttentionHookCache,
    HookHandleGroup,
    get_nested_module,
    register_eva_attention_pre_hooks,
    register_packed_qkv_hooks,
    register_projection_hooks,
)
from .base import AttentionHooksNotSupported, UnsupportedModelError, ensure_image_stats


IMAGENET_DEFAULT_MEAN = (0.5, 0.5, 0.5)
IMAGENET_DEFAULT_STD = (0.5, 0.5, 0.5)


def _as_tuple3(values: Any) -> tuple[float, float, float] | None:
    if values is None:
        return None
    if isinstance(values, torch.Tensor):
        values = values.detach().flatten().tolist()
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or len(values) != 3:
        return None
    stats = ensure_image_stats(values, name="image stats")
    if max(abs(value) for value in stats) > 2.0:
        return tuple(value / 255.0 for value in stats)
    return stats


def _nested_attr(source: Any, dotted_path: str) -> Any:
    current = source
    for part in dotted_path.split("."):
        if isinstance(current, dict):
            current = current[part]
        else:
            current = getattr(current, part)
    return current


def _first_existing_attr(source: Any, paths: Iterable[str]) -> Any:
    for path in paths:
        try:
            value = _nested_attr(source, path)
        except (AttributeError, KeyError):
            continue
        if value is not None:
            return value
    return None


def _image_stats(model: nn.Module, *, fallback_mean=IMAGENET_DEFAULT_MEAN, fallback_std=IMAGENET_DEFAULT_STD):
    mean = _as_tuple3(
        _first_existing_attr(
            model,
            (
                "default_cfg.mean",
                "pretrained_cfg.mean",
                "config.image_mean",
                "config.vision_config.image_mean",
                "pixel_mean",
            ),
        )
    )
    std = _as_tuple3(
        _first_existing_attr(
            model,
            (
                "default_cfg.std",
                "pretrained_cfg.std",
                "config.image_std",
                "config.vision_config.image_std",
                "pixel_std",
            ),
        )
    )
    return mean or fallback_mean, std or fallback_std


def _patch_size(model: nn.Module, override: int | None = None) -> int:
    if override is not None:
        return int(override)
    value = _first_existing_attr(
        model,
        (
            "patch_size",
            "patch_embed.patch_size",
            "embeddings.patch_size",
            "config.patch_size",
            "config.vision_config.patch_size",
        ),
    )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return int(value[0])
    if value is not None:
        return int(value)
    raise UnsupportedModelError(f"Cannot infer patch size for {model.__class__.__name__}.")


def _prefix_tokens(model: nn.Module, override: int | None = None) -> int:
    if override is not None:
        return int(override)
    value = _first_existing_attr(model, ("num_prefix_tokens", "config.num_prefix_tokens"))
    if value is not None:
        return int(value)
    if any(hasattr(model, attr) for attr in ("cls_token", "class_token")):
        return 1
    return 0


def _extract_vision_module(model: nn.Module, candidates: Sequence[str]) -> nn.Module:
    for name in candidates:
        child = getattr(model, name, None)
        if isinstance(child, nn.Module):
            return child
    return model


def _coerce_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, dict):
        for key in ("x_norm_patchtokens", "patch_tokens", "last_hidden_state", "image_embeddings", "features"):
            value = output.get(key)
            if isinstance(value, torch.Tensor):
                return value
    for attr in ("last_hidden_state", "image_embeddings", "image_embeds"):
        value = getattr(output, attr, None)
        if isinstance(value, torch.Tensor):
            return value
    if isinstance(output, (tuple, list)):
        for value in output:
            if isinstance(value, torch.Tensor):
                return value
    raise TypeError(f"Cannot extract a dense tensor from output type {type(output)!r}.")


def _flatten_dense_tensor(tensor: torch.Tensor, *, skip_prefix_tokens: int) -> torch.Tensor:
    if tensor.ndim == 3:
        return tensor[:, skip_prefix_tokens:, :] if skip_prefix_tokens > 0 else tensor
    if tensor.ndim == 4:
        if tensor.shape[-1] >= tensor.shape[1]:
            return tensor.reshape(tensor.shape[0], -1, tensor.shape[-1])
        return tensor.flatten(2).transpose(1, 2)
    raise ValueError(f"Dense output must have rank 3 or 4, got {tuple(tensor.shape)}.")


def _forward_dense(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    if callable(getattr(model, "forward_features", None)):
        return _coerce_tensor(model.forward_features(images))
    try:
        return _coerce_tensor(model(images))
    except TypeError:
        return _coerce_tensor(model(pixel_values=images))


def _layers_from_model(model: nn.Module) -> Sequence[nn.Module]:
    value = _first_existing_attr(model, ("blocks", "encoder.layers", "encoder.layer", "layers"))
    if isinstance(value, nn.ModuleList):
        return value
    if isinstance(value, (list, tuple)) and all(isinstance(item, nn.Module) for item in value):
        return value
    raise AttentionHooksNotSupported(f"Cannot locate transformer layers for {model.__class__.__name__}.")


def _has_module(layer: nn.Module, path: str) -> bool:
    try:
        get_nested_module(layer, path)
    except (AttributeError, AttentionHooksNotSupported):
        return False
    return True


def _is_eva_attention(module: nn.Module) -> bool:
    return module.__class__.__name__ == "EvaAttention" or (
        hasattr(module, "qkv")
        and hasattr(module, "q_norm")
        and hasattr(module, "k_norm")
        and hasattr(module, "num_prefix_tokens")
    )


def _register_auto_attention_hooks(
    layers: Sequence[nn.Module],
    cache: AttentionHookCache,
    *,
    capture: tuple[str, ...],
    skip_prefix_tokens: int,
    get_states: bool,
) -> HookHandleGroup:
    first = layers[0]
    if _has_module(first, "attn"):
        attn = get_nested_module(first, "attn")
        if _is_eva_attention(attn):
            return register_eva_attention_pre_hooks(
                layers,
                cache,
                attn_path="attn",
                capture=capture,
                skip_prefix_tokens=skip_prefix_tokens,
                get_states=get_states,
            )
        if getattr(attn, "qkv", None) is not None:
            return register_packed_qkv_hooks(
                layers,
                cache,
                qkv_path="attn.qkv",
                capture=capture,
                skip_prefix_tokens=skip_prefix_tokens,
                get_states=get_states,
            )
        if _has_module(first, "attn.q_proj") and _has_module(first, "attn.k_proj"):
            return register_projection_hooks(
                layers,
                cache,
                q_path="attn.q_proj",
                k_path="attn.k_proj",
                v_path="attn.v_proj" if _has_module(first, "attn.v_proj") else None,
                capture=capture,
                skip_prefix_tokens=skip_prefix_tokens,
                get_states=get_states,
            )
    for prefix in ("self_attn", "attention.attention"):
        q_name = "query" if prefix == "attention.attention" else "q_proj"
        k_name = "key" if prefix == "attention.attention" else "k_proj"
        v_name = "value" if prefix == "attention.attention" else "v_proj"
        if _has_module(first, f"{prefix}.{q_name}") and _has_module(first, f"{prefix}.{k_name}"):
            return register_projection_hooks(
                layers,
                cache,
                q_path=f"{prefix}.{q_name}",
                k_path=f"{prefix}.{k_name}",
                v_path=f"{prefix}.{v_name}" if _has_module(first, f"{prefix}.{v_name}") else None,
                capture=capture,
                skip_prefix_tokens=skip_prefix_tokens,
                get_states=get_states,
            )
    raise AttentionHooksNotSupported(f"Cannot infer attention hook paths from {first.__class__.__name__}.")


def wrap_generic_dense_model(
    model: nn.Module,
    *,
    vision_candidates: Sequence[str] = ("vision_model", "vision_encoder", "image_encoder"),
    patch_size: int | None = None,
    skip_prefix_tokens: int | None = None,
    fallback_mean=IMAGENET_DEFAULT_MEAN,
    fallback_std=IMAGENET_DEFAULT_STD,
) -> nn.Module:
    """Attach UniRefiner dense-token methods to a ViT-like model."""

    vision = _extract_vision_module(model, vision_candidates)
    prefix_tokens = _prefix_tokens(vision, skip_prefix_tokens)

    def encode_dense(self, images: torch.Tensor) -> torch.Tensor:
        return _flatten_dense_tensor(_forward_dense(self, images), skip_prefix_tokens=self.num_prefix_tokens)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        dense = self.encode_dense(images)
        return dense.mean(dim=1)

    def prepare_attention_hooks(
        self,
        cache: AttentionHookCache,
        layers: range | list[int] | None = None,
        capture: tuple[str, ...] = ("q", "k"),
        *,
        get_states: bool = False,
    ) -> HookHandleGroup:
        all_layers = _layers_from_model(self)
        selected_layers = all_layers if layers is None else [all_layers[index] for index in layers]
        return _register_auto_attention_hooks(
            selected_layers,
            cache,
            capture=capture,
            skip_prefix_tokens=self.num_prefix_tokens,
            get_states=get_states,
        )

    def hook_prepare(self, dense_features, get_v: bool = False, get_states: bool = False):
        capture = ("q", "k", "v") if get_v else ("q", "k")
        return self.prepare_attention_hooks(dense_features, capture=capture, get_states=get_states)

    vision.encode_dense = encode_dense.__get__(vision)
    vision.encode_image = encode_image.__get__(vision)
    vision.prepare_attention_hooks = prepare_attention_hooks.__get__(vision)
    vision.hook_prepare = hook_prepare.__get__(vision)
    vision.patch_size = _patch_size(vision, patch_size)
    vision.image_mean, vision.image_std = _image_stats(vision, fallback_mean=fallback_mean, fallback_std=fallback_std)
    vision.num_prefix_tokens = prefix_tokens
    if not hasattr(vision, "num_register_tokens"):
        vision.num_register_tokens = 0

    torch.cuda.empty_cache()
    return vision
