"""SAM3 vision-trunk wrapper."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from .attention_hooks import (
    AttentionHookCache,
    HookHandleGroup,
    register_packed_qkv_hooks,
)
from .base import MissingModelDependency
from .generic import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


SAM3_PATCH_SIZE = 14
SAM3_EMBED_DIM = 1024
SAM3_DEPTH = 32
SAM3_GLOBAL_ATTN_BLOCKS = (7, 15, 23, 31)


def _ensure_local_sam3_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    local_sam3_root = repo_root / "sam3"
    if local_sam3_root.is_dir() and str(local_sam3_root) not in sys.path:
        sys.path.insert(0, str(local_sam3_root))
    cached = sys.modules.get("sam3")
    if cached is not None and getattr(cached, "__file__", None) is None:
        sys.modules.pop("sam3", None)


def _import_sam3_vit():
    try:
        from sam3.model.vitdet import ViT
    except Exception:  # pragma: no cover - depends on optional checkout/install
        _ensure_local_sam3_on_path()
        try:
            from sam3.model.vitdet import ViT
        except Exception as nested_error:
            raise MissingModelDependency(
                "SAM3 requires the `sam3` package or the local `sam3/` checkout."
            ) from nested_error
        else:
            return ViT
    else:
        return ViT


def _unwrap_checkpoint(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(checkpoint, Mapping) and "model" in checkpoint and isinstance(checkpoint["model"], Mapping):
        checkpoint = checkpoint["model"]
    if isinstance(checkpoint, Mapping) and "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], Mapping):
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Expected a SAM3 checkpoint with a state-dict-like payload.")
    return checkpoint


def _adapt_sam3_state_dict(checkpoint: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    state_dict = {str(key): value for key, value in checkpoint.items()}
    sam3_image_ckpt = {key.replace("detector.", ""): value for key, value in state_dict.items() if "detector" in key}
    sam3_image_ckpt = {
        key.replace("backbone.vision_backbone.trunk.", ""): value for key, value in sam3_image_ckpt.items()
    }
    drop_keys = [key for key in sam3_image_ckpt if key.endswith("freqs_cis")]
    for key in drop_keys:
        sam3_image_ckpt.pop(key)
    return sam3_image_ckpt


def load_sam3_checkpoint(model: nn.Module, checkpoint_path: str | Path) -> None:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = _adapt_sam3_state_dict(_unwrap_checkpoint(checkpoint))
    load_result = model.load_state_dict(state_dict, strict=False)
    missing = getattr(load_result, "missing_keys", [])
    if missing:
        print(f"loaded {checkpoint_path} and found missing and/or unexpected keys:\nmissing_keys={missing}")


def build_sam3_vision_model(
    *,
    img_size: int = 1008,
    checkpoint_path: str | Path | None = None,
    compile_mode: str | None = None,
) -> nn.Module:
    """Build the SAM3 image ViT exactly as the reference mmseg backbone does."""

    ViT = _import_sam3_vit()
    model = ViT(
        img_size=img_size,
        pretrain_img_size=336,
        patch_size=SAM3_PATCH_SIZE,
        embed_dim=SAM3_EMBED_DIM,
        depth=SAM3_DEPTH,
        num_heads=16,
        mlp_ratio=4.625,
        norm_layer="LayerNorm",
        drop_path_rate=0.1,
        qkv_bias=True,
        use_abs_pos=True,
        tile_abs_pos=True,
        global_att_blocks=SAM3_GLOBAL_ATTN_BLOCKS,
        rel_pos_blocks=(),
        use_rope=True,
        use_interp_rope=True,
        window_size=24,
        pretrain_use_cls_token=True,
        retain_cls_token=False,
        ln_pre=True,
        ln_post=False,
        return_interm_layers=True,
        bias_patch_embed=False,
        compile_mode=compile_mode,
    )
    if checkpoint_path is not None:
        load_sam3_checkpoint(model, checkpoint_path)
    return model.eval()


def _coerce_last_feature(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (list, tuple)) and output:
        for value in reversed(output):
            if isinstance(value, torch.Tensor):
                return value
    if isinstance(output, Mapping):
        for key in ("image_embeddings", "features", "last_hidden_state"):
            value = output.get(key)
            if isinstance(value, torch.Tensor):
                return value
    raise TypeError(f"Cannot extract SAM3 dense features from output type {type(output)!r}.")


def _flatten_feature_map(features: torch.Tensor) -> torch.Tensor:
    if features.ndim == 3:
        return features
    if features.ndim != 4:
        raise ValueError(f"SAM3 features must have rank 3 or 4, got {tuple(features.shape)}.")
    if features.shape[1] == SAM3_EMBED_DIM:
        return features.flatten(2).transpose(1, 2)
    return features.reshape(features.shape[0], -1, features.shape[-1])


def wrap_sam3(model: nn.Module) -> nn.Module:
    """Expose dense image-encoder tokens for a SAM3 ViT trunk."""

    def encode_dense(self, images: torch.Tensor) -> torch.Tensor:
        return _flatten_feature_map(_coerce_last_feature(self(images)))

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return self.encode_dense(images).mean(dim=1)

    def prepare_attention_hooks(
        self,
        cache: AttentionHookCache,
        layers: range | list[int] | None = None,
        capture: tuple[str, ...] = ("q", "k"),
        *,
        get_states: bool = False,
    ) -> HookHandleGroup:
        all_layers = list(self.blocks)
        selected_layers = all_layers if layers is None else [all_layers[index] for index in layers]
        return register_packed_qkv_hooks(
            selected_layers,
            cache,
            qkv_path="attn.qkv",
            capture=capture,
            skip_prefix_tokens=0,
            get_states=get_states,
        )

    def hook_prepare(self, dense_features, get_v: bool = False, get_states: bool = False):
        capture = ("q", "k", "v") if get_v else ("q", "k")
        return self.prepare_attention_hooks(dense_features, capture=capture, get_states=get_states)

    model.encode_dense = encode_dense.__get__(model)
    model.encode_image = encode_image.__get__(model)
    model.prepare_attention_hooks = prepare_attention_hooks.__get__(model)
    model.hook_prepare = hook_prepare.__get__(model)
    model.patch_size = SAM3_PATCH_SIZE
    model.image_mean = IMAGENET_DEFAULT_MEAN
    model.image_std = IMAGENET_DEFAULT_STD
    model.num_prefix_tokens = 0
    model.num_register_tokens = 0
    model.requires_square_inputs = True
    return model


SAM3_Wrapper = wrap_sam3
