"""DINOv2 ViT wrapper."""

from __future__ import annotations

import torch

from .attention_hooks import AttentionHookCache, HookHandleGroup, register_projection_hooks


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def wrap_dinov2_giant(model):
    """Expose dense visual tokens for HF DINOv2 ViTs."""

    def encode_dense(self, images: torch.Tensor) -> torch.Tensor:
        embedding_output = self.embeddings(images)
        encoder_outputs = self.encoder(embedding_output)
        sequence_output = self.layernorm(encoder_outputs.last_hidden_state)
        return sequence_output[:, 1:, :]

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        embedding_output = self.embeddings(images)
        encoder_outputs = self.encoder(embedding_output)
        sequence_output = self.layernorm(encoder_outputs.last_hidden_state)
        return sequence_output[:, 0, :]

    def prepare_attention_hooks(
        self,
        cache: AttentionHookCache,
        layers: range | list[int] | None = None,
        capture: tuple[str, ...] = ("q", "k"),
        *,
        get_states: bool = False,
    ) -> HookHandleGroup:
        selected_layers = self.encoder.layer if layers is None else [self.encoder.layer[index] for index in layers]
        return register_projection_hooks(
            selected_layers,
            cache,
            q_path="attention.attention.query",
            k_path="attention.attention.key",
            v_path="attention.attention.value",
            capture=capture,
            skip_prefix_tokens=1,
            get_states=get_states,
        )

    def hook_prepare(self, dense_features, get_v: bool = False, get_states: bool = False):
        capture = ("q", "k", "v") if get_v else ("q", "k")
        return self.prepare_attention_hooks(dense_features, capture=capture, get_states=get_states)

    model.encode_dense = encode_dense.__get__(model)
    model.encode_image = encode_image.__get__(model)
    model.prepare_attention_hooks = prepare_attention_hooks.__get__(model)
    model.hook_prepare = hook_prepare.__get__(model)
    model.patch_size = int(model.embeddings.patch_size)
    model.image_mean = IMAGENET_DEFAULT_MEAN
    model.image_std = IMAGENET_DEFAULT_STD
    model.num_register_tokens = 0
    torch.cuda.empty_cache()
    return model


DINOV2_Wrapper = wrap_dinov2_giant
