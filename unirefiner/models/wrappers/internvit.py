"""InternViT wrapper.

This is for InternViT/InternVL3-era vision towers, not InternVL3.5-ViT.
"""

from __future__ import annotations

import torch

from .attention_hooks import AttentionHookCache, HookHandleGroup, register_packed_qkv_hooks


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def wrap_internvit(model, *, remove_last_layers: bool = True):
    """Expose dense visual tokens for InternViT backbones."""

    def encode_dense(self, images: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embeddings(images)

        if self.num_register_tokens > 0:
            hidden_states = torch.cat([hidden_states, self.reg_token.expand(hidden_states.size(0), -1, -1)], dim=1)

        encoder_outputs = self.encoder(inputs_embeds=hidden_states)
        dense_tokens = encoder_outputs.last_hidden_state[:, 1:, :]

        if self.channel_msk is not None:
            dense_tokens = dense_tokens * self.channel_msk

        return dense_tokens

    def prepare_attention_hooks(
        self,
        cache: AttentionHookCache,
        layers: range | list[int] | None = None,
        capture: tuple[str, ...] = ("q", "k"),
        *,
        get_states: bool = False,
    ) -> HookHandleGroup:
        selected_layers = self.encoder.layers if layers is None else [self.encoder.layers[index] for index in layers]
        return register_packed_qkv_hooks(
            selected_layers,
            cache,
            qkv_path="attn.qkv",
            capture=capture,
            skip_prefix_tokens=1,
            get_states=get_states,
        )

    def hook_prepare(self, dense_features, get_v: bool = False, get_states: bool = False):
        capture = ("q", "k", "v") if get_v else ("q", "k")
        return self.prepare_attention_hooks(dense_features, capture=capture, get_states=get_states)

    model.encode_dense = encode_dense.__get__(model)
    model.prepare_attention_hooks = prepare_attention_hooks.__get__(model)
    model.hook_prepare = hook_prepare.__get__(model)
    model.patch_size = int(model.embeddings.patch_size)
    model.image_mean = IMAGENET_DEFAULT_MEAN
    model.image_std = IMAGENET_DEFAULT_STD
    model.channel_msk = None
    model.num_register_tokens = 0

    if remove_last_layers:
        model.encoder.layers = model.encoder.layers[:-3]

    torch.cuda.empty_cache()
    return model


InternViT_Wrapper = wrap_internvit
InternViT6B_InternVL3_Wrapper = wrap_internvit
