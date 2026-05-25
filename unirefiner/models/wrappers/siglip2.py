"""SigLIP2 ViT wrappers for SO400M and Giant release models."""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from .attention_hooks import AttentionHookCache, HookHandleGroup, register_projection_hooks


SIGLIP_IMAGE_MEAN = (0.5, 0.5, 0.5)
SIGLIP_IMAGE_STD = (0.5, 0.5, 0.5)


def wrap_siglip2(model):
    """Wrap a non-NaFlex HF SigLIP2 model.

    The release configs target `siglip2-so400m` and `siglip2-giant`; both use
    this dense-token path.
    """

    def encode_dense(self, images: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embeddings(images, interpolate_pos_encoding=True)

        if self.num_register_tokens > 0:
            hidden_states = torch.cat([hidden_states, self.reg_token.expand(hidden_states.size(0), -1, -1)], dim=1)

        encoder_outputs = self.encoder(inputs_embeds=hidden_states)
        return self.post_layernorm(encoder_outputs.last_hidden_state)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embeddings(images, interpolate_pos_encoding=True)

        if self.num_register_tokens > 0:
            hidden_states = torch.cat([hidden_states, self.reg_token.expand(hidden_states.size(0), -1, -1)], dim=1)

        encoder_outputs = self.encoder(inputs_embeds=hidden_states)
        dense_tokens = self.post_layernorm(encoder_outputs.last_hidden_state)
        return self.head(dense_tokens)

    def encode_dense_w_proj(self, images: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embeddings(images, interpolate_pos_encoding=True)

        if self.num_register_tokens > 0:
            hidden_states = torch.cat([hidden_states, self.reg_token.expand(hidden_states.size(0), -1, -1)], dim=1)

        encoder_outputs = self.encoder(inputs_embeds=hidden_states)
        dense_tokens = encoder_outputs.last_hidden_state
        if self.num_register_tokens > 0:
            dense_tokens = dense_tokens[:, : -self.num_register_tokens]

        dense_tokens = self.post_layernorm(dense_tokens)
        attn_weight = self.head.attention.in_proj_weight
        attn_bias = self.head.attention.in_proj_bias
        query, key, value = F.linear(dense_tokens, attn_weight, attn_bias).chunk(3, dim=-1)
        _ = query, key
        dense_tokens = self.head.attention.out_proj(value)

        residual = dense_tokens
        dense_tokens = self.head.layernorm(dense_tokens)
        return residual + self.head.mlp(dense_tokens)

    def unlock_last_n_layers(self, n: int) -> None:
        self.requires_grad_(False)
        self.head.requires_grad_(True)
        for index in range(n):
            self.encoder.layers[-(index + 1)].requires_grad_(True)

    def prepare_attention_hooks(
        self,
        cache: AttentionHookCache,
        layers: range | list[int] | None = None,
        capture: tuple[str, ...] = ("q", "k"),
        *,
        get_states: bool = False,
    ) -> HookHandleGroup:
        selected_layers = self.encoder.layers if layers is None else [self.encoder.layers[index] for index in layers]
        return register_projection_hooks(
            selected_layers,
            cache,
            q_path="self_attn.q_proj",
            k_path="self_attn.k_proj",
            v_path="self_attn.v_proj",
            capture=capture,
            skip_prefix_tokens=0,
            get_states=get_states,
        )

    def hook_prepare(self, dense_features, get_states: bool = False, get_v: bool = False):
        capture = ("q", "k", "v") if get_v else ("q", "k")
        return self.prepare_attention_hooks(dense_features, capture=capture, get_states=get_states)

    vision = copy.deepcopy(model.vision_model)
    vision.encode_dense = encode_dense.__get__(vision)
    vision.encode_image = encode_image.__get__(vision)
    vision.encode_dense_w_proj = encode_dense_w_proj.__get__(vision)
    vision.unlock_last_n_layers = unlock_last_n_layers.__get__(vision)
    vision.prepare_attention_hooks = prepare_attention_hooks.__get__(vision)
    vision.hook_prepare = hook_prepare.__get__(vision)
    vision.patch_size = int(vision.embeddings.patch_size)
    vision.image_mean = SIGLIP_IMAGE_MEAN
    vision.image_std = SIGLIP_IMAGE_STD
    vision.num_register_tokens = 0

    del model
    torch.cuda.empty_cache()
    return vision


Siglip2_ViT_Wrapper = wrap_siglip2
