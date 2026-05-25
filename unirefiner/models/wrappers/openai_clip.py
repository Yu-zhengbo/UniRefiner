"""OpenAI and LAION CLIP ViT wrappers."""

from __future__ import annotations

import copy

import torch

from .attention_hooks import AttentionHookCache, HookHandleGroup, register_projection_hooks


OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def wrap_openai_clip(model):
    """Wrap a HF CLIPModel-like object and expose dense patch tokens.

    The wrapper copies `vision_model`, uses the CLIP embedding interpolation
    path, runs the encoder, drops CLS, then applies the vision post-layernorm to
    dense tokens.
    """

    def encode_dense(self, images: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embeddings(images, interpolate_pos_encoding=True)
        hidden_states = self.pre_layrnorm(hidden_states)

        if self.num_register_tokens > 0:
            hidden_states = torch.cat([hidden_states, self.reg_token.expand(hidden_states.size(0), -1, -1)], dim=1)

        encoder_outputs = self.encoder(inputs_embeds=hidden_states)
        dense_tokens = encoder_outputs.last_hidden_state[:, 1:, :]
        return self.post_layernorm(dense_tokens)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        vision_outputs = self.forward(pixel_values=images)
        pooled_output = vision_outputs[1]
        return self.visual_projection(pooled_output)

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
            skip_prefix_tokens=1,
            get_states=get_states,
        )

    def hook_prepare(self, dense_features, get_v: bool = False, get_states: bool = False):
        capture = ("q", "k", "v") if get_v else ("q", "k")
        return self.prepare_attention_hooks(dense_features, capture=capture, get_states=get_states)

    vision = copy.deepcopy(model.vision_model)
    vision.encode_dense = encode_dense.__get__(vision)
    vision.encode_image = encode_image.__get__(vision)
    vision.prepare_attention_hooks = prepare_attention_hooks.__get__(vision)
    vision.hook_prepare = hook_prepare.__get__(vision)
    vision.patch_size = int(vision.embeddings.patch_size)
    vision.image_mean = OPENAI_CLIP_MEAN
    vision.image_std = OPENAI_CLIP_STD
    vision.visual_projection = model.visual_projection
    vision.num_register_tokens = 0

    del model
    torch.cuda.empty_cache()
    return vision


OpenAI_CLIPViT_Wrapper = wrap_openai_clip
