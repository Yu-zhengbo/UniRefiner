"""EVA-CLIP-8B wrapper."""

from __future__ import annotations

import copy
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .attention_hooks import AttentionHookCache, HookHandleGroup, register_projection_hooks


OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class HeadwiseGateAttention(nn.Module):
    """Optional per-head gate for EVA attention."""

    def __init__(self, num_heads: int, head_dim: int) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.weights = nn.Parameter(torch.zeros(num_heads, head_dim, 1))

    def forward(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        batch_size, token_count, _ = x.size()
        x = x.view(batch_size, token_count, self.num_heads, self.head_dim)
        gate_logits = torch.einsum("bnhd,hdm->bnhm", x, self.weights)
        return y * (2 * torch.sigmoid(gate_logits))


def patch_embeddings_with_interpolation(embeddings):
    """Patch EVA embeddings to interpolate absolute positions at runtime."""

    class InterpolatingEmbeddings(embeddings.__class__):
        def forward(self, pixel_values):
            batch_size = pixel_values.shape[0]
            patch_embeds = self.patch_embedding(pixel_values)
            _, _, grid_h, grid_w = patch_embeds.shape
            patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

            class_embeds = self.class_embedding.expand(batch_size, 1, -1)
            embeddings_out = torch.cat([class_embeds, patch_embeds], dim=1)
            pos_embed = self.position_embedding(self.position_ids)

            pretrained_side = int((pos_embed.shape[1] - 1) ** 0.5)
            if not (grid_h == grid_w == pretrained_side):
                patch_pos = pos_embed[:, 1:, :].reshape(1, pretrained_side, pretrained_side, -1).permute(0, 3, 1, 2)
                patch_pos = F.interpolate(
                    patch_pos,
                    size=(grid_h, grid_w),
                    mode="bicubic",
                    align_corners=False,
                )
                patch_pos = patch_pos.reshape(1, -1, grid_h * grid_w).permute(0, 2, 1)
                pos_embed = torch.cat([pos_embed[:, :1, :], patch_pos], dim=1)

            return embeddings_out + pos_embed

    embeddings.__class__ = InterpolatingEmbeddings
    return embeddings


def _attention_forward_with_gate(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    causal_attention_mask: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """EVA attention forward with the optional per-head gate."""

    batch_size, target_len, embed_dim = hidden_states.size()

    query_states = self.q_proj(hidden_states) * self.scale
    key_states = self._shape(self.k_proj(hidden_states), -1, batch_size)
    value_states = self._shape(self.v_proj(hidden_states), -1, batch_size)

    proj_shape = (batch_size * self.num_heads, -1, self.head_dim)
    query_states = self._shape(query_states, target_len, batch_size).view(*proj_shape)
    key_states = key_states.view(*proj_shape)
    value_states = value_states.view(*proj_shape)

    src_len = key_states.size(1)
    attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

    if attn_weights.size() != (batch_size * self.num_heads, target_len, src_len):
        raise ValueError(
            f"Attention weights should be {(batch_size * self.num_heads, target_len, src_len)}, "
            f"got {tuple(attn_weights.size())}."
        )

    if causal_attention_mask is not None:
        if causal_attention_mask.size() != (batch_size, 1, target_len, src_len):
            raise ValueError(
                f"Causal attention mask should be {(batch_size, 1, target_len, src_len)}, "
                f"got {tuple(causal_attention_mask.size())}."
            )
        attn_weights = attn_weights.view(batch_size, self.num_heads, target_len, src_len) + causal_attention_mask
        attn_weights = attn_weights.view(batch_size * self.num_heads, target_len, src_len)

    if attention_mask is not None:
        if attention_mask.size() != (batch_size, 1, target_len, src_len):
            raise ValueError(
                f"Attention mask should be {(batch_size, 1, target_len, src_len)}, "
                f"got {tuple(attention_mask.size())}."
            )
        attn_weights = attn_weights.view(batch_size, self.num_heads, target_len, src_len) + attention_mask
        attn_weights = attn_weights.view(batch_size * self.num_heads, target_len, src_len)

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    if output_attentions:
        attn_weights_reshaped = attn_weights.view(batch_size, self.num_heads, target_len, src_len)
        attn_weights = attn_weights_reshaped.view(batch_size * self.num_heads, target_len, src_len)
    else:
        attn_weights_reshaped = None

    attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_output = torch.bmm(attn_probs, value_states)

    if attn_output.size() != (batch_size * self.num_heads, target_len, self.head_dim):
        raise ValueError(
            f"Attention output should be {(batch_size * self.num_heads, target_len, self.head_dim)}, "
            f"got {tuple(attn_output.size())}."
        )

    attn_output = attn_output.view(batch_size, self.num_heads, target_len, self.head_dim).transpose(1, 2)
    if hasattr(self, "attn_gate"):
        attn_output = self.attn_gate(attn_output, hidden_states)

    attn_output = attn_output.reshape(batch_size, target_len, embed_dim)
    attn_output = self.out_proj(attn_output)
    return attn_output, attn_weights_reshaped


def wrap_evaclip8b(model, *, gate_attention: bool = False):
    """Wrap a HF EVA-CLIP model and expose dense patch tokens."""

    def encode_dense(self, images: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embeddings(images)

        if self.num_register_tokens > 0:
            hidden_states = torch.cat([hidden_states, self.reg_token.expand(hidden_states.size(0), -1, -1)], dim=1)

        encoder_outputs = self.encoder(inputs_embeds=hidden_states)
        return encoder_outputs[0][:, 1:, :]

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
    vision.embeddings = patch_embeddings_with_interpolation(vision.embeddings)
    vision.encode_dense = encode_dense.__get__(vision)
    vision.encode_image = encode_image.__get__(vision)
    vision.prepare_attention_hooks = prepare_attention_hooks.__get__(vision)
    vision.hook_prepare = hook_prepare.__get__(vision)
    vision.patch_size = int(vision.embeddings.patch_size)
    vision.image_mean = OPENAI_CLIP_MEAN
    vision.image_std = OPENAI_CLIP_STD
    vision.visual_projection = model.visual_projection
    vision.num_register_tokens = 0

    if gate_attention:
        for layer in getattr(vision.encoder, "layers", []):
            self_attn = getattr(layer, "self_attn", None)
            if self_attn is None or hasattr(self_attn, "attn_gate"):
                continue
            self_attn.attn_gate = HeadwiseGateAttention(self_attn.num_heads, self_attn.head_dim)
            self_attn.forward = _attention_forward_with_gate.__get__(self_attn)

    del model
    torch.cuda.empty_cache()
    return vision


EvaCLIPViT_Wrapper = wrap_evaclip8b
