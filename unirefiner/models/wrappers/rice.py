"""RICE-ViT wrapper."""

from __future__ import annotations

import copy
from typing import Callable, Optional

import torch
from torch import nn


OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


try:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
except Exception:  # pragma: no cover - transformers-version compatibility
    ALL_ATTENTION_FUNCTIONS = {}


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q = q.float()
    k = k.float()
    cos = cos.unsqueeze(-2).float()
    sin = sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)


def apply_rotary_pos_emb_vision_to_q(q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    orig_dtype = q.dtype
    q = q.float()
    cos = cos.unsqueeze(-2).float()
    sin = sin.unsqueeze(-2).float()
    return ((q * cos) + (rotate_half(q) * sin)).to(orig_dtype)


def apply_rotary_pos_emb_vision_to_k(k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    orig_dtype = k.dtype
    k = k.float()
    cos = cos.unsqueeze(-2).float()
    sin = sin.unsqueeze(-2).float()
    return ((k * cos) + (rotate_half(k) * sin)).to(orig_dtype)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    del kwargs
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous()
    return attn_output, attn_weights


def wrap_rice_vit(model):
    """Wrap RICE-ViT and preserve its RoPE-aware Q/K hook behavior."""

    def attn_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_length = hidden_states.shape[:-1]

        self.q_proj.rope = position_embeddings
        self.k_proj.rope = position_embeddings
        self.q_proj.num_heads = self.num_heads
        self.k_proj.num_heads = self.num_heads
        self.q_proj.head_dim = self.head_dim
        self.k_proj.head_dim = self.head_dim

        query_states = self.q_proj(hidden_states).reshape((batch_size, seq_length, self.num_heads, self.head_dim))
        key_states = self.k_proj(hidden_states).reshape((batch_size, seq_length, self.num_heads, self.head_dim))
        value_states = self.v_proj(hidden_states).reshape((batch_size, seq_length, self.num_heads, self.head_dim))

        cos = position_embeddings[0].unsqueeze(0).float()
        sin = position_embeddings[1].unsqueeze(0).float()
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        query_states = query_states.permute(0, 2, 1, 3).contiguous()
        key_states = key_states.permute(0, 2, 1, 3).contiguous()
        value_states = value_states.permute(0, 2, 1, 3).contiguous()

        attention_interface: Callable = eager_attention_forward
        attn_impl = getattr(getattr(self, "config", None), "_attn_implementation", "eager")
        if attn_impl != "eager" and attn_impl in ALL_ATTENTION_FUNCTIONS:
            attention_interface = ALL_ATTENTION_FUNCTIONS[attn_impl]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.dropout,
            scaling=self.scale,
            is_causal=self.is_causal,
            **kwargs,
        )

        attn_output = attn_output.permute(1, 0, 2, 3).contiguous()
        attn_output = attn_output.view(seq_length, batch_size, -1)
        attn_output = self.out_proj(attn_output)
        return attn_output.permute(1, 0, 2).contiguous(), attn_weights

    def hook_prepare(self, dense_features, get_states: bool = False, get_v: bool = False):
        del get_states

        class MidHook:
            def __init__(self, layer_id: int, kind: str) -> None:
                self.layer_id = layer_id
                self.kind = kind

            def __call__(self, module, inputs, output) -> None:
                if dense_features["record"] is not True:
                    return
                tensor = output[0] if isinstance(output, tuple) else output
                tensor = tensor.detach().clone()
                if self.kind in {"q", "k"}:
                    tensor = tensor.reshape(tensor.shape[0], -1, module.num_heads, module.head_dim)
                    cos = module.rope[0].unsqueeze(0).float()
                    sin = module.rope[1].unsqueeze(0).float()
                    if self.kind == "q":
                        tensor = apply_rotary_pos_emb_vision_to_q(tensor, cos, sin)
                    else:
                        tensor = apply_rotary_pos_emb_vision_to_k(tensor, cos, sin)
                    tensor = tensor.reshape(tensor.shape[0], -1, module.num_heads * module.head_dim)
                dense_features[f"{self.layer_id}_{self.kind}"] = tensor[:, 1:]

        handles = []
        for layer_id, layer in enumerate(self.encoder.layers):
            handles.append(layer.self_attn.q_proj.register_forward_hook(MidHook(layer_id, "q")))
            handles.append(layer.self_attn.k_proj.register_forward_hook(MidHook(layer_id, "k")))
            if get_v:
                handles.append(layer.self_attn.v_proj.register_forward_hook(MidHook(layer_id, "v")))
        return handles

    def encode_dense(self, images: torch.Tensor) -> torch.Tensor:
        outputs = self.forward(images)
        return outputs.last_hidden_state[:, 1:]

    vision = copy.deepcopy(model.vision_model)
    vision.encode_dense = encode_dense.__get__(vision)
    vision.hook_prepare = hook_prepare.__get__(vision)
    vision.patch_size = int(vision.embeddings.patch_size)
    vision.image_mean = OPENAI_CLIP_MEAN
    vision.image_std = OPENAI_CLIP_STD

    for layer in vision.encoder.layers:
        layer.self_attn.forward = attn_forward.__get__(layer.self_attn)

    del model
    torch.cuda.empty_cache()
    return vision


RICE_ViT_Wrapper = wrap_rice_vit
