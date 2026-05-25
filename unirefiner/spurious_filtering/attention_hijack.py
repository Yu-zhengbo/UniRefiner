"""Attention hijacker-hijackee filtering for foreground teacher tokens.

Attention hijacking describes one-way attention information flow: hijacker
tokens send excessive information outward and overwrite other token semantics,
while hijackee tokens mainly receive information and provide little semantic
support outward. This module implements the AH clean-token mask used by
UniRefiner.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .sampling import safe_multinomial


ATTENTION_HIJACK_SAMPLE_RATIO = 0.2


@torch.no_grad()
def filter_attention_hijackees(
    teacher_hook_cache,
    layer_start: int,
    layer_end: int,
    zscore_sigma: float = 0.25,
    foreground_mask: torch.Tensor | None = None,
    pre_rejected_token_mask: torch.Tensor | None = None,
    sampled_token_count: int = 0,
):
    """Filter foreground tokens by teacher AH information-flow support.

    `teacher_hook_cache` stores per-layer normalized-projection inputs with
    keys such as `{layer_id}_q` and `{layer_id}_k`, each shaped `[B, tokens, C]`.
    When `foreground_mask` is provided, the score is computed only over crop
    foreground tokens. `pre_rejected_token_mask` marks tokens already rejected by
    earlier filters; those keys are zeroed and excluded from the threshold
    estimate.

    Returns a boolean clean-token mask with `True=clean`. If
    `sampled_token_count > 0`, also returns sampled high-AH-score indices.
    """

    attention_flow_maps = []
    for layer_id in range(layer_start, layer_end + 1):
        batch_size, _, channels = teacher_hook_cache[f"{layer_id}_q"].shape
        query_tokens = teacher_hook_cache[f"{layer_id}_q"]
        key_tokens = teacher_hook_cache[f"{layer_id}_k"]
        if foreground_mask is not None:
            query_tokens = F.normalize(query_tokens[foreground_mask].view(batch_size, -1, channels), dim=-1)
            key_tokens = F.normalize(key_tokens[foreground_mask].view(batch_size, -1, channels), dim=-1)
        else:
            query_tokens = F.normalize(query_tokens, dim=-1)
            key_tokens = F.normalize(key_tokens, dim=-1)
        if pre_rejected_token_mask is not None:
            key_tokens[pre_rejected_token_mask] = 0
        attention_flow_maps.append(F.softmax(query_tokens @ key_tokens.transpose(1, 2), dim=-1))

    attention_hijackee_score = 1 - torch.stack(attention_flow_maps, dim=0).mean(dim=0).sum(1)
    valid_hijackee_scores = (
        attention_hijackee_score[~pre_rejected_token_mask]
        if pre_rejected_token_mask is not None
        else attention_hijackee_score
    )
    mean, std = valid_hijackee_scores.mean(), valid_hijackee_scores.std()
    threshold = mean + zscore_sigma * std
    ah_clean_mask = attention_hijackee_score <= threshold

    if sampled_token_count > 0:
        token_count = attention_hijackee_score.shape[1]
        if pre_rejected_token_mask is not None:
            attention_hijackee_score[pre_rejected_token_mask] = -10
        sampled_count = int(token_count * ATTENTION_HIJACK_SAMPLE_RATIO)
        threshold_cos = attention_hijackee_score.topk(sampled_count, dim=1, largest=True)[0][:, -1]
        sampled_mask = attention_hijackee_score >= threshold_cos.unsqueeze(1)
        sampled_ids = safe_multinomial(
            sampled_mask.float(),
            sampled_token_count,
            replacement=True,
            fallback_scores=attention_hijackee_score,
        )
        return ah_clean_mask, sampled_ids
    return ah_clean_mask


def analyze_attention_hijacking(
    teacher_hook_cache,
    layer_start,
    layer_end,
    thres_sigma=0.25,
    foreground_mask=None,
    rejected_mask=None,
    sampled_token_count=0,
):
    """Method-facing attention hijacker/hijackee filter."""

    return filter_attention_hijackees(
        teacher_hook_cache,
        layer_start=layer_start,
        layer_end=layer_end,
        zscore_sigma=thres_sigma,
        foreground_mask=foreground_mask,
        pre_rejected_token_mask=rejected_mask,
        sampled_token_count=sampled_token_count,
    )
