"""Adaptive-register filtering for teacher tokens.

The current student registers act as an adaptive detector for tokens already
absorbed by the register region. Teacher tokens with high similarity to these
detector registers are treated as spurious and removed from clean-token
refinement supervision.
"""

from __future__ import annotations

import torch

from .sampling import safe_multinomial


ADAPTIVE_REGISTER_SAMPLE_RATIO = 0.4
ADAPTIVE_REGISTER_SAMPLE_MARGIN = 0.05


@torch.no_grad()
def filter_by_adaptive_register(
    adaptive_detector_tokens: torch.Tensor,
    teacher_tokens: torch.Tensor,
    cosine_threshold: float = 0.7,
    sampled_token_count: int = 0,
):
    """Reject teacher tokens that are too similar to adaptive detector registers.

    Returns a boolean clean-token mask with `True=clean`. If
    `sampled_token_count > 0`, also returns high register-similarity teacher
    tokens as spurious candidates for register absorption. The sampling set uses
    a small margin below the high-similarity top-ratio threshold to keep the
    absorption targets sufficiently broad.
    """

    if adaptive_detector_tokens.shape[0] == 1:
        adaptive_detector_tokens = adaptive_detector_tokens.expand(teacher_tokens.shape[0], -1, -1)
    _, token_count, _ = teacher_tokens.shape
    max_register_similarity = (teacher_tokens @ adaptive_detector_tokens.permute(0, 2, 1)).max(dim=2)[0]
    clean_token_mask = max_register_similarity < cosine_threshold

    if sampled_token_count > 0:
        sampled_count = int(token_count * ADAPTIVE_REGISTER_SAMPLE_RATIO)
        threshold_cos = max_register_similarity.topk(sampled_count, dim=1, largest=True)[0][:, -1]
        threshold_cos = threshold_cos - ADAPTIVE_REGISTER_SAMPLE_MARGIN
        sampled_mask = max_register_similarity >= threshold_cos.unsqueeze(1)
        sampled_ids = safe_multinomial(
            sampled_mask.float(),
            sampled_token_count,
            replacement=True,
            fallback_scores=max_register_similarity,
        )
        sampled_tokens = teacher_tokens.gather(
            1,
            sampled_ids.unsqueeze(-1).expand(-1, -1, teacher_tokens.shape[-1]),
        )
        return clean_token_mask, sampled_tokens
    return clean_token_mask


def analyze_adaptive_spurious_detection(
    register_tokens,
    teacher_tokens,
    thres_cos=0.7,
    sampled_token_count=0,
):
    """Method-facing adaptive-register spurious-token filter."""

    return filter_by_adaptive_register(
        register_tokens,
        teacher_tokens,
        cosine_threshold=thres_cos,
        sampled_token_count=sampled_token_count,
    )
