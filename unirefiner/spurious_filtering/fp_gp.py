"""FP-GP filtering by similarity to irrelevant image contents.

Both fixed-pattern and global-proxy tokens exhibit high similarity with
irrelevant image contents, so UniRefiner jointly identifies FP-GP tokens with
inter-image and intra-image similarity checks.
"""

from __future__ import annotations

import torch

from .sampling import safe_multinomial


FP_GP_SAMPLE_RATIO = 0.4


@torch.no_grad()
def filter_by_fp_gp_similarity(
    foreground_teacher_tokens: torch.Tensor,
    comparison_tokens: torch.Tensor,
    zscore_sigma: float | None = None,
    cosine_threshold: float | None = None,
    sampled_token_count: int = 0,
):
    """Apply one FP-GP similarity criterion to foreground teacher tokens.

    Returns a boolean clean-token mask with `True=clean`. If
    `sampled_token_count > 0`, also returns foreground tokens sampled from the
    high-similarity region; those sampled tokens are absorption targets, not
    clean supervision targets.

    `zscore_sigma` applies a per-sample `mean + sigma * std` threshold to each
    foreground token's maximum similarity to the comparison tokens.
    `cosine_threshold` uses a fixed threshold on the same score. The caller
    decides whether the comparison is inter-image or intra-image.
    """

    if zscore_sigma is None and cosine_threshold is None:
        raise ValueError("At least one of zscore_sigma or cosine_threshold must be provided.")
    if zscore_sigma is not None and cosine_threshold is not None:
        raise ValueError("Only one of zscore_sigma or cosine_threshold should be provided.")

    batch_size, token_count, _ = foreground_teacher_tokens.shape
    comparison_tokens = comparison_tokens.expand(batch_size, -1, -1)
    fp_gp_similarity = foreground_teacher_tokens @ comparison_tokens.permute(0, 2, 1)
    max_fp_gp_similarity = fp_gp_similarity.max(dim=2)[0]

    if zscore_sigma is not None:
        similarity_mean = max_fp_gp_similarity.mean(dim=-1, keepdim=True)
        similarity_std = max_fp_gp_similarity.std(dim=-1, keepdim=True)
        clean_token_mask = max_fp_gp_similarity < (similarity_mean + zscore_sigma * similarity_std)
    else:
        clean_token_mask = max_fp_gp_similarity < cosine_threshold

    if sampled_token_count > 0:
        sampled_count = int(token_count * FP_GP_SAMPLE_RATIO)
        threshold_cos = max_fp_gp_similarity.topk(sampled_count, dim=1, largest=True)[0][:, -1]
        sampled_mask = max_fp_gp_similarity >= threshold_cos.unsqueeze(1)
        sampled_ids = safe_multinomial(
            sampled_mask.float(),
            sampled_token_count,
            replacement=True,
            fallback_scores=max_fp_gp_similarity,
        )
        sampled_tokens = foreground_teacher_tokens.gather(
            1,
            sampled_ids.unsqueeze(-1).expand(-1, -1, foreground_teacher_tokens.shape[-1]),
        )
        return clean_token_mask, sampled_tokens
    return clean_token_mask


def analyze_fp_gp_similarity(
    token_features,
    comparison_features,
    thres_sigma=None,
    thres_cos=None,
    sampled_token_count=0,
):
    """Method-facing FP-GP similarity filter.

    The method calls this for both the inter-image and intra-image FP-GP
    criteria. 
    """

    return filter_by_fp_gp_similarity(
        token_features,
        comparison_features,
        zscore_sigma=thres_sigma,
        cosine_threshold=thres_cos,
        sampled_token_count=sampled_token_count,
    )
