"""Sampling utilities for spurious-token candidate selection."""

from __future__ import annotations

import torch


def safe_multinomial(
    weights: torch.Tensor,
    num_samples: int,
    replacement: bool = True,
    fallback_scores: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample token indices from row-wise weights with a stable empty-row fallback.

    Some filters construct binary masks for high-spurious candidate regions. A
    row can be empty when no token passes that candidate criterion. In that case,
    `fallback_scores` selects the highest-scoring token; without fallback scores,
    the row falls back to uniform sampling.
    """

    weights = weights.float()
    row_sums = weights.sum(dim=1, keepdim=True)
    valid_rows = row_sums > 0
    if valid_rows.all():
        return torch.multinomial(weights, num_samples, replacement=replacement)

    safe_weights = weights.clone()
    invalid_rows = ~valid_rows.squeeze(1)
    if fallback_scores is not None:
        fallback = torch.zeros_like(safe_weights[invalid_rows])
        top_idx = fallback_scores[invalid_rows].argmax(dim=1, keepdim=True)
        fallback.scatter_(1, top_idx, 1.0)
        safe_weights[invalid_rows] = fallback
    else:
        safe_weights[invalid_rows] = 1.0
    return torch.multinomial(safe_weights, num_samples, replacement=replacement)


_safe_multinomial = safe_multinomial
