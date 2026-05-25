"""Feature-channel masking for cosine-based UniRefiner computations.

Some vision backbones expose high-scale, low-variance lazy channels. UniRefiner
relies on cosine similarity, so those channels can dominate normalized
features and break refinement. Channel masking is therefore a backbone feature
safeguard.
"""

from __future__ import annotations

from collections.abc import Iterable
import json
import logging
import os
import random

import torch
import torch.nn.functional as F


EPS = 1e-6


def resolve_channel_mask_channels(args=None, channels: Iterable[int] | None = None) -> list[int]:
    """Resolve hidden runtime channel-mask indices.

    Public configs should not expose low-level channel-mask knobs. Runtime code
    may still carry precomputed channel indices for backbones that need this
    cosine-similarity safeguard.
    """

    if channels is None:
        channels = getattr(args, "channel_mask_channels", []) if args is not None else []
    return [int(channel) for channel in (channels or [])]


def apply_channel_mask(
    features: torch.Tensor,
    *,
    args=None,
    channels: Iterable[int] | None = None,
) -> torch.Tensor:
    """Suppress known lazy channels before normalization or cosine similarity."""

    masked_channels = resolve_channel_mask_channels(args=args, channels=channels)
    if not masked_channels:
        return features

    feature_dim = int(features.shape[-1])
    invalid = [channel for channel in masked_channels if channel < 0 or channel >= feature_dim]
    if invalid:
        raise ValueError(f"Channel-mask indices {invalid} are outside feature dim {feature_dim}.")

    keep_mask = features.new_ones(feature_dim)
    keep_mask[masked_channels] = 0
    return features * keep_mask


def _chunked(items: list[int], chunk_size: int) -> Iterable[list[int]]:
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def _sample_indices(num_items: int, num_samples: int, seed: int) -> list[int]:
    if num_items <= 0:
        return []
    if num_samples >= num_items:
        return list(range(num_items))
    rng = random.Random(seed)
    return rng.sample(range(num_items), num_samples)


def _strip_register_tokens(features: torch.Tensor, model) -> torch.Tensor:
    num_register_tokens = int(getattr(model, "num_register_tokens", 0) or 0)
    if num_register_tokens <= 0 or features.shape[1] <= num_register_tokens:
        return features
    return features[:, :-num_register_tokens, :]


def _sample_tokens(features: torch.Tensor, tokens_per_image: int, rng: random.Random) -> torch.Tensor:
    if tokens_per_image <= 0 or features.shape[1] <= tokens_per_image:
        return features
    sampled = []
    token_count = features.shape[1]
    for image_index in range(features.shape[0]):
        token_indices = torch.tensor(
            rng.sample(range(token_count), tokens_per_image),
            device=features.device,
            dtype=torch.long,
        )
        sampled.append(features[image_index].index_select(0, token_indices))
    return torch.stack(sampled, dim=0)


def _compute_flip_neg_mean(features: torch.Tensor) -> torch.Tensor:
    if features.shape[0] < 2:
        return features.new_zeros(())
    paired = features.flip(0).flip(1)
    return (features * paired).sum(dim=-1).mean()


def _compute_channel_stability(features: torch.Tensor) -> torch.Tensor:
    channel_mean = features.mean(dim=1)
    channel_var = features.var(dim=1, unbiased=False)
    return (channel_mean.pow(2) / (channel_var + EPS)).mean(dim=0)


def _compute_flip_channel_bias(features: torch.Tensor) -> torch.Tensor:
    if features.shape[0] < 2:
        return features.new_zeros(features.shape[-1])
    paired = features.flip(0).flip(1)
    return (features * paired).mean(dim=(0, 1))


def _top_channel_summary(
    scores: torch.Tensor,
    flip_bias: torch.Tensor,
    stability: torch.Tensor,
    excluded_channels: list[int],
    topk: int = 10,
) -> list[dict]:
    if scores.numel() == 0:
        return []
    visible_scores = scores.clone()
    if excluded_channels:
        visible_scores[excluded_channels] = float("-inf")
    top_values, top_indices = torch.topk(visible_scores, k=min(max(int(topk), 1), visible_scores.numel()))
    summary = []
    for value, index in zip(top_values.tolist(), top_indices.tolist()):
        if not torch.isfinite(torch.tensor(value)):
            continue
        summary.append(
            {
                "channel": int(index),
                "selection_score": float(value),
                "flip_bias": float(flip_bias[index].item()),
                "stability": float(stability[index].item()),
            }
        )
    return summary


def _evaluate_candidates(current_features: torch.Tensor, candidate_channels: list[int]) -> torch.Tensor:
    if current_features.shape[0] < 2 or not candidate_channels:
        return current_features.new_zeros((0,))

    paired_features = current_features.flip(0).flip(1)
    base_dot = (current_features * paired_features).sum(dim=-1)
    candidate_tensor = torch.tensor(candidate_channels, device=current_features.device, dtype=torch.long)
    channel_values = current_features.index_select(dim=-1, index=candidate_tensor).permute(2, 0, 1).contiguous()
    paired_channel_values = paired_features.index_select(dim=-1, index=candidate_tensor).permute(2, 0, 1).contiguous()
    weights = torch.rsqrt((1.0 - channel_values.pow(2)).clamp_min(EPS))
    paired_weights = torch.rsqrt((1.0 - paired_channel_values.pow(2)).clamp_min(EPS))
    masked_dot = (base_dot.unsqueeze(0) - channel_values * paired_channel_values) * weights * paired_weights
    return masked_dot.mean(dim=(1, 2))


def save_channel_mask_artifacts(selection: dict, args) -> None:
    output_path = getattr(args, "channel_mask_report_path", None)
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(selection, handle, indent=2)

    tensor_path = getattr(args, "channel_mask_tensor_path", None)
    if tensor_path:
        os.makedirs(os.path.dirname(tensor_path), exist_ok=True)
        torch.save(torch.tensor(selection.get("masked_channels", []), dtype=torch.long), tensor_path)


@torch.no_grad()
def collect_channel_mask_feature_bank(model, dataset, args, cast_dtype):
    sample_count = min(max(int(getattr(args, "channel_mask_samples", 64)), 0), len(dataset))
    if sample_count <= 0:
        return None

    batch_size = max(int(getattr(args, "channel_mask_batch_size", 8)), 1)
    tokens_per_image = max(int(getattr(args, "channel_mask_tokens_per_image", 128)), 1)
    image_indices = _sample_indices(len(dataset), sample_count, getattr(args, "seed", 0) + 17)
    token_rng = random.Random(getattr(args, "seed", 0) + 29)

    device = next(model.parameters()).device
    input_dtype = cast_dtype if cast_dtype is not None else torch.float32
    feature_batches = []

    was_training = model.training
    model.eval()
    try:
        for batch_indices in _chunked(image_indices, batch_size):
            batch_images = [dataset[index][0] for index in batch_indices]
            images = torch.stack(batch_images, dim=0).to(device=device, dtype=input_dtype, non_blocking=True)
            features = model.encode_dense(images)
            features = _strip_register_tokens(features, model).float()
            feature_batches.append(_sample_tokens(features, tokens_per_image=tokens_per_image, rng=token_rng).cpu())
    finally:
        model.train(was_training)

    feature_bank = torch.cat(feature_batches, dim=0)
    logging.info(
        "Collected channel-mask feature bank: %d images x %d tokens x %d channels.",
        feature_bank.shape[0],
        feature_bank.shape[1],
        feature_bank.shape[2],
    )
    return {
        "feature_bank": feature_bank,
        "image_indices": image_indices,
        "tokens_per_image": int(feature_bank.shape[1]),
        "channel_dim": int(feature_bank.shape[2]),
    }


@torch.no_grad()
def run_greedy_channel_mask(feature_bank_info, args) -> dict:
    feature_bank = feature_bank_info["feature_bank"].float()
    image_indices = feature_bank_info["image_indices"]
    tokens_per_image = feature_bank_info["tokens_per_image"]

    threshold = float(getattr(args, "channel_mask_threshold", 0.2))
    max_channels = max(int(getattr(args, "channel_mask_max_channels", 5)), 0)
    candidate_pool = max(int(getattr(args, "channel_mask_candidate_pool", 16)), 1)
    min_delta = float(getattr(args, "channel_mask_min_delta", 0.005))

    target_device = feature_bank.device
    if torch.cuda.is_available():
        try:
            feature_bank = feature_bank.to(device=torch.device(args.device), non_blocking=True)
            target_device = feature_bank.device
        except RuntimeError as error:
            logging.warning("Falling back to CPU for channel-mask selection: %s", error)
            target_device = feature_bank.device

    current_raw = feature_bank
    current_norm = F.normalize(current_raw, dim=-1)
    current_neg_mean = float(_compute_flip_neg_mean(current_norm).item())
    masked_channels: list[int] = []
    rounds = []

    for _ in range(max_channels):
        flip_bias = _compute_flip_channel_bias(current_norm)
        stability = _compute_channel_stability(current_raw)
        channel_score = flip_bias * stability
        for channel in masked_channels:
            channel_score[channel] = float("-inf")

        candidate_indices = torch.topk(channel_score, k=min(candidate_pool, channel_score.numel())).indices.tolist()
        candidate_neg_means = _evaluate_candidates(current_norm, candidate_indices)
        if candidate_neg_means.numel() == 0:
            stop_reason = "no_candidates"
            break

        best_index = int(torch.argmin(candidate_neg_means).item())
        best_channel = int(candidate_indices[best_index])
        best_neg_mean = float(candidate_neg_means[best_index].item())
        improvement = current_neg_mean - best_neg_mean

        rounds.append(
            {
                "masked_channels_so_far": list(masked_channels),
                "candidate_summary": _top_channel_summary(channel_score, flip_bias, stability, masked_channels),
                "best_channel": best_channel,
                "neg_mean_before": current_neg_mean,
                "neg_mean_after": best_neg_mean,
                "improvement": improvement,
            }
        )

        if best_neg_mean <= threshold:
            masked_channels.append(best_channel)
            current_raw[..., best_channel] = 0
            current_norm = F.normalize(current_raw, dim=-1)
            current_neg_mean = best_neg_mean
            stop_reason = "threshold"
            break

        if improvement < min_delta:
            stop_reason = "min_delta"
            break

        masked_channels.append(best_channel)
        current_raw[..., best_channel] = 0
        current_norm = F.normalize(current_raw, dim=-1)
        current_neg_mean = best_neg_mean
    else:
        stop_reason = "max_channels"

    return {
        "enabled": True,
        "sample_count": int(feature_bank.shape[0]),
        "sampled_image_indices": [int(index) for index in image_indices],
        "tokens_per_image": int(tokens_per_image),
        "final_neg_mean": float(current_neg_mean),
        "masked_channels": [int(channel) for channel in masked_channels],
        "stop_reason": stop_reason,
        "rounds": rounds,
        "device": str(target_device),
    }
