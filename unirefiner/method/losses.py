"""UniRefiner loss terms."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def shift_image_without_wrap(
    images: torch.Tensor,
    *,
    shift_tokens_y: int,
    shift_tokens_x: int,
    patch_size: int,
) -> torch.Tensor:
    """Translate images by whole patch tokens using pad+crop, without wraparound."""

    shift_pixels_y = int(shift_tokens_y) * int(patch_size)
    shift_pixels_x = int(shift_tokens_x) * int(patch_size)
    if shift_pixels_y < 0 or shift_pixels_x < 0:
        raise ValueError("Window-phase shifts must be non-negative.")
    if shift_pixels_y == 0 and shift_pixels_x == 0:
        return images

    height, width = images.shape[-2:]
    if shift_pixels_y >= height or shift_pixels_x >= width:
        raise ValueError(
            f"Shift {(shift_tokens_y, shift_tokens_x)} is larger than the image grid."
        )

    cropped = images[:, :, : height - shift_pixels_y, : width - shift_pixels_x]
    return F.pad(cropped, (shift_pixels_x, 0, shift_pixels_y, 0), value=0.0)


def build_window_phase_shifts(window_size: int) -> tuple[tuple[int, int], ...]:
    quarter = max(int(window_size) // 4, 1)
    half = max(int(window_size) // 2, 1)
    return (
        (quarter, 0),
        (0, quarter),
        (half, 0),
        (0, half),
        (half, half),
    )


def compute_window_phase_artifact_loss(
    model,
    images: torch.Tensor,
    *,
    patch_size: int,
    window_size: int,
    encode_dense=None,
) -> torch.Tensor:
    """Penalize dense-token sensitivity to window partition phase.

    The shifted views are constructed with padding and cropping instead of
    roll-style wraparound. Only the valid overlapping token region contributes
    to the cosine-distance loss.
    """

    if encode_dense is None:
        encode_dense = model.encode_dense

    base_tokens = encode_dense(images)
    batch_size, token_count, feature_dim = base_tokens.shape
    grid_h = images.shape[-2] // int(patch_size)
    grid_w = images.shape[-1] // int(patch_size)
    if token_count != grid_h * grid_w:
        raise ValueError(
            "Window-phase artifact loss requires dense patch tokens with "
            f"N=H*W, got N={token_count}, H={grid_h}, W={grid_w}."
        )

    base_grid = base_tokens.reshape(batch_size, grid_h, grid_w, feature_dim)
    losses = []
    for shift_y, shift_x in build_window_phase_shifts(window_size):
        if shift_y >= grid_h or shift_x >= grid_w:
            continue

        shifted_images = shift_image_without_wrap(
            images,
            shift_tokens_y=shift_y,
            shift_tokens_x=shift_x,
            patch_size=patch_size,
        )
        shifted_tokens = encode_dense(shifted_images)
        shifted_grid = shifted_tokens.reshape(batch_size, grid_h, grid_w, feature_dim)

        valid_h = grid_h - shift_y
        valid_w = grid_w - shift_x
        base_overlap = base_grid[:, :valid_h, :valid_w, :]
        shifted_aligned = shifted_grid[:, shift_y : shift_y + valid_h, shift_x : shift_x + valid_w, :]
        base_overlap = F.normalize(base_overlap, dim=-1)
        shifted_aligned = F.normalize(shifted_aligned, dim=-1)
        cosine_similarity = (base_overlap * shifted_aligned).sum(dim=-1).clamp(min=-1.0, max=1.0)
        cosine_distance = 1.0 - cosine_similarity
        losses.append(cosine_distance.mean())

    if not losses:
        return base_tokens.new_zeros(())
    return torch.stack(losses).mean()


def valid_clean_tokens_or_all(clean_token_mask: torch.Tensor) -> torch.Tensor:
    """Use all crop tokens when filtering rejects every token in the batch."""

    if clean_token_mask.any():
        return clean_token_mask
    return torch.ones_like(clean_token_mask, dtype=torch.bool)


def estimate_teacher_dot_sparsity_threshold(
    teacher_crop_tokens: torch.Tensor,
    valid_clean_mask: torch.Tensor,
) -> torch.Tensor:
    """Estimate teacher-space sparsity from normalized dot products."""

    valid_teacher_tokens = teacher_crop_tokens[valid_clean_mask]
    return (valid_teacher_tokens * valid_teacher_tokens.flip(0)).sum(dim=-1).mean()


def detach_low_similarity_gradient(
    similarity: torch.Tensor,
    threshold: torch.Tensor | float,
) -> torch.Tensor:
    """Stop gradients for already sparse similarities without changing values."""

    with torch.no_grad():
        sparse_mask = similarity < threshold
    return torch.where(sparse_mask, similarity.detach(), similarity)


def apply_teacher_sparsity_balance(
    similarity: torch.Tensor,
    teacher_sparsity_threshold: torch.Tensor,
) -> torch.Tensor:
    """Relax register-spurious uniformity using teacher feature-space sparsity.

    The threshold is estimated from clean teacher crop tokens, so it adapts to
    the teacher backbone and current batch. Similarities below this teacher-space
    sparsity estimate are anchored at threshold + 0.1 with zero gradient.
    """

    with torch.no_grad():
        sparse_mask = similarity < teacher_sparsity_threshold
    replacement = torch.as_tensor(
        teacher_sparsity_threshold + 0.1,
        dtype=similarity.dtype,
        device=similarity.device,
    ).detach()
    return torch.where(sparse_mask, replacement, similarity)


def logsumexp_uniformity(
    similarity: torch.Tensor,
    temp: float,
    *,
    reduce_mean: bool = False,
) -> torch.Tensor:
    uniformity = torch.logsumexp(similarity / temp, dim=-1)
    if reduce_mean:
        return uniformity.mean()
    return uniformity


def rematch_sampled_spurious_tokens_to_full_image(
    sampled_spurious_teacher_tokens: torch.Tensor,
    teacher_full_image_tokens: torch.Tensor,
    feature_dim: int,
) -> torch.Tensor:
    with torch.no_grad():
        closest_full_image_indices = (
            sampled_spurious_teacher_tokens @ teacher_full_image_tokens.transpose(1, 2)
        ).argmax(dim=-1)
        return torch.gather(
            teacher_full_image_tokens,
            1,
            closest_full_image_indices.unsqueeze(-1).expand(-1, -1, feature_dim),
        )


def compute_register_absorption_loss(
    student_register_tokens: torch.Tensor,
    student_crop_tokens: torch.Tensor,
    matched_spurious_teacher_tokens: torch.Tensor,
    teacher_sparsity_threshold: torch.Tensor,
    temp: float,
) -> torch.Tensor:
    """Default register absorption loss.

    Register tokens align to spurious teacher tokens, retain a weak image-token
    auxiliary alignment, and use sparsity-balanced register-spurious uniformity.
    """

    feature_dim = student_register_tokens.shape[-1]
    register_to_spurious_similarity = (
        student_register_tokens @ matched_spurious_teacher_tokens.detach().transpose(1, 2)
    )
    register_spurious_alignment, register_spurious_indices = register_to_spurious_similarity.max(dim=-1)

    register_to_crop_similarity = (
        student_register_tokens @ student_crop_tokens.detach().transpose(1, 2)
    )
    register_crop_alignment, _ = register_to_crop_similarity.max(dim=-1)
    top_register_crop_match_count = max((register_crop_alignment.shape[1] + 9) // 10, 1)
    strongest_register_crop_alignment = register_crop_alignment.topk(
        k=top_register_crop_match_count,
        dim=-1,
    ).values
    valid_register_crop_alignment = strongest_register_crop_alignment > teacher_sparsity_threshold
    if valid_register_crop_alignment.any():
        register_crop_alignment_term = strongest_register_crop_alignment[
            valid_register_crop_alignment
        ].mean()
    else:
        register_crop_alignment_term = strongest_register_crop_alignment.new_zeros(())

    register_matched_spurious_targets = matched_spurious_teacher_tokens.detach().gather(
        1,
        register_spurious_indices.unsqueeze(-1).expand(-1, -1, feature_dim),
    )
    register_to_matched_spurious_similarity = (
        student_register_tokens @ register_matched_spurious_targets.transpose(1, 2)
    )
    balanced_spurious_similarity = apply_teacher_sparsity_balance(
        register_to_matched_spurious_similarity,
        teacher_sparsity_threshold,
    )
    register_spurious_uniformity = logsumexp_uniformity(
        balanced_spurious_similarity,
        temp,
        reduce_mean=True,
    )
    return (
        -register_spurious_alignment.mean() / temp
        - 0.2 * register_crop_alignment_term / temp
        + register_spurious_uniformity
    )


def compute_refinement_loss(
    student_crop_tokens: torch.Tensor,
    teacher_target_tokens: torch.Tensor,
    clean_token_mask: torch.Tensor,
    student_register_tokens: torch.Tensor,
    sampled_spurious_teacher_tokens: torch.Tensor,
    repeated_student_full_image_tokens: torch.Tensor,
    teacher_full_image_tokens: torch.Tensor,
    temp: float = 0.2,
    args=None,
    *,
    teacher_negative_pool_tokens: torch.Tensor | None = None,
    teacher_spurious_rematch_pool_tokens: torch.Tensor | None = None,
):
    """Compute the UniRefiner refinement objective.

    Clean crop tokens receive NCE supervision from teacher targets. Teacher and
    register uniformity terms stop gradients for already sparse similarities,
    while register absorption moves register tokens toward sampled spurious
    teacher tokens.
    """

    uniformity_strength = float(getattr(args, "uniformity_strength", 1.0)) if args is not None else 1.0
    with torch.no_grad():
        valid_clean_mask = valid_clean_tokens_or_all(clean_token_mask)
        feature_dim = student_crop_tokens.shape[-1]
        teacher_sparsity_threshold = estimate_teacher_dot_sparsity_threshold(
            teacher_target_tokens,
            valid_clean_mask,
        )

    if teacher_negative_pool_tokens is None:
        teacher_negative_pool_tokens = teacher_full_image_tokens
    if teacher_spurious_rematch_pool_tokens is None:
        teacher_spurious_rematch_pool_tokens = teacher_full_image_tokens

    positive_alignment = (student_crop_tokens * teacher_target_tokens).sum(dim=-1)
    student_teacher_negative_affinity = student_crop_tokens @ teacher_negative_pool_tokens.transpose(1, 2)
    student_register_affinity = repeated_student_full_image_tokens @ student_register_tokens.transpose(1, 2)

    balanced_teacher_affinity = detach_low_similarity_gradient(
        student_teacher_negative_affinity,
        teacher_sparsity_threshold,
    )
    balanced_register_affinity = detach_low_similarity_gradient(
        student_register_affinity,
        0.55,
    )
    teacher_uniformity = logsumexp_uniformity(balanced_teacher_affinity, temp)
    register_uniformity = logsumexp_uniformity(balanced_register_affinity, temp, reduce_mean=True)
    refinement_loss = -(
        positive_alignment / temp - uniformity_strength * teacher_uniformity
    )[valid_clean_mask].mean() + register_uniformity

    matched_spurious_teacher_tokens = rematch_sampled_spurious_tokens_to_full_image(
        sampled_spurious_teacher_tokens,
        teacher_spurious_rematch_pool_tokens,
        feature_dim,
    )
    register_absorption_loss = compute_register_absorption_loss(
        student_register_tokens,
        student_crop_tokens,
        matched_spurious_teacher_tokens,
        teacher_sparsity_threshold,
        temp,
    )
    return (
        refinement_loss,
        positive_alignment.detach()[valid_clean_mask].mean(),
        register_absorption_loss,
        register_uniformity,
    )


def compute_spatial_consistency_loss(
    student_crop_tokens: torch.Tensor,
    register_expanded_crop_tokens: torch.Tensor,
    temp: float = 0.2,
) -> torch.Tensor:
    """Spatial Correlation Distillation for register-expanded crops.

    Enlarging the register region at test time naturally creates more register
    tokens and usually yields a cleaner dense feature map. Following SCD from
    arXiv:2504.02328, UniRefiner uses the register-expanded feature map only as
    a soft relational target: it distills token-token spatial correlations, not
    the feature vectors themselves.
    """

    def minmax_normalize(similarity: torch.Tensor) -> torch.Tensor:
        # Rescale per sample before softmax so relation distributions share a stable range.
        sim_min = similarity.min(dim=-1, keepdim=True).values
        sim_max = similarity.max(dim=-1, keepdim=True).values
        return (similarity - sim_min) / (sim_max - sim_min + 1e-10)

    # Match self-relation distributions between the normal crop and the cleaner
    # register-expanded crop, while leaving absolute feature directions free.
    student_relation_logits = student_crop_tokens @ student_crop_tokens.detach().transpose(1, 2)
    with torch.no_grad():
        target_relation_logits = register_expanded_crop_tokens @ register_expanded_crop_tokens.transpose(1, 2)
        target_relation_logits = minmax_normalize(target_relation_logits)
        target_relation_distribution = F.softmax(target_relation_logits / temp, dim=-1)

    student_relation_logits = minmax_normalize(student_relation_logits)
    student_relation_distribution = F.softmax(student_relation_logits / temp, dim=-1)

    return (
        -target_relation_distribution * torch.log(student_relation_distribution + 1e-10)
    ).sum(dim=-1).mean()
