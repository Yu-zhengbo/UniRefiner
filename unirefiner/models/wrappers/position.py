"""Position and RoPE helpers for model wrappers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .base import GridHW, ResolutionError


@dataclass(frozen=True, slots=True)
class RopeGrid:
    """Spatial metadata used by RoPE-based vision backbones."""

    grid_hw: GridHW
    patch_size: int
    merge_size: int = 1

    @property
    def token_count(self) -> int:
        return int(self.grid_hw[0]) * int(self.grid_hw[1])


def image_hw(images: torch.Tensor) -> GridHW:
    if images.ndim != 4:
        raise ResolutionError(f"Expected images with shape [B, C, H, W], got {tuple(images.shape)}.")
    return (int(images.shape[-2]), int(images.shape[-1]))


def infer_patch_grid(
    images_or_hw: torch.Tensor | tuple[int, int],
    patch_size: int,
    *,
    merge_size: int = 1,
) -> GridHW:
    """Infer the final visual-token grid for a patch or merged-patch model."""

    height, width = image_hw(images_or_hw) if isinstance(images_or_hw, torch.Tensor) else images_or_hw
    stride = int(patch_size) * int(merge_size)
    if height % stride != 0 or width % stride != 0:
        raise ResolutionError(
            f"Image size {(height, width)} must be divisible by patch_size * merge_size = {stride}."
        )
    return (height // stride, width // stride)


def infer_pretrained_grid(
    patch_pos_count: int,
    *,
    expected_grid: GridHW | None = None,
) -> GridHW:
    if expected_grid is not None:
        if expected_grid[0] * expected_grid[1] != patch_pos_count:
            raise ResolutionError(
                f"Expected grid {expected_grid} does not match {patch_pos_count} patch positions."
            )
        return expected_grid
    side = int(patch_pos_count**0.5)
    if side * side != patch_pos_count:
        raise ResolutionError(
            f"Cannot infer a square pretrained grid from {patch_pos_count} patch positions."
        )
    return (side, side)


def interpolate_abs_pos_embed(
    pos_embed: torch.Tensor,
    target_grid: GridHW,
    *,
    prefix_tokens: int = 1,
    source_grid: GridHW | None = None,
    mode: str = "bicubic",
) -> torch.Tensor:
    """Resize absolute position embeddings while preserving prefix tokens."""

    if pos_embed.ndim != 3 or pos_embed.shape[0] != 1:
        raise ResolutionError(f"Expected position embedding [1, N, C], got {tuple(pos_embed.shape)}.")

    prefix = pos_embed[:, :prefix_tokens, :] if prefix_tokens else pos_embed[:, :0, :]
    patch_pos = pos_embed[:, prefix_tokens:, :]
    source_grid = infer_pretrained_grid(patch_pos.shape[1], expected_grid=source_grid)
    if source_grid == target_grid:
        return pos_embed

    channels = patch_pos.shape[-1]
    patch_pos = patch_pos.reshape(1, source_grid[0], source_grid[1], channels).permute(0, 3, 1, 2)
    resized = F.interpolate(
        patch_pos.float(),
        size=target_grid,
        mode=mode,
        align_corners=False if mode in {"linear", "bilinear", "bicubic", "trilinear"} else None,
    )
    resized = resized.to(dtype=pos_embed.dtype).reshape(1, channels, target_grid[0] * target_grid[1]).permute(0, 2, 1)
    return torch.cat([prefix, resized], dim=1) if prefix_tokens else resized


def make_rope_grid(images_or_hw: torch.Tensor | tuple[int, int], patch_size: int, *, merge_size: int = 1) -> RopeGrid:
    return RopeGrid(
        grid_hw=infer_patch_grid(images_or_hw, patch_size, merge_size=merge_size),
        patch_size=int(patch_size),
        merge_size=int(merge_size),
    )
