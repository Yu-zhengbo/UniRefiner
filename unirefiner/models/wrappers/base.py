"""Shared model-wrapper contract.

Built-in wrappers may be more involved, but a user-provided wrapper only needs
to expose `encode_dense(images)`, `patch_size`, `image_mean`, and `image_std`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

import torch
from torch import nn


ImageStats = tuple[float, float, float]
GridHW = tuple[int, int]


class ModelWrapperError(RuntimeError):
    """Base error for model-wrapper failures."""


class ResolutionError(ModelWrapperError):
    """Raised when a wrapper cannot support the requested input resolution."""


class AttentionHooksNotSupported(ModelWrapperError):
    """Raised when AH filtering asks for hooks that a wrapper cannot provide."""


class MissingModelDependency(ModelWrapperError):
    """Raised when a requested model backend is not installed."""


class UnsupportedModelError(ModelWrapperError):
    """Raised when no built-in wrapper matches a model request."""


@dataclass(slots=True)
class DenseFeatures:
    """Dense visual tokens and the spatial metadata needed by UniRefiner."""

    tokens: torch.Tensor
    grid_hw: GridHW
    patch_size: int
    valid_mask: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.tokens.ndim != 3:
            raise ValueError(f"`tokens` must have shape [B, N, C], got {tuple(self.tokens.shape)}.")
        if len(self.grid_hw) != 2:
            raise ValueError(f"`grid_hw` must be a pair, got {self.grid_hw!r}.")
        grid_tokens = int(self.grid_hw[0]) * int(self.grid_hw[1])
        if self.valid_mask is None and grid_tokens != self.tokens.shape[1]:
            raise ValueError(
                "`grid_hw` does not match dense token count: "
                f"{self.grid_hw} gives {grid_tokens}, tokens have {self.tokens.shape[1]}."
            )
        if self.valid_mask is not None and self.valid_mask.shape[:2] != self.tokens.shape[:2]:
            raise ValueError(
                "`valid_mask` must match the first two token dimensions, "
                f"got {tuple(self.valid_mask.shape)} for tokens {tuple(self.tokens.shape)}."
            )

    @property
    def batch_size(self) -> int:
        return int(self.tokens.shape[0])

    @property
    def token_count(self) -> int:
        return int(self.tokens.shape[1])

    @property
    def channels(self) -> int:
        return int(self.tokens.shape[2])


@runtime_checkable
class DenseTokenWrapper(Protocol):
    """Minimal protocol expected from custom UniRefiner vision wrappers."""

    patch_size: int
    image_mean: Sequence[float]
    image_std: Sequence[float]

    def encode_dense(self, images: torch.Tensor) -> torch.Tensor:
        """Return dense patch tokens in raster order."""


def ensure_image_stats(values: Sequence[float], *, name: str) -> ImageStats:
    if len(values) != 3:
        raise ValueError(f"`{name}` must contain three channel values, got {values!r}.")
    return (float(values[0]), float(values[1]), float(values[2]))


def infer_square_grid(token_count: int) -> GridHW:
    side = int(token_count**0.5)
    if side * side != token_count:
        raise ResolutionError(
            "Cannot infer a square patch grid from dense token count "
            f"{token_count}; return DenseFeatures with explicit grid_hw instead."
        )
    return (side, side)


def infer_grid_from_images(images: torch.Tensor, patch_size: int) -> GridHW:
    if images.ndim != 4:
        raise ResolutionError(f"Expected image tensor [B, C, H, W], got {tuple(images.shape)}.")
    height, width = int(images.shape[-2]), int(images.shape[-1])
    if height % patch_size != 0 or width % patch_size != 0:
        raise ResolutionError(
            f"Image size {(height, width)} must be divisible by patch size {patch_size}."
        )
    return (height // patch_size, width // patch_size)


def as_dense_features(
    output: torch.Tensor | DenseFeatures,
    *,
    images: torch.Tensor | None = None,
    patch_size: int | None = None,
    grid_hw: GridHW | None = None,
    valid_mask: torch.Tensor | None = None,
) -> DenseFeatures:
    """Normalize raw wrapper output into `DenseFeatures`.

    The public wrapper contract only requires raw dense-token tensors. This
    helper remains available for internal utilities that need explicit grid
    metadata.
    """

    if isinstance(output, DenseFeatures):
        return output
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Expected Tensor-like dense features, got {type(output)!r}.")
    if output.ndim != 3:
        raise ValueError(f"Dense tokens must have shape [B, N, C], got {tuple(output.shape)}.")
    if patch_size is None:
        raise ValueError("`patch_size` is required when converting raw dense tokens.")
    if grid_hw is None:
        grid_hw = infer_grid_from_images(images, patch_size) if images is not None else infer_square_grid(output.shape[1])
    return DenseFeatures(tokens=output, grid_hw=grid_hw, patch_size=int(patch_size), valid_mask=valid_mask)


def require_encode_dense(module: nn.Module) -> None:
    if not callable(getattr(module, "encode_dense", None)):
        raise UnsupportedModelError(f"{module.__class__.__name__} does not expose encode_dense(images).")
