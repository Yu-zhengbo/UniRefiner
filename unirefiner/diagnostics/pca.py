"""PCA visualization for dense feature maps."""

from __future__ import annotations

import math

from PIL import Image
import torch


def compute_pca_reference(
    features: torch.Tensor,
    reference_features: torch.Tensor | None = None,
    quantile_low: float = 0.01,
    quantile_high: float = 0.99,
) -> dict[str, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
    """Compute reusable PCA projection metadata from dense features."""

    features = features.float()
    reference_features = features if reference_features is None else reference_features.float()
    _, dim = features.shape
    center = features.mean(dim=0, keepdim=True)
    centered = features - center

    if features.shape[0] < 2:
        eigenvectors = torch.eye(dim, device=features.device, dtype=features.dtype)
    else:
        covariance = centered.T @ centered / (features.shape[0] - 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        eigenvectors = eigenvectors[:, torch.argsort(eigenvalues, descending=True)]

    projected = (reference_features - center) @ eigenvectors[:, :3]
    proj_low = torch.quantile(projected, quantile_low, dim=0)
    proj_high = torch.quantile(projected, quantile_high, dim=0)
    return {"center": center, "eigenvectors": eigenvectors, "value_range": (proj_low, proj_high)}


def pca_visualize(
    features: torch.Tensor,
    width: int | None = None,
    height: int | None = None,
    output_size: int = 512,
    return_eigen: bool = False,
    eigen: torch.Tensor | None = None,
    center: torch.Tensor | None = None,
    value_range: tuple[torch.Tensor, torch.Tensor] | None = None,
):
    """Render dense tokens as an RGB PCA image.

    The first call can return eigenvectors; later calls can reuse them so
    student/teacher snapshots share a color basis during training.
    """

    if features.ndim != 2:
        raise ValueError(f"`features` must have shape [tokens, channels], got {tuple(features.shape)}.")

    token_count, dim = features.shape
    if token_count == 0:
        image = Image.new("RGB", (1, 1), (0, 0, 0))
        return (image, torch.eye(dim, device=features.device, dtype=features.dtype)) if return_eigen else image

    features = features.float()
    if center is None:
        center = features.mean(dim=0, keepdim=True)
    elif center.dim() == 1:
        center = center.unsqueeze(0)
    center = center.to(device=features.device, dtype=features.dtype)
    features_centered = features - center

    if eigen is None:
        covariance = features_centered.T @ features_centered / max(token_count - 1, 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        eigenvectors = eigenvectors[:, torch.argsort(eigenvalues, descending=True)]
    else:
        eigenvectors = eigen.to(device=features.device, dtype=features.dtype)

    pca_features = features_centered @ eigenvectors[:, :3]
    if value_range is None:
        pca_min = pca_features.min(dim=0, keepdim=True)[0]
        pca_max = pca_features.max(dim=0, keepdim=True)[0]
    else:
        pca_min, pca_max = value_range
        pca_min = pca_min.unsqueeze(0) if pca_min.dim() == 1 else pca_min
        pca_max = pca_max.unsqueeze(0) if pca_max.dim() == 1 else pca_max
        pca_min = pca_min.to(device=features.device, dtype=features.dtype)
        pca_max = pca_max.to(device=features.device, dtype=features.dtype)

    pca_features = (pca_features - pca_min) / (pca_max - pca_min + 1e-8) * 255
    pca_features = torch.clamp(pca_features, 0, 255).to(torch.uint8)

    if width is None or height is None:
        side = int(math.sqrt(token_count))
        if side * side != token_count:
            raise ValueError("`width` and `height` are required for non-square token grids.")
        width = height = side
    if token_count != width * height:
        raise ValueError(f"Token count {token_count} does not match grid {width}x{height}.")

    array = pca_features.view(height, width, 3).cpu().numpy()
    image = Image.fromarray(array, mode="RGB")
    scale = output_size / min(width, height)
    image = image.resize((int(width * scale), int(height * scale)), Image.NEAREST)

    if return_eigen:
        return image, eigenvectors
    return image
