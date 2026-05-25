"""Register-region construction helpers."""

from __future__ import annotations

import torch


def normalize_register_fill(register_fill: str) -> str:
    if register_fill not in {"rand", "randn", "zero"}:
        raise ValueError(f"Unsupported register_fill: {register_fill}")
    return register_fill


def surround_image_with_registers(
    image_tensor: torch.Tensor,
    patch_size: int = 16,
    register_size: int = 1,
    register_fill: str = "rand",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add LRUD register regions around the image and return a register-token mask."""

    batch_size, channels, height, width = image_tensor.shape
    register_fill = normalize_register_fill(register_fill)
    output_height = height + 2 * register_size * patch_size
    output_width = width + 2 * register_size * patch_size

    if register_fill == "rand":
        canvas_with_registers = torch.rand(
            batch_size,
            channels,
            output_height,
            output_width,
            device=image_tensor.device,
            dtype=image_tensor.dtype,
        )
    elif register_fill == "randn":
        canvas_with_registers = torch.randn(
            batch_size,
            channels,
            output_height,
            output_width,
            device=image_tensor.device,
            dtype=image_tensor.dtype,
        )
    else:
        canvas_with_registers = torch.zeros(
            batch_size,
            channels,
            output_height,
            output_width,
            device=image_tensor.device,
            dtype=image_tensor.dtype,
        )

    canvas_with_registers[
        :,
        :,
        register_size * patch_size : -register_size * patch_size,
        register_size * patch_size : -register_size * patch_size,
    ] = image_tensor

    token_height = height // patch_size + 2 * register_size
    token_width = width // patch_size + 2 * register_size
    mask_vertical = torch.zeros(1, token_height, token_width, device=image_tensor.device, dtype=torch.bool)
    mask_horizontal = torch.zeros_like(mask_vertical)

    mask_vertical[:, :register_size, :] = 1
    mask_vertical[:, -register_size:, :] = 1
    mask_horizontal[:, register_size:-register_size, :register_size] = 1
    mask_horizontal[:, register_size:-register_size, -register_size:] = 1

    register_mask = (mask_vertical | mask_horizontal).reshape(1, -1).expand(batch_size, -1)
    return canvas_with_registers, register_mask
