"""Crop proposal and crop-background composition helpers."""

from __future__ import annotations

import math
import random

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.ops as tvops


def sample_random_crop_boxes(scale=(0.4, 0.6), ratio=(0.5, 2.0), num_boxes: int = 8):
    boxes = np.zeros((num_boxes, 4))
    width, height = 1.0, 1.0
    area = width * height

    for box_id in range(num_boxes):
        target_area = random.uniform(*scale) * area
        attempts = 0
        while True:
            current_ratio = ratio
            if attempts >= 20:
                current_ratio = (min(ratio[0], width / height), max(ratio[1], width / height))
            attempts += 1
            aspect_ratio = math.exp(random.uniform(math.log(current_ratio[0]), math.log(current_ratio[1])))
            box_width = math.sqrt(target_area * aspect_ratio)
            box_height = math.sqrt(target_area / aspect_ratio)

            if box_width < width and box_height < height:
                center_x = np.random.uniform(0, width)
                center_y = np.random.uniform(0, height)
                x1 = max(0.0, min(1.0 - box_width, center_x - box_width / 2))
                y1 = max(0.0, min(1.0 - box_height, center_y - box_height / 2))
                x2 = x1 + box_width
                y2 = y1 + box_height
                boxes[box_id] = np.array([x1, y1, x2, y2])
                break
    return boxes


def add_batch_indices_and_scale_boxes(boxes: torch.Tensor, height: int, width: int) -> torch.Tensor:
    batch_size, num_boxes, _ = boxes.size()
    batch_index = torch.arange(batch_size, device=boxes.device).reshape(batch_size, 1).repeat(1, num_boxes)
    batch_index = batch_index.reshape(batch_size, num_boxes, 1)

    scaled_boxes = boxes[:, :, :4].clone()
    scaled_boxes[:, :, 0] *= width
    scaled_boxes[:, :, 2] *= width
    scaled_boxes[:, :, 1] *= height
    scaled_boxes[:, :, 3] *= height
    return torch.cat([batch_index, scaled_boxes], dim=2).reshape(batch_size * num_boxes, 5)


def roi_align_feature_map(feature_map: torch.Tensor, boxes: torch.Tensor, size=None) -> torch.Tensor:
    original_dtype = feature_map.dtype
    if feature_map.dtype in {torch.bfloat16, torch.float16}:
        feature_map = feature_map.float()
    boxes = boxes.to(dtype=feature_map.dtype)
    _, _, height, width = feature_map.shape
    if size is None:
        output_size = (height, width)
    elif isinstance(size, tuple):
        output_size = size
    else:
        output_size = (size, size)
    aligned_features = tvops.roi_align(
        input=feature_map,
        boxes=add_batch_indices_and_scale_boxes(boxes, height, width),
        output_size=output_size,
        aligned=True,
    )
    return aligned_features.to(dtype=original_dtype)


def roi_align_token_grid(
    token_features: torch.Tensor,
    boxes: torch.Tensor,
    normalize: bool = False,
    size=None,
) -> torch.Tensor:
    batch_size, token_count, channels = token_features.shape
    if normalize:
        token_features = F.normalize(token_features, dim=2)
    side = int(token_count**0.5)
    feature_map = token_features.permute(0, 2, 1).reshape(batch_size, channels, side, side)
    output_size = side if size is None else int(size)
    aligned = roi_align_feature_map(feature_map, boxes, size=output_size)
    return aligned.reshape(-1, channels, output_size**2).permute(0, 2, 1)


def compose_crop_with_background(
    crop_images: torch.Tensor,
    background_images: torch.Tensor,
    ratio: float = 0.5,
    patch_size: int = 16,
    square: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, channels, height, width = crop_images.shape
    background_images = background_images[:1]
    num_tokens = width // patch_size
    num_token_bg = int(num_tokens * ratio)
    if num_token_bg % 2 != 0:
        num_token_bg += 1

    layout = np.random.randint(0, 4)
    if square:
        total_tokens = num_tokens + num_token_bg
        output_size = patch_size * total_tokens
        new_img = F.interpolate(
            background_images.float(),
            size=(output_size, output_size),
            mode="bilinear",
            align_corners=False,
        ).to(dtype=crop_images.dtype)
        new_img = new_img.expand(batch_size, -1, -1, -1).clone()
        fg_mask = torch.zeros(
            (batch_size, total_tokens, total_tokens),
            dtype=torch.bool,
            device=crop_images.device,
        )

        row_start = 0
        col_start = 0
        if layout == 0:
            col_start = num_token_bg
        elif layout == 1:
            row_start = num_token_bg

        row_px = row_start * patch_size
        col_px = col_start * patch_size
        new_img[:, :, row_px : row_px + height, col_px : col_px + width] = crop_images
        fg_mask[:, row_start : row_start + num_tokens, col_start : col_start + num_tokens] = 1
        return new_img, fg_mask.reshape(batch_size, -1)

    if layout == 0:
        new_img = torch.zeros(
            (batch_size, channels, height, patch_size * (num_tokens + num_token_bg)),
            dtype=crop_images.dtype,
            device=crop_images.device,
        )
        new_img[:, :, :, : num_token_bg * patch_size] = background_images[:, :, :, : num_token_bg * patch_size]
        new_img[:, :, :, num_token_bg * patch_size :] = crop_images
        fg_mask = torch.zeros((batch_size, num_tokens, num_tokens + num_token_bg), dtype=torch.bool, device=crop_images.device)
        fg_mask[:, :, num_token_bg:] = 1
    elif layout == 2:
        new_img = torch.zeros(
            (batch_size, channels, height, patch_size * (num_tokens + num_token_bg)),
            dtype=crop_images.dtype,
            device=crop_images.device,
        )
        new_img[:, :, :, : num_tokens * patch_size] = crop_images
        new_img[:, :, :, num_tokens * patch_size :] = background_images[:, :, :, -num_token_bg * patch_size :]
        fg_mask = torch.zeros((batch_size, num_tokens, num_tokens + num_token_bg), dtype=torch.bool, device=crop_images.device)
        fg_mask[:, :, :num_tokens] = 1
    elif layout == 1:
        new_img = torch.zeros(
            (batch_size, channels, patch_size * (num_tokens + num_token_bg), width),
            dtype=crop_images.dtype,
            device=crop_images.device,
        )
        new_img[:, :, : num_token_bg * patch_size, :] = background_images[:, :, : num_token_bg * patch_size, :]
        new_img[:, :, num_token_bg * patch_size :, :] = crop_images
        fg_mask = torch.zeros((batch_size, num_tokens + num_token_bg, num_tokens), dtype=torch.bool, device=crop_images.device)
        fg_mask[:, num_token_bg:, :] = 1
    else:
        new_img = torch.zeros(
            (batch_size, channels, patch_size * (num_tokens + num_token_bg), width),
            dtype=crop_images.dtype,
            device=crop_images.device,
        )
        new_img[:, :, -num_token_bg * patch_size :, :] = background_images[:, :, -num_token_bg * patch_size :, :]
        new_img[:, :, : -num_token_bg * patch_size, :] = crop_images
        fg_mask = torch.zeros((batch_size, num_tokens + num_token_bg, num_tokens), dtype=torch.bool, device=crop_images.device)
        fg_mask[:, :-num_token_bg, :] = 1

    return new_img, fg_mask.reshape(batch_size, -1)
