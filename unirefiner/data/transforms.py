"""Image preprocessing helpers for UniRefiner training data."""

from __future__ import annotations

from collections.abc import Sequence

from PIL import Image
from torchvision.transforms import Compose, InterpolationMode, Normalize, Resize, ToTensor


OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)


def convert_to_rgb(image: Image.Image) -> Image.Image:
    """Convert any input image mode to RGB before normalization."""

    return image.convert("RGB")


def build_image_transform(
    image_size: int,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
):
    """Build the training image transform used for both image and background."""

    mean = tuple(mean or OPENAI_DATASET_MEAN)
    std = tuple(std or OPENAI_DATASET_STD)
    return Compose(
        [
            Resize((int(image_size), int(image_size)), interpolation=InterpolationMode.BICUBIC),
            convert_to_rgb,
            ToTensor(),
            Normalize(mean=mean, std=std),
        ]
    )


build_transform = build_image_transform
