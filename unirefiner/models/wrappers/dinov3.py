"""DINOv3 wrappers."""

from __future__ import annotations

from .generic import wrap_generic_dense_model

IMAGENET_DEFAULT_MEAN = [109.65/255., 104.805/255., 75.48/255.]
IMAGENET_DEFAULT_STD = [54.315/255., 39.78/255., 36.465/255.]

def wrap_dinov3(model):
    """Expose dense patch tokens and EvaAttention hooks for timm DINOv3 ViTs."""

    return wrap_generic_dense_model(
        model,
        vision_candidates=(),
        skip_prefix_tokens=None,
        fallback_mean=IMAGENET_DEFAULT_MEAN,
        fallback_std=IMAGENET_DEFAULT_STD,
    )


DINOv3_Wrapper = wrap_dinov3
