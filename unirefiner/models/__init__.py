"""Model registry, weights, wrappers, channel masks, and LoRA."""

from .channel_mask import (
    apply_channel_mask,
    collect_channel_mask_feature_bank,
    resolve_channel_mask_channels,
    run_greedy_channel_mask,
    save_channel_mask_artifacts,
)
from .registry import (
    ModelRequest,
    create_model,
    create_model_from_config,
    create_model_from_request,
    infer_builtin_wrapper_key,
    list_builtin_wrappers,
    wrap_model,
)

__all__ = [
    "apply_channel_mask",
    "collect_channel_mask_feature_bank",
    "ModelRequest",
    "create_model",
    "create_model_from_config",
    "create_model_from_request",
    "infer_builtin_wrapper_key",
    "list_builtin_wrappers",
    "resolve_channel_mask_channels",
    "run_greedy_channel_mask",
    "save_channel_mask_artifacts",
    "wrap_model",
]
