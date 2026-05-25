"""Backbone wrappers exposing the common UniRefiner model interface."""

from .base import (
    AttentionHooksNotSupported,
    DenseFeatures,
    DenseTokenWrapper,
    MissingModelDependency,
    ModelWrapperError,
    ResolutionError,
    UnsupportedModelError,
    as_dense_features,
)

__all__ = [
    "AttentionHooksNotSupported",
    "DenseFeatures",
    "DenseTokenWrapper",
    "MissingModelDependency",
    "ModelWrapperError",
    "ResolutionError",
    "UnsupportedModelError",
    "as_dense_features",
]
