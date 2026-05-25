"""Backbone registry and model construction."""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .weights import load_checkpoint_if_needed
from .wrappers.base import DenseTokenWrapper, UnsupportedModelError, require_encode_dense


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclass(frozen=True, slots=True)
class ModelRequest:
    name: str
    wrapper: str | None = None
    precision: str = "fp32"
    device: str | torch.device = "cpu"
    lora: str | None = None
    role: str = "student"
    trust_remote_code: bool = True
    cache_dir: str | None = None
    student_checkpoint: str | None = None
    teacher_checkpoint: str | None = None


BUILTIN_WRAPPERS: dict[str, str] = {
    "evaclip8b": "unirefiner.models.wrappers.evaclip8b:wrap_evaclip8b",
    "eva_clip_8b": "unirefiner.models.wrappers.evaclip8b:wrap_evaclip8b",
    "internvit": "unirefiner.models.wrappers.internvit:wrap_internvit",
    "internvit_6b": "unirefiner.models.wrappers.internvit:wrap_internvit",
    "openai_clip": "unirefiner.models.wrappers.openai_clip:wrap_openai_clip",
    "laion_clip": "unirefiner.models.wrappers.openai_clip:wrap_openai_clip",
    "clip": "unirefiner.models.wrappers.openai_clip:wrap_openai_clip",
    "dinov2": "unirefiner.models.wrappers.dinov2:wrap_dinov2_giant",
    "dinov2_giant": "unirefiner.models.wrappers.dinov2:wrap_dinov2_giant",
    "siglip2": "unirefiner.models.wrappers.siglip2:wrap_siglip2",
    "siglip2_so400m": "unirefiner.models.wrappers.siglip2:wrap_siglip2",
    "siglip2_giant": "unirefiner.models.wrappers.siglip2:wrap_siglip2",
    "rice": "unirefiner.models.wrappers.rice:wrap_rice_vit",
    "rice_vit": "unirefiner.models.wrappers.rice:wrap_rice_vit",
}


def get_cast_dtype(precision: str | None) -> torch.dtype | None:
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return None


def _get_field(source: Any, key: str, default: Any = None) -> Any:
    if source is None:
        return default
    if isinstance(source, dict):
        return source.get(key, default)
    return getattr(source, key, default)


def _get_nested(source: Any, section: str, key: str, default: Any = None) -> Any:
    section_value = _get_field(source, section, None)
    if section_value is None:
        return _get_field(source, key, default)
    return _get_field(section_value, key, default)


def request_from_config(config: Any, *, role: str = "student", device: str | torch.device = "cpu") -> ModelRequest:
    model_name = _get_nested(config, "model", "name")
    if model_name is None:
        raise ValueError("Model config must define `model.name`.")
    runtime_precision = _get_nested(config, "runtime", "precision", "fp32")
    return ModelRequest(
        name=str(model_name),
        wrapper=_get_nested(config, "model", "wrapper", None),
        precision=str(runtime_precision),
        device=device,
        lora=_get_nested(config, "model", "lora", None),
        role=role,
        trust_remote_code=bool(_get_nested(config, "model", "trust_remote_code", True)),
        cache_dir=_get_nested(config, "model", "cache_dir", None),
        student_checkpoint=_get_nested(config, "model", "student_checkpoint", _get_field(config, "student_model_ckpt", None)),
        teacher_checkpoint=_get_nested(config, "model", "teacher_checkpoint", _get_field(config, "teacher_model_ckpt", None)),
    )


def _parse_model_spec(model_name: str) -> tuple[str, str]:
    if "-register" in model_name or "-clearclip" in model_name or "-maskclip" in model_name:
        raise ValueError(f"Deprecated model suffixes are not part of the public API: `{model_name}`.")
    if model_name.endswith("-hf"):
        return model_name[:-3], "hf"
    if model_name.endswith("-timm"):
        return model_name[:-5], "timm"
    return model_name, "hf"


def _load_base_model(
    model_name: str,
    *,
    backend: str,
    cast_dtype: torch.dtype | None,
    trust_remote_code: bool,
    cache_dir: str | None,
) -> nn.Module:
    if backend == "hf":
        from transformers import AutoModel

        return AutoModel.from_pretrained(
            model_name,
            torch_dtype=cast_dtype,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
        ).eval()

    if backend == "timm":
        from timm import create_model as timm_create_model

        kwargs: dict[str, Any] = {"dynamic_img_size": True}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        return timm_create_model(model_name, pretrained=True, **kwargs).eval()

    raise NotImplementedError(f"Unsupported model backend: {backend}.")


def infer_builtin_wrapper_key(model_name: str) -> str:
    normalized = model_name.lower()
    if "internvl3_5" in normalized or "internvl35" in normalized:
        raise UnsupportedModelError("InternVL3.5-ViT is not included in this release; use InternViT instead.")
    if "eva-clip-8b" in normalized or "evaclip8b" in normalized:
        return "evaclip8b"
    if "internvit" in normalized:
        return "internvit"
    if "dinov2-giant" in normalized or "dinov2_giant" in normalized or "dinov2" in normalized:
        return "dinov2_giant"
    if "siglip2" in normalized:
        return "siglip2"
    if "rice-vit" in normalized or "rice_vit" in normalized or "ricevit" in normalized:
        return "rice_vit"
    if (
        "clip-vit-g-14-laion2b" in normalized
        or "clip-vit-base-patch16" in normalized
        or "clip-vit-large-patch14" in normalized
        or "openai/clip" in normalized
        or "laion" in normalized
    ):
        return "openai_clip"
    raise UnsupportedModelError(
        f"No built-in wrapper matched `{model_name}`. Set `model.wrapper` to one of "
        f"{sorted(BUILTIN_WRAPPERS)} or to `module.path:Wrapper`."
    )


def _import_object(spec: str):
    if ":" not in spec:
        raise ValueError(f"Wrapper import path must have the form `module.path:object`, got `{spec}`.")
    module_name, object_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, object_name)


def resolve_wrapper(wrapper: str | None, *, model_name: str):
    wrapper_key = infer_builtin_wrapper_key(model_name) if wrapper is None else str(wrapper)
    import_spec = BUILTIN_WRAPPERS.get(wrapper_key, wrapper_key)
    return _import_object(import_spec)


def wrap_model(base_model: nn.Module, *, model_name: str, wrapper: str | None = None) -> DenseTokenWrapper:
    wrapper_fn = resolve_wrapper(wrapper, model_name=model_name)
    wrapped = wrapper_fn(base_model)
    require_encode_dense(wrapped)
    return wrapped


def _apply_lora_if_needed(model: nn.Module, lora: str | None) -> nn.Module:
    if not lora:
        return model
    from .lora.factory import apply_lora

    return apply_lora(model, lora, merge=False)


def _checkpoint_for_role(request: ModelRequest) -> str | None:
    if request.role == "student":
        return request.student_checkpoint
    if request.role == "teacher":
        return request.teacher_checkpoint
    return None


def create_model(
    model_name: str | None = None,
    precision: str | None = None,
    device: str | torch.device = "cpu",
    args: Any = None,
    role: str = "student",
    *,
    wrapper: str | None = None,
    lora: str | None = None,
) -> nn.Module | None:
    """Create and wrap a UniRefiner vision backbone.

    Construction order is load base model, wrap dense-token behavior, attach
    student LoRA, move dtype/device, then load role-specific checkpoints.
    """

    if model_name is None:
        model_name = _get_nested(args, "model", "name", None)
    if model_name is None:
        return None

    request = ModelRequest(
        name=str(model_name),
        wrapper=wrapper or _get_nested(args, "model", "wrapper", _get_field(args, "model_wrapper", None)),
        precision=str(precision if precision is not None else _get_nested(args, "runtime", "precision", "fp32")),
        device=device,
        lora=lora if lora is not None else _get_nested(args, "model", "lora", _get_field(args, "lora_model", None)),
        role=role,
        trust_remote_code=bool(_get_field(args, "trust_remote_code", True)),
        cache_dir=_get_field(args, "cache_dir", None),
        student_checkpoint=_get_nested(args, "model", "student_checkpoint", _get_field(args, "student_model_ckpt", None)),
        teacher_checkpoint=_get_nested(args, "model", "teacher_checkpoint", _get_field(args, "teacher_model_ckpt", None)),
    )
    return create_model_from_request(request)


def create_model_from_config(config: Any, *, role: str = "student", device: str | torch.device = "cpu") -> nn.Module:
    return create_model_from_request(request_from_config(config, role=role, device=device))


def create_model_from_request(request: ModelRequest) -> nn.Module:
    if isinstance(request.device, str):
        device = torch.device(request.device)
    else:
        device = request.device

    cast_dtype = get_cast_dtype(request.precision)
    normalized_name, backend = _parse_model_spec(request.name)
    logging.info("Loading %s model `%s` with wrapper `%s`.", request.role, normalized_name, request.wrapper or "auto")

    base_model = _load_base_model(
        normalized_name,
        backend=backend,
        cast_dtype=cast_dtype,
        trust_remote_code=request.trust_remote_code,
        cache_dir=request.cache_dir,
    )
    wrapped_model = wrap_model(base_model, model_name=normalized_name, wrapper=request.wrapper)

    if request.role == "student":
        wrapped_model = _apply_lora_if_needed(wrapped_model, request.lora)

    if cast_dtype is None:
        wrapped_model.to(device=device)
    else:
        wrapped_model.to(device=device, dtype=cast_dtype)

    return load_checkpoint_if_needed(wrapped_model, _checkpoint_for_role(request), role=request.role)


def list_builtin_wrappers() -> list[str]:
    return sorted(BUILTIN_WRAPPERS)
