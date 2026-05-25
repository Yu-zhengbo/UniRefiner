"""Strict YAML loading and dotted override parsing."""

from __future__ import annotations

from dataclasses import MISSING, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin, get_type_hints

import yaml

from .schema import UniRefinerConfig


T = TypeVar("T")


def _parse_override_value(raw_value: str) -> Any:
    if raw_value == "":
        return ""
    return yaml.safe_load(raw_value)


def _deep_update(payload: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    if any(key == "" for key in keys):
        raise ValueError(f"Invalid override key: `{dotted_key}`.")

    cursor = payload
    for key in keys[:-1]:
        next_value = cursor.setdefault(key, {})
        if not isinstance(next_value, dict):
            raise ValueError(f"Cannot override `{dotted_key}` because `{key}` is not a section.")
        cursor = next_value
    cursor[keys[-1]] = value


def _is_dataclass_type(field_type: Any) -> bool:
    return isinstance(field_type, type) and is_dataclass(field_type)


def _unwrap_optional(field_type: Any) -> Any:
    origin = get_origin(field_type)
    if origin is None:
        return field_type
    args = [arg for arg in get_args(field_type) if arg is not type(None)]
    if len(args) == 1:
        return args[0]
    return field_type


def _construct_dataclass(cls: type[T], payload: dict[str, Any]) -> T:
    type_hints = get_type_hints(cls)
    known_fields = {field_info.name for field_info in fields(cls)}
    unknown_fields = sorted(set(payload) - known_fields)
    if unknown_fields:
        joined = ", ".join(unknown_fields)
        raise ValueError(f"Unknown config field(s) for {cls.__name__}: {joined}")

    kwargs: dict[str, Any] = {}
    for field_info in fields(cls):
        if field_info.name not in payload:
            continue
        value = payload[field_info.name]
        field_type = _unwrap_optional(type_hints.get(field_info.name, field_info.type))
        if _is_dataclass_type(field_type):
            if value is None:
                kwargs[field_info.name] = None
            elif isinstance(value, dict):
                kwargs[field_info.name] = _construct_dataclass(field_type, value)
            else:
                raise ValueError(f"`{field_info.name}` must be a mapping for {field_type.__name__}.")
        else:
            kwargs[field_info.name] = value

    try:
        return cls(**kwargs)
    except TypeError as error:
        missing = [
            field_info.name
            for field_info in fields(cls)
            if field_info.default is MISSING
            and field_info.default_factory is MISSING
            and field_info.name not in payload
        ]
        if missing:
            raise ValueError(f"Missing required config field(s) for {cls.__name__}: {', '.join(missing)}") from error
        raise


def _validate_config(config: UniRefinerConfig) -> None:
    if not config.experiment.name:
        raise ValueError("`experiment.name` must be non-empty.")
    if int(config.runtime.batch_size) <= 0:
        raise ValueError("`runtime.batch_size` must be positive.")
    if int(config.runtime.accum_freq) <= 0:
        raise ValueError("`runtime.accum_freq` must be positive.")
    if int(config.runtime.epochs) <= 0:
        raise ValueError("`runtime.epochs` must be positive.")
    if int(config.data.image_size) <= 0:
        raise ValueError("`data.image_size` must be positive.")
    if int(config.method.reg_factor) <= 0:
        raise ValueError("`method.reg_factor` must be positive.")
    if int(config.method.num_proposals) <= 0:
        raise ValueError("`method.num_proposals` must be positive.")
    if config.method.name != "unirefiner":
        raise ValueError("`method.name` must be `unirefiner`.")
    if isinstance(config.model.lora, list) or (
        isinstance(config.model.lora, str) and "," in config.model.lora
    ):
        raise ValueError("Only a single LoRA preset is supported in this release.")


def load_config(config_path: str | Path, overrides: list[str] | None = None) -> UniRefinerConfig:
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")

    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Override must have the form `section.field=value`, got `{override}`.")
        key, raw_value = override.split("=", 1)
        _deep_update(payload, key, _parse_override_value(raw_value))

    config = _construct_dataclass(UniRefinerConfig, payload)
    _validate_config(config)
    return config


def dump_config(config: UniRefinerConfig, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)
