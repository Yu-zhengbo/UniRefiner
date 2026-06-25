"""Public UniRefiner configuration schema."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def default_background_path() -> str:
    project_root = Path(__file__).resolve().parents[2]
    return str(project_root / "assets" / "backgrounds" / "fixed_reference.png")


def default_pca_test_image_path() -> str:
    project_root = Path(__file__).resolve().parents[2]
    return str(project_root / "assets" / "diagnostics" / "pca_test_image.jpg")


@dataclass(slots=True)
class ExperimentConfig:
    name: str
    output_dir: str = "outputs"


@dataclass(slots=True)
class RuntimeConfig:
    seed: int = 0
    workers: int = 8
    batch_size: int = 2
    epochs: int = 2
    accum_freq: int = 1
    precision: str = "bf16"
    dist_url: str = "env://"
    dist_backend: str = "nccl"
    no_set_device_rank: bool = False
    ddp_static_graph: bool = False
    use_bn_sync: bool = False
    save_freq: int = 1
    log_every_n_steps: int = 50
    wandb_log_every_n_steps: int = 1
    grad_clip_norm: float | None = None
    resume: str | None = None
    device: str | None = None


@dataclass(slots=True)
class BackgroundConfig:
    path: str = field(default_factory=default_background_path)


@dataclass(slots=True)
class DataConfig:
    train_image_root: str | list[str] = ""
    image_size: int = 448
    background: BackgroundConfig = field(default_factory=BackgroundConfig)


@dataclass(slots=True)
class ModelConfig:
    name: str = ""
    wrapper: str | None = None
    teacher_name: str | None = None
    student_checkpoint: str | None = None
    teacher_checkpoint: str | None = None
    lora: str | None = None
    cache_dir: str | None = None
    trust_remote_code: bool = True


@dataclass(slots=True)
class MethodConfig:
    name: str = "unirefiner"
    reg_factor: int = 24
    register_type: str = "randn"
    enable_window_phase_artifact_loss: bool = False
    fp_gp_thres: float | None = 0.5
    adaptive_register_thres: float = 0.6
    disable_attention_hijack_filter: bool = False
    attention_hijack_layer_start: int = 10
    attention_hijack_layer_end: int = 15
    attention_hijack_thres: float = 0.5
    disable_student_teacher_matching: bool = False
    uniformity_strength: float = 1.0
    num_proposals: int = 3


@dataclass(slots=True)
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 1.0e-4
    lr_last: float = 0.0
    wd: float = 0.1
    warmup: int = 0
    scheduler: str = "const"
    beta1: float | None = None
    beta2: float | None = None
    eps: float | None = None
    skip_scheduler: bool = False
    loraplus_lr_ratio: float = 4.0
    loraplus_weight_decay: float | None = None


@dataclass(slots=True)
class LoggingConfig:
    wandb: bool = False
    wandb_project: str = "uni_refiner"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_mode: str = "offline"
    log_local: bool = False


@dataclass(slots=True)
class DiagnosticsConfig:
    vis_pca_interval: int = 0
    vis_pca_test_image: str | None = field(default_factory=default_pca_test_image_path)
    vis_pca_save_dir: str | None = None


@dataclass(slots=True)
class UniRefinerConfig:
    experiment: ExperimentConfig
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    method: MethodConfig = field(default_factory=MethodConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_runtime_namespace(self) -> SimpleNamespace:
        from .runtime import to_runtime_namespace

        return to_runtime_namespace(self)
