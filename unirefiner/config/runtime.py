"""Map public configs to the runtime namespace expected by training code."""

from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace

from .schema import UniRefinerConfig


def to_runtime_namespace(config: UniRefinerConfig) -> SimpleNamespace:
    """Build the runtime namespace consumed by training code.

    Public YAMLs use paper-facing names. Training code uses a few concise
    runtime aliases, so they are added here rather than exposed in default
    configs.
    """

    runtime_values = asdict(config.runtime)
    method_values = asdict(config.method)
    method_name = method_values.pop("name")

    values = {
        "name": config.experiment.name,
        "logs_path": config.experiment.output_dir,
        "train_image_root": config.data.train_image_root,
        "image_size": config.data.image_size,
        "background_image_path": config.data.background.path,
        "model": config.model.name,
        "t_model": config.model.teacher_name,
        "model_wrapper": config.model.wrapper,
        "student_model_ckpt": config.model.student_checkpoint,
        "teacher_model_ckpt": config.model.teacher_checkpoint,
        "lora_model": config.model.lora,
        "cache_dir": config.model.cache_dir,
        "trust_remote_code": config.model.trust_remote_code,
        "method": method_name,
        "optimizer": config.optimizer.name,
        "lr": config.optimizer.lr,
        "lr_last": config.optimizer.lr_last,
        "wd": config.optimizer.wd,
        "warmup": config.optimizer.warmup,
        "lr_scheduler": config.optimizer.scheduler,
        "beta1": config.optimizer.beta1,
        "beta2": config.optimizer.beta2,
        "eps": config.optimizer.eps,
        "skip_scheduler": config.optimizer.skip_scheduler,
        "loraplus_lr_ratio": config.optimizer.loraplus_lr_ratio,
        "loraplus_weight_decay": config.optimizer.loraplus_weight_decay,
        "wandb": config.logging.wandb,
        "wandb_project": config.logging.wandb_project,
        "wandb_entity": config.logging.wandb_entity,
        "wandb_run_name": config.logging.wandb_run_name,
        "wandb_mode": config.logging.wandb_mode,
        "log_local": config.logging.log_local,
        "vis_pca_interval": config.diagnostics.vis_pca_interval,
        "vis_pca_test_image": config.diagnostics.vis_pca_test_image,
        "vis_pca_save_dir": config.diagnostics.vis_pca_save_dir,
        "fp_gp_sigma": config.method.fp_gp_thres,
        "fp_gp_cosine_threshold": None,
        "attention_hijack_sigma": config.method.attention_hijack_thres,
        "adaptive_spurious_detector_cosine_threshold": config.method.adaptive_register_thres,
        "register_fill": config.method.register_type,
        "auto_channel_mask": True,
        "channel_mask_channels": [],
        "channel_mask_samples": 64,
        "channel_mask_tokens_per_image": 128,
        "channel_mask_batch_size": 8,
        "channel_mask_threshold": 0.2,
        "channel_mask_max_channels": 5,
        "channel_mask_candidate_pool": 16,
        "channel_mask_min_delta": 0.005,
        "channel_mask_final_neg_mean": None,
        "channel_mask_stop_reason": None,
        "distributed": False,
        "rank": 0,
        "local_rank": 0,
        "world_size": 1,
        "train_stage": 0.0,
        "global_step": 0,
    }
    values.update(runtime_values)
    values.update(method_values)
    return SimpleNamespace(**values)
