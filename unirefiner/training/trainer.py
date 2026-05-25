"""Main UniRefiner training loop."""

from __future__ import annotations

import logging
import os
from contextlib import suppress

import torch
from torch.cuda.amp import GradScaler

from unirefiner.config.schema import UniRefinerConfig
from unirefiner.data import build_data
from unirefiner.method import UniRefinerMethod
from unirefiner.models import (
    collect_channel_mask_feature_bank,
    create_model,
    run_greedy_channel_mask,
    save_channel_mask_artifacts,
)

from .checkpoint import load_resume_checkpoint, save_epoch_checkpoint, save_final_model
from .distributed import broadcast_object, init_distributed_device, is_master
from .logger import setup_logging
from .optimizer import (
    REFERENCE_TOTAL_BATCH_SIZE,
    apply_linear_lr_scaling,
    build_optimizer,
    fill_default_optimizer_params,
)
from .scheduler import build_scheduler
from .seed import set_random_seed


def get_cast_dtype(precision: str):
    if precision in {"bf16", "amp_bf16", "amp_bfloat16"}:
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return None


def get_autocast(precision: str):
    if precision == "amp":
        return torch.cuda.amp.autocast
    if precision in {"amp_bfloat16", "amp_bf16"}:
        return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
    return suppress


def _init_torch_backend() -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def _init_wandb(args, config: UniRefinerConfig, log_base_path: str):
    if not args.wandb or args.wandb_mode == "disabled" or not is_master(args):
        return None

    try:
        import wandb
    except ImportError:
        logging.warning("wandb is not installed; continuing without W&B logging.")
        return None

    api_key = os.environ.get("WANDB_API_KEY")
    if api_key:
        try:
            wandb.login(key=api_key, relogin=True)
        except Exception as error:
            logging.warning("wandb login failed: %s", error)

    try:
        return wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or args.name,
            dir=log_base_path,
            config=config.to_dict(),
            mode=args.wandb_mode,
        )
    except Exception as error:
        logging.warning("wandb init failed: %s", error)
        return None


def _prepare_output_paths(args) -> str:
    log_base_path = os.path.join(args.logs_path, args.name)
    args.log_path = None
    if is_master(args, local=args.log_local):
        os.makedirs(log_base_path, exist_ok=True)
        args.log_path = os.path.join(log_base_path, f"out-{args.rank}" if args.log_local else "out.log")
    args.checkpoint_path = os.path.join(log_base_path, "checkpoints")
    args.channel_mask_report_path = os.path.join(log_base_path, "channel_mask.json")
    args.channel_mask_tensor_path = os.path.join(log_base_path, "channel_mask.pt")
    os.makedirs(args.checkpoint_path, exist_ok=True)
    return log_base_path


def _resolve_channel_mask(model, data, args) -> None:
    args.channel_mask_channels = []
    args.channel_mask_final_neg_mean = None
    args.channel_mask_stop_reason = "disabled"
    if not bool(getattr(args, "auto_channel_mask", True)):
        return

    selection = {
        "enabled": True,
        "masked_channels": [],
        "final_neg_mean": None,
        "stop_reason": "not_master",
    }
    if is_master(args):
        try:
            feature_bank_info = collect_channel_mask_feature_bank(
                model=model,
                dataset=data.dataloader.dataset,
                args=args,
                cast_dtype=get_cast_dtype(args.precision),
            )
            if feature_bank_info is None:
                selection = {
                    "enabled": True,
                    "masked_channels": [],
                    "final_neg_mean": None,
                    "stop_reason": "no_samples",
                }
            else:
                selection = run_greedy_channel_mask(feature_bank_info, args)
        except Exception as error:
            logging.exception("Auto channel mask failed: %s", error)
            selection = {
                "enabled": True,
                "masked_channels": [],
                "final_neg_mean": None,
                "stop_reason": "error",
            }
        save_channel_mask_artifacts(selection, args)

    selection = broadcast_object(args, selection)
    args.channel_mask_channels = selection.get("masked_channels", [])
    args.channel_mask_final_neg_mean = selection.get("final_neg_mean")
    args.channel_mask_stop_reason = selection.get("stop_reason")
    if is_master(args):
        logging.info(
            "channel_mask enabled=%s channels=%s final_neg_mean=%s stop_reason=%s",
            selection.get("enabled", False),
            args.channel_mask_channels,
            args.channel_mask_final_neg_mean,
            args.channel_mask_stop_reason,
        )


def _wrap_distributed_student(model, args):
    if not args.distributed:
        return model
    if args.use_bn_sync:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    ddp_kwargs = {}
    if args.ddp_static_graph:
        ddp_kwargs["static_graph"] = True
    if torch.cuda.is_available():
        return torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            **ddp_kwargs,
        )
    return torch.nn.parallel.DistributedDataParallel(model, **ddp_kwargs)


def _log_loss_meters(losses_meter: dict[str, dict[str, float]]) -> str:
    parts = []
    for loss_name, meter in losses_meter.items():
        average = meter["sum"] / max(meter["count"], 1)
        parts.append(f"{loss_name}={average:.4f}")
    return " ".join(parts)


def _record_losses(losses_meter: dict[str, dict[str, float]], losses: dict[str, torch.Tensor], batch_size: int) -> None:
    for loss_name, value in losses.items():
        meter = losses_meter.setdefault(loss_name, {"sum": 0.0, "count": 0})
        meter["sum"] += float(value.detach().item()) * batch_size
        meter["count"] += batch_size


def train_one_epoch(model, teacher_model, method, data, epoch: int, optimizer, scaler, scheduler, args) -> None:
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    model.train()
    data.set_epoch(epoch)
    dataloader = data.dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    total_iterations = len(dataloader) * args.epochs
    losses_meter: dict[str, dict[str, float]] = {}

    for batch_index, batch in enumerate(dataloader):
        args.train_stage = (batch_index + len(dataloader) * epoch) / max(total_iterations, 1)
        optimizer.zero_grad()
        step = num_batches_per_epoch * epoch + batch_index // args.accum_freq
        args.global_step = step
        if not args.skip_scheduler:
            scheduler(step)

        with autocast():
            losses, batch_size = method(batch, model, teacher_model, device, cast_dtype, args.distributed, args)
            total_loss = losses["loss_final"]
            losses["loss"] = total_loss

        if scaler is not None:
            scaler.scale(total_loss).backward()
            if args.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        _record_losses(losses_meter, losses, batch_size)

        if is_master(args) and (
            batch_index % args.log_every_n_steps == 0 or (batch_index + 1) == dataloader.num_batches
        ):
            logging.info(
                "epoch=%d step=%d/%d lr=%.6g %s",
                epoch,
                batch_index + 1,
                dataloader.num_batches,
                optimizer.param_groups[0]["lr"],
                _log_loss_meters(losses_meter),
            )

        if is_master(args) and getattr(args, "wandb_run", None) is not None:
            wandb_log_freq = max(getattr(args, "wandb_log_every_n_steps", 1), 1)
            if batch_index % wandb_log_freq == 0:
                args.wandb_run.log(
                    {
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "train/epoch": epoch,
                        **{f"train/{name}": float(value.detach().item()) for name, value in losses.items()},
                    },
                    step=step,
                    commit=True,
                )


def run_training(config: UniRefinerConfig) -> None:
    args = config.to_runtime_namespace()
    if args.method != "unirefiner":
        raise ValueError(
            f"Unsupported method `{args.method}` in open-source training package. "
            "Supported value is `unirefiner`."
        )
    args.wandb_run = None

    _init_torch_backend()
    device = init_distributed_device(args)
    apply_linear_lr_scaling(args)
    if args.t_model is None:
        args.t_model = args.model

    log_base_path = _prepare_output_paths(args)
    setup_logging(args.log_path, logging.INFO)
    args.wandb_run = _init_wandb(args, config, log_base_path)

    set_random_seed(args.seed, 0)
    student_model = create_model(args.model, precision=args.precision, device=device, args=args, role="student")
    data = build_data(
        args,
        mean=getattr(student_model, "image_mean", None),
        std=getattr(student_model, "image_std", None),
    )
    teacher_model = create_model(args.t_model, precision=args.precision, device=device, args=args, role="teacher")
    teacher_model.requires_grad_(False)
    load_resume_checkpoint(student_model, args.resume, map_location=device)
    _resolve_channel_mask(student_model, data, args)

    set_random_seed(args.seed, args.rank)
    if is_master(args):
        logging.info("student_model=%s", student_model.__class__.__name__)
        logging.info("teacher_model=%s", teacher_model.__class__.__name__)
        logging.info("config=%s", config.to_dict())

    student_model = _wrap_distributed_student(student_model, args)
    fill_default_optimizer_params(args)
    optimizer = build_optimizer(student_model, args)
    scaler = GradScaler() if args.precision == "amp" else None
    total_steps = max((data.dataloader.num_batches // args.accum_freq) * args.epochs, 1)
    scheduler = build_scheduler(optimizer, args, total_steps)

    if is_master(args):
        logging.info(
            "optimizer=%s lr=%s reference_lr=%s total_batch_size=%s reference_total_batch_size=%s lr_scale=%s wd=%s",
            args.optimizer,
            args.lr,
            getattr(args, "reference_lr", args.lr),
            getattr(args, "total_batch_size", None),
            getattr(args, "reference_total_batch_size", REFERENCE_TOTAL_BATCH_SIZE),
            getattr(args, "lr_scale", 1.0),
            args.wd,
        )

    method = UniRefinerMethod()
    for epoch in range(args.epochs):
        if is_master(args):
            logging.info("Start epoch %d", epoch)
        args.cur_epoch = epoch
        train_one_epoch(student_model, teacher_model, method, data, epoch, optimizer, scaler, scheduler, args)

        if is_master(args) and (
            epoch + 1 == args.epochs or (args.save_freq > 0 and (epoch + 1) % args.save_freq == 0)
        ):
            save_epoch_checkpoint(student_model, os.path.join(args.checkpoint_path, f"epoch_{epoch + 1}.pt"))

    if is_master(args):
        save_final_model(student_model, os.path.join(args.checkpoint_path, "model_final.pt"))
    if args.wandb_run is not None:
        args.wandb_run.finish()
