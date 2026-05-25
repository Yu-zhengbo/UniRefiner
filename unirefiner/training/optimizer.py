"""Optimizer construction and hidden-reference LR scaling."""

from __future__ import annotations

from torch import optim


REFERENCE_TOTAL_BATCH_SIZE = 16


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def default_optimizer_params(model_name: str) -> dict[str, float]:
    lower_name = model_name.lower()
    if "vit" in lower_name:
        return {"beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    return {"beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8}


def apply_linear_lr_scaling(args) -> None:
    args.reference_lr = args.lr
    args.reference_lr_last = args.lr_last
    args.total_batch_size = int(args.batch_size) * int(getattr(args, "world_size", 1)) * int(args.accum_freq)
    args.reference_total_batch_size = REFERENCE_TOTAL_BATCH_SIZE
    args.lr_scale = args.total_batch_size / REFERENCE_TOTAL_BATCH_SIZE
    args.lr = args.reference_lr * args.lr_scale
    if args.lr_last is not None:
        args.lr_last = args.reference_lr_last * args.lr_scale


def fill_default_optimizer_params(args) -> None:
    defaults = default_optimizer_params(args.model)
    args.beta1 = args.beta1 if args.beta1 is not None else defaults["beta1"]
    args.beta2 = args.beta2 if args.beta2 is not None else defaults["beta2"]
    args.eps = args.eps if args.eps is not None else defaults["eps"]


def _parameter_groups_for_adamw(model, weight_decay: float):
    named_parameters = list(model.named_parameters())

    def exclude(name, parameter) -> bool:
        return parameter.ndim < 2 or "bn" in name or "ln" in name or "bias" in name or "logit" in name

    gain_or_bias_params = [
        parameter
        for name, parameter in named_parameters
        if exclude(name, parameter) and parameter.requires_grad
    ]
    rest_params = [
        parameter
        for name, parameter in named_parameters
        if not exclude(name, parameter) and parameter.requires_grad
    ]
    return [
        {"params": gain_or_bias_params, "weight_decay": 0.0},
        {"params": rest_params, "weight_decay": weight_decay},
    ]


def build_optimizer(model, args):
    model_to_optimize = unwrap_model(model)

    if args.optimizer == "adamw":
        return optim.AdamW(
            _parameter_groups_for_adamw(model, args.wd),
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
        )

    if args.optimizer == "loraplus":
        if not hasattr(model_to_optimize, "peft_config"):
            raise ValueError("optimizer.name=loraplus requires a PEFT LoRA student model. Please set model.lora.")
        from peft.optimizers import create_loraplus_optimizer

        loraplus_weight_decay = args.loraplus_weight_decay if args.loraplus_weight_decay is not None else args.wd
        return create_loraplus_optimizer(
            model_to_optimize,
            optim.AdamW,
            lr=args.lr,
            loraplus_lr_ratio=args.loraplus_lr_ratio,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
            loraplus_weight_decay=loraplus_weight_decay,
        )

    raise ValueError(f"Unsupported optimizer: {args.optimizer}")
