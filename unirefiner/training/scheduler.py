"""Learning-rate schedules."""

from __future__ import annotations

import numpy as np


def assign_learning_rate(optimizer, new_lr: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def warmup_lr(base_lr: float, warmup_length: int, step: int) -> float:
    return base_lr * (step + 1) / warmup_length


def const_lr(optimizer, base_lr: float, warmup_length: int, steps: int):
    def _lr_adjuster(step: int) -> float:
        if step < warmup_length:
            lr = warmup_lr(base_lr, warmup_length, step)
        else:
            lr = base_lr
        assign_learning_rate(optimizer, lr)
        return lr

    return _lr_adjuster


def cosine_lr(optimizer, base_lr: float, warmup_length: int, steps: int, end_lr: float = 0.0):
    def _lr_adjuster(step: int) -> float:
        if step < warmup_length:
            lr = warmup_lr(base_lr, warmup_length, step)
        else:
            decay_steps = steps - warmup_length
            if decay_steps <= 1:
                lr = end_lr
            else:
                elapsed = min(step - warmup_length, decay_steps - 1)
                decay = 0.5 * (1 + np.cos(np.pi * elapsed / (decay_steps - 1)))
                lr = decay * (base_lr - end_lr) + end_lr
        assign_learning_rate(optimizer, lr)
        return lr

    return _lr_adjuster


def build_scheduler(optimizer, args, total_steps: int):
    if args.lr_scheduler == "cosine":
        return cosine_lr(optimizer, args.lr, args.warmup, total_steps, end_lr=args.lr_last)
    if args.lr_scheduler == "const":
        return const_lr(optimizer, args.lr, args.warmup, total_steps)
    raise ValueError(f"Unsupported scheduler: {args.lr_scheduler}")
