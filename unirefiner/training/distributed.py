"""Distributed training helpers."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def is_global_master(args) -> bool:
    return int(getattr(args, "rank", 0)) == 0


def is_local_master(args) -> bool:
    return int(getattr(args, "local_rank", 0)) == 0


def is_master(args, local: bool = False) -> bool:
    return is_local_master(args) if local else is_global_master(args)


def world_info_from_env() -> tuple[int, int, int]:
    local_rank = 0
    for variable in ("LOCAL_RANK", "MPI_LOCALRANKID", "SLURM_LOCALID", "OMPI_COMM_WORLD_LOCAL_RANK"):
        if variable in os.environ:
            local_rank = int(os.environ[variable])
            break

    global_rank = 0
    for variable in ("RANK", "PMI_RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
        if variable in os.environ:
            global_rank = int(os.environ[variable])
            break

    world_size = 1
    for variable in ("WORLD_SIZE", "PMI_SIZE", "SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE"):
        if variable in os.environ:
            world_size = int(os.environ[variable])
            break
    return local_rank, global_rank, world_size


def init_distributed_device(args) -> torch.device:
    args.distributed = False
    args.world_size = 1
    args.rank = 0
    args.local_rank = 0

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 or "SLURM_PROCID" in os.environ:
        if "SLURM_PROCID" in os.environ:
            args.local_rank, args.rank, args.world_size = world_info_from_env()
            os.environ["LOCAL_RANK"] = str(args.local_rank)
            os.environ["RANK"] = str(args.rank)
            os.environ["WORLD_SIZE"] = str(args.world_size)
        else:
            args.local_rank, _, _ = world_info_from_env()

        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
        )
        args.world_size = dist.get_world_size()
        args.rank = dist.get_rank()
        args.distributed = True

    if torch.cuda.is_available():
        if args.distributed and not args.no_set_device_rank:
            device = f"cuda:{args.local_rank}"
        else:
            device = "cuda:0"
        torch.cuda.set_device(device)
    else:
        device = "cpu"
    args.device = device
    return torch.device(device)


def broadcast_object(args, obj, src: int = 0):
    if not getattr(args, "distributed", False):
        return obj
    objects = [obj] if args.rank == src else [None]
    dist.broadcast_object_list(objects, src=src)
    return objects[0]
