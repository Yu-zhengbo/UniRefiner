"""Random seed helpers."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_random_seed(seed: int, rank: int = 0) -> None:
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)
