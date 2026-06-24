"""
Training dataset implementation.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from multiprocessing import Value
from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from .transforms import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD, build_image_transform


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def default_background_path() -> str:
    project_root = Path(__file__).resolve().parents[2]
    return str(project_root / "assets" / "backgrounds" / "fixed_reference.png")


class SharedEpoch:
    """Small shared epoch container for distributed samplers."""

    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value("i", epoch)

    def set_value(self, epoch: int) -> None:
        self.shared_epoch.value = int(epoch)

    def get_value(self) -> int:
        return int(self.shared_epoch.value)


@dataclass(slots=True)
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler | None = None
    shared_epoch: SharedEpoch | None = None

    def set_epoch(self, epoch: int) -> None:
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


class RecursiveImageDataset(Dataset):
    """Recursively load training images and pair each image with one background."""

    def __init__(
        self,
        image_root: str | Path | Sequence[str | Path],
        background_path: str | Path,
        transform: Callable,
        *,
        image_extensions: set[str] | None = None,
    ) -> None:
        self.image_roots = self._normalize_image_roots(image_root)
        self.image_root = self.image_roots[0] if len(self.image_roots) == 1 else self.image_roots
        self.background_path = Path(background_path)
        self.transform = transform
        self.image_extensions = image_extensions or IMAGE_EXTENSIONS

        for root in self.image_roots:
            if not root.exists():
                raise FileNotFoundError(f"Training image root does not exist: {root}")
        if not self.background_path.exists():
            raise FileNotFoundError(f"Background image does not exist: {self.background_path}")

        self.samples = []
        for root in self.image_roots:
            root_samples = sorted(path for path in root.rglob("*") if path.suffix.lower() in self.image_extensions)
            if not root_samples:
                raise RuntimeError(f"No images found under {root}")
            self.samples.extend(root_samples)
        if not self.samples:
            raise RuntimeError(f"No images found under {self.image_roots}")

    @staticmethod
    def _normalize_image_roots(image_root: str | Path | Sequence[str | Path]) -> tuple[Path, ...]:
        if isinstance(image_root, (str, Path)):
            roots = (Path(image_root),)
        else:
            roots = tuple(Path(root) for root in image_root)
        if not roots:
            raise ValueError("`train_image_root` must contain at least one path.")
        return roots

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: Path) -> Image.Image:
        with Image.open(path) as image:
            return image.copy()

    def __getitem__(self, index: int):
        image_path = self.samples[index]
        try:
            image = self._load_image(image_path)
        except Exception:
            fallback_index = random.randrange(len(self.samples))
            return self.__getitem__(fallback_index)

        background = self._load_image(self.background_path)
        return self.transform(image), self.transform(background)


def _get_arg(args, name: str, default=None):
    if isinstance(args, dict):
        return args.get(name, default)
    return getattr(args, name, default)


def build_transform(image_size: int, mean=None, std=None):
    return build_image_transform(image_size, mean=mean, std=std)


def build_dataset(args, mean=None, std=None) -> RecursiveImageDataset:
    image_size = _get_arg(args, "image_size")
    image_root = _get_arg(args, "train_image_root")
    background_path = _get_arg(args, "background_image_path", None)
    if background_path is None:
        background_path = default_background_path()
    return RecursiveImageDataset(
        image_root=image_root,
        background_path=background_path,
        transform=build_transform(image_size, mean=mean, std=std),
    )


def build_data(args, mean=None, std=None) -> DataInfo:
    dataset = build_dataset(args, mean=mean, std=std)
    distributed = bool(_get_arg(args, "distributed", False))
    sampler = DistributedSampler(dataset) if distributed else None
    dataloader = DataLoader(
        dataset,
        batch_size=int(_get_arg(args, "batch_size")),
        num_workers=int(_get_arg(args, "workers", 0)),
        pin_memory=True,
        sampler=sampler,
        drop_last=True,
    )
    dataloader.num_samples = len(dataset)
    dataloader.num_batches = len(dataloader)
    return DataInfo(dataloader=dataloader, sampler=sampler, shared_epoch=SharedEpoch())
