from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class TinyImageNetValDataset(Dataset):
    """Tiny ImageNet validation split with labels from val_annotations.txt."""

    def __init__(
        self,
        root: str | Path,
        class_to_idx: Dict[str, int],
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.root = Path(root)
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.val_dir = self.root / "val"
        self.image_dir = self.val_dir / "images"
        annotations_path = self.val_dir / "val_annotations.txt"

        if not annotations_path.exists():
            raise FileNotFoundError(
                f"Missing Tiny ImageNet validation annotations: {annotations_path}"
            )

        samples = []
        with annotations_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                image_name, class_id = parts[0], parts[1]
                if class_id not in class_to_idx:
                    raise ValueError(
                        f"Validation class {class_id!r} is not present in training classes."
                    )
                image_path = self.image_dir / image_name
                samples.append((image_path, class_to_idx[class_id]))

        self.samples = sorted(samples, key=lambda item: item[0].name)
        self.targets = [target for _, target in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        image_path, target = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def build_transforms(image_size: int = 64) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def build_train_dataset(
    data_root: str | Path,
    image_size: int = 64,
) -> datasets.ImageFolder:
    data_root = Path(data_root)
    train_dir = data_root / "train"
    if not train_dir.exists():
        raise FileNotFoundError(f"Missing Tiny ImageNet train directory: {train_dir}")

    train_transform, _ = build_transforms(image_size=image_size)
    return datasets.ImageFolder(train_dir, transform=train_transform)


def build_val_dataset(
    data_root: str | Path,
    class_to_idx: Dict[str, int],
    image_size: int = 64,
) -> TinyImageNetValDataset:
    _, eval_transform = build_transforms(image_size=image_size)
    return TinyImageNetValDataset(data_root, class_to_idx, transform=eval_transform)


def make_dataloaders(
    data_root: str | Path,
    batch_size: int,
    num_workers: int,
    image_size: int = 64,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader, Dict[str, int]]:
    train_dataset = build_train_dataset(data_root, image_size=image_size)
    val_dataset = build_val_dataset(
        data_root,
        class_to_idx=train_dataset.class_to_idx,
        image_size=image_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader, train_dataset.class_to_idx
