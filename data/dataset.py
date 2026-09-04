"""Oxford-IIIT Pet data loading and reproducible splitting."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet


IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for a repeatable experiment."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_transform() -> transforms.Compose:
    """Augmentation used only for the training split."""
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def eval_transform() -> transforms.Compose:
    """Deterministic preprocessing for validation and test."""
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class TransformSubset(Dataset):
    """A subset that applies its own transform to raw PIL images."""

    def __init__(self, dataset: Dataset, indices: np.ndarray, transform: Any) -> None:
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image, label = self.dataset[int(self.indices[index])]
        return self.transform(image), int(label)


def _create_stratified_splits(labels: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    """Create the required 70/15/15 split, stratified by the 37 labels."""
    all_indices = np.arange(len(labels))

    train_indices, temporary_indices = train_test_split(
        all_indices,
        test_size=0.30,
        random_state=seed,
        stratify=labels,
    )
    validation_indices, test_indices = train_test_split(
        temporary_indices,
        test_size=0.50,
        random_state=seed,
        stratify=labels[temporary_indices],
    )

    return {
        "train": np.asarray(train_indices, dtype=np.int64),
        "validation": np.asarray(validation_indices, dtype=np.int64),
        "test": np.asarray(test_indices, dtype=np.int64),
    }


def save_split_indices(splits: dict[str, np.ndarray], path: str | Path, seed: int) -> None:
    """Save split indices so the exact data partition can be inspected later."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {"seed": seed, **{name: values.tolist() for name, values in splits.items()}}
    path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")


def build_pet_dataloaders(
    data_root: str | Path,
    batch_size: int = 32,
    seed: int = 42,
    num_workers: int = 0,
    download: bool = True,
    split_file: str | Path | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str], dict[str, np.ndarray]]:
    """Build 70/15/15 loaders from all 7,349 Oxford-IIIT Pet images.

    The assignment requires a split over all images. Therefore the official
    trainval and test partitions are first combined, then split once with
    seed=42 and stratification. num_workers=0 is intentionally the stable
    default for Kaggle notebooks.
    """
    data_root = Path(data_root)
    raw_trainval = OxfordIIITPet(
        root=data_root,
        split="trainval",
        target_types="category",
        download=download,
    )
    raw_official_test = OxfordIIITPet(
        root=data_root,
        split="test",
        target_types="category",
        download=download,
    )

    all_raw_images = ConcatDataset([raw_trainval, raw_official_test])
    all_labels = np.asarray(raw_trainval._labels + raw_official_test._labels, dtype=np.int64)
    splits = _create_stratified_splits(all_labels, seed)

    expected_sizes = {"train": 5144, "validation": 1102, "test": 1103}
    actual_sizes = {name: len(indices) for name, indices in splits.items()}
    if actual_sizes != expected_sizes:
        raise RuntimeError(f"Unexpected split sizes: {actual_sizes}")

    if split_file is not None:
        save_split_indices(splits, split_file, seed)

    train_dataset = TransformSubset(all_raw_images, splits["train"], train_transform())
    validation_dataset = TransformSubset(all_raw_images, splits["validation"], eval_transform())
    test_dataset = TransformSubset(all_raw_images, splits["test"], eval_transform())

    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": False,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)

    return train_loader, validation_loader, test_loader, list(raw_trainval.classes), splits

