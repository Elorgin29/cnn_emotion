"""
Chargement et prétraitement du dataset FER2013.

Les images PNG (format ImageFolder) sont converties une seule fois en tableaux
NumPy dans ``data/processed/``. Les époques suivantes lisent ce cache
directement en mémoire, ce qui évite de décoder 28 000 PNG à chaque époque
(gain d'un facteur ~3 sur CPU).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets
from torchvision.transforms import v2

from src.constants import CLASSES, IMAGE_SIZE, NUM_CLASSES

__all__ = [
    "CLASSES",
    "NUM_CLASSES",
    "IMAGE_SIZE",
    "FERDataset",
    "get_transforms",
    "get_dataloaders",
    "compute_class_weights",
]

MEAN = 0.5
STD = 0.5


def get_transforms(augment: bool = False):
    """Augmentation légère appliquée à un tenseur float dans [0, 1]."""
    if not augment:
        return None
    return v2.Compose([
        v2.RandomRotation(10),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness=0.1),
    ])


class FERDataset(Dataset):
    """Dataset en mémoire : tableaux uint8 (N, 48, 48) + labels."""

    def __init__(self, images: np.ndarray, labels: np.ndarray,
                 augment: bool = False):
        assert images.shape[1:] == (IMAGE_SIZE, IMAGE_SIZE), images.shape
        self.images = images
        self.labels = labels.astype(np.int64, copy=False)
        self.augment = get_transforms(augment)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        image = torch.from_numpy(self.images[index]).unsqueeze(0)  # 1x48x48 uint8
        image = image.float().div_(255.0)
        if self.augment is not None:
            image = self.augment(image)
        image = image.sub_(MEAN).div_(STD)
        return image, int(self.labels[index])

    def subset(self, indices) -> "FERDataset":
        return FERDataset(self.images[indices], self.labels[indices],
                          augment=self.augment is not None)


def _load_images(samples) -> np.ndarray:
    images = np.empty((len(samples), IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    for i, (path, _) in enumerate(samples):
        with Image.open(path) as img:
            array = np.asarray(img.convert("L"), dtype=np.uint8)
        if array.shape != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"Taille inattendue pour {path} : {array.shape}")
        images[i] = array
    return images


def _processed_dir(data_dir: str) -> Path:
    return Path(data_dir).resolve().parent / "processed"


def _load_split(data_dir: str, split: str):
    """Charge (images, labels) depuis le cache, ou le construit."""
    folder = datasets.ImageFolder(os.path.join(data_dir, split))
    expected = {c: i for i, c in enumerate(CLASSES)}
    if folder.class_to_idx != expected:
        raise ValueError(f"Ordre des classes inattendu : {folder.class_to_idx}")

    labels = np.array([label for _, label in folder.samples], dtype=np.int64)
    processed = _processed_dir(data_dir)
    processed.mkdir(parents=True, exist_ok=True)
    images_path = processed / f"{split}_images.npy"
    labels_path = processed / f"{split}_labels.npy"
    meta_path = processed / f"{split}_meta.json"

    if images_path.exists() and labels_path.exists() and meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("count") == len(labels):
            images = np.load(images_path)
            cached_labels = np.load(labels_path)
            print(f"  {split}: cache chargé ({images.shape[0]} images)")
            return images, cached_labels

    print(f"  {split}: construction du cache ({len(labels)} images)...")
    t0 = time.time()
    images = _load_images(folder.samples)
    np.save(images_path, images)
    np.save(labels_path, labels)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"count": int(len(labels)), "shape": list(images.shape)}, f)
    print(f"  {split}: cache écrit en {time.time() - t0:.1f}s -> {images_path}")
    return images, labels


def get_dataloaders(data_dir: str = "data/raw",
                    batch_size: int = 64,
                    augment: bool = False,
                    val_ratio: float = 0.1,
                    num_workers: int = 0,
                    seed: int = 42):
    """Construit les loaders train / val / test.

    - ``data_dir`` doit contenir ``train/`` et ``test/`` (ImageFolder).
    - Le train est découpé en train/val de façon stratifiée.
    - La validation et le test ne sont jamais augmentés.
    """
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio doit être dans ]0, 1[")

    train_images, train_labels = _load_split(data_dir, "train")
    test_images, test_labels = _load_split(data_dir, "test")

    indices = np.arange(len(train_labels))
    train_idx, val_idx = train_test_split(
        indices, test_size=val_ratio, stratify=train_labels, random_state=seed
    )

    train_ds = FERDataset(train_images[train_idx], train_labels[train_idx],
                          augment=augment)
    val_ds = FERDataset(train_images[val_idx], train_labels[val_idx],
                        augment=False)
    test_ds = FERDataset(test_images, test_labels, augment=False)

    generator = torch.Generator()
    generator.manual_seed(seed)

    common = dict(num_workers=num_workers, pin_memory=False)
    if num_workers > 0:
        common["persistent_workers"] = True

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              generator=generator, **common)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common)

    return train_loader, val_loader, test_loader


def compute_class_weights(data_dir: str = "data/raw") -> torch.Tensor:
    """Poids inversement proportionnels à la fréquence de chaque classe."""
    _, labels = _load_split(data_dir, "train")
    targets = torch.from_numpy(labels).long()
    counts = torch.bincount(targets, minlength=NUM_CLASSES).float()
    if (counts == 0).any():
        raise ValueError(f"Classes vides détectées : {counts.tolist()}")
    weights = counts.sum() / (NUM_CLASSES * counts)
    return weights


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()
    print(f"Train batches : {len(train_loader)} ({len(train_loader.dataset)} images)")
    print(f"Val batches   : {len(val_loader)} ({len(val_loader.dataset)} images)")
    print(f"Test batches  : {len(test_loader)} ({len(test_loader.dataset)} images)")
    xb, yb = next(iter(train_loader))
    print(f"Shape batch   : {tuple(xb.shape)} | labels : {tuple(yb.shape)}")
    print(f"Min/Max pixel : {xb.min():.3f} / {xb.max():.3f}")
    print(f"Class weights : {[round(w, 4) for w in compute_class_weights().tolist()]}")
