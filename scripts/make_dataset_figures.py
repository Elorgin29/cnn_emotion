"""
Génère les figures descriptives du dataset (distribution + exemples).

Usage :
    python scripts/make_dataset_figures.py
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

import matplotlib

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constants import CLASSES  # noqa: E402

FIG_DIR = ROOT / "results" / "figures"
RANDOM = np.random.default_rng(0)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    train_counts = Counter()
    test_counts = Counter()
    for split, counter in (("train", train_counts), ("test", test_counts)):
        for idx, name in enumerate(CLASSES):
            counter[idx] = len(list((ROOT / "data" / "raw" / split / name).glob("*.png")))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, counts, title in ((axes[0], train_counts, "Train (28 709)"),
                              (axes[1], test_counts, "Test (7 178)")):
        values = [counts[i] for i in range(len(CLASSES))]
        bars = ax.bar(CLASSES, values, color="#4C72B0")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=40)
        ax.bar_label(bars, fontsize=7, padding=2)
        ax.margins(y=0.12)
    fig.suptitle("Distribution des classes dans FER2013")
    fig.tight_layout()
    path = FIG_DIR / "dataset_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure : {path}")

    files = []
    labels = []
    for idx, name in enumerate(CLASSES):
        candidates = sorted((ROOT / "data" / "raw" / "train" / name).glob("*.png"))
        if candidates:
            files.append(candidates[int(RANDOM.integers(len(candidates)))])
            labels.append(idx)

    fig, axes = plt.subplots(1, len(files), figsize=(2 * len(files), 2.6))
    for ax, path, label in zip(axes, files, labels):
        ax.imshow(plt.imread(path), cmap="gray")
        ax.set_title(CLASSES[label], fontsize=10)
        ax.axis("off")
    fig.suptitle("Un exemple par classe (48x48, niveaux de gris)")
    fig.tight_layout()
    out = FIG_DIR / "dataset_samples.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure : {out}")


if __name__ == "__main__":
    main()
