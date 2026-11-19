"""
Convertit les fichiers Parquet FER2013 en arborescence ImageFolder.

Source attendue (format compétition Kaggle) :
    <src>/train.parquet     colonnes: label (int64), image {bytes, path}
    <src>/public.parquet
    <src>/private.parquet

Produit :
    data/raw/train/<classe>/*.png   (28 709 images)
    data/raw/test/<classe>/*.png    (7 178 images = public + private)

Usage :
    python scripts/prepare_data.py
    python scripts/prepare_data.py --src ../cnn_2/data/raw_parquet --force
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

# Permet d'exécuter le script depuis la racine du dépôt.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constants import CLASSES, IMAGE_SIZE, NUM_CLASSES  # noqa: E402

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def read_split(path: Path) -> list[tuple[int, bytes]]:
    """Retourne la liste (label, bytes PNG) d'un fichier parquet."""
    table = pq.read_table(path, columns=["label", "image"])
    labels = table.column("label").to_pylist()
    images = table.column("image").to_pylist()
    rows: list[tuple[int, bytes]] = []
    for label, image in zip(labels, images):
        raw = image["bytes"]
        if raw is None or raw[:8] != PNG_MAGIC:
            raise ValueError(f"Image non PNG dans {path.name} (label={label})")
        if not 0 <= int(label) < NUM_CLASSES:
            raise ValueError(f"Label invalide {label} dans {path.name}")
        rows.append((int(label), raw))
    return rows


def write_split(rows: list[tuple[int, bytes]], split: str, dst: Path,
                manifest: list[list[str]]) -> Counter:
    counts: Counter = Counter()
    for i, (label, raw) in enumerate(rows):
        class_name = CLASSES[label]
        out_dir = dst / split / class_name
        out_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{split}_{i:05d}.png"
        (out_dir / filename).write_bytes(raw)
        counts[class_name] += 1
        manifest.append([split, f"{split}/{class_name}/{filename}", str(label), class_name])
        if (i + 1) % 5000 == 0:
            print(f"  {split}: {i + 1}/{len(rows)} images écrites...")
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parquet FER2013 -> ImageFolder PNG")
    parser.add_argument("--src", type=Path,
                        default=ROOT.parent / "cnn_2" / "data" / "raw_parquet",
                        help="Dossier contenant train/public/private.parquet")
    parser.add_argument("--dst", type=Path, default=ROOT / "data" / "raw",
                        help="Dossier de sortie (data/raw par défaut)")
    parser.add_argument("--force", action="store_true",
                        help="Écrase les dossiers train/ et test/ existants")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    src, dst = args.src.resolve(), args.dst.resolve()

    required = ["train.parquet", "public.parquet", "private.parquet"]
    missing = [name for name in required if not (src / name).exists()]
    if missing:
        print(f"ERREUR : fichiers manquants dans {src} : {', '.join(missing)}")
        print("Voir data/download.md pour récupérer le dataset.")
        return 1

    if args.force:
        import shutil
        for split in ("train", "test"):
            if (dst / split).exists():
                print(f"Suppression de {dst / split} (--force)")
                shutil.rmtree(dst / split)

    if (dst / "train").exists() and any((dst / "train").iterdir()):
        print(f"{dst / 'train'} existe déjà. Utilisez --force pour régénérer.")
        return 0

    print(f"Source : {src}")
    print(f"Sortie : {dst}")

    print("\nLecture des parquet...")
    train_rows = read_split(src / "train.parquet")
    public_rows = read_split(src / "public.parquet")
    private_rows = read_split(src / "private.parquet")
    test_rows = public_rows + private_rows
    print(f"  train   : {len(train_rows)} images")
    print(f"  public  : {len(public_rows)} images")
    print(f"  private : {len(private_rows)} images")
    print(f"  test    : {len(test_rows)} images (public + private)")

    manifest: list[list[str]] = []
    print("\nÉcriture des PNG...")
    train_counts = write_split(train_rows, "train", dst, manifest)
    test_counts = write_split(test_rows, "test", dst, manifest)

    manifest_path = dst / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "path", "label", "class"])
        writer.writerows(manifest)

    print("\n=== Distribution ===")
    print(f"{'classe':>10s} {'train':>8s} {'test':>8s}")
    for name in CLASSES:
        print(f"{name:>10s} {train_counts[name]:>8d} {test_counts[name]:>8d}")
    print(f"{'TOTAL':>10s} {sum(train_counts.values()):>8d} {sum(test_counts.values()):>8d}")

    total = sum(train_counts.values()) + sum(test_counts.values())
    print(f"\nTotal : {total} images (attendu : 35 887)")
    print(f"Manifeste : {manifest_path}")
    print(f"Taille attendue des images : {IMAGE_SIZE}x{IMAGE_SIZE}, 1 canal")
    return 0 if total == 35887 else 2


if __name__ == "__main__":
    raise SystemExit(main())
