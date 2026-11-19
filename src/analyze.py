"""
Analyse des résultats :

- courbes d'apprentissage (loss / accuracy / F1)
- matrices de confusion (brutes et normalisées)
- F1 par classe
- tableau comparatif des variantes (summary.csv)
- rapport de classification texte
- paires de classes les plus confondues

Usage :
    python -m src.analyze
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

# Cache matplotlib dans un dossier inscriptible du projet (Windows : AppData
# peut être protégé en écriture).
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torchvision import datasets

from src.constants import CLASSES

FIG_DIR = Path("results/figures")
MET_DIR = Path("results/metrics")
FIG_DIR.mkdir(parents=True, exist_ok=True)
MET_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")
COLORS = ["#4C72B0", "#DD8452"]


def plot_learning_curves(results: dict, run_name: str) -> None:
    h = pd.DataFrame(results["history"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(h["epoch"], h["train_loss"], label="train", color=COLORS[0])
    axes[0].plot(h["epoch"], h["val_loss"], label="val", color=COLORS[1])
    axes[0].set_title("Perte (cross-entropy)")
    axes[0].set_xlabel("Époque")
    axes[0].legend()

    axes[1].plot(h["epoch"], h["train_acc"], label="train", color=COLORS[0])
    axes[1].plot(h["epoch"], h["val_acc"], label="val", color=COLORS[1])
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Époque")
    axes[1].legend()

    axes[2].plot(h["epoch"], h["train_f1"], label="train", color=COLORS[0])
    axes[2].plot(h["epoch"], h["val_f1"], label="val", color=COLORS[1])
    axes[2].set_title("F1 macro")
    axes[2].set_xlabel("Époque")
    axes[2].legend()

    best = results.get("best_epoch")
    if best:
        for ax in axes:
            ax.axvline(best, color="grey", linestyle="--", linewidth=1, alpha=0.7)

    fig.suptitle(f"Courbes d'apprentissage — {run_name}", y=1.02)
    fig.tight_layout()
    path = FIG_DIR / f"{run_name}_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure : {path}")


def plot_confusion(results: dict, run_name: str) -> dict:
    y_true = np.asarray(results["test_labels"])
    y_pred = np.asarray(results["test_preds"])
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=CLASSES, yticklabels=CLASSES, ax=axes[0])
    axes[0].set_title("Effectifs")
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
                xticklabels=CLASSES, yticklabels=CLASSES, ax=axes[1])
    axes[1].set_title("Proportions (par ligne)")
    for ax in axes:
        ax.set_xlabel("Prédit")
        ax.set_ylabel("Vrai")
    fig.suptitle(f"Matrice de confusion — {run_name}", y=1.02)
    fig.tight_layout()
    path = FIG_DIR / f"{run_name}_confusion.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure : {path}")

    # Paires les plus confondues (hors diagonale)
    pairs = []
    for i, true_name in enumerate(CLASSES):
        for j, pred_name in enumerate(CLASSES):
            if i != j and cm[i, j] > 0:
                pairs.append({
                    "true": true_name,
                    "pred": pred_name,
                    "count": int(cm[i, j]),
                    "share_of_true": float(cm[i, j] / cm[i].sum()),
                })
    pairs.sort(key=lambda x: x["count"], reverse=True)
    return {"confusion_matrix": cm.tolist(), "top_confusions": pairs[:10]}


def plot_per_class_f1(results: dict, run_name: str) -> None:
    report = classification_report(
        results["test_labels"], results["test_preds"],
        labels=list(range(len(CLASSES))), target_names=CLASSES,
        output_dict=True, zero_division=0,
    )
    f1s = [report[c]["f1-score"] for c in CLASSES]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(CLASSES, f1s, color="#4C72B0")
    ax.axhline(results["test_f1_macro"], color="#C44E52", linestyle="--",
               label=f"F1 macro = {results['test_f1_macro']:.3f}")
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1-score")
    ax.set_title(f"F1 par classe — {run_name}")
    ax.legend()
    for bar, value in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}",
                ha="center", fontsize=8)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    path = FIG_DIR / f"{run_name}_per_class_f1.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure : {path}")


def plot_confusion_grid(confusions: dict[str, np.ndarray]) -> None:
    """Grille des matrices de confusion normalisées (une par variante)."""
    names = list(confusions)
    if len(names) < 2:
        return
    cols = 2
    rows = math.ceil(len(names) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6.2 * cols, 4.8 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, name in zip(axes, names):
        cm = confusions[name]
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
                    cbar=False, xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
        ax.set_title(name)
        ax.set_xlabel("Prédit")
        ax.set_ylabel("Vrai")
    for ax in axes[len(names):]:
        ax.axis("off")

    fig.suptitle("Matrices de confusion normalisées (test)", y=1.01)
    fig.tight_layout()
    path = FIG_DIR / "confusion_grid.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure : {path}")


def plot_error_examples(results: dict, run_name: str, n: int = 12,
                        data_dir: str = "data/raw/test") -> None:
    """Galerie d'images mal classées (vrai vs prédit)."""
    folder = datasets.ImageFolder(data_dir, transform=None)
    labels = [label for _, label in folder.samples]
    if labels != list(results["test_labels"]):
        print("  (galerie d'erreurs ignorée : ordre des labels différent)")
        return

    preds = results["test_preds"]
    wrong = [i for i, (true, pred) in enumerate(zip(labels, preds)) if true != pred]
    if not wrong:
        return

    step = max(1, len(wrong) // n)
    chosen = wrong[::step][:n]
    cols = 4
    rows = math.ceil(len(chosen) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 3.2 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, index in zip(axes, chosen):
        path, true = folder.samples[index]
        with Image.open(path) as img:
            array = np.asarray(img.convert("L"))
        ax.imshow(array, cmap="gray")
        ax.axis("off")
        ax.set_title(f"vrai : {CLASSES[true]}\nprédit : {CLASSES[preds[index]]}",
                     fontsize=8, color="#B22222")
    for ax in axes[len(chosen):]:
        ax.axis("off")

    fig.suptitle(f"Exemples mal classés — {run_name} ({len(wrong)} erreurs)", y=1.0)
    fig.tight_layout()
    path = FIG_DIR / f"{run_name}_errors.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure : {path}")


def plot_comparison(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    x = np.arange(len(df))
    width = 0.35

    axes[0].bar(x - width / 2, df["test_acc"], width, label="Accuracy", color="#4C72B0")
    axes[0].bar(x + width / 2, df["test_f1_macro"], width, label="F1 macro", color="#DD8452")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df["variant"], rotation=20, ha="right")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Performance sur le test")
    axes[0].legend()

    axes[1].bar(x, df["train_time_min"], color="#55A868")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df["variant"], rotation=20, ha="right")
    axes[1].set_title("Temps d'entraînement (min)")
    axes[1].set_ylabel("minutes")

    fig.suptitle("Comparaison des variantes", y=1.02)
    fig.tight_layout()
    path = FIG_DIR / "comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure : {path}")


def main() -> None:
    files = sorted(MET_DIR.glob("*_results.json"))
    if not files:
        print("Aucun résultat. Lancez d'abord python -m src.train.")
        return

    rows = []
    all_confusions = {}
    confusion_matrices = {}
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            results = json.load(f)
        run_name = results.get("run_name", path.stem.replace("_results", ""))
        print(f"\n--- Analyse : {run_name} ---")

        plot_learning_curves(results, run_name)
        analysis = plot_confusion(results, run_name)
        plot_per_class_f1(results, run_name)
        plot_error_examples(results, run_name)
        all_confusions[run_name] = analysis["top_confusions"]
        confusion_matrices[run_name] = np.asarray(analysis["confusion_matrix"])

        report_txt = classification_report(
            results["test_labels"], results["test_preds"],
            labels=list(range(len(CLASSES))), target_names=CLASSES,
            digits=4, zero_division=0,
        )
        (MET_DIR / f"{run_name}_classification_report.txt").write_text(
            report_txt, encoding="utf-8"
        )
        print(report_txt)

        rows.append({
            "variant": run_name,
            "test_acc": results["test_acc"],
            "test_f1_macro": results["test_f1_macro"],
            "best_val_f1": results["best_val_f1"],
            "best_epoch": results.get("best_epoch", ""),
            "epochs_run": results.get("epochs_run", len(results["history"])),
            "train_time_min": results["total_train_time_min"],
            "num_params": results["num_params"],
            "augment": results["config"].get("augment"),
            "class_weights": results["config"].get("use_class_weights"),
        })

    df = pd.DataFrame(rows).sort_values("test_f1_macro", ascending=False)
    df.to_csv(MET_DIR / "summary.csv", index=False)

    with open(MET_DIR / "error_analysis.json", "w", encoding="utf-8") as f:
        json.dump(all_confusions, f, indent=2, ensure_ascii=False)

    print("\n=== Tableau comparatif ===")
    print(df.to_string(index=False))
    print(f"\nSauvegardé : {MET_DIR / 'summary.csv'}")

    if len(df) > 1:
        plot_confusion_grid(confusion_matrices)
        plot_comparison(df.reset_index(drop=True))

    best = df.iloc[0]
    print(f"\nMeilleure variante : {best['variant']} "
          f"(F1 macro = {best['test_f1_macro']:.4f}, accuracy = {best['test_acc']:.4f})")


if __name__ == "__main__":
    main()
