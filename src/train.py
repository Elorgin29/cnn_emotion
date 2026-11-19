"""
Entraînement d'une variante.

Usage :
    python -m src.train --config configs/v1_baseline.yaml
    python -m src.train --config configs/v1_baseline.yaml --epochs 5 --tag smoke
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from src.constants import CLASSES, NUM_CLASSES
from src.dataset import compute_class_weights, get_dataloaders
from src.model import EmotionCNN, count_parameters

DEFAULTS = {
    "seed": 42,
    "data_dir": "data/raw",
    "batch_size": 64,
    "num_workers": 0,
    "val_ratio": 0.1,
    "epochs": 30,
    "early_stop_patience": 6,
    "lr": 0.001,
    "weight_decay": 0.0001,
    "dropout_conv": 0.25,
    "dropout_fc": 0.5,
    "augment": False,
    "use_class_weights": False,
    "num_threads": 6,
}

METRICS_DIR = Path("results/metrics")
MODELS_DIR = Path("results/models")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    unknown = set(cfg) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Clés inconnues dans {path} : {sorted(unknown)}")
    return {**DEFAULTS, **cfg}


def train_one_epoch(model, loader, criterion, optimizer, device, max_batches=None):
    model.train()
    total_loss, seen, preds, labels = 0.0, 0, [], []
    iterator = tqdm(loader, desc="train", leave=False, ncols=90,
                    disable=not sys.stderr.isatty())
    for step, (xb, yb) in enumerate(iterator, start=1):
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        seen += xb.size(0)
        preds.extend(logits.argmax(dim=1).cpu().tolist())
        labels.extend(yb.cpu().tolist())
        if max_batches is not None and step >= max_batches:
            break

    return (total_loss / seen, accuracy_score(labels, preds),
            f1_score(labels, preds, average="macro", zero_division=0))


@torch.no_grad()
def evaluate(model, loader, criterion, device, max_batches=None):
    model.eval()
    total_loss, seen, preds, labels = 0.0, 0, [], []
    iterator = tqdm(loader, desc="eval", leave=False, ncols=90,
                    disable=not sys.stderr.isatty())
    for step, (xb, yb) in enumerate(iterator, start=1):
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item() * xb.size(0)
        seen += xb.size(0)
        preds.extend(logits.argmax(dim=1).cpu().tolist())
        labels.extend(yb.cpu().tolist())
        if max_batches is not None and step >= max_batches:
            break

    return (total_loss / seen, accuracy_score(labels, preds),
            f1_score(labels, preds, average="macro", zero_division=0),
            preds, labels)


def main(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs

    run_name = Path(args.config).stem
    if args.tag:
        run_name = f"{run_name}_{args.tag}"

    print(f"=== Config : {args.config} ===")
    print(json.dumps(cfg, indent=2))

    set_seed(cfg["seed"])
    if cfg["num_threads"]:
        torch.set_num_threads(int(cfg["num_threads"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device} | threads : {torch.get_num_threads()}")

    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=cfg["data_dir"],
        batch_size=cfg["batch_size"],
        augment=cfg["augment"],
        val_ratio=cfg["val_ratio"],
        num_workers=cfg["num_workers"],
        seed=cfg["seed"],
    )
    print(f"Train : {len(train_loader.dataset)} | Val : {len(val_loader.dataset)} | "
          f"Test : {len(test_loader.dataset)}")

    model = EmotionCNN(
        num_classes=NUM_CLASSES,
        dropout_conv=cfg["dropout_conv"],
        dropout_fc=cfg["dropout_fc"],
    ).to(device)
    num_params = count_parameters(model)
    print(f"Paramètres : {num_params:,}")

    # CE pondérée pour l'entraînement, CE standard pour val/test (comparaison fair-play)
    if cfg["use_class_weights"]:
        weights = compute_class_weights(cfg["data_dir"]).to(device)
        train_criterion = nn.CrossEntropyLoss(weight=weights)
        print(f"Class weights : {[round(w, 4) for w in weights.tolist()]}")
    else:
        train_criterion = nn.CrossEntropyLoss()
    eval_criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                                 weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_val_f1, best_val_loss, best_epoch = -1.0, float("inf"), 0
    best_state = None
    patience_counter = 0
    history = []
    start = time.time()

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, train_criterion, optimizer, device, args.max_batches
        )
        val_loss, val_acc, val_f1, _, _ = evaluate(
            model, val_loader, eval_criterion, device, args.max_batches
        )
        lr_now = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        dt = time.time() - t0

        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss), "train_acc": float(train_acc),
            "train_f1": float(train_f1),
            "val_loss": float(val_loss), "val_acc": float(val_acc),
            "val_f1": float(val_f1),
            "lr": float(lr_now), "time_s": float(dt),
        })
        print(f"[Epoch {epoch:02d}] train_loss={train_loss:.4f} acc={train_acc:.4f} "
              f"f1={train_f1:.4f} | val_loss={val_loss:.4f} acc={val_acc:.4f} "
              f"f1={val_f1:.4f} | lr={lr_now:.2e} | {dt:.1f}s")

        if val_f1 > best_val_f1:
            best_val_f1, best_val_loss, best_epoch = val_f1, val_loss, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            print(f"  -> nouveau meilleur modèle (val_f1={val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg["early_stop_patience"]:
                print(f"Early stopping à l'époque {epoch} "
                      f"(meilleure époque : {best_epoch}).")
                break

    total_time = time.time() - start
    print(f"\nTemps total d'entraînement : {total_time / 60:.1f} min")

    if best_state is None:  # sécurité
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    test_loss, test_acc, test_f1, test_preds, test_labels = evaluate(
        model, test_loader, eval_criterion, device
    )
    print("\n=== TEST ===")
    print(f"loss={test_loss:.4f} acc={test_acc:.4f} f1_macro={test_f1:.4f}")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        "run_name": run_name,
        "config_path": str(args.config),
        "config": cfg,
        "classes": CLASSES,
        "num_params": int(num_params),
        "best_epoch": int(best_epoch),
        "best_val_f1": float(best_val_f1),
        "best_val_loss": float(best_val_loss),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "test_f1_macro": float(test_f1),
        "total_train_time_min": float(total_time / 60),
        "epochs_run": len(history),
        "history": history,
        "test_preds": [int(x) for x in test_preds],
        "test_labels": [int(x) for x in test_labels],
    }
    results_path = METRICS_DIR / f"{run_name}_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    pd.DataFrame(history).to_csv(METRICS_DIR / f"{run_name}_history.csv", index=False)
    torch.save(best_state, MODELS_DIR / f"{run_name}_best.pth")

    print(f"Résultats : {results_path}")
    print(f"Historique : {METRICS_DIR / (run_name + '_history.csv')}")
    print(f"Modèle : {MODELS_DIR / (run_name + '_best.pth')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Chemin du fichier YAML")
    parser.add_argument("--epochs", type=int, default=None, help="Surcharge le nombre d'époques")
    parser.add_argument("--tag", default=None, help="Suffixe pour le nom du run")
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Limite le nombre de batchs par époque (tests rapides)")
    main(parser.parse_args())
