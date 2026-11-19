"""
Génère les éléments LaTeX du rapport à partir des résultats réels.

Produit :
    reports/generated/macros.tex          (nombres, noms et écarts vs V1)
    reports/generated/results_table.tex   (tableau de résultats)
    reports/generated/per_class_table.tex (F1 par classe)
    reports/generated/per_class_macros.tex
    reports/generated/numbers.json        (chiffres pour la discussion)

Usage :
    python scripts/make_report_assets.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
METRICS = ROOT / "results" / "metrics"
OUT = ROOT / "reports" / "generated"

ORDER = ["v1_baseline", "v2_augmentation", "v3_classweights", "v4_regularization"]
SUFFIX = {"v1_baseline": "One", "v2_augmentation": "Two",
          "v3_classweights": "Three", "v4_regularization": "Four"}
LABEL = {
    "v1_baseline": "V1 --- baseline",
    "v2_augmentation": "V2 --- augmentation",
    "v3_classweights": "V3 --- class weights",
    "v4_regularization": "V4 --- régularisation",
}
HEADER = "% Généré par scripts/make_report_assets.py --- ne pas éditer à la main."


def num(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def signed(value: float, digits: int = 3) -> str:
    """Nombre signé en notation française (+0,012), sûr en mode texte LaTeX."""
    text = f"{value:+.{digits}f}".replace(".", ",")
    return text.replace("-", "$-$")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary_path = METRICS / "summary.csv"
    if not summary_path.exists():
        raise SystemExit("results/metrics/summary.csv introuvable : lancez d'abord src.analyze")

    df = pd.read_csv(summary_path).set_index("variant")
    best = df["test_f1_macro"].idxmax()
    best_row = df.loc[best]

    lines = [
        HEADER,
        f"\\newcommand{{\\BestRun}}{{{best}}}",
        f"\\newcommand{{\\BestName}}{{{LABEL.get(best, best)}}}",
        f"\\newcommand{{\\BestAcc}}{{{num(best_row['test_acc'])}}}",
        f"\\newcommand{{\\BestF}}{{{num(best_row['test_f1_macro'])}}}",
        f"\\newcommand{{\\BestEpoch}}{{{int(best_row['best_epoch'])}}}",
        f"\\newcommand{{\\BestTime}}{{{num(best_row['train_time_min'], 1)}}}",
        "\\newcommand{\\BestParams}{"
        + f"{int(best_row['num_params']):,}".replace(",", "\\,") + "}",
    ]

    for run in ORDER:
        suffix = SUFFIX[run]
        if run not in df.index:
            for key in ("Acc", "F", "Time", "Epoch", "GapAcc", "GapF"):
                lines.append(f"\\newcommand{{\\{key}{suffix}}}{{--}}")
            continue
        row = df.loc[run]
        lines.append(f"\\newcommand{{\\Acc{suffix}}}{{{num(row['test_acc'])}}}")
        lines.append(f"\\newcommand{{\\F{suffix}}}{{{num(row['test_f1_macro'])}}}")
        lines.append(f"\\newcommand{{\\Time{suffix}}}{{{num(row['train_time_min'], 1)}}}")
        lines.append(f"\\newcommand{{\\Epoch{suffix}}}{{{int(row['best_epoch'])}}}")

        # Écart signé par rapport à V1, en points de métrique.
        if "v1_baseline" in df.index and run != "v1_baseline":
            d_acc = row["test_acc"] - df.loc["v1_baseline", "test_acc"]
            d_f1 = row["test_f1_macro"] - df.loc["v1_baseline", "test_f1_macro"]
            lines.append(f"\\newcommand{{\\GapAcc{suffix}}}{{{signed(d_acc)}}}")
            lines.append(f"\\newcommand{{\\GapF{suffix}}}{{{signed(d_f1)}}}")
        else:
            lines.append(f"\\newcommand{{\\GapAcc{suffix}}}{{--}}")
            lines.append(f"\\newcommand{{\\GapF{suffix}}}{{--}}")

    (OUT / "macros.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Écrit : {OUT / 'macros.tex'}")

    # --- Tableau de résultats -------------------------------------------------
    rows = []
    for run in ORDER:
        if run not in df.index:
            continue
        row = df.loc[run]
        rows.append(
            f"{LABEL[run]} & {num(row['test_acc'])} & {num(row['test_f1_macro'])} & "
            f"{int(row['best_epoch'])} & {num(row['train_time_min'], 1)} \\\\"
        )

    table = [
        HEADER,
        "\\begin{table}[H]",
        "\\centering",
        "\\small",
        "\\caption{Résultats sur le jeu de test (\\TestSize{} images). Le temps est le",
        "temps total d'entraînement mesuré sur CPU.}",
        "\\label{tab:resultats}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Variante & Accuracy & F1 macro & Meilleure époque & Temps (min) \\\\",
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ]
    (OUT / "results_table.tex").write_text("\n".join(table), encoding="utf-8")
    print(f"Écrit : {OUT / 'results_table.tex'}")

    # --- F1 par classe -------------------------------------------------------
    from sklearn.metrics import f1_score

    from src.constants import CLASSES

    per_class: dict[str, list[float]] = {}
    for run in ORDER:
        path = METRICS / f"{run}_results.json"
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        per_class[run] = f1_score(
            payload["test_labels"], payload["test_preds"], average=None,
            labels=list(range(7)), zero_division=0,
        ).tolist()

    runs = [run for run in ORDER if run in per_class]
    header = "Classe & " + " & ".join(
        f"V{ORDER.index(run) + 1}" for run in runs
    ) + " \\\\"

    body = []
    for index, name in enumerate(CLASSES):
        cells = [num(per_class[run][index], 2) for run in runs]
        body.append(f"{name} & " + " & ".join(cells) + " \\\\")
    if per_class:
        means = [num(sum(per_class[run]) / len(CLASSES), 2) for run in runs]
        body.append("\\midrule")
        body.append("Moyenne (F1 macro) & " + " & ".join(means) + " \\\\")

    per_class_tex = [
        HEADER,
        "\\begin{table}[H]",
        "\\centering",
        "\\small",
        "\\caption{F1 par classe sur le jeu de test (classes par ordre alphabétique).}",
        "\\label{tab:perclass}",
        "\\begin{tabular}{l" + "c" * len(runs) + "}",
        "\\toprule",
        header,
        "\\midrule",
        *body,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ]
    (OUT / "per_class_table.tex").write_text("\n".join(per_class_tex), encoding="utf-8")
    print(f"Écrit : {OUT / 'per_class_table.tex'}")

    # --- Macros ciblées sur les classes critiques ----------------------------
    targets = {"Disgust": 1, "Fear": 2, "Happy": 3}
    class_lines = [HEADER]
    for label, index in targets.items():
        for run in ORDER:
            suffix = SUFFIX[run]
            if run in per_class:
                value = num(per_class[run][index], 2)
            else:
                value = "--"
            class_lines.append(f"\\newcommand{{\\F{label}{suffix}}}{{{value}}}")
    (OUT / "per_class_macros.tex").write_text("\n".join(class_lines) + "\n", encoding="utf-8")
    print(f"Écrit : {OUT / 'per_class_macros.tex'}")

    # --- Chiffres pour la discussion ----------------------------------------
    payload = {
        "best": best,
        "best_acc_gain_vs_v1": float(best_row["test_acc"] - df.loc["v1_baseline", "test_acc"])
        if "v1_baseline" in df.index else None,
        "best_f_gain_vs_v1": float(best_row["test_f1_macro"] - df.loc["v1_baseline", "test_f1_macro"])
        if "v1_baseline" in df.index else None,
        "epochs": {run: int(df.loc[run, "epochs_run"]) for run in df.index},
        "per_class_f1": {run: [round(v, 4) for v in values]
                         for run, values in per_class.items()},
    }
    (OUT / "numbers.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
