# CNN léger pour la reconnaissance des émotions faciales (FER2013)

Projet final — *Réseaux de neurones : vision par ordinateur*.

Reconnaissance de 7 émotions faciales sur **FER2013** avec un **CNN léger
(683 751 paramètres)**, entraînable **sur CPU uniquement**.
Le projet compare quatre variantes d'entraînement et analyse les erreurs du
meilleur modèle.

---

## 1. Objectifs et méthode

### Dataset

**FER2013** : 35 887 images en niveaux de gris, 48×48 pixels, 7 classes :

| Classe | Train | Test |
|---|---:|---:|
| angry | 3 995 | 958 |
| disgust | 436 | 111 |
| fear | 4 097 | 1 024 |
| happy | 7 215 | 1 774 |
| neutral | 4 830 | 1 247 |
| sad | 3 171 | 831 |
| surprise | 4 965 | 1 233 |
| **Total** | **28 709** | **7 178** |

Le train est découpé en **90 % entraînement / 10 % validation** (stratifié,
`seed=42`). Le jeu de test officiel (7 178 images) n'est utilisé qu'une fois,
pour l'évaluation finale.

Le dataset est fortement déséquilibré : `disgust` ne représente que 1,5 % du
train, contre 25 % pour `happy` (**ratio 16,5×**).

### Modèle

CNN volontairement compact :

```
Entrée 1×48×48
  ├─ Conv 3×3 (32) + BN + ReLU + MaxPool 2 + Dropout
  ├─ Conv 3×3 (64) + BN + ReLU + MaxPool 2 + Dropout
  ├─ Conv 3×3 (128) + BN + ReLU + MaxPool 2 + Dropout
  ├─ Flatten (128×6×6 = 4 608)
  ├─ Linear 128 + ReLU + Dropout
  └─ Linear 7
```

**683 751 paramètres** entraînables. Aucune donnée externe, aucun
pré-entraînement : le modèle est appris from scratch sur CPU.

### Contribution personnelle : quatre variantes comparées

| Variante | Augmentation | Class weights | Dropout (conv / FC) |
|---|:---:|:---:|:---:|
| **V1** — baseline | ✗ | ✗ | 0,25 / 0,50 |
| **V2** — augmentation | ✓ | ✗ | 0,25 / 0,50 |
| **V3** — class weights | ✓ | ✓ | 0,25 / 0,50 |
| **V4** — régularisation | ✓ | ✓ | 0,30 / 0,60 |

- **Augmentation** : rotation ±10°, flip horizontal (p=0,5), variation de
  luminosité ±10 %.
- **Class weights** : poids inversement proportionnels à la fréquence
  (`w_c = N / (7 · n_c)`), appliqués uniquement à la loss d'entraînement.
- **Métriques** : accuracy et **F1 macro** (adapté au déséquilibre), matrice de
  confusion, F1 par classe, temps d'entraînement.

---

## 2. Installation et exécution

### 2.1 Environnement

```bash
py -3.12 -m venv .venv

# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

> Python 3.12 est requis (PyTorch CPU). Les versions sont figées dans
> `requirements.txt`.
>
> **Windows** : si `import torch` échoue avec une erreur `WinError 1114`
> (DLL init failed), le runtime Visual C++ est trop ancien. Installez
> [vc_redist.x64.exe](https://aka.ms/vs/17/release/vc_redist.x64.exe)
> (14.40 ou plus récent), puis relancez le terminal.

Vérification de l'installation :

```bash
python scripts/check_env.py
```

### 2.2 Données

```bash
python scripts/prepare_data.py
```

Cette commande convertit les fichiers Parquet FER2013 en arborescence
`ImageFolder` dans `data/raw/` (voir `data/download.md` pour les autres
options). Au premier appel de l'entraîneur, les images sont converties en
tableaux NumPy dans `data/processed/` : ce cache (~100 s pour le train) évite
de décoder 28 000 PNG à chaque époque et réduit le temps d'époque d'un facteur
3 environ. Vérification rapide :

```bash
python -m src.dataset
```

### 2.3 Entraînement des quatre variantes

```bash
python -m src.train --config configs/v1_baseline.yaml
python -m src.train --config configs/v2_augmentation.yaml
python -m src.train --config configs/v3_classweights.yaml
python -m src.train --config configs/v4_regularization.yaml
```

Ou en une commande :

```bash
# Linux / macOS / Git Bash
bash scripts/run_all.sh
# Windows PowerShell
.\scripts\run_all.ps1
```

> Important : utiliser `python -m src.train` (et non `python src/train.py`)
> pour que le package `src` soit importable.

### 2.4 Analyse et figures

```bash
python -m src.analyze
```

Génère dans `results/` :

- `metrics/summary.csv` — tableau comparatif des variantes ;
- `metrics/*_results.json` — métriques, historique, prédictions de test ;
- `metrics/*_classification_report.txt` — précision/rappel/F1 par classe ;
- `metrics/error_analysis.json` — paires de classes les plus confondues ;
- `figures/*_curves.png` — courbes d'apprentissage ;
- `figures/*_confusion.png` — matrices de confusion ;
- `figures/*_per_class_f1.png` — F1 par classe ;
- `figures/comparison.png` — comparaison des variantes.

### 2.5 Inférence sur une image

```bash
python -m src.predict --model results/models/v1_baseline_best.pth \
                      --image chemin/vers/visage.png
```

Le rapport LaTeX (**6–8 pages**) se compile avec `pdflatex` :

```bash
cd reports
pdflatex final_report.tex
pdflatex final_report.tex
```

Les macros et tableaux du rapport sont régénérés depuis les résultats réels :

```bash
python scripts/make_report_assets.py
```

---

## 3. Résultats

<!-- RESULTATS_DEBUT -->
Entraînements réalisés sur le CPU décrit en section 5 (30 époques maximum,
early stopping patience 6, `seed=42`).

| Variante | Accuracy | F1 macro | Meilleure époque | Temps (min) |
|---|---:|---:|---:|---:|
| **V1 — baseline** | **0,5811** | **0,4765** | 30 | 27,1 |
| V2 — augmentation | 0,5603 | 0,4506 | 29 | 33,0 |
| V3 — class weights | 0,2799 | 0,2465 | 18 | 26,4 |
| V4 — régularisation | 0,4533 | 0,3764 | 30 | 33,5 |

F1 par classe sur le test :

| Classe | V1 | V2 | V3 | V4 |
|---|---:|---:|---:|---:|
| angry | 0,48 | 0,41 | 0,22 | 0,20 |
| disgust | 0,00 | 0,00 | **0,31** | 0,20 |
| fear | 0,27 | 0,22 | 0,11 | 0,10 |
| happy | **0,82** | **0,82** | 0,03 | 0,75 |
| neutral | 0,47 | 0,46 | 0,04 | 0,33 |
| sad | 0,73 | 0,70 | 0,66 | 0,65 |
| surprise | 0,56 | 0,54 | 0,36 | 0,40 |
| **F1 macro** | **0,48** | 0,45 | 0,25 | 0,38 |

Figures : `results/figures/` (`comparison.png`, `*_curves.png`,
`*_confusion.png`, `confusion_grid.png`, `*_errors.png`,
`dataset_distribution.png`, `dataset_samples.png`). Tableau machine :
`results/metrics/summary.csv`.
<!-- RESULTATS_FIN -->

---

## 4. Analyse

<!-- ANALYSE_DEBUT -->
**La baseline gagne.** Contrairement à l'hypothèse initiale, aucun des trois
leviers testés n'améliore la référence à budget d'époques constant :
V1 obtient 58,11 % d'accuracy et 0,4765 de F1 macro.

**Augmentation (V2) : −2,1 pts d'accuracy, −2,6 pts de F1.** La rotation
±10° sur des images 48×48 déplace des détails fins déjà peu informatifs, et
surtout elle ralentit la convergence : à l'époque 30, la perte d'entraînement
diminue encore. Le coût augmente de 20 % (33,0 min contre 27,1).

**Class weights (V3) : effondrement.** Le poids `disgust ≈ 9,4` (facteur 16
avec `happy`) domine la perte : le rappel de `disgust` monte à 0,39 et son F1 à
0,31, mais `happy` tombe à 0,03 et `neutral` à 0,04. L'optimisation devient
instable avec `lr = 10⁻³` et l'early stopping se déclenche à l'époque 24.
La pondération par l'inverse exact des fréquences n'est donc pas robuste ici ;
il faudrait des poids plus doux (racine carrée, écrêtage à 3–4) ou un `lr`
plus faible.

**Régularisation (V4) : correction partielle.** Le dropout renforcé
(0,30/0,60) limite la confiance excessive du modèle pondéré et restaure
`happy` à 0,75, mais reste 10 points de F1 sous la baseline. Le décrochage de
`lr` à l'époque 19 relance la validation, ce qui confirme le diagnostic
d'optimisation.

**Erreurs.** Les confusions dominantes de la baseline sont `fear → neutral`
(34 % des *fear*), `angry → neutral` (24 %), `surprise ↔ neutral` et
`fear → angry` ; `neutral` agit comme l'attracteur principal. La classe
`disgust` est prédite `angry` dans 67 % des cas. Les visages peu contrastés,
occlus ou de profil concentrent l'essentiel des erreurs.

**Limites.** Un seul découpage et une seule graine ; 30 époques imposées par
le temps de calcul ; pas de pré-entraînement. Une pondération mieux calibrée et
un entraînement plus long sont les pistes les plus prometteuses.
<!-- ANALYSE_FIN -->

---

## 5. Configuration matérielle

| Élément | Valeur |
|---|---|
| CPU | AMD Ryzen 5 5500GT (6 cœurs / 12 threads, 3,6 GHz) |
| RAM | 32 Go |
| GPU | aucun — entraînement **CPU uniquement** |
| OS | Windows 11 64 bits |
| Python | 3.12 |
| PyTorch | version CPU (voir `requirements.txt`) |

Le temps par époque et le temps total sont mesurés et enregistrés dans
`results/metrics/*_results.json`.

---

## 6. Reproductibilité

- Seed fixée (`seed: 42`) : Python, NumPy, PyTorch et le générateur du
  `DataLoader`.
- Split train/validation stratifié et déterministe.
- Hyperparamètres centralisés dans `configs/*.yaml`.
- Versions figées dans `requirements.txt`.
- Temps, nombre de paramètres et historique complet sauvegardés à chaque run.
- Le test n'est jamais utilisé pour le early stopping ni le choix du modèle
  (sélection sur le F1 de validation).

---

## 7. Structure du dépôt

```
.
├── README.md
├── requirements.txt
├── data/
│   ├── download.md          # instructions dataset
│   └── raw/                 # images (non versionnées)
├── src/
│   ├── constants.py         # classes, taille d'image
│   ├── dataset.py           # chargement + split + class weights
│   ├── model.py             # CNN léger
│   ├── train.py             # entraînement + évaluation
│   ├── analyze.py           # figures + tableau comparatif
│   └── predict.py           # inférence
├── configs/                 # 4 variantes YAML
├── scripts/
│   ├── prepare_data.py      # Parquet -> ImageFolder
│   ├── check_env.py         # diagnostic de l'installation
│   ├── make_dataset_figures.py
│   ├── make_report_assets.py # macros/tableaux LaTeX depuis les résultats
│   ├── run_all.sh
│   ├── run_all.ps1
│   └── run_experiments.ps1  # 4 runs + analyse (avec logs)
├── notebooks/exploration.ipynb
├── results/
│   ├── figures/
│   ├── metrics/
│   └── models/
└── reports/
    ├── final_report.tex     # rapport (6-8 pages)
    ├── final_report.pdf
    └── generated/           # macros et tableaux générés
```

---

## 8. Références

- Goodfellow et al., *Challenges in Representation Learning: A report on three
  machine learning contests*, ICONIP 2013 (FER2013).
- Modi et al., *Facial Emotion Recognition using Convolutional Neural
  Networks*.
- Documentation PyTorch : <https://pytorch.org/docs/stable/>
- torchvision : <https://pytorch.org/vision/stable/>
- scikit-learn : <https://scikit-learn.org/stable/>
