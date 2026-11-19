# Téléchargement du jeu de données FER2013

Le projet utilise **FER2013** : 35 887 images en niveaux de gris 48×48,
réparties en 7 classes (`angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`,
`surprise`).

Structure attendue par le code :

```
data/raw/
├── train/
│   ├── angry/*.png
│   ├── disgust/*.png
│   ├── fear/*.png
│   ├── happy/*.png
│   ├── neutral/*.png
│   ├── sad/*.png
│   └── surprise/*.png
└── test/
    └── (mêmes 7 sous-dossiers, 7 178 images)
```

## Option 1 — Depuis les fichiers Parquet (recommandé, sans compte Kaggle)

Si vous disposez des fichiers `train.parquet`, `public.parquet` et
`private.parquet` (format de la compétition Kaggle FER2013) :

```bash
python scripts/prepare_data.py --src chemin/vers/raw_parquet
```

Le script écrit les PNG dans `data/raw/train/` et `data/raw/test/`
(`test` = `public` + `private`, soit 7 178 images) et génère
`data/raw/manifest.csv`.

## Option 2 — Kaggle CLI

1. Installer le client :
   ```bash
   pip install kaggle
   ```
2. Placer `kaggle.json` dans :
   - Linux/macOS : `~/.kaggle/kaggle.json`
   - Windows : `C:\Users\<user>\.kaggle\kaggle.json`
3. Télécharger :
   ```bash
   kaggle datasets download -d msambare/fer2013 -p data/raw --unzip
   ```

## Option 3 — Téléchargement manuel

1. Aller sur <https://www.kaggle.com/datasets/msambare/fer2013>
2. Télécharger `archive.zip`
3. Extraire dans `data/raw/` de sorte à obtenir `train/` et `test/`.

## Vérification

```bash
python -m src.dataset
```

Sortie attendue : 28 709 images d'entraînement, 25 840 de validation
(90/10 stratifié) et 7 178 de test.

> Les images ne sont pas versionnées (voir `.gitignore`) : elles sont
> régénérables à partir des parquet ou téléchargeables sur Kaggle.

## Classes (7)

`angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`
