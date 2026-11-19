#!/usr/bin/env bash
# Lance les 4 entraînements puis l'analyse.
set -e

if [ -x ".venv/Scripts/python.exe" ]; then
  PY=".venv/Scripts/python.exe"   # Windows / Git Bash
elif [ -x ".venv/bin/python" ]; then
  PY=".venv/bin/python"           # Linux / macOS
else
  PY="python"
fi

echo "Python : $PY"

for cfg in v1_baseline v2_augmentation v3_classweights v4_regularization; do
  echo "=== $cfg ==="
  "$PY" -m src.train --config "configs/$cfg.yaml"
done

echo "=== Analyse ==="
"$PY" -m src.analyze

echo "Terminé. Voir results/."
