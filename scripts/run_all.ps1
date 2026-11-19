# Lance les 4 entraînements puis l'analyse (Windows PowerShell)
$ErrorActionPreference = "Stop"

$py = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "python" }
Write-Host "Python : $py"

$configs = @(
    "v1_baseline",
    "v2_augmentation",
    "v3_classweights",
    "v4_regularization"
)

foreach ($cfg in $configs) {
    Write-Host "=== $cfg ==="
    & $py -m src.train --config "configs/$cfg.yaml"
}

Write-Host "=== Analyse ==="
& $py -m src.analyze

Write-Host "Termine. Voir results/."
