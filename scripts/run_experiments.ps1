# Lance les 4 variantes sequentiellement, avec un log par run et une analyse
# intermediaire apres chaque entrainement.
$ErrorActionPreference = "Stop"

$env:MPLCONFIGDIR = Join-Path (Get-Location) ".mplconfig"
$py = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "python" }
$logDir = "results/logs"
New-Item -ItemType Directory -Force $logDir | Out-Null

$configs = @(
    "v1_baseline",
    "v2_augmentation",
    "v3_classweights",
    "v4_regularization"
)

$overall = Get-Date
foreach ($cfg in $configs) {
    $log = Join-Path $logDir "$cfg.log"
    $start = Get-Date
    Write-Host "=== $cfg - debut $(Get-Date -Format 'HH:mm:ss') ==="
    & $py -u -m src.train --config "configs/$cfg.yaml" 2>&1 | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ECHEC : $cfg (code $LASTEXITCODE)"
        exit $LASTEXITCODE
    }
    $dur = (Get-Date) - $start
    Write-Host ("=== $cfg termine en {0:N1} min ===" -f $dur.TotalMinutes)

    Write-Host "--- analyse intermediaire ---"
    & $py -u -m src.analyze 2>&1 | Tee-Object -FilePath (Join-Path $logDir "analyze.log") -Append
}

$total = (Get-Date) - $overall
Write-Host ("TOUT TERMINE en {0:N1} min" -f $total.TotalMinutes)
