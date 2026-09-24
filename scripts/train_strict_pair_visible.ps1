$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$Host.UI.RawUI.WindowTitle = 'EE5438 strict baseline vs P2 experiment'

$comparisonId = 'strict-b8w0-s0-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
Write-Host "Comparison ID: $comparisonId" -ForegroundColor Cyan
Write-Host 'Sequential runs: YOLO11n baseline, then P2/P3/P4.'
Write-Host 'Both: 50 epochs, 640 px, batch 8, workers 0, seed 0, same VisDrone data and COCO checkpoint.'
Write-Host 'Terminal shows the native progress bar. Each run saves a short result JSON and normal YOLO results.'
Write-Host 'Keep this window open and prevent the computer from sleeping.' -ForegroundColor Yellow

& .\.venv\Scripts\python.exe .\src\run_experiment.py `
    --variant baseline `
    --comparison-id $comparisonId `
    --epochs 50 --batch 8 --workers 0 --imgsz 640 --seed 0 --save-period 10 --eval-batch 16

if ($LASTEXITCODE -ne 0) {
    Write-Host "Baseline failed with exit code $LASTEXITCODE. P2 was not started." -ForegroundColor Red
    return
}

Write-Host 'Baseline complete and evaluated. Starting P2/P3/P4.' -ForegroundColor Green
& .\.venv\Scripts\python.exe .\src\run_experiment.py `
    --variant p2 `
    --comparison-id $comparisonId `
    --epochs 50 --batch 8 --workers 0 --imgsz 640 --seed 0 --save-period 10 --eval-batch 16

if ($LASTEXITCODE -ne 0) {
    Write-Host "P2 failed with exit code $LASTEXITCODE. Baseline results remain saved." -ForegroundColor Red
    return
}

Write-Host "Strict pair completed and evaluated: $comparisonId" -ForegroundColor Green
