$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$Host.UI.RawUI.WindowTitle = 'EE5438 P2/P3/P4/P5 strict experiment'

$comparisonId = 'strict-b8w0-s0-20260925-011512'
Write-Host "Four-scale comparison ID: $comparisonId" -ForegroundColor Cyan
Write-Host '50 epochs, 640 px, batch 8, workers 0, seed 0, same VisDrone data and COCO checkpoint.'
Write-Host 'Native training progress remains visible. Results are saved in runs/records and runs/evaluation.'
Write-Host 'Keep this window open and prevent the computer from sleeping.' -ForegroundColor Yellow

& .\.venv\Scripts\python.exe .\src\run_experiment.py `
    --variant p2p3p4p5 `
    --comparison-id $comparisonId `
    --epochs 50 --batch 8 --workers 0 --imgsz 640 --seed 0 --save-period 10 --eval-batch 16

if ($LASTEXITCODE -ne 0) {
    Write-Host "Four-scale run failed with exit code $LASTEXITCODE." -ForegroundColor Red
    return
}

Write-Host "Four-scale run completed and evaluated: $comparisonId" -ForegroundColor Green
