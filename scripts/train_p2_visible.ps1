$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$Host.UI.RawUI.WindowTitle = 'EE5438 P2-P3-P4 training'

Write-Host 'EE5438 YOLO11n P2/P3/P4 experiment' -ForegroundColor Cyan
Write-Host 'First: one-epoch 1% data smoke test at batch 16.'
Write-Host 'If that succeeds: 50-epoch full-data run at batch 16.'
Write-Host 'Keep this window open while training is running.' -ForegroundColor Yellow

& .\.venv\Scripts\python.exe .\src\train_p2.py `
    --epochs 1 `
    --batch 16 `
    --imgsz 640 `
    --workers 2 `
    --seed 0 `
    --fraction 0.01 `
    --name smoke_yolo11n_p2p3p4_640_b16_seed0

if ($LASTEXITCODE -ne 0) {
    Write-Host "Smoke test failed with exit code $LASTEXITCODE; formal training was not started." -ForegroundColor Red
    return
}

Write-Host 'Smoke test passed. Starting full-data 50-epoch training.' -ForegroundColor Green
& .\.venv\Scripts\python.exe .\src\train_p2.py `
    --epochs 50 `
    --batch 16 `
    --imgsz 640 `
    --workers 2 `
    --seed 0 `
    --name formal_yolo11n_p2p3p4_640_b16_seed0

if ($LASTEXITCODE -eq 0) {
    Write-Host 'P2/P3/P4 training completed successfully.' -ForegroundColor Green
} else {
    Write-Host "P2/P3/P4 training stopped with exit code $LASTEXITCODE." -ForegroundColor Red
}
