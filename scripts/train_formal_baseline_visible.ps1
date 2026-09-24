$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

Write-Host 'EE5438 VisDrone formal YOLO11n baseline training' -ForegroundColor Cyan
Write-Host 'Run: runs/baseline/formal_yolo11n_640_b16_seed0_live'
Write-Host 'Configuration: 50 epochs, 640 px, batch 16, 2 workers, seed 0'
Write-Host 'Keep this window open while training is running.' -ForegroundColor Yellow

& .\.venv\Scripts\python.exe .\src\train_baseline.py `
    --epochs 50 `
    --batch 16 `
    --imgsz 640 `
    --workers 2 `
    --seed 0 `
    --name formal_yolo11n_640_b16_seed0_live

$trainingExitCode = $LASTEXITCODE
if ($trainingExitCode -eq 0) {
    Write-Host 'Training completed successfully.' -ForegroundColor Green
} else {
    Write-Host "Training stopped with exit code $trainingExitCode." -ForegroundColor Red
}
