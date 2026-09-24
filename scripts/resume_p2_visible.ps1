$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$Host.UI.RawUI.WindowTitle = 'EE5438 P2-P3-P4 resumed training'

Write-Host 'Resuming P2/P3/P4 training from last.pt with the original checkpoint settings.' -ForegroundColor Cyan
Write-Host 'Expected settings: 50 epochs, 640 px, batch 16, 2 workers, seed 0.'
Write-Host 'Keep this window open while training is running.' -ForegroundColor Yellow

& .\.venv\Scripts\python.exe .\src\resume_p2.py

if ($LASTEXITCODE -eq 0) {
    Write-Host 'P2/P3/P4 training completed successfully.' -ForegroundColor Green
} else {
    Write-Host "P2/P3/P4 training stopped with exit code $LASTEXITCODE." -ForegroundColor Red
}
