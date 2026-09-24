$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$Host.UI.RawUI.WindowTitle = 'EE5438 P2-P3-P4 safe resume'

Write-Host 'Resuming P2/P3/P4 from last.pt after repeated DataLoader worker exits.' -ForegroundColor Cyan
Write-Host 'Overrides: batch 8, workers 0. Other checkpoint settings remain unchanged.'
Write-Host 'Keep this window open while training is running.' -ForegroundColor Yellow

& .\.venv\Scripts\python.exe .\src\resume_p2.py --batch 8 --workers 0

if ($LASTEXITCODE -eq 0) {
    Write-Host 'P2/P3/P4 training completed successfully.' -ForegroundColor Green
} else {
    Write-Host "P2/P3/P4 training stopped with exit code $LASTEXITCODE." -ForegroundColor Red
}
