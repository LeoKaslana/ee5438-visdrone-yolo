$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$Host.UI.RawUI.WindowTitle = 'EE5438 strict repeat seeds 1 and 2'

Write-Host 'Repeating baseline and P2/P3/P4 with seeds 1 and 2.' -ForegroundColor Cyan
Write-Host 'Each run: 50 epochs, 640 px, batch 8, workers 0, full VisDrone; native progress and separate records.'
Write-Host 'Four runs will take many hours. Keep this window open and prevent computer sleep.' -ForegroundColor Yellow

foreach ($seed in @(1, 2)) {
    $comparisonId = 'strict-b8w0-s' + $seed + '-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
    foreach ($variant in @('baseline', 'p2')) {
        Write-Host "Starting $variant, seed $seed, comparison $comparisonId" -ForegroundColor Cyan
        & .\.venv\Scripts\python.exe .\src\run_experiment.py `
            --variant $variant `
            --comparison-id $comparisonId `
            --epochs 50 --batch 8 --workers 0 --imgsz 640 --seed $seed --save-period 10 --eval-batch 16

        if ($LASTEXITCODE -ne 0) {
            Write-Host "$variant seed $seed failed with exit code $LASTEXITCODE. Completed results remain saved; later runs were not started." -ForegroundColor Red
            return
        }
        Write-Host "$variant seed $seed completed and evaluated." -ForegroundColor Green
    }
}

Write-Host 'All four repeat runs completed and evaluated.' -ForegroundColor Green
