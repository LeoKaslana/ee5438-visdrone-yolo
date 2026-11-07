$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$Host.UI.RawUI.WindowTitle = 'EE5438 continue strict repeats'

$experiments = @(
    @{ Seed = 1; Variant = 'baseline'; Id = 'strict-b8w0-s1-20260925-172635' },
    @{ Seed = 1; Variant = 'p2'; Id = 'strict-b8w0-s1-20260927-retry' },
    @{ Seed = 2; Variant = 'baseline'; Id = 'strict-b8w0-s2-20260927-continued' },
    @{ Seed = 2; Variant = 'p2'; Id = 'strict-b8w0-s2-20260927-retry' }
)

Write-Host 'Skip completed experiments and resume remaining strict repeats from last.pt.' -ForegroundColor Cyan
Write-Host 'Each run keeps batch 8, workers 0, 640 px, 50 epochs and its own result record.'
Write-Host 'Keep this visible window open; prevent the computer from sleeping.' -ForegroundColor Yellow

foreach ($experiment in $experiments) {
        $variant = $experiment.Variant
        $comparisonId = $experiment.Id
        $seed = $experiment.Seed
        $recordPath = Join-Path $projectRoot "runs\records\$comparisonId-$variant.json"
        $common = @(
            '.\src\run_experiment.py', '--variant', $variant, '--comparison-id', $comparisonId,
            '--epochs', '50', '--batch', '8', '--workers', '0', '--imgsz', '640',
            '--seed', [string]$seed, '--save-period', '10', '--eval-batch', '16'
        )
        if (Test-Path -LiteralPath $recordPath) {
            $record = Get-Content -LiteralPath $recordPath -Raw | ConvertFrom-Json
            if ($record.status -eq 'complete') {
                Write-Host "$variant seed $seed already complete; skipping." -ForegroundColor Green
                continue
            }
            $lastCheckpoint = Join-Path $projectRoot "runs\$variant\$comparisonId-$variant\weights\last.pt"
            if (-not (Test-Path -LiteralPath $lastCheckpoint)) {
                Write-Host "Incomplete $variant seed $seed has no checkpoint: $lastCheckpoint. Keep its record and select a fresh comparison ID." -ForegroundColor Red
                return
            }
            Write-Host "Resuming $variant seed $seed from its saved checkpoint." -ForegroundColor Cyan
            $common += '--resume'
        } else {
            Write-Host "Starting $variant seed $seed as a new run." -ForegroundColor Cyan
        }
        & .\.venv\Scripts\python.exe @common
        if ($LASTEXITCODE -ne 0) {
            Write-Host "$variant seed $seed stopped with exit code $LASTEXITCODE. Later runs were not started." -ForegroundColor Red
            return
        }
        Write-Host "$variant seed $seed complete and evaluated." -ForegroundColor Green
}

Write-Host 'All remaining strict repeats completed and evaluated.' -ForegroundColor Green
