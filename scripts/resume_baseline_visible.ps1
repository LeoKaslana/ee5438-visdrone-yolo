$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$runDir = Join-Path $projectRoot 'runs\baseline\yolo11n_640_seed0-4'
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
$resumeScript = Join-Path $projectRoot 'src\resume_baseline.py'
$logPath = Join-Path $runDir 'resume_terminal.log'

Set-Location -LiteralPath $projectRoot
Start-Transcript -Path $logPath -Append | Out-Null
try {
    Write-Host "Resuming VisDrone baseline from last.pt with workers=0"
    Write-Host "Run directory: $runDir"
    & $python $resumeScript
    $exitCode = $LASTEXITCODE
    Write-Host "Training process exited with code $exitCode"
    Write-Host "Transcript: $logPath"
} finally {
    Stop-Transcript | Out-Null
}
