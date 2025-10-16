<#
Run AI4Mars downloader in the project venv (PowerShell helper).

Usage (from repo root):
  .\scripts\run_ai4mars.ps1                # runs ai.data_fetcher.ai4mars_loader
  .\scripts\run_ai4mars.ps1 -- --help      # pass args to the script

This script looks for `venv` or `.venv` in the repo root and uses its python.
If no venv is found, it falls back to the current `python` on PATH.
#>

param(
    [Parameter(ValueFromRemainingArguments=$true)]
    $RemainingArgs
)

Set-StrictMode -Version Latest

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir

# Try common venv locations
$venvCandidates = @(
    Join-Path $repoRoot 'venv'
    Join-Path $repoRoot '.venv'
)

$python = $null
foreach ($v in $venvCandidates) {
    $candidate = Join-Path $v 'Scripts\python.exe'
    if (Test-Path $candidate) {
        $python = $candidate
        break
    }
}

if (-not $python) {
    Write-Host "No repo venv found; using 'python' on PATH"
    $python = 'python'
}

# Build module invocation
$module = 'ai.data_fetcher.ai4mars_loader'

# Execute the module from the repository root so package imports resolve
Push-Location $repoRoot
try {
    & $python -m $module @RemainingArgs
    $exit = $LASTEXITCODE
} finally {
    Pop-Location
}

if ($exit -ne 0) {
    Write-Host "Module exited with code $exit" -ForegroundColor Red
    exit $exit
}
