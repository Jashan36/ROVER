# PowerShell setup script for project
# Usage: Open PowerShell (run as normal or admin if needed) and run:
# .\setup_env.ps1

param(
    [string]$venvPath = ".venv"
)

Write-Host "Creating Python virtual environment at: $venvPath"
python -m venv $venvPath

Write-Host "Activating virtual environment"
& "$venvPath\Scripts\Activate.ps1"

Write-Host "Upgrading pip and installing Python dependencies"
python -m pip install --upgrade pip
pip install -r ..\requirements.txt

Write-Host "Environment setup complete. Activate with: & $venvPath\Scripts\Activate.ps1"
