# ==============================================================================
# ENVIRONMENT SETUP SCRIPT (PowerShell for Windows)
# Sets up Mars Rover AI environment on Windows
# ==============================================================================

# Set error action
$ErrorActionPreference = "Stop"

# Colors
function Write-Status {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Header
Write-Host ""
Write-Host "============================================" -ForegroundColor Blue
Write-Host "   MARS ROVER AI - ENVIRONMENT SETUP" -ForegroundColor Blue
Write-Host "   Windows PowerShell" -ForegroundColor Blue
Write-Host "============================================" -ForegroundColor Blue
Write-Host ""

# Get script directory
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$PROJECT_ROOT = Split-Path -Parent $SCRIPT_DIR

Write-Status "Project root: $PROJECT_ROOT"

# ==============================================================================
# CHECK PREREQUISITES
# ==============================================================================

Write-Status "Checking prerequisites..."

# Check Python
try {
    $pythonVersion = python --version 2>&1
    Write-Success "Python found: $pythonVersion"
} catch {
    Write-Error "Python not found. Please install Python 3.8+ from python.org"
    exit 1
}

# Check pip
try {
    $pipVersion = pip --version 2>&1
    Write-Success "pip found: $pipVersion"
} catch {
    Write-Error "pip not found. Please install pip"
    exit 1
}

# Check Git
try {
    $gitVersion = git --version 2>&1
    Write-Success "Git found: $gitVersion"
} catch {
    Write-Warning "Git not found. Some features may not work"
}

Write-Success "Prerequisites check completed"

# ==============================================================================
# CREATE VIRTUAL ENVIRONMENT
# ==============================================================================

Write-Status "Setting up Python virtual environment..."

Set-Location $PROJECT_ROOT

if (-not (Test-Path "venv")) {
    Write-Status "Creating virtual environment..."
    python -m venv venv
    Write-Success "Virtual environment created"
} else {
    Write-Status "Virtual environment already exists"
}

# Activate virtual environment
Write-Status "Activating virtual environment..."
& ".\venv\Scripts\Activate.ps1"

# Upgrade pip
Write-Status "Upgrading pip..."
python -m pip install --upgrade pip --quiet

# ==============================================================================
# INSTALL DEPENDENCIES
# ==============================================================================

Write-Status "Installing Python dependencies..."

if (Test-Path "requirements.txt") {
    Write-Status "Installing from requirements.txt..."
    pip install -r requirements.txt --quiet
    Write-Success "Dependencies installed"
} else {
    Write-Warning "requirements.txt not found"
}

# Install PyTorch
Write-Status "Installing PyTorch..."

# Check for NVIDIA GPU
$hasGPU = $false
try {
    nvidia-smi | Out-Null
    $hasGPU = $true
    Write-Status "NVIDIA GPU detected"
} catch {
    Write-Status "No NVIDIA GPU detected"
}

if ($hasGPU) {
    Write-Status "Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --quiet
} else {
    Write-Status "Installing PyTorch CPU version..."
    pip install torch torchvision torchaudio --quiet
}

Write-Success "PyTorch installed"

# ==============================================================================
# INSTALL DEVELOPMENT TOOLS
# ==============================================================================

Write-Status "Installing development tools..."

pip install pytest pytest-cov black flake8 mypy --quiet

Write-Success "Development tools installed"

# ==============================================================================
# CREATE DIRECTORIES
# ==============================================================================

Write-Status "Creating project directories..."

$directories = @(
    "data\raw\perseverance",
    "data\raw\ai4mars",
    "data\raw\dem",
    "data\processed",
    "data\cache",
    "logs",
    "ai\models\weights",
    "demos\videos",
    "demos\screenshots"
)

foreach ($dir in $directories) {
    $fullPath = Join-Path $PROJECT_ROOT $dir
    if (-not (Test-Path $fullPath)) {
        New-Item -ItemType Directory -Path $fullPath -Force | Out-Null
    }
}

Write-Success "Directories created"

# ==============================================================================
# SET ENVIRONMENT VARIABLES
# ==============================================================================

Write-Status "Setting environment variables..."

# Set PYTHONPATH
$env:PYTHONPATH = $PROJECT_ROOT

# NASA API Key
if (-not $env:NASA_API_KEY) {
    $env:NASA_API_KEY = "DEMO_KEY"
    Write-Warning "NASA_API_KEY not set. Using DEMO_KEY (limited to 30 requests/hour)"
    Write-Warning "Get a free API key at: https://api.nasa.gov/"
}

Write-Success "Environment variables set"

# ==============================================================================
# CREATE CONVENIENCE SCRIPTS
# ==============================================================================

Write-Status "Creating convenience scripts..."

# Create activate.ps1
$activateScript = @"
# Activate Mars Rover AI environment
Write-Host "Activating Mars Rover AI environment..." -ForegroundColor Blue

Set-Location "$PROJECT_ROOT"
& ".\venv\Scripts\Activate.ps1"

`$env:PYTHONPATH = "$PROJECT_ROOT"
`$env:NASA_API_KEY = if (`$env:NASA_API_KEY) { `$env:NASA_API_KEY } else { "DEMO_KEY" }

Write-Host "Environment activated!" -ForegroundColor Green
Write-Host "  Python: `$(python --version)" -ForegroundColor Cyan
Write-Host "  Location: $PROJECT_ROOT" -ForegroundColor Cyan
"@

$activateScript | Out-File -FilePath "$PROJECT_ROOT\activate.ps1" -Encoding UTF8

# Create run_tests.ps1
$runTestsScript = @"
# Run tests
& ".\activate.ps1"
pytest tests\ -v
"@

$runTestsScript | Out-File -FilePath "$PROJECT_ROOT\run_tests.ps1" -Encoding UTF8

# Create launch_dashboard.ps1
$launchDashboardScript = @"
# Launch dashboard
& ".\activate.ps1"
Set-Location dashboard
streamlit run app.py
"@

$launchDashboardScript | Out-File -FilePath "$PROJECT_ROOT\launch_dashboard.ps1" -Encoding UTF8

Write-Success "Convenience scripts created"

# ==============================================================================
# VERIFY INSTALLATION
# ==============================================================================

Write-Status "Verifying installation..."

# Check Python imports
$verifyScript = @"
try:
    import numpy
    import cv2
    import torch
    import streamlit
    print('✓ All core packages importable')
except ImportError as e:
    print(f'✗ Import error: {e}')
    exit(1)
"@

$verifyScript | python

if ($LASTEXITCODE -eq 0) {
    Write-Success "Installation verified"
} else {
    Write-Error "Installation verification failed"
    exit 1
}

# ==============================================================================
# SUMMARY
# ==============================================================================

Write-Host ""
Write-Success "============================================"
Write-Success "   SETUP COMPLETED SUCCESSFULLY!"
Write-Success "============================================"
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Activate environment:      .\activate.ps1" -ForegroundColor White
Write-Host "  2. Download data:             python scripts\download_dataset.py --minimal" -ForegroundColor White
Write-Host "  3. Run tests:                 .\run_tests.ps1" -ForegroundColor White
Write-Host "  4. Launch dashboard:          .\launch_dashboard.ps1" -ForegroundColor White
Write-Host ""
Write-Host "For ROS2 support, please install ROS2 Humble separately" -ForegroundColor Yellow
Write-Host "Visit: https://docs.ros.org/en/humble/Installation.html" -ForegroundColor Yellow
Write-Host ""

# Keep window open
Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")