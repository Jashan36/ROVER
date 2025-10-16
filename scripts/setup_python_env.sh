#!/bin/bash
# ==============================================================================
# PYTHON ENVIRONMENT SETUP SCRIPT
# Sets up Python virtual environment and dependencies
# ==============================================================================

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "${BLUE}"
echo "============================================"
echo "   PYTHON ENVIRONMENT SETUP"
echo "============================================"
echo -e "${NC}"

# ==============================================================================
# FUNCTIONS
# ==============================================================================

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        print_error "$1 is not installed"
        return 1
    fi
    return 0
}

# ==============================================================================
# CHECK PREREQUISITES
# ==============================================================================

print_status "Checking prerequisites..."

# Check Python version
if ! check_command "python3"; then
    print_error "Python 3 not found. Please install Python 3.8 or higher"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
print_success "Python $PYTHON_VERSION found"

# Check pip
if ! check_command "pip3"; then
    print_error "pip3 not found. Installing..."
    python3 -m ensurepip --upgrade
fi

print_success "pip3 found"

# Check for venv module
python3 -c "import venv" 2>/dev/null
if [ $? -ne 0 ]; then
    print_error "venv module not found. Installing..."
    
    # Try to install based on OS
    if [ -f /etc/debian_version ]; then
        sudo apt-get update
        sudo apt-get install -y python3-venv
    elif [ -f /etc/redhat-release ]; then
        sudo yum install -y python3-virtualenv
    else
        print_error "Please install python3-venv manually"
        exit 1
    fi
fi

print_success "Prerequisites check completed"

# ==============================================================================
# CREATE VIRTUAL ENVIRONMENT
# ==============================================================================

print_status "Setting up virtual environment..."

cd "$PROJECT_ROOT"

if [ ! -d "venv" ]; then
    print_status "Creating virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_status "Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip setuptools wheel -q

print_success "Virtual environment activated"

# ==============================================================================
# INSTALL DEPENDENCIES
# ==============================================================================

print_status "Installing Python dependencies..."

# Install from requirements.txt
if [ -f "requirements.txt" ]; then
    print_status "Installing from requirements.txt..."
    pip install -r requirements.txt -q
    print_success "Core dependencies installed"
else
    print_warning "requirements.txt not found, installing manually..."
    
    # Core dependencies
    pip install numpy opencv-python pillow -q
    pip install pyyaml tqdm requests -q
    pip install matplotlib plotly -q
    
    # ML dependencies
    print_status "Installing ML dependencies..."
    pip install scikit-learn scikit-image -q
    
    # Dashboard dependencies
    pip install streamlit -q
    
    # Testing dependencies
    pip install pytest pytest-cov -q
fi

# ==============================================================================
# INSTALL PYTORCH
# ==============================================================================

print_status "Installing PyTorch..."

# Detect GPU
if command -v nvidia-smi &> /dev/null; then
    print_status "NVIDIA GPU detected, installing PyTorch with CUDA support..."
    
    # Check CUDA version
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
        print_status "CUDA version: $CUDA_VERSION"
    fi
    
    # Install PyTorch with CUDA 11.8
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -q
    
else
    print_status "No GPU detected, installing PyTorch CPU version..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu -q
fi

print_success "PyTorch installed"

# ==============================================================================
# INSTALL DEVELOPMENT TOOLS
# ==============================================================================

print_status "Installing development tools..."

pip install black flake8 mypy pylint -q
pip install ipython jupyter -q

print_success "Development tools installed"

# ==============================================================================
# VERIFY INSTALLATION
# ==============================================================================

print_status "Verifying installation..."

python3 - <<EOF
import sys

packages = {
    'numpy': 'NumPy',
    'cv2': 'OpenCV',
    'torch': 'PyTorch',
    'PIL': 'Pillow',
    'yaml': 'PyYAML',
    'streamlit': 'Streamlit',
    'pytest': 'pytest',
}

failed = []

for module, name in packages.items():
    try:
        __import__(module)
        print(f'✓ {name}')
    except ImportError:
        print(f'✗ {name} (failed)')
        failed.append(name)

if failed:
    print(f'\nFailed to import: {", ".join(failed)}')
    sys.exit(1)
else:
    print('\n✓ All packages verified')
EOF

if [ $? -eq 0 ]; then
    print_success "Installation verified"
else
    print_error "Installation verification failed"
    exit 1
fi

# ==============================================================================
# CREATE ACTIVATION SCRIPT
# ==============================================================================

print_status "Creating activation script..."

cat > "$PROJECT_ROOT/activate_env.sh" <<'EOF'
#!/bin/bash
# Activate Python virtual environment

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -d "$SCRIPT_DIR/venv" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
    export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
    echo "✓ Python environment activated"
    echo "  Python: $(python --version)"
    echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
else
    echo "✗ Virtual environment not found"
    echo "  Run: ./scripts/setup_python_env.sh"
    exit 1
fi
EOF

chmod +x "$PROJECT_ROOT/activate_env.sh"

print_success "Activation script created: activate_env.sh"

# ==============================================================================
# SUMMARY
# ==============================================================================

echo ""
print_success "============================================"
print_success "   PYTHON ENVIRONMENT SETUP COMPLETE!"
print_success "============================================"
echo ""
echo "To activate the environment:"
echo "  source activate_env.sh"
echo ""
echo "Or manually:"
echo "  source venv/bin/activate"
echo ""
echo "Installed packages:"
pip list | grep -E "torch|numpy|opencv|streamlit|pytest"
echo ""