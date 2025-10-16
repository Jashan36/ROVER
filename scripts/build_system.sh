#!/bin/bash
# ==============================================================================
# SYSTEM BUILD SCRIPT
# Builds ROS2 packages and sets up the Mars Rover AI system
# ==============================================================================

set -e  # Exit on error

# Colors for output
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
echo "   MARS ROVER AI - SYSTEM BUILD"
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
# ENVIRONMENT CHECK
# ==============================================================================

print_status "Checking build environment..."

# Check for ROS2
if [ -z "$ROS_DISTRO" ]; then
    print_error "ROS2 not sourced. Run: source /opt/ros/humble/setup.bash"
    exit 1
fi

print_success "ROS2 $ROS_DISTRO detected"

# Check for required tools
check_command "colcon" || exit 1
check_command "python3" || exit 1

print_success "Build tools available"

# ==============================================================================
# PYTHON VIRTUAL ENVIRONMENT
# ==============================================================================

setup_python_env() {
    print_status "Setting up Python environment..."
    
    cd "$PROJECT_ROOT"
    
    if [ ! -d "venv" ]; then
        print_status "Creating virtual environment..."
        python3 -m venv venv
        print_success "Virtual environment created"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    print_status "Upgrading pip..."
    pip install --upgrade pip -q
    
    # Install requirements
    if [ -f "requirements.txt" ]; then
        print_status "Installing Python dependencies..."
        pip install -r requirements.txt -q
        print_success "Python dependencies installed"
    fi
    
    # Install PyTorch (check for GPU)
    if command -v nvidia-smi &> /dev/null; then
        print_status "NVIDIA GPU detected, installing PyTorch with CUDA support..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -q
    else
        print_status "No GPU detected, installing PyTorch CPU version..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu -q
    fi
    
    print_success "Python environment ready"
}

# ==============================================================================
# ROS2 WORKSPACE BUILD
# ==============================================================================

build_ros_workspace() {
    print_status "Building ROS2 workspace..."
    
    cd "$PROJECT_ROOT"
    
    # Clean previous build (optional)
    if [ "$CLEAN_BUILD" = true ]; then
        print_status "Cleaning previous build..."
        rm -rf build/ install/ log/
        print_success "Build cleaned"
    fi
    
    # Build with colcon
    print_status "Running colcon build..."
    
    colcon build \
        --symlink-install \
        --cmake-args -DCMAKE_BUILD_TYPE=Release \
        --parallel-workers $(nproc) \
        2>&1 | tee build.log
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        print_success "ROS2 packages built successfully"
    else
        print_error "Build failed. Check build.log for details"
        return 1
    fi
    
    # Source the workspace
    source install/setup.bash
    
    print_success "Workspace sourced"
}

# ==============================================================================
# VERIFY BUILD
# ==============================================================================

verify_build() {
    print_status "Verifying build..."
    
    # Check for key packages
    local packages=(
        "rover_description"
        "rover_integration"
    )
    
    for pkg in "${packages[@]}"; do
        if [ -d "install/$pkg" ]; then
            print_success "Package '$pkg' built"
        else
            print_warning "Package '$pkg' not found"
        fi
    done
    
    # Check Python imports
    print_status "Checking Python imports..."
    
    python3 - <<EOF
try:
    import ai
    import integration
    import dashboard
    print("✓ All Python packages importable")
except ImportError as e:
    print(f"✗ Import error: {e}")
    exit(1)
EOF
    
    if [ $? -eq 0 ]; then
        print_success "Python imports verified"
    else
        print_error "Python import verification failed"
        return 1
    fi
}

# ==============================================================================
# CREATE LAUNCH SCRIPTS
# ==============================================================================

create_launch_scripts() {
    print_status "Creating convenience launch scripts..."
    
    cd "$PROJECT_ROOT"
    
    # Create launch_simulation.sh
    cat > launch_simulation.sh <<'EOF'
#!/bin/bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch rover_description full_system.launch.py
EOF
    chmod +x launch_simulation.sh
    
    # Create launch_dashboard.sh
    cat > launch_dashboard.sh <<'EOF'
#!/bin/bash
source venv/bin/activate
cd dashboard
streamlit run app.py
EOF
    chmod +x launch_dashboard.sh
    
    # Create quick_test.sh
    cat > quick_test.sh <<'EOF'
#!/bin/bash
source /opt/ros/humble/setup.bash
source install/setup.bash
source venv/bin/activate
pytest tests/ -v
EOF
    chmod +x quick_test.sh
    
    print_success "Launch scripts created"
}

# ==============================================================================
# SETUP ENVIRONMENT FILE
# ==============================================================================

create_env_file() {
    print_status "Creating environment setup file..."
    
    cat > "$PROJECT_ROOT/setup_env.sh" <<'EOF'
#!/bin/bash
# Mars Rover AI Environment Setup

# ROS2
if [ -f "/opt/ros/humble/setup.bash" ]; then
    source /opt/ros/humble/setup.bash
fi

# Workspace
if [ -f "install/setup.bash" ]; then
    source install/setup.bash
fi

# Python virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Environment variables
export NASA_API_KEY="${NASA_API_KEY:-DEMO_KEY}"
export PYTHONPATH="${PWD}:${PYTHONPATH}"

echo "Mars Rover AI environment loaded"
echo "  ROS2: $ROS_DISTRO"
echo "  Python: $(python3 --version)"
EOF
    
    chmod +x "$PROJECT_ROOT/setup_env.sh"
    
    print_success "Environment file created: setup_env.sh"
}

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

main() {
    local CLEAN_BUILD=false
    local SKIP_PYTHON=false
    local SKIP_ROS=false
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --clean)
                CLEAN_BUILD=true
                shift
                ;;
            --skip-python)
                SKIP_PYTHON=true
                shift
                ;;
            --skip-ros)
                SKIP_ROS=true
                shift
                ;;
            --help|-h)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --clean        Clean previous build before building"
                echo "  --skip-python  Skip Python environment setup"
                echo "  --skip-ros     Skip ROS2 workspace build"
                echo "  --help         Show this help message"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                echo "Run '$0 --help' for usage information"
                exit 1
                ;;
        esac
    done
    
    # Build process
    echo ""
    print_status "Starting build process..."
    echo ""
    
    if [ "$SKIP_PYTHON" != true ]; then
        setup_python_env
        echo ""
    fi
    
    if [ "$SKIP_ROS" != true ]; then
        build_ros_workspace
        echo ""
    fi
    
    verify_build
    echo ""
    
    create_launch_scripts
    echo ""
    
    create_env_file
    echo ""
    
    # Final summary
    echo -e "${GREEN}"
    echo "============================================"
    echo "   BUILD COMPLETED SUCCESSFULLY!"
    echo "============================================"
    echo -e "${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Source environment:    source setup_env.sh"
    echo "  2. Download data:         ./scripts/download_data.sh --minimal"
    echo "  3. Run tests:             ./quick_test.sh"
    echo "  4. Launch simulation:     ./launch_simulation.sh"
    echo "  5. Launch dashboard:      ./launch_dashboard.sh"
    echo ""
    echo "For full documentation, see README.md"
    echo ""
}

# Run main function
main "$@"