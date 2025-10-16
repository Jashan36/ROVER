#!/bin/bash
# ==============================================================================
# DATA DOWNLOAD SCRIPT
# Downloads all required datasets for Mars Rover AI system
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
echo "   MARS ROVER AI - DATA DOWNLOAD"
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
        print_error "$1 is not installed. Please install it first."
        return 1
    fi
    return 0
}

check_python_package() {
    python3 -c "import $1" 2>/dev/null
    if [ $? -ne 0 ]; then
        print_error "Python package '$1' is not installed."
        return 1
    fi
    return 0
}

# ==============================================================================
# ENVIRONMENT CHECK
# ==============================================================================

print_status "Checking environment..."

# Check for required commands
check_command "python3" || exit 1
check_command "wget" || exit 1
check_command "unzip" || exit 1

# Check for required Python packages
check_python_package "requests" || exit 1

print_success "Environment check passed"

# ==============================================================================
# CREATE DIRECTORIES
# ==============================================================================

print_status "Creating data directories..."

DATA_DIR="$PROJECT_ROOT/data"
mkdir -p "$DATA_DIR/raw/perseverance"
mkdir -p "$DATA_DIR/raw/ai4mars"
mkdir -p "$DATA_DIR/raw/dem"
mkdir -p "$DATA_DIR/processed"
mkdir -p "$DATA_DIR/cache"

print_success "Directories created"

# ==============================================================================
# DOWNLOAD AI4MARS DATASET
# ==============================================================================

download_ai4mars() {
    print_status "Downloading AI4Mars dataset..."
    
    local DATASET_TYPE=${1:-"sample"}  # "sample" or "full"
    
    cd "$DATA_DIR/raw/ai4mars"
    
    # Activate virtual environment if exists
    if [ -d "$PROJECT_ROOT/venv" ]; then
        source "$PROJECT_ROOT/venv/bin/activate"
    fi
    
    # Run Python downloader
    python3 - <<EOF
import sys
sys.path.insert(0, "$PROJECT_ROOT")

from ai.data_fetcher.ai4mars_loader import AI4MarsLoader

print("Initializing AI4Mars loader...")
loader = AI4MarsLoader(
    root_dir="$DATA_DIR/raw/ai4mars",
    version="$DATASET_TYPE"
)

print("Downloading dataset...")
archive_path = loader.download()

print("Extracting dataset...")
loader.extract(archive_path)

print("Dataset ready!")
stats = loader.get_statistics()
print(f"Total images: {stats.get('total_images', 0)}")
EOF
    
    if [ $? -eq 0 ]; then
        print_success "AI4Mars dataset downloaded ($DATASET_TYPE version)"
    else
        print_error "Failed to download AI4Mars dataset"
        return 1
    fi
}

# ==============================================================================
# DOWNLOAD NASA PERSEVERANCE IMAGES
# ==============================================================================

download_perseverance_images() {
    print_status "Downloading NASA Perseverance images..."
    
    # Check for NASA API key
    if [ -z "$NASA_API_KEY" ]; then
        print_warning "NASA_API_KEY not set. Using DEMO_KEY (limited to 30 requests/hour)"
        print_warning "Get a free API key at: https://api.nasa.gov/"
        export NASA_API_KEY="DEMO_KEY"
    fi
    
    cd "$DATA_DIR/raw/perseverance"
    
    # Activate virtual environment if exists
    if [ -d "$PROJECT_ROOT/venv" ]; then
        source "$PROJECT_ROOT/venv/bin/activate"
    fi
    
    # Run Python downloader
    python3 - <<EOF
import sys
sys.path.insert(0, "$PROJECT_ROOT")

from ai.data_fetcher.perseverance_api import fetch_latest_images

print("Fetching latest Perseverance images...")
images = fetch_latest_images(
    camera='NAVCAM_LEFT',
    limit=25,
    download=True,
    cache_dir="$DATA_DIR/raw/perseverance"
)

print(f"Downloaded {len(images)} images")
for img in images[:5]:
    print(f"  - Sol {img.sol}: {img.camera}")
EOF
    
    if [ $? -eq 0 ]; then
        print_success "Perseverance images downloaded"
    else
        print_error "Failed to download Perseverance images"
        return 1
    fi
}

# ==============================================================================
# GENERATE DEM HEIGHTMAP
# ==============================================================================

generate_dem() {
    print_status "Generating Mars DEM heightmap..."
    
    cd "$DATA_DIR/raw/dem"
    
    # Activate virtual environment if exists
    if [ -d "$PROJECT_ROOT/venv" ]; then
        source "$PROJECT_ROOT/venv/bin/activate"
    fi
    
    # Run Python generator
    python3 - <<EOF
import sys
sys.path.insert(0, "$PROJECT_ROOT")

from ai.data_fetcher.dem_processor import generate_mars_heightmap

print("Generating heightmap for Gazebo simulation...")
heightmap_path, texture_path = generate_mars_heightmap(
    output_dir="$DATA_DIR/raw/dem",
    size=(1024, 1024),
    z_range_meters=(0.0, 10.0),
    with_texture=True
)

print(f"Heightmap: {heightmap_path}")
print(f"Texture: {texture_path}")
EOF
    
    if [ $? -eq 0 ]; then
        print_success "DEM heightmap generated"
        
        # Copy to simulation directory
        if [ -d "$PROJECT_ROOT/sim/rover_description/worlds" ]; then
            cp "$DATA_DIR/raw/dem/mars_heightmap.png" "$PROJECT_ROOT/sim/rover_description/worlds/"
            cp "$DATA_DIR/raw/dem/mars_texture.png" "$PROJECT_ROOT/sim/rover_description/worlds/" 2>/dev/null || true
            print_success "Heightmap copied to simulation directory"
        fi
    else
        print_error "Failed to generate DEM heightmap"
        return 1
    fi
}

# ==============================================================================
# DOWNLOAD PRETRAINED MODELS
# ==============================================================================

download_models() {
    print_status "Checking for pretrained models..."
    
    MODELS_DIR="$PROJECT_ROOT/ai/models/weights"
    mkdir -p "$MODELS_DIR"
    
    if [ -f "$MODELS_DIR/terrain_unet_best.pth" ]; then
        print_success "Pretrained model already exists"
    else
        print_warning "No pretrained model found"
        print_warning "You will need to either:"
        print_warning "  1. Train your own model: python ai/training/train_segmentation.py"
        print_warning "  2. Download pretrained weights (if available)"
        print_warning ""
        print_warning "The system can still run, but with reduced accuracy"
    fi
}

# ==============================================================================
# MAIN MENU
# ==============================================================================

show_menu() {
    echo ""
    echo "Select datasets to download:"
    echo "  1) AI4Mars Sample Dataset (32 MB) - Fast, for testing"
    echo "  2) AI4Mars Full Dataset (326 MB) - Complete training data"
    echo "  3) NASA Perseverance Images (25 images, ~50 MB)"
    echo "  4) Generate DEM Heightmap for simulation"
    echo "  5) Check pretrained models"
    echo "  6) Download ALL (Full AI4Mars + Perseverance + DEM)"
    echo "  7) Download MINIMAL (Sample AI4Mars + DEM)"
    echo "  0) Exit"
    echo ""
}

process_choice() {
    local choice=$1
    
    case $choice in
        1)
            download_ai4mars "sample"
            ;;
        2)
            download_ai4mars "full"
            ;;
        3)
            download_perseverance_images
            ;;
        4)
            generate_dem
            ;;
        5)
            download_models
            ;;
        6)
            print_status "Downloading all datasets..."
            download_ai4mars "full"
            download_perseverance_images
            generate_dem
            download_models
            print_success "All datasets downloaded!"
            ;;
        7)
            print_status "Downloading minimal datasets..."
            download_ai4mars "sample"
            generate_dem
            download_models
            print_success "Minimal datasets downloaded!"
            ;;
        0)
            print_status "Exiting..."
            exit 0
            ;;
        *)
            print_error "Invalid choice"
            ;;
    esac
}

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

# Check if running with arguments
if [ $# -eq 0 ]; then
    # Interactive mode
    while true; do
        show_menu
        read -p "Enter choice [0-7]: " choice
        process_choice "$choice"
        echo ""
        read -p "Press Enter to continue..."
    done
else
    # Command line mode
    case $1 in
        --sample)
            download_ai4mars "sample"
            ;;
        --full)
            download_ai4mars "full"
            ;;
        --perseverance)
            download_perseverance_images
            ;;
        --dem)
            generate_dem
            ;;
        --all)
            download_ai4mars "full"
            download_perseverance_images
            generate_dem
            download_models
            ;;
        --minimal)
            download_ai4mars "sample"
            generate_dem
            download_models
            ;;
        --help|-h)
            echo "Usage: $0 [OPTION]"
            echo ""
            echo "Options:"
            echo "  --sample         Download AI4Mars sample dataset"
            echo "  --full           Download AI4Mars full dataset"
            echo "  --perseverance   Download NASA Perseverance images"
            echo "  --dem            Generate DEM heightmap"
            echo "  --all            Download everything"
            echo "  --minimal        Download minimal datasets (sample + DEM)"
            echo "  --help           Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  NASA_API_KEY     NASA API key (get at https://api.nasa.gov/)"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Run '$0 --help' for usage information"
            exit 1
            ;;
    esac
fi

echo ""
print_success "Data download script completed!"
echo ""
echo "Data location: $DATA_DIR"
echo ""