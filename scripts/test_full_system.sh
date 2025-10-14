#!/bin/bash
# ==============================================================================
# FULL SYSTEM TEST SCRIPT
# Comprehensive end-to-end testing of the Mars Rover AI system
# ==============================================================================

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Test results
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_SKIPPED=0

echo -e "${BLUE}"
echo "============================================"
echo "   MARS ROVER AI - FULL SYSTEM TEST"
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

print_test() {
    echo -e "${CYAN}[TEST]${NC} $1"
}

print_section() {
    echo ""
    echo -e "${MAGENTA}==================== $1 ====================${NC}"
    echo ""
}

# Test tracking
test_passed() {
    TESTS_PASSED=$((TESTS_PASSED + 1))
    print_success "✓ $1"
}

test_failed() {
    TESTS_FAILED=$((TESTS_FAILED + 1))
    print_error "✗ $1"
}

test_skipped() {
    TESTS_SKIPPED=$((TESTS_SKIPPED + 1))
    print_warning "⊘ $1 (skipped)"
}

check_command() {
    if command -v $1 &> /dev/null; then
        return 0
    else
        return 1
    fi
}

# ==============================================================================
# ENVIRONMENT SETUP
# ==============================================================================

setup_environment() {
    print_section "ENVIRONMENT SETUP"
    
    cd "$PROJECT_ROOT"
    
    # Source ROS2
    if [ -f "/opt/ros/humble/setup.bash" ]; then
        source /opt/ros/humble/setup.bash
        test_passed "ROS2 environment sourced"
    else
        test_skipped "ROS2 not installed"
    fi
    
    # Source workspace
    if [ -f "install/setup.bash" ]; then
        source install/setup.bash
        test_passed "Workspace sourced"
    else
        test_failed "Workspace not built"
        print_error "Run: ./scripts/build_system.sh"
        exit 1
    fi
    
    # Activate Python environment
    if [ -d "venv" ]; then
        source venv/bin/activate
        test_passed "Python virtual environment activated"
    else
        test_failed "Python virtual environment not found"
        print_error "Run: ./scripts/setup_python_env.sh"
        exit 1
    fi
    
    # Set environment variables
    export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
    export NASA_API_KEY="${NASA_API_KEY:-DEMO_KEY}"
    
    test_passed "Environment variables set"
}

# ==============================================================================
# SYSTEM CHECKS
# ==============================================================================

test_system_requirements() {
    print_section "SYSTEM REQUIREMENTS"
    
    # Check Python version
    print_test "Checking Python version..."
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
    
    if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
        test_passed "Python $PYTHON_VERSION (>= 3.8)"
    else
        test_failed "Python $PYTHON_VERSION (requires >= 3.8)"
    fi
    
    # Check ROS2
    print_test "Checking ROS2..."
    if [ -n "$ROS_DISTRO" ]; then
        test_passed "ROS2 $ROS_DISTRO"
    else
        test_skipped "ROS2 not available"
    fi
    
    # Check CUDA
    print_test "Checking CUDA..."
    if check_command "nvidia-smi"; then
        CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
        test_passed "CUDA $CUDA_VERSION available"
    else
        test_skipped "CUDA not available (CPU-only mode)"
    fi
    
    # Check disk space
    print_test "Checking disk space..."
    AVAILABLE_GB=$(df -BG "$PROJECT_ROOT" | tail -1 | awk '{print $4}' | sed 's/G//')
    if [ "$AVAILABLE_GB" -gt 5 ]; then
        test_passed "Disk space: ${AVAILABLE_GB}GB available"
    else
        test_warning "Low disk space: ${AVAILABLE_GB}GB (recommend > 5GB)"
    fi
}

# ==============================================================================
# PYTHON DEPENDENCIES
# ==============================================================================

test_python_dependencies() {
    print_section "PYTHON DEPENDENCIES"
    
    python3 - <<EOF
import sys

packages = {
    'numpy': 'NumPy',
    'cv2': 'OpenCV',
    'torch': 'PyTorch',
    'torchvision': 'Torchvision',
    'PIL': 'Pillow',
    'yaml': 'PyYAML',
    'streamlit': 'Streamlit',
    'plotly': 'Plotly',
    'sklearn': 'scikit-learn',
    'pytest': 'pytest',
    'requests': 'Requests',
    'tqdm': 'tqdm'
}

failed = []
passed = 0

for module, name in packages.items():
    try:
        mod = __import__(module)
        version = getattr(mod, '__version__', 'unknown')
        print(f'[SUCCESS] ✓ {name} ({version})')
        passed += 1
    except ImportError:
        print(f'[ERROR] ✗ {name}')
        failed.append(name)

print(f'\nPassed: {passed}/{len(packages)}')

if failed:
    print(f'Failed: {", ".join(failed)}')
    sys.exit(1)
EOF
    
    if [ $? -eq 0 ]; then
        test_passed "All Python dependencies available"
    else
        test_failed "Missing Python dependencies"
    fi
}

# ==============================================================================
# PROJECT STRUCTURE
# ==============================================================================

test_project_structure() {
    print_section "PROJECT STRUCTURE"
    
    local required_dirs=(
        "ai/models"
        "ai/training"
        "ai/inference"
        "ai/data_fetcher"
        "integration"
        "sim/rover_description"
        "dashboard"
        "tests"
        "data"
    )
    
    for dir in "${required_dirs[@]}"; do
        print_test "Checking $dir..."
        if [ -d "$PROJECT_ROOT/$dir" ]; then
            test_passed "$dir exists"
        else
            test_failed "$dir missing"
        fi
    done
    
    # Check for key files
    local required_files=(
        "requirements.txt"
        "README.md"
        "ai/models/terrain_segmentation.py"
        "integration/ros_ai_bridge.py"
        "dashboard/app.py"
    )
    
    for file in "${required_files[@]}"; do
        print_test "Checking $file..."
        if [ -f "$PROJECT_ROOT/$file" ]; then
            test_passed "$file exists"
        else
            test_failed "$file missing"
        fi
    done
}

# ==============================================================================
# AI MODELS TEST
# ==============================================================================

test_ai_models() {
    print_section "AI MODELS"
    
    print_test "Testing terrain segmentation model..."
    python3 - <<EOF
import sys
sys.path.insert(0, "$PROJECT_ROOT")

try:
    from ai.models.terrain_segmentation import TerrainSegmentationModel, UNet
    import numpy as np
    
    # Test U-Net creation
    model = UNet(n_channels=3, n_classes=5)
    print("[SUCCESS] ✓ U-Net model created")
    
    # Test segmentation model
    seg_model = TerrainSegmentationModel(device='cpu', input_size=(512, 512))
    print("[SUCCESS] ✓ Segmentation model initialized")
    
    # Test prediction
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    result = seg_model.predict(test_image)
    
    assert 'classes' in result
    assert 'confidence' in result
    print("[SUCCESS] ✓ Model prediction works")
    
    print("\nAI models test: PASSED")
    
except Exception as e:
    print(f"[ERROR] ✗ AI models test failed: {e}")
    sys.exit(1)
EOF
    
    if [ $? -eq 0 ]; then
        test_passed "AI models functional"
    else
        test_failed "AI models test failed"
    fi
    
    print_test "Testing traversability analyzer..."
    python3 - <<EOF
import sys
sys.path.insert(0, "$PROJECT_ROOT")

try:
    from ai.models.traversability import TraversabilityAnalyzer
    import numpy as np
    
    analyzer = TraversabilityAnalyzer()
    print("[SUCCESS] ✓ Traversability analyzer created")
    
    # Test analysis
    seg = np.random.randint(0, 5, (512, 512), dtype=np.uint8)
    conf = np.random.rand(512, 512).astype(np.float32)
    
    result = analyzer.analyze(seg, conf)
    
    assert 'traversability_map' in result
    assert 'best_direction' in result
    print("[SUCCESS] ✓ Traversability analysis works")
    
except Exception as e:
    print(f"[ERROR] ✗ Traversability test failed: {e}")
    sys.exit(1)
EOF
    
    if [ $? -eq 0 ]; then
        test_passed "Traversability analyzer functional"
    else
        test_failed "Traversability analyzer test failed"
    fi
    
    print_test "Testing hazard detector..."
    python3 - <<EOF
import sys
sys.path.insert(0, "$PROJECT_ROOT")

try:
    from ai.models.hazard_detector import HazardDetector
    import numpy as np
    
    detector = HazardDetector()
    print("[SUCCESS] ✓ Hazard detector created")
    
    # Test detection
    img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    seg = np.random.randint(0, 5, (512, 512), dtype=np.uint8)
    conf = np.random.rand(512, 512).astype(np.float32)
    
    hazards = detector.detect(img, seg, conf)
    print(f"[SUCCESS] ✓ Hazard detection works (found {len(hazards)} hazards)")
    
except Exception as e:
    print(f"[ERROR] ✗ Hazard detector test failed: {e}")
    sys.exit(1)
EOF
    
    if [ $? -eq 0 ]; then
        test_passed "Hazard detector functional"
    else
        test_failed "Hazard detector test failed"
    fi
}

# ==============================================================================
# INFERENCE PIPELINE TEST
# ==============================================================================

test_inference_pipeline() {
    print_section "INFERENCE PIPELINE"
    
    print_test "Testing complete inference pipeline..."
    python3 - <<EOF
import sys
sys.path.insert(0, "$PROJECT_ROOT")

try:
    from ai.inference.realtime_pipeline import RealtimePipeline, PipelineConfig
    import numpy as np
    import time
    
    # Create pipeline
    config = PipelineConfig(device='cpu', input_size=(512, 512))
    pipeline = RealtimePipeline(config)
    print("[SUCCESS] ✓ Pipeline created")
    
    # Test processing
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    start = time.time()
    result = pipeline.process(test_image)
    elapsed = (time.time() - start) * 1000
    
    print(f"[SUCCESS] ✓ Inference completed in {elapsed:.1f}ms")
    
    # Verify results
    assert result.classes is not None
    assert result.confidence is not None
    assert result.traversability_map is not None
    print("[SUCCESS] ✓ All outputs generated")
    
    # Performance check
    if elapsed < 500:
        print(f"[SUCCESS] ✓ Performance acceptable ({elapsed:.1f}ms < 500ms)")
    else:
        print(f"[WARNING] ⚠ Performance slow ({elapsed:.1f}ms > 500ms)")
    
except Exception as e:
    print(f"[ERROR] ✗ Inference pipeline test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF
    
    if [ $? -eq 0 ]; then
        test_passed "Inference pipeline functional"
    else
        test_failed "Inference pipeline test failed"
    fi
}

# ==============================================================================
# INTEGRATION TEST
# ==============================================================================

test_integration() {
    print_section "INTEGRATION COMPONENTS"
    
    print_test "Testing costmap generator..."
    python3 - <<EOF
import sys
sys.path.insert(0, "$PROJECT_ROOT")

try:
    from integration.costmap_generator import CostmapGenerator
    import numpy as np
    
    generator = CostmapGenerator()
    print("[SUCCESS] ✓ Costmap generator created")
    
    # Test generation
    trav = np.random.rand(512, 512).astype(np.float32)
    costmap = generator.generate_costmap(trav)
    
    assert costmap.shape == (200, 200)
    print("[SUCCESS] ✓ Costmap generation works")
    
except Exception as e:
    print(f"[ERROR] ✗ Costmap test failed: {e}")
    sys.exit(1)
EOF
    
    if [ $? -eq 0 ]; then
        test_passed "Costmap generator functional"
    else
        test_failed "Costmap generator test failed"
    fi
    
    # Test ROS2 integration (if ROS2 available)
    if [ -n "$ROS_DISTRO" ]; then
        print_test "Testing ROS2 packages..."
        
        # Check if packages are built
        if ros2 pkg list | grep -q "rover_description"; then
            test_passed "ROS2 packages found"
        else
            test_failed "ROS2 packages not built"
        fi
    else
        test_skipped "ROS2 integration (ROS2 not available)"
    fi
}

# ==============================================================================
# DASHBOARD TEST
# ==============================================================================

test_dashboard() {
    print_section "DASHBOARD COMPONENTS"
    
    print_test "Testing dashboard utilities..."
    python3 - <<EOF
import sys
sys.path.insert(0, "$PROJECT_ROOT")

try:
    from dashboard.utils.ros_connector import ROSConnector, TerrainData
    from dashboard.utils.data_processor import DataProcessor
    from dashboard.utils.visualizations import create_terrain_plot
    import time
    
    # Test connector
    connector = ROSConnector()
    print("[SUCCESS] ✓ ROS connector created")
    
    # Test data processor
    processor = DataProcessor(history_size=50)
    print("[SUCCESS] ✓ Data processor created")
    
    # Test with sample data
    data = TerrainData(
        timestamp=time.time(),
        avg_traversability=0.75,
        num_hazards=3,
        fps=5.0
    )
    
    processed = processor.process(data)
    print("[SUCCESS] ✓ Data processing works")
    
    # Test visualization
    if len(processed.time_history) > 0:
        fig = create_terrain_plot(
            processed.time_history,
            processed.traversability_history
        )
        print("[SUCCESS] ✓ Visualization generation works")
    
    print("\nDashboard test: PASSED")
    
except Exception as e:
    print(f"[ERROR] ✗ Dashboard test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF
    
    if [ $? -eq 0 ]; then
        test_passed "Dashboard components functional"
    else
        test_failed "Dashboard test failed"
    fi
}

# ==============================================================================
# UNIT TESTS
# ==============================================================================

run_unit_tests() {
    print_section "UNIT TESTS"
    
    if [ -d "tests" ]; then
        print_test "Running pytest suite..."
        
        # Run fast tests only
        pytest tests/ -v -m "not slow" --tb=short 2>&1 | tee test_output.log
        
        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            test_passed "Unit tests passed"
        else
            test_failed "Some unit tests failed"
            print_warning "Check test_output.log for details"
        fi
    else
        test_skipped "Unit tests (tests/ directory not found)"
    fi
}

# ==============================================================================
# DATA CHECK
# ==============================================================================

check_data() {
    print_section "DATA AVAILABILITY"
    
    # Check for data directories
    print_test "Checking data directories..."
    
    if [ -d "data/raw" ]; then
        test_passed "data/raw exists"
        
        # Check subdirectories
        for subdir in perseverance ai4mars dem; do
            if [ -d "data/raw/$subdir" ] && [ "$(ls -A data/raw/$subdir 2>/dev/null)" ]; then
                test_passed "data/raw/$subdir has data"
            else
                test_warning "data/raw/$subdir is empty"
            fi
        done
    else
        test_warning "data/raw not found"
        print_warning "Run: ./scripts/download_data.sh --minimal"
    fi
    
    # Check for model weights
    print_test "Checking model weights..."
    if [ -f "ai/models/weights/terrain_unet_best.pth" ]; then
        test_passed "Pretrained model found"
    else
        test_warning "No pretrained model (system will use untrained model)"
    fi
}

# ==============================================================================
# PERFORMANCE BENCHMARK
# ==============================================================================

performance_benchmark() {
    print_section "PERFORMANCE BENCHMARK"
    
    print_test "Running performance benchmark..."
    python3 - <<EOF
import sys
sys.path.insert(0, "$PROJECT_ROOT")

try:
    from ai.inference.realtime_pipeline import RealtimePipeline, PipelineConfig
    import numpy as np
    import time
    
    config = PipelineConfig(device='cpu', input_size=(512, 512))
    pipeline = RealtimePipeline(config)
    
    # Warm up
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    pipeline.process(test_image)
    
    # Benchmark
    times = []
    for _ in range(10):
        start = time.time()
        pipeline.process(test_image)
        times.append((time.time() - start) * 1000)
    
    avg_time = sum(times) / len(times)
    fps = 1000 / avg_time if avg_time > 0 else 0
    
    print(f"[SUCCESS] Average inference time: {avg_time:.1f}ms")
    print(f"[SUCCESS] Estimated FPS: {fps:.1f}")
    
    if avg_time < 500:
        print("[SUCCESS] ✓ Performance: GOOD")
    elif avg_time < 1000:
        print("[WARNING] ⚠ Performance: ACCEPTABLE")
    else:
        print("[WARNING] ⚠ Performance: SLOW (consider GPU)")
    
except Exception as e:
    print(f"[ERROR] ✗ Benchmark failed: {e}")
    sys.exit(1)
EOF
    
    if [ $? -eq 0 ]; then
        test_passed "Performance benchmark completed"
    else
        test_failed "Performance benchmark failed"
    fi
}

# ==============================================================================
# GENERATE REPORT
# ==============================================================================

generate_report() {
    print_section "TEST SUMMARY"
    
    local total=$((TESTS_PASSED + TESTS_FAILED + TESTS_SKIPPED))
    local pass_rate=0
    
    if [ $total -gt 0 ]; then
        pass_rate=$((TESTS_PASSED * 100 / total))
    fi
    
    echo ""
    echo "Total Tests: $total"
    echo -e "${GREEN}Passed: $TESTS_PASSED${NC}"
    echo -e "${RED}Failed: $TESTS_FAILED${NC}"
    echo -e "${YELLOW}Skipped: $TESTS_SKIPPED${NC}"
    echo ""
    echo "Pass Rate: ${pass_rate}%"
    echo ""
    
    # Save report
    cat > test_report.txt <<EOF
MARS ROVER AI - FULL SYSTEM TEST REPORT
Generated: $(date)

SUMMARY
=======
Total Tests: $total
Passed: $TESTS_PASSED
Failed: $TESTS_FAILED
Skipped: $TESTS_SKIPPED
Pass Rate: ${pass_rate}%

ENVIRONMENT
===========
OS: $(uname -s)
Python: $(python3 --version)
ROS2: ${ROS_DISTRO:-Not installed}
CUDA: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo "Not available")

PROJECT
=======
Location: $PROJECT_ROOT
Git Branch: $(git branch --show-current 2>/dev/null || echo "Unknown")
Git Commit: $(git rev-parse --short HEAD 2>/dev/null || echo "Unknown")
EOF
    
    print_success "Report saved: test_report.txt"
    
    # Exit code
    if [ $TESTS_FAILED -gt 0 ]; then
        echo ""
        print_error "Some tests failed. Please review the output above."
        return 1
    else
        echo ""
        print_success "All tests passed!"
        return 0
    fi
}

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

main() {
    local run_all=true
    local run_quick=false
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --quick|-q)
                run_quick=true
                shift
                ;;
            --help|-h)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --quick, -q    Run quick tests only"
                echo "  --help, -h     Show this help message"
                echo ""
                echo "This script performs comprehensive system testing including:"
                echo "  - Environment checks"
                echo "  - Dependency verification"
                echo "  - AI model tests"
                echo "  - Integration tests"
                echo "  - Dashboard tests"
                echo "  - Unit tests"
                echo "  - Performance benchmarks"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                echo "Run '$0 --help' for usage information"
                exit 1
                ;;
        esac
    done
    
    # Start time
    START_TIME=$(date +%s)
    
    # Run tests
    setup_environment
    test_system_requirements
    test_python_dependencies
    test_project_structure
    
    if [ "$run_quick" = false ]; then
        test_ai_models
        test_inference_pipeline
        test_integration
        test_dashboard
        run_unit_tests
        check_data
        performance_benchmark
    fi
    
    # End time
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    echo ""
    print_status "Test duration: ${DURATION}s"
    
    # Generate report
    generate_report
    
    # Exit with appropriate code
    if [ $TESTS_FAILED -gt 0 ]; then
        exit 1
    else
        exit 0
    fi
}

# Run main function
main "$@"