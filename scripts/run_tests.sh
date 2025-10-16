#!/bin/bash
# ==============================================================================
# TEST RUNNER SCRIPT
# Runs the Mars Rover AI test suite with various options
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
echo "   MARS ROVER AI - TEST RUNNER"
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

# ==============================================================================
# ENVIRONMENT SETUP
# ==============================================================================

setup_environment() {
    print_status "Setting up test environment..."
    
    cd "$PROJECT_ROOT"
    
    # Source ROS2 if available
    if [ -f "/opt/ros/humble/setup.bash" ]; then
        source /opt/ros/humble/setup.bash
    fi
    
    # Source workspace if built
    if [ -f "install/setup.bash" ]; then
        source install/setup.bash
    fi
    
    # Activate Python virtual environment
    if [ -d "venv" ]; then
        source venv/bin/activate
    else
        print_warning "Virtual environment not found. Run ./scripts/build_system.sh first"
    fi
    
    # Install test dependencies
    pip install pytest pytest-cov pytest-html -q
    
    print_success "Environment ready"
}

# ==============================================================================
# TEST FUNCTIONS
# ==============================================================================

run_all_tests() {
    print_status "Running all tests..."
    pytest tests/ -v --tb=short
}

run_perception_tests() {
    print_status "Running perception tests..."
    pytest tests/test_perception.py -v
}

run_navigation_tests() {
    print_status "Running navigation tests..."
    pytest tests/test_navigation.py -v
}

run_integration_tests() {
    print_status "Running integration tests..."
    pytest tests/test_integration.py -v
}

run_dashboard_tests() {
    print_status "Running dashboard tests..."
    pytest tests/test_dashboard.py -v
}

run_fast_tests() {
    print_status "Running fast tests only..."
    pytest tests/ -v -m "not slow"
}

run_with_coverage() {
    print_status "Running tests with coverage..."
    pytest tests/ \
        --cov=ai \
        --cov=integration \
        --cov=dashboard \
        --cov-report=html \
        --cov-report=term \
        -v
    
    print_success "Coverage report generated: htmlcov/index.html"
}

run_specific_test() {
    local test_path=$1
    print_status "Running specific test: $test_path"
    pytest "$test_path" -v
}

run_with_html_report() {
    print_status "Running tests with HTML report..."
    pytest tests/ \
        --html=test_report.html \
        --self-contained-html \
        -v
    
    print_success "HTML report generated: test_report.html"
}

run_performance_tests() {
    print_status "Running performance tests..."
    pytest tests/ -v -k "performance or speed or latency"
}

run_parallel_tests() {
    print_status "Running tests in parallel..."
    
    # Install pytest-xdist if not available
    pip install pytest-xdist -q
    
    pytest tests/ -v -n auto
}

# ==============================================================================
# MAIN MENU
# ==============================================================================

show_menu() {
    echo ""
    echo "Select test suite to run:"
    echo "  1) All tests"
    echo "  2) Perception tests only"
    echo "  3) Navigation tests only"
    echo "  4) Integration tests only"
    echo "  5) Dashboard tests only"
    echo "  6) Fast tests only (skip slow tests)"
    echo "  7) Tests with coverage report"
    echo "  8) Tests with HTML report"
    echo "  9) Performance tests"
    echo " 10) Parallel test execution"
    echo "  0) Exit"
    echo ""
}

process_choice() {
    local choice=$1
    
    case $choice in
        1)
            run_all_tests
            ;;
        2)
            run_perception_tests
            ;;
        3)
            run_navigation_tests
            ;;
        4)
            run_integration_tests
            ;;
        5)
            run_dashboard_tests
            ;;
        6)
            run_fast_tests
            ;;
        7)
            run_with_coverage
            ;;
        8)
            run_with_html_report
            ;;
        9)
            run_performance_tests
            ;;
        10)
            run_parallel_tests
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

main() {
    setup_environment
    
    # Check if running with arguments
    if [ $# -eq 0 ]; then
        # Interactive mode
        while true; do
            show_menu
            read -p "Enter choice [0-10]: " choice
            echo ""
            process_choice "$choice"
            
            echo ""
            if [ $? -eq 0 ]; then
                print_success "Tests completed"
            else
                print_error "Tests failed"
            fi
            echo ""
            read -p "Press Enter to continue..."
        done
    else
        # Command line mode
        case $1 in
            --all|-a)
                run_all_tests
                ;;
            --perception)
                run_perception_tests
                ;;
            --navigation)
                run_navigation_tests
                ;;
            --integration)
                run_integration_tests
                ;;
            --dashboard)
                run_dashboard_tests
                ;;
            --fast)
                run_fast_tests
                ;;
            --coverage|-c)
                run_with_coverage
                ;;
            --html)
                run_with_html_report
                ;;
            --performance)
                run_performance_tests
                ;;
            --parallel|-p)
                run_parallel_tests
                ;;
            --test)
                if [ -z "$2" ]; then
                    print_error "Please specify test path"
                    echo "Usage: $0 --test tests/test_perception.py::TestUNet"
                    exit 1
                fi
                run_specific_test "$2"
                ;;
            --help|-h)
                echo "Usage: $0 [OPTION]"
                echo ""
                echo "Options:"
                echo "  --all, -a          Run all tests"
                echo "  --perception       Run perception tests"
                echo "  --navigation       Run navigation tests"
                echo "  --integration      Run integration tests"
                echo "  --dashboard        Run dashboard tests"
                echo "  --fast             Run fast tests only"
                echo "  --coverage, -c     Run with coverage report"
                echo "  --html             Generate HTML test report"
                echo "  --performance      Run performance tests"
                echo "  --parallel, -p     Run tests in parallel"
                echo "  --test PATH        Run specific test"
                echo "  --help, -h         Show this help message"
                echo ""
                echo "Examples:"
                echo "  $0 --all"
                echo "  $0 --coverage"
                echo "  $0 --test tests/test_perception.py::TestUNet::test_forward_pass"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                echo "Run '$0 --help' for usage information"
                exit 1
                ;;
        esac
    fi
}

# Run main function
main "$@"