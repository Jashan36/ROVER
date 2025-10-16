#!/bin/bash

# ============================================
# Mars Rover AI - Full System Launcher
# ============================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"
VENV_DIR="$ROOT_DIR/venv"
INSTALL_DIR="$ROOT_DIR/install"
LOG_DIR="$ROOT_DIR/logs"

# Logging functions
log_info() {
    echo -e "${BLUE}📘 INFO:${NC} $1"
}

log_success() {
    echo -e "${GREEN}✅ SUCCESS:${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}⚠️ WARNING:${NC} $1"
}

log_error() {
    echo -e "${RED}❌ ERROR:${NC} $1"
}

# Banner
print_banner() {
    echo -e "${BLUE}"
    echo "==========================================="
    echo "   🚀 MARS ROVER AI - FULL SYSTEM LAUNCH"
    echo "==========================================="
    echo -e "${NC}"
}

# Check system requirements
check_requirements() {
    log_info "Checking system requirements..."
    
    # Check ROS2
    if [ -z "$ROS_DISTRO" ]; then
        log_error "ROS2 not sourced. Please run: source /opt/ros/humble/setup.bash"
        exit 1
    fi
    
    if [ "$ROS_DISTRO" != "humble" ]; then
        log_warning "ROS2 distribution is $ROS_DISTRO, expected 'humble'"
    fi
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 not found"
        exit 1
    fi
    
    # Check Gazebo
    if ! command -v gazebo &> /dev/null; then
        log_warning "Gazebo not found - simulation will not work"
    fi
    
    log_success "System requirements check passed"
}

# Setup environment
setup_environment() {
    log_info "Setting up environment..."
    
    # Create log directory
    mkdir -p "$LOG_DIR"
    
    # Source ROS2
    source /opt/ros/humble/setup.bash
    
    # Source workspace if built
    if [ -f "$INSTALL_DIR/setup.bash" ]; then
        source "$INSTALL_DIR/setup.bash"
    else
        log_warning "Workspace not built, running initial setup..."
        ./scripts/build_system.sh
        source "$INSTALL_DIR/setup.bash"
    fi
    
    # Activate Python virtual environment
    if [ -f "$VENV_DIR/bin/activate" ]; then
        source "$VENV_DIR/bin/activate"
    else
        log_warning "Python virtual environment not found"
    fi
    
    # Set environment variables
    export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"
    export GAZEBO_MODEL_PATH="$ROOT_DIR/sim/rover_description:$GAZEBO_MODEL_PATH"
    
    log_success "Environment setup complete"
}

# Check if components are built
check_built_components() {
    log_info "Checking built components..."
    
    local missing_components=()
    
    # Check ROS2 packages
    if [ ! -f "$INSTALL_DIR/setup.bash" ]; then
        missing_components+=("ROS2 workspace")
    fi
    
    # Check Python packages
    if ! python3 -c "import ai" &> /dev/null; then
        missing_components+=("AI package")
    fi
    
    if ! python3 -c "import integration" &> /dev/null; then
        missing_components+=("Integration package")
    fi
    
    if [ ${#missing_components[@]} -ne 0 ]; then
        log_warning "Missing components: ${missing_components[*]}"
        log_info "Running build script..."
        ./scripts/build_system.sh
    fi
    
    log_success "All components built and ready"
}

# Launch simulation and navigation
launch_simulation() {
    log_info "Launching simulation and navigation..."
    
    gnome-terminal --title="🚀 Rover System" --working-directory="$ROOT_DIR" -- bash -c "
        echo '🔄 Starting Gazebo, Nav2, and AI Vision...'
        source /opt/ros/humble/setup.bash
        source '$INSTALL_DIR/setup.bash'
        
        # Set log level
        export RCUTILS_CONSOLE_OUTPUT_FORMAT='[{severity} {time}] [{name}]: {message}'
        export RCUTILS_LOGGING_USE_STDOUT=1
        export RCUTILS_LOGGING_BUFFERED_STREAM=1
        
        # Launch full system
        ros2 launch rover_description full_system.launch.py \
            use_sim_time:=true \
            launch_gazebo:=true \
            launch_nav2:=true \
            launch_ai:=true \
            slam:=true \
            2>&1 | tee '$LOG_DIR/rover_system.log'
        
        echo '🛑 Rover system stopped'
        exec bash
    "
    
    # Wait for simulation to initialize
    sleep 15
}

# Launch dashboard
launch_dashboard() {
    log_info "Launching web dashboard..."
    
    gnome-terminal --title="🌐 Dashboard" --working-directory="$ROOT_DIR" -- bash -c "
        echo '🔄 Starting Streamlit Dashboard...'
        source '$VENV_DIR/bin/activate'
        cd '$ROOT_DIR/dashboard'
        
        # Set Streamlit configuration
        export STREAMLIT_SERVER_PORT=8501
        export STREAMLIT_SERVER_ADDRESS=0.0.0.0
        export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
        
        # Launch dashboard
        streamlit run app.py \
            --server.port=8501 \
            --server.address=0.0.0.0 \
            --theme.base=dark \
            --browser.serverAddress=localhost \
            2>&1 | tee '$LOG_DIR/dashboard.log'
        
        echo '🛑 Dashboard stopped'
        exec bash
    "
    
    sleep 5
}

# Launch waypoint navigation
launch_navigation() {
    log_info "Starting autonomous navigation..."
    
    gnome-terminal --title="🎯 Navigation" --working-directory="$ROOT_DIR" -- bash -c "
        echo '🔄 Starting waypoint navigation...'
        source /opt/ros/humble/setup.bash
        source '$INSTALL_DIR/setup.bash'
        
        # Wait for system to be ready
        sleep 10
        
        # Start goal publisher
        ros2 run rover_description goal_publisher.py \
            2>&1 | tee '$LOG_DIR/navigation.log'
        
        echo '🛑 Navigation stopped'
        exec bash
    "
    
    sleep 3
}

# Launch monitoring tools
launch_monitoring() {
    log_info "Starting system monitoring..."
    
    gnome-terminal --title="📊 Monitoring" --working-directory="$ROOT_DIR" -- bash -c "
        echo '📊 Starting system monitoring...'
        source /opt/ros/humble/setup.bash
        source '$INSTALL_DIR/setup.bash'
        
        # Monitor key topics
        echo '=== Active Topics ==='
        ros2 topic list
        
        echo ''
        echo '=== System Status ==='
        while true; do
            echo '--- $(date) ---'
            echo 'CPU Usage:'
            top -bn1 | grep 'Cpu(s)' | awk '{print \$2}'
            echo 'Memory Usage:'
            free -h | grep Mem | awk '{print \$3\"/\"\$2}'
            echo 'ROS2 Nodes:'
            ros2 node list | wc -l
            echo 'Active Topics:'
            ros2 topic list | wc -l
            echo '-------------------'
            sleep 10
        done
    "
}

# Health check
health_check() {
    log_info "Performing system health check..."
    
    local errors=0
    
    # Check if ROS2 master is running
    if ! ros2 node list &> /dev/null; then
        log_error "ROS2 master not running"
        ((errors++))
    fi
    
    # Check if essential topics are available
    local essential_topics=("/camera/image_raw" "/scan" "/odom")
    for topic in "${essential_topics[@]}"; do
        if ! ros2 topic list | grep -q "$topic"; then
            log_warning "Topic $topic not found"
        fi
    done
    
    if [ $errors -eq 0 ]; then
        log_success "System health check passed"
    else
        log_warning "System health check completed with $errors error(s)"
    fi
}

# Main execution
main() {
    print_banner
    
    # Check if we're in the right directory
    if [ ! -f "package.xml" ]; then
        log_error "Please run this script from the project root directory"
        exit 1
    fi
    
    # Run checks and setup
    check_requirements
    setup_environment
    check_built_components
    
    # Launch components
    launch_simulation
    launch_dashboard
    launch_navigation
    launch_monitoring
    
    # Final health check
    sleep 10
    health_check
    
    # Display access information
    echo ""
    echo -e "${GREEN}===========================================${NC}"
    echo -e "${GREEN}       🚀 SYSTEM LAUNCH COMPLETE!${NC}"
    echo -e "${GREEN}===========================================${NC}"
    echo ""
    echo -e "${BLUE}📊 Access Points:${NC}"
    echo "   - Gazebo Simulation: Running in terminal 1"
    echo "   - RViz Navigation: Included in simulation"
    echo "   - Streamlit Dashboard: http://localhost:8501"
    echo "   - ROS2 Topics: Run 'ros2 topic list'"
    echo ""
    echo -e "${BLUE}🔧 Monitoring:${NC}"
    echo "   - System logs: $LOG_DIR/"
    echo "   - ROS2 graph: Run 'rqt_graph'"
    echo "   - Topic monitoring: Run 'ros2 topic hz /camera/image_raw'"
    echo ""
    echo -e "${YELLOW}⚠️  To stop the system:${NC}"
    echo "   - Close any terminal window"
    echo "   - Or press Ctrl+C in each terminal"
    echo ""
    echo -e "${GREEN}🎯 Next steps:${NC}"
    echo "   - Open the dashboard at http://localhost:8501"
    echo "   - Monitor rover navigation in RViz"
    echo "   - Check AI perception results in the dashboard"
    echo ""
}

# Handle script interruption
cleanup() {
    echo ""
    log_warning "Received interrupt signal, cleaning up..."
    
    # Kill all background processes
    pkill -f "ros2 launch" || true
    pkill -f "streamlit" || true
    pkill -f "goal_publisher" || true
    
    log_info "Cleanup complete"
    exit 0
}

# Set trap for cleanup
trap cleanup INT TERM

# Run main function
main "$@"