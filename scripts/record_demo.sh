#!/bin/bash
# ==============================================================================
# DEMO RECORDING SCRIPT
# Records demonstration videos and creates presentation materials
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
echo "   MARS ROVER AI - DEMO RECORDER"
echo "============================================"
echo -e "${NC}"

# ==============================================================================
# CONFIGURATION
# ==============================================================================

DEMO_DIR="$PROJECT_ROOT/demos"
VIDEO_DIR="$DEMO_DIR/videos"
SCREENSHOTS_DIR="$DEMO_DIR/screenshots"
ROSBAG_DIR="$DEMO_DIR/rosbags"

# Create directories
mkdir -p "$VIDEO_DIR"
mkdir -p "$SCREENSHOTS_DIR"
mkdir -p "$ROSBAG_DIR"

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
# ENVIRONMENT SETUP
# ==============================================================================

setup_environment() {
    print_status "Setting up demo environment..."
    
    # Source ROS2
    if [ -f "/opt/ros/humble/setup.bash" ]; then
        source /opt/ros/humble/setup.bash
    else
        print_error "ROS2 not found"
        exit 1
    fi
    
    # Source workspace
    if [ -f "$PROJECT_ROOT/install/setup.bash" ]; then
        source "$PROJECT_ROOT/install/setup.bash"
    fi
    
    print_success "Environment ready"
}

# ==============================================================================
# SCREEN RECORDING
# ==============================================================================

record_screen() {
    local duration=$1
    local output_name=$2
    
    print_status "Recording screen for ${duration} seconds..."
    print_warning "Recording will start in 3 seconds. Get ready!"
    sleep 3
    
    # Check for recording tools
    if check_command "ffmpeg"; then
        # Use ffmpeg for screen recording
        ffmpeg -f x11grab -framerate 30 -video_size 1920x1080 \
            -i :0.0 -t "$duration" \
            -c:v libx264 -preset ultrafast -crf 18 \
            "$VIDEO_DIR/${output_name}.mp4" \
            2>&1 | grep -v "frame="
        
        print_success "Recording saved: $VIDEO_DIR/${output_name}.mp4"
    elif check_command "recordmydesktop"; then
        # Use recordmydesktop as fallback
        recordmydesktop --duration="$duration" \
            --output="$VIDEO_DIR/${output_name}.ogv" \
            --no-sound --fps=30
        
        print_success "Recording saved: $VIDEO_DIR/${output_name}.ogv"
    else
        print_error "No screen recording tool found. Install ffmpeg or recordmydesktop"
        return 1
    fi
}

# ==============================================================================
# SCREENSHOT CAPTURE
# ==============================================================================

take_screenshot() {
    local name=$1
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local filename="${name}_${timestamp}.png"
    
    if check_command "import"; then
        # ImageMagick
        import -window root "$SCREENSHOTS_DIR/$filename"
    elif check_command "scrot"; then
        # scrot
        scrot "$SCREENSHOTS_DIR/$filename"
    elif check_command "gnome-screenshot"; then
        # GNOME
        gnome-screenshot -f "$SCREENSHOTS_DIR/$filename"
    else
        print_error "No screenshot tool found"
        return 1
    fi
    
    print_success "Screenshot saved: $SCREENSHOTS_DIR/$filename"
}

# ==============================================================================
# ROSBAG RECORDING
# ==============================================================================

record_rosbag() {
    local duration=$1
    local bag_name=$2
    
    print_status "Recording ROS2 bag for ${duration} seconds..."
    
    cd "$ROSBAG_DIR"
    
    # Record important topics
    timeout "$duration" ros2 bag record \
        /camera/image_raw \
        /terrain/segmentation \
        /terrain/overlay \
        /terrain/hazards \
        /terrain/costmap \
        /cmd_vel \
        /odom \
        -o "$bag_name" \
        || true
    
    print_success "ROS bag saved: $ROSBAG_DIR/$bag_name"
}

# ==============================================================================
# AUTOMATED DEMO SCENARIOS
# ==============================================================================

demo_full_system() {
    print_status "Recording full system demo..."
    
    # Start recording
    record_screen 120 "full_system_demo" &
    RECORD_PID=$!
    
    # Give time for recording to start
    sleep 2
    
    # Launch system
    print_status "Launching system..."
    gnome-terminal -- bash -c "source setup_env.sh && ./launch_simulation.sh" &
    SIM_PID=$!
    
    sleep 10
    
    gnome-terminal -- bash -c "source venv/bin/activate && cd dashboard && streamlit run app.py" &
    DASH_PID=$!
    
    # Wait for recording to complete
    wait $RECORD_PID
    
    # Cleanup
    kill $SIM_PID $DASH_PID 2>/dev/null || true
    
    print_success "Full system demo recorded"
}

demo_perception() {
    print_status "Recording perception demo..."
    
    # Record screen
    record_screen 60 "perception_demo" &
    RECORD_PID=$!
    
    sleep 2
    
    # Run perception demo
    gnome-terminal -- bash -c "
        source setup_env.sh
        python3 -c '
import numpy as np
import cv2
from ai.inference.realtime_pipeline import RealtimePipeline, PipelineConfig
import time

# Load test image
img = cv2.imread(\"data/raw/perseverance/sample.jpg\")
if img is None:
    img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
else:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Create pipeline
config = PipelineConfig(device=\"cpu\")
pipeline = RealtimePipeline(config)

# Process and display
print(\"Processing image...\")
result = pipeline.process(img)

print(f\"Detected {len(result.hazards)} hazards\")
print(f\"Avg traversability: {result.stats.get(\"avg_traversability\", 0):.2f}\")

# Show results
if result.overlay is not None:
    cv2.imshow(\"Perception Result\", cv2.cvtColor(result.overlay, cv2.COLOR_RGB2BGR))
    cv2.waitKey(30000)  # Show for 30 seconds
'
        " &
    
    wait $RECORD_PID
    
    print_success "Perception demo recorded"
}

demo_navigation() {
    print_status "Recording navigation demo..."
    
    # Record with ROS bag
    record_rosbag 60 "navigation_demo" &
    BAG_PID=$!
    
    # Record screen
    record_screen 60 "navigation_demo_screen" &
    RECORD_PID=$!
    
    # Launch navigation
    gnome-terminal -- bash -c "
        source setup_env.sh
        ros2 launch rover_description nav2.launch.py
    " &
    NAV_PID=$!
    
    # Wait for recordings
    wait $RECORD_PID $BAG_PID
    
    # Cleanup
    kill $NAV_PID 2>/dev/null || true
    
    print_success "Navigation demo recorded"
}

# ==============================================================================
# CREATE PRESENTATION
# ==============================================================================

create_presentation() {
    print_status "Creating presentation materials..."
    
    # Create README for demos
    cat > "$DEMO_DIR/README.md" <<'EOF'
# Mars Rover AI - Demo Materials

## Videos

### Full System Demo
- File: `videos/full_system_demo.mp4`
- Duration: 2 minutes
- Shows: Complete system with simulation, AI processing, and dashboard

### Perception Demo
- File: `videos/perception_demo.mp4`
- Duration: 1 minute
- Shows: AI terrain analysis and hazard detection

### Navigation Demo
- File: `videos/navigation_demo_screen.mp4`
- Duration: 1 minute
- Shows: Autonomous navigation with obstacle avoidance

## Screenshots

Screenshots are automatically timestamped and saved in `screenshots/`

## ROS Bags

ROS bag recordings for playback:
- `rosbags/navigation_demo/`

### Playback Instructions
```bash
# Source ROS2
source /opt/ros/humble/setup.bash

# Play bag
ros2 bag play rosbags/navigation_demo