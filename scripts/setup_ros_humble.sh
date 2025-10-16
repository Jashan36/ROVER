#!/bin/bash
# ==============================================================================
# ROS2 HUMBLE INSTALLATION SCRIPT
# Installs ROS2 Humble on Ubuntu 22.04
# ==============================================================================

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "============================================"
echo "   ROS2 HUMBLE INSTALLATION"
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
# CHECK OS
# ==============================================================================

print_status "Checking operating system..."

if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
    VER=$VERSION_ID
else
    print_error "Cannot detect OS"
    exit 1
fi

if [ "$OS" != "ubuntu" ]; then
    print_error "This script is for Ubuntu only"
    print_error "For other OS, see: https://docs.ros.org/en/humble/Installation.html"
    exit 1
fi

if [ "$VER" != "22.04" ]; then
    print_warning "ROS2 Humble is designed for Ubuntu 22.04"
    print_warning "You are running Ubuntu $VER"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

print_success "OS check passed: Ubuntu $VER"

# ==============================================================================
# SET LOCALE
# ==============================================================================

print_status "Setting locale..."

sudo apt update && sudo apt install -y locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

print_success "Locale configured"

# ==============================================================================
# ADD ROS2 APT REPOSITORY
# ==============================================================================

print_status "Adding ROS2 apt repository..."

# Enable Ubuntu Universe repository
sudo apt install -y software-properties-common
sudo add-apt-repository universe -y

# Add ROS2 GPG key
sudo apt update && sudo apt install -y curl
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg

# Add repository to sources list
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
    | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

print_success "ROS2 repository added"

# ==============================================================================
# INSTALL ROS2 HUMBLE
# ==============================================================================

print_status "Installing ROS2 Humble (this may take a while)..."

sudo apt update
sudo apt upgrade -y

# Install ROS2 Desktop (Full)
print_status "Installing ros-humble-desktop..."
sudo apt install -y ros-humble-desktop

# Install development tools
print_status "Installing development tools..."
sudo apt install -y \
    ros-dev-tools \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool

print_success "ROS2 Humble installed"

# ==============================================================================
# INSTALL ADDITIONAL PACKAGES
# ==============================================================================

print_status "Installing additional ROS2 packages..."

# Gazebo
print_status "Installing Gazebo..."
sudo apt install -y ros-humble-gazebo-ros-pkgs

# Navigation2
print_status "Installing Nav2..."
sudo apt install -y ros-humble-navigation2 ros-humble-nav2-bringup

# Vision & perception
print_status "Installing vision packages..."
sudo apt install -y \
    ros-humble-vision-opencv \
    ros-humble-image-transport \
    ros-humble-cv-bridge

# Visualization
print_status "Installing RViz..."
sudo apt install -y ros-humble-rviz2

print_success "Additional packages installed"

# ==============================================================================
# INITIALIZE ROSDEP
# ==============================================================================

print_status "Initializing rosdep..."

if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
    sudo rosdep init
fi

rosdep update

print_success "rosdep initialized"

# ==============================================================================
# SETUP ENVIRONMENT
# ==============================================================================

print_status "Setting up environment..."

# Add to bashrc
if ! grep -q "source /opt/ros/humble/setup.bash" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# ROS2 Humble" >> ~/.bashrc
    echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
    print_success "Added ROS2 to ~/.bashrc"
else
    print_status "ROS2 already in ~/.bashrc"
fi

# Source for current session
source /opt/ros/humble/setup.bash

print_success "Environment configured"

# ==============================================================================
# INSTALL GAZEBO GARDEN (Optional)
# ==============================================================================

read -p "Install Gazebo Garden (recommended for better simulation)? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Installing Gazebo Garden..."
    
    sudo apt install -y \
        gz-garden \
        ros-humble-ros-gzgarden
    
    print_success "Gazebo Garden installed"
fi

# ==============================================================================
# VERIFY INSTALLATION
# ==============================================================================

print_status "Verifying installation..."

# Check ROS2 version
ROS_VERSION=$(ros2 --version 2>&1 || echo "not found")

if [[ $ROS_VERSION == *"not found"* ]]; then
    print_error "ROS2 installation verification failed"
    exit 1
fi

print_success "ROS2 installed: $ROS_VERSION"

# Test basic commands
print_status "Testing ROS2 commands..."

ros2 pkg list > /dev/null 2>&1
if [ $? -eq 0 ]; then
    print_success "ROS2 packages accessible"
else
    print_error "Cannot access ROS2 packages"
    exit 1
fi

# ==============================================================================
# SUMMARY
# ==============================================================================

echo ""
print_success "============================================"
print_success "   ROS2 HUMBLE INSTALLATION COMPLETE!"
print_success "============================================"
echo ""
echo "Installed components:"
echo "  - ROS2 Humble Desktop"
echo "  - Gazebo"
echo "  - Navigation2"
echo "  - Vision packages"
echo "  - Development tools"
echo ""
echo "To use ROS2:"
echo "  1. Open new terminal (or run: source ~/.bashrc)"
echo "  2. Verify: ros2 --version"
echo "  3. Test: ros2 run demo_nodes_cpp talker"
echo ""
echo "Next steps:"
echo "  - Build workspace: ./scripts/build_system.sh"
echo "  - Read docs: https://docs.ros.org/en/humble/"
echo ""