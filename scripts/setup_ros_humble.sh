#!/usr/bin/env bash
# Install ROS2 Humble on Ubuntu 22.04
# Usage: sudo bash scripts/setup_ros_humble.sh

set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
  echo "This script must be run with sudo. Re-run as: sudo $0"
  exit 1
fi

apt update
apt install -y software-properties-common
add-apt-repository universe
apt update && apt install -y curl gnupg lsb-release

# Add ROS 2 apt repository key and source
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
  http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" \
  | tee /etc/apt/sources.list.d/ros2.list > /dev/null

apt update

# Core desktop + Gazebo + Nav2 + robot_localization
apt install -y ros-humble-desktop ros-humble-gazebo-ros-pkgs ros-humble-nav2-bringup ros-humble-robot-localization

# rosdep (needs network access)
apt install -y python3-rosdep python3-colcon-common-extensions
if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
  rosdep init || true
fi
rosdep update || true

# Add ROS environment sourcing to /etc/profile.d for all users
cat > /etc/profile.d/ros2_humble.sh <<'EOF'
# Source ROS2 Humble
if [ -f /opt/ros/humble/setup.bash ]; then
  source /opt/ros/humble/setup.bash
fi
EOF

chmod +x /etc/profile.d/ros2_humble.sh

echo "ROS2 Humble and related packages installed. Reboot recommended if using GPU or virtualization features."

echo "To use ROS in your shell: source /opt/ros/humble/setup.bash"

exit 0
