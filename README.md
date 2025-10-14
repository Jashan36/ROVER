# 🚀 Mars Rover AI Project

Autonomous Mars Rover with Real-time AI Perception and Navigation

![Mars Rover](docs/images/architecture.png)

## 🌟 Overview

Build a working autonomous Mars rover prototype that can:
- Navigate autonomously in simulated extraterrestrial terrain (Moon/Mars)
- Analyze real planetary images for terrain, rocks, slopes, and hazards
- Output scientific insights in real-time through a dashboard
- Use only real NASA data (no synthetic/fake data)

## 🛠️ Core Technologies

| Component | Technology | Purpose |
|-----------|------------|---------|
| Simulation | Gazebo + ROS2 Humble | Physics-based Mars environment |
| Navigation | Nav2 + Cartographer | Autonomous path planning & SLAM |
| AI Perception | PyTorch U-Net | Terrain segmentation |
| Real Data | NASA Perseverance API | Authentic Mars imagery |
| Training Data | AI4Mars Dataset | Labeled terrain annotations |
| Visualization | Streamlit | Real-time web dashboard |
| Integration | ROS2 Python | Camera → AI → Navigation pipeline |

## 🚀 Key Features

- ✅ Real-time camera-based terrain detection (5 Hz)
- ✅ Autonomous waypoint navigation with obstacle avoidance
- ✅ Hazard detection (rocks, sand, slopes)
- ✅ Traversability analysis for safe path planning
- ✅ Live web dashboard with scientific notes
- ✅ Mars-accurate physics (3.71 m/s² gravity)
- ✅ Real NASA Perseverance images for validation

## 📋 Quick Start

### Prerequisites

- **OS**: Ubuntu 22.04 LTS (recommended)
- **ROS2**: Humble Hawksbill
- **Python**: 3.10+
- **GPU**: Optional (CUDA 11.8+ for acceleration)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/space_rover.git
cd space_rover

# Run automated setup
./scripts/build_system.sh

# Or manual setup:
# 1. Install ROS2 Humble
# 2. Create Python environment
# 3. Build ROS2 workspace
# 4. Download datasets