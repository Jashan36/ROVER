# Mars Rover AI Platform

## Overview

This repository contains a modular prototype for an autonomous Mars rover
software stack. It combines real-time perception, navigation, simulation, and a
web dashboard designed to visualize rover state. The current codebase is
production-quality, but ships with lightweight placeholder models so that the
entire pipeline can run without GPU acceleration.

Use this project as a foundation for experimentation, algorithm development, or
integration with a full ROS2-based rover stack.

## Project Status

- Completed: Architecture and integration scaffolding are in place
- Completed: End-to-end tests run locally with CPU-only fallback models
- Pending: Trained weights are **not** included; performance is illustrative
- Pending: ROS2 simulation requires additional setup (see optional section below)

## Key Features

- Real-time terrain segmentation, hazard detection, and traversability analysis
- Direct IP camera ingestion with on-device inference fallback for rapid demos
- Python-based navigation utilities that integrate with ROS2 Nav2
- Streamlit dashboard for monitoring perception, navigation, and science data
- Scripts for dataset management, testing, and ROS2 workspace bootstrap

## Repository Layout

| Path | Description |
| --- | --- |
| `ai/` | Inference models, training utilities, and data fetchers |
| `dashboard/` | Streamlit dashboard components and assets |
| `sim/` | ROS2 packages, launch files, and Gazebo worlds |
| `scripts/` | Helper scripts for setup, testing, and automation |
| `tests/` | Pytest-based regression suite |
| `data/` | Placeholder location for raw and processed datasets |
| `docs/` | Additional setup and architecture documentation |

## Prerequisites

The core Python components run on Windows, macOS, or Linux. ROS2 simulation is
typically performed on Ubuntu.

- Python 3.10 or newer
- `pip` and virtual environment tooling (`venv` or `conda`)
- Optional: CUDA-enabled GPU for accelerated training/inference
- Optional (for full simulation): Ubuntu 22.04/24.04 with ROS2 Humble/Jazzy
- Optional (for full ROS bridge): Install ROS2 Python packages (`rclpy`,
  `cv-bridge`, message packages) from the ROS2 distribution rather than PyPI

## Quick Start (Python-Only Workflow)

Follow the steps below to install dependencies, run tests, and launch the
dashboard. Commands are shown for both Unix-like shells and PowerShell.

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-org/space_rover.git
   cd space_rover
   ```

2. **Create a virtual environment**
   ```bash
   # Unix / macOS
   python -m venv .venv

   # Windows (PowerShell)
   python -m venv .venv
   ```

3. **Activate the environment**
   ```bash
   # Unix / macOS
   source .venv/bin/activate

   # Windows (PowerShell)
   .\.venv\Scripts\Activate.ps1
   ```

4. **Install Python dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. **(Optional) Download sample datasets**
   ```bash
   python scripts/download_datasets.py --output-dir data/raw
   ```
   The project includes lightweight fixtures so tests pass even without large
   datasets. Use this step when you want to experiment with real imagery.

6. **Run the automated tests**
   ```bash
   python -m pytest -q
   ```
   All tests should pass (the suite executes quickly using CPU-friendly mocks).

## Launch the Dashboard

1. Activate the virtual environment (if not already active).
2. Start Streamlit:
   ```bash
   streamlit run dashboard/app.py
   ```
3. Open the URL shown in the console (typically http://localhost:8501).

The dashboard connects to mocked data by default. To integrate with live ROS2
topics, configure `dashboard/config/dashboard_config.yaml` and ensure the ROS
bridge nodes are operating (see optional ROS section).

### Direct Camera Mode

When a ROS/WebSocket feed is unavailable, switch the sidebar source to **Direct
IP Camera** and provide any MJPEG/HTTP stream URL. The dashboard will fetch
frames locally, run the real-time inference pipeline, and populate the terrain,
hazard, and metrics panels without ROS2 dependencies. This is ideal for quick
field tests or demo environments.

## Run Sample Inference

To exercise the CPU-friendly inference pipeline with randomly generated imagery:

```bash
python ai/inference/real_time_pipeline.py
```

The script prints timing statistics and mock hazard summaries to the console.

## Optional: ROS2 & Simulation Workflow

For full rover simulation with Gazebo, Nav2, and ROS2, complete the following
after the Python setup:

1. Install ROS2 Humble (Ubuntu 22.04) or Jazzy (Ubuntu 24.04). See
   `docs/SETUP.md` for platform-specific instructions.
   - This also provides the Python bindings (`rclpy`, `cv-bridge`, and message
     packages). These are not available on PyPI for Windows/macOS and must come
     from the ROS2 installation.
2. Source the ROS2 environment and build the integration packages:
   ```bash
   colcon build --symlink-install
   source install/setup.bash
   ```
3. Launch the full system simulation:
   ```bash
   ros2 launch sim/launch/full_system.launch.py
   ```
4. Start the dashboard in another terminal to visualize the live ROS2 data.

> **Tip:** Use `scripts/build_system.sh` on Ubuntu to automate steps such as ROS
> dependency installation, workspace creation, and dataset downloads.

## Data Management & Training

- Raw datasets should be stored under `data/raw`. The repository includes helper
  scripts for fetching NASA imagery and AI4Mars assets.
- Training utilities live in `ai/training/`. For example, to fine-tune the
  segmentation network:
  ```bash
  python ai/training/train_segmentation.py --config ai/config/model_config.yaml
  ```
- Update `ai/config/inference_config.yaml` with the path to your trained weights
  before running real-time inference.

## Troubleshooting

| Issue | Resolution |
| --- | --- |
| `ModuleNotFoundError: rclpy` | ROS2 packages are optional; install ROS2 or use the mocked integration layer provided in `integration/`. |
| Streamlit dashboard cannot connect | Ensure the dashboard config points to the correct WebSocket/ROS bridge or run in mock mode. |
| Tests fail after modifying code | Run `python -m pytest -q` to pinpoint regressions. Many components include unit tests covering edge cases. |

## Publishing Your Changes

Before pushing updates to GitHub:

1. Verify formatting and linting (optional but recommended):
   ```bash
   python -m compileall dashboard ai sim
   ```
2. Run the automated tests:
   ```bash
   python -m pytest -q
   ```
3. Review the diff and stage files:
   ```bash
   git status
   git add <paths>
   ```
4. Craft a descriptive commit message:
   ```bash
   git commit -m "Describe the change"
   ```
5. Push to your fork or origin:
   ```bash
   git push origin <branch-name>
   ```

Include screenshots or relevant logs in your pull request to help reviewers
understand the behavior change, especially when updating dashboards or ROS2
bridges.

## Contributing

1. Fork the repository and create a feature branch.
2. Run the full test suite (`python -m pytest -q`).
3. Submit a pull request describing the motivation and design decisions.

## License

This project is released under the MIT License. See `LICENSE` for details.
