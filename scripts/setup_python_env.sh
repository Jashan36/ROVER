#!/usr/bin/env bash
# Setup Python virtual environment and install Python dependencies
# Usage (Linux/macOS): bash scripts/setup_python_env.sh

set -euo pipefail

VENV_DIR=${1:-venv}
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip

# Install PyTorch CPU wheel (change if you have CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install project requirements
pip install -r requirements.txt

# Rasterio prerequisites hint
cat <<'EOF'
Note: rasterio may require system libraries (libgdal). On Ubuntu install:
  sudo apt install -y gdal-bin libgdal-dev
Then re-run: pip install rasterio
EOF

exit 0
