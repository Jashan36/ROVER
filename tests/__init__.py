"""
Test Suite for Mars Rover AI System
Comprehensive tests for perception, navigation, integration, and dashboard
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

__version__ = "1.0.0"

# Test configuration
TEST_DATA_DIR = Path(__file__).parent / "test_data"
TEST_IMAGES_DIR = TEST_DATA_DIR / "images"
TEST_MODELS_DIR = TEST_DATA_DIR / "models"

# Create test directories if they don't exist
TEST_DATA_DIR.mkdir(exist_ok=True)
TEST_IMAGES_DIR.mkdir(exist_ok=True)
TEST_MODELS_DIR.mkdir(exist_ok=True)