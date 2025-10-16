"""
Pytest Configuration
Shared fixtures and configuration for all tests
"""

import pytest
import numpy as np
import sys
from pathlib import Path
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging for tests
logging.basicConfig(
    level=logging.WARNING,  # Reduce noise during tests
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Disable verbose logging from external libraries
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)


# ============================================
# GLOBAL FIXTURES
# ============================================

@pytest.fixture(scope="session")
def test_data_dir():
    """Test data directory"""
    data_dir = Path(__file__).parent / "test_data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


@pytest.fixture(scope="session")
def test_images_dir(test_data_dir):
    """Test images directory"""
    images_dir = test_data_dir / "images"
    images_dir.mkdir(exist_ok=True)
    return images_dir


@pytest.fixture
def random_seed():
    """Set random seed for reproducibility"""
    np.random.seed(42)
    yield
    # Reset after test
    np.random.seed(None)


@pytest.fixture
def sample_image_small():
    """Small sample image (256x256)"""
    return np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)


@pytest.fixture
def sample_image_large():
    """Large sample image (1024x1024)"""
    return np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)


# ============================================
# TEST MARKERS
# ============================================

def pytest_configure(config):
    """Configure custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "ros: marks tests that require ROS2"
    )


# ============================================
# TEST COLLECTION
# ============================================

def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    # Add markers to tests based on their path
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        if "performance" in item.name.lower():
            item.add_marker(pytest.mark.slow)