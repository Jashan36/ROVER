"""
AI Perception Package
Complete AI pipeline for Mars terrain analysis

This package provides:
- Terrain segmentation using U-Net
- Traversability analysis for navigation
- Hazard detection for safety
- Real-time and batch inference pipelines
- Data fetching from NASA APIs
- Training utilities for model development
"""

__version__ = "1.0.0"
__author__ = "Mars Rover AI Team"

# Core models
from .models.terrain_segmentation import (
    UNet,
    TerrainSegmentationModel,
)

from .models.traversability import (
    TraversabilityAnalyzer,
    TerrainClass,
)

from .models.hazard_detector import (
    HazardDetector,
    DetectedHazard,
    HazardType,
    HazardSeverity,
)

# Inference
from .inference.real_time_pipeline import (
    RealtimePipeline,
    InferenceResult,
    PipelineConfig,
    create_pipeline,
)

from .inference.batch_processor import (
    BatchProcessor,
    BatchResult,
    process_image_directory,
)

from .inference.model_cache import (
    ModelCache,
    CacheConfig,
)

# Data fetching
from .data_fetcher.perseverance_api import (
    PerseveranceAPIClient,
    MarsImage,
    fetch_latest_images,
    fetch_images_by_sol,
)

from .data_fetcher.ai4mars_loader import (
    AI4MarsLoader,
)

from .data_fetcher.dem_processor import (
    DEMProcessor,
    generate_mars_heightmap,
)

# Training (optional import)
try:
    from .training.train_segmentation import (
        train_model,
        Trainer,
        TrainingConfig,
    )
    
    from .training.dataset import (
        AI4MarsDataset,
        TerrainDataset,
        get_data_loaders,
    )
    
    from .training.augmentations import (
        get_training_augmentations,
        get_validation_augmentations,
    )
    
    TRAINING_AVAILABLE = True
except ImportError:
    TRAINING_AVAILABLE = False

# Configuration utilities
import yaml
from pathlib import Path


def load_config(config_name: str = "inference_config") -> dict:
    """
    Load configuration from YAML file
    
    Args:
        config_name: Name of config file (without .yaml extension)
        
    Returns:
        Configuration dictionary
    """
    config_dir = Path(__file__).parent / "config"
    config_path = config_dir / f"{config_name}.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


# Convenience function for quick setup
def setup_inference_pipeline(
    model_path: str = None,
    config_name: str = "inference_config",
    device: str = "auto"
) -> RealtimePipeline:
    """
    Quick setup for inference pipeline with config file
    
    Args:
        model_path: Path to model weights (overrides config)
        config_name: Name of config file
        device: Device to use ('auto', 'cuda', 'cpu')
        
    Returns:
        Configured RealtimePipeline
    """
    # Load config
    config_dict = load_config(config_name)
    
    # Override with provided parameters
    if model_path:
        config_dict['inference']['model']['path'] = model_path
    
    if device != "auto":
        config_dict['inference']['model']['device'] = device
    
    # Create pipeline config
    pipeline_config = PipelineConfig(
        segmentation_model_path=Path(config_dict['inference']['model']['path']) 
            if config_dict['inference']['model']['path'] else None,
        device=config_dict['inference']['model']['device'],
        input_size=tuple(config_dict['inference']['input']['size']),
        enable_half_precision=config_dict['inference']['performance']['enable_half_precision'],
        confidence_weight=config_dict['traversability']['confidence_weight'],
        safety_margin=config_dict['traversability']['safety_margin_pixels'],
        min_traversability=config_dict['traversability']['min_traversability'],
        enable_hazard_detection=config_dict['hazard_detection']['enabled'],
        generate_overlays=config_dict['inference']['output']['generate_overlays'],
        overlay_alpha=config_dict['inference']['output']['overlay_alpha'],
    )
    
    return RealtimePipeline(pipeline_config)


# Package exports
__all__ = [
    # Core models
    'UNet',
    'TerrainSegmentationModel',
    'TraversabilityAnalyzer',
    'TerrainClass',
    'HazardDetector',
    'DetectedHazard',
    'HazardType',
    'HazardSeverity',
    
    # Inference
    'RealtimePipeline',
    'InferenceResult',
    'PipelineConfig',
    'create_pipeline',
    'BatchProcessor',
    'BatchResult',
    'process_image_directory',
    'ModelCache',
    'CacheConfig',
    
    # Data fetching
    'PerseveranceAPIClient',
    'MarsImage',
    'fetch_latest_images',
    'fetch_images_by_sol',
    'AI4MarsLoader',
    'download_ai4mars_dataset',
    'DEMProcessor',
    'generate_mars_heightmap',
    
    # Utilities
    'load_config',
    'setup_inference_pipeline',
]

# Add training exports if available
if TRAINING_AVAILABLE:
    __all__.extend([
        'train_model',
        'Trainer',
        'TrainingConfig',
        'AI4MarsDataset',
        'TerrainDataset',
        'get_data_loaders',
        'get_training_augmentations',
        'get_validation_augmentations',
    ])


# Package info
def get_package_info() -> dict:
    """Get package information"""
    return {
        'version': __version__,
        'author': __author__,
        'training_available': TRAINING_AVAILABLE,
        'modules': {
            'models': True,
            'inference': True,
            'data_fetcher': True,
            'training': TRAINING_AVAILABLE,
        }
    }


# Initialize logging
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())

# Print package info on import (optional, can be disabled)
if __name__ != "__main__":
    _logger = logging.getLogger(__name__)
    _logger.info(f"Mars Rover AI Package v{__version__} loaded")
    if not TRAINING_AVAILABLE:
        _logger.debug("Training modules not available (albumentations not installed)")
