"""
AI Models Package
Terrain segmentation, traversability analysis, and hazard detection
"""

from .terrain_segmentation import (
    UNet,
    TerrainSegmentationModel,
    DoubleConv
)

from .traversability import (
    TraversabilityAnalyzer,
    TerrainClass,
    Hazard as TraversabilityHazard
)

from .hazard_detector import (
    HazardDetector,
    DetectedHazard,
    HazardType,
    HazardSeverity
)

from .mars_classifier import MarsClassifier

__all__ = [
    # Segmentation
    'UNet',
    'TerrainSegmentationModel',
    'DoubleConv',
    
    # Traversability
    'TraversabilityAnalyzer',
    'TerrainClass',
    'TraversabilityHazard',
    
    # Hazard Detection
    'HazardDetector',
    'DetectedHazard',
    'HazardType',
    'HazardSeverity',

    # Classification
    'MarsClassifier',
]

__version__ = '1.0.0'
