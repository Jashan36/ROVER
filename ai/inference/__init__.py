"""
Inference Package
Real-time and batch processing pipelines for terrain analysis
"""

# Import real-time pipeline (filename is real_time_pipeline.py)
from .real_time_pipeline import (
    RealtimePipeline,
    InferenceResult,
    PipelineConfig
)

from .batch_processor import (
    BatchProcessor,
    BatchResult,
    process_image_directory
)

from .model_cache import (
    ModelCache,
    CacheConfig
)

__all__ = [
    # Real-time
    'RealtimePipeline',
    'InferenceResult',
    'PipelineConfig',
    
    # Batch
    'BatchProcessor',
    'BatchResult',
    'process_image_directory',
    
    # Cache
    'ModelCache',
    'CacheConfig',
]
