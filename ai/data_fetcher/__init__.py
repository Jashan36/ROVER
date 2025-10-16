"""
Data Fetcher Package
Tools for downloading and processing Mars data from various sources
"""

from .perseverance_api import (
    PerseveranceAPIClient,
    MarsImage,
    fetch_latest_images,
    fetch_images_by_sol
)

from .ai4mars_loader import AI4MarsLoader
from .base_loader import BaseDataLoader, DatasetMetadata, DownloadResult
from .universal_loader import (
    UniversalDataLoader,
    detect_source,
    load_config,
    register_loader,
)

# Register source-specific loaders
from . import kaggle_loader  # noqa: F401
from . import huggingface_loader  # noqa: F401
from . import nasa_loader  # noqa: F401
from . import gdrive_loader  # noqa: F401
from . import url_loader  # noqa: F401
from . import s3_loader  # noqa: F401
from . import roboflow_loader  # noqa: F401
from . import zenodo_loader  # noqa: F401

from .dem_processor import (
    DEMProcessor,
    generate_mars_heightmap,
    create_crater_profile
)

__all__ = [
    # Perseverance API
    'PerseveranceAPIClient',
    'MarsImage',
    'fetch_latest_images',
    'fetch_images_by_sol',
    
    # AI4Mars
    'AI4MarsLoader',

    # Universal loader
    'BaseDataLoader',
    'DatasetMetadata',
    'DownloadResult',
    'UniversalDataLoader',
    'detect_source',
    'load_config',
    'register_loader',

    # DEM Processing
    'DEMProcessor',
    'generate_mars_heightmap',
    'create_crater_profile',
]
