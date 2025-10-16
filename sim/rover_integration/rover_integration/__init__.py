"""
Integration Package
ROS2-AI Bridge for connecting perception to navigation and dashboard

This package avoids importing ROS2 dependencies at import time so that
non-ROS environments (e.g., unit tests) can import modules safely.
"""

# Expose costmap utilities (pure NumPy/OpenCV logic)
from .costmap_generator import (
    CostmapGenerator,
    CostmapConfig,
)

# Lazily expose ROS-dependent modules; import guarded to work without ROS2
try:
    from .ros_ai_bridge import (
        ROSAIBridge,
        BridgeConfig,
    )
    from .image_subscriber import (
        ImageSubscriber,
        ImageBuffer,
    )
    from .analysis_publisher import (
        AnalysisPublisher,
        PublisherConfig,
    )
    from .dashboard_bridge import (
        DashboardBridge,
        DashboardData,
    )
    _ROS_AVAILABLE = True
except Exception:
    # ROS not available; allow package import without these symbols
    _ROS_AVAILABLE = False

__all__ = [
    'CostmapGenerator',
    'CostmapConfig',
]

if _ROS_AVAILABLE:
    __all__ += [
        'ROSAIBridge', 'BridgeConfig',
        'ImageSubscriber', 'ImageBuffer',
        'AnalysisPublisher', 'PublisherConfig',
        'DashboardBridge', 'DashboardData',
    ]

__version__ = "1.0.0"
