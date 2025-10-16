"""
Integration (Test-Friendly)
Lightweight, ROS-free shims that provide the interfaces used in tests.
These mirror the public API expected by the test suite without requiring ROS2.
"""

from .image_subscriber import ImageSubscriber, ImageBuffer
from .analysis_publisher import AnalysisPublisher, PublisherConfig
from .dashboard_bridge import DashboardBridge, DashboardData

__all__ = [
    'ImageSubscriber', 'ImageBuffer',
    'AnalysisPublisher', 'PublisherConfig',
    'DashboardBridge', 'DashboardData',
]

