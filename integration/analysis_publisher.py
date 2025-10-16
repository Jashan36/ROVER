"""
Test-friendly integration.analysis_publisher
Provides ROS-free stubs required by the tests.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PublisherConfig:
    """Configuration for publishers (mirrors expected defaults)."""
    publish_segmentation: bool = True
    publish_overlay: bool = True
    publish_traversability: bool = True
    publish_hazards: bool = True
    publish_costmap: bool = True
    publish_markers: bool = True

    # Topic names
    segmentation_topic: str = '/terrain/segmentation'
    overlay_topic: str = '/terrain/overlay'
    traversability_topic: str = '/terrain/traversability_map'
    hazards_topic: str = '/terrain/hazards'
    costmap_topic: str = '/terrain/costmap'
    markers_topic: str = '/terrain/hazard_markers'

    # QoS
    qos_depth: int = 2


class AnalysisPublisher:
    """Minimal stub for tests (no ROS required)."""

    def __init__(self, node_name: str = 'analysis_publisher', config: Optional[PublisherConfig] = None):
        self.node_name = node_name
        self.config = config or PublisherConfig()

