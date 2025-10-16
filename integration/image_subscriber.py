"""
Test-friendly integration.image_subscriber
Provides ROS-free stubs required by the tests.
"""

from dataclasses import dataclass
from typing import Optional, Callable
import numpy as np


@dataclass
class ImageBuffer:
    """Thread-safe image buffer (simplified for tests)"""
    image: np.ndarray
    timestamp: float
    frame_id: str
    seq: int


class ImageSubscriber:
    """Minimal stub for tests (no ROS required)."""

    def __init__(self, node_name: str = 'image_subscriber', topic: str = '/camera/image_raw', buffer_size: int = 5, qos_depth: int = 10):
        self.node_name = node_name
        self.topic = topic
        self.buffer_size = buffer_size
        self.qos_depth = qos_depth
        self.callback: Optional[Callable[[ImageBuffer], None]] = None

    def set_callback(self, cb: Callable[[ImageBuffer], None]):
        self.callback = cb

