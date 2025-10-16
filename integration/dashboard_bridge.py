"""
Test-friendly integration.dashboard_bridge
Provides ROS-free stubs and utilities required by the tests.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import base64
import cv2
import numpy as np
import time


@dataclass
class DashboardData:
    timestamp: float
    avg_traversability: float
    num_hazards: int
    fps: float


class DashboardBridge:
    """Minimal stub for tests (no ROS required)."""

    def __init__(self, host: str = 'localhost', port: int = 8765):
        self.host = host
        self.port = port
        self._running = False
        self._connected_clients = 0

    @staticmethod
    def encode_image(image: np.ndarray, format: str = 'JPEG') -> str:
        """Encode an RGB image as a base64 data URI string."""
        if image is None or image.size == 0:
            raise ValueError('Invalid image')

        ext = '.jpg' if format.upper() == 'JPEG' else '.png'
        success, buffer = cv2.imencode(ext, image[:, :, ::-1])  # RGB->BGR for OpenCV
        if not success:
            raise RuntimeError('Failed to encode image')
        b64 = base64.b64encode(buffer).decode('utf-8')
        mime = 'image/jpeg' if format.upper() == 'JPEG' else 'image/png'
        return f"data:{mime};base64,{b64}"

    def get_statistics(self) -> Dict[str, Any]:
        return {
            'running': self._running,
            'connected_clients': self._connected_clients,
            'host': self.host,
            'port': self.port,
        }

