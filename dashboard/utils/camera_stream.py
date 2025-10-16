"""
Camera Stream Utilities
Shared helpers for fetching frames from IP/MJPEG camera endpoints.
"""

from __future__ import annotations

from typing import Optional
from urllib.parse import urljoin

import cv2
import numpy as np
import requests


def fetch_ip_camera_frame(stream_url: str, timeout: float = 3.0) -> Optional[np.ndarray]:
    """Fetch a single RGB frame from an IP camera stream or snapshot endpoint."""
    if not stream_url:
        return None

    # Try direct video stream capture first
    cap = cv2.VideoCapture(stream_url)
    if cap.isOpened():
        try:
            ret, frame = cap.read()
            if ret and frame is not None:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        finally:
            cap.release()
    else:
        cap.release()

    # Attempt common snapshot endpoints (e.g., IP Webcam, MJPEG servers)
    base_url = stream_url.rstrip("/") + "/"
    snapshot_candidates = [
        urljoin(base_url, "shot.jpg"),
        urljoin(base_url, "photo.jpg"),
        urljoin(base_url, "image.jpg"),
    ]

    for snapshot_url in snapshot_candidates:
        try:
            response = requests.get(snapshot_url, timeout=timeout)
            if response.status_code == 200 and response.content:
                arr = np.frombuffer(response.content, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except requests.RequestException:
            continue

    return None
