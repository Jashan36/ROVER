"""
Direct camera processing utilities.

Fetch raw frames from an IP camera, run the real-time inference pipeline,
and convert results into the TerrainData format expected by the dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import logging

import numpy as np

from dashboard.utils.ros_connector import TerrainData

logger = logging.getLogger(__name__)


@dataclass
class DirectCameraConfig:
    """Configuration for the direct camera inference processor."""

    model_path: Optional[str] = None
    device: str = "auto"
    fast_mode: bool = True
    overlay_alpha: float = 0.5


class DirectCameraProcessor:
    """Run the AI inference pipeline on raw camera frames."""

    def __init__(self, config: Optional[DirectCameraConfig] = None):
        self.config = config or DirectCameraConfig()
        self._pipeline = None
        self._last_fps = 0.0

    def _ensure_pipeline(self):
        if self._pipeline is not None:
            return

        try:
            from ai.inference.real_time_pipeline import create_pipeline
        except Exception as exc:  # pragma: no cover - protective import
            raise RuntimeError(
                "The real-time inference pipeline is unavailable. "
                "Ensure the AI models and dependencies are installed."
            ) from exc

        self._pipeline = create_pipeline(
            model_path=self.config.model_path,
            device=self.config.device,
            fast_mode=self.config.fast_mode,
        )
        self._pipeline.config.overlay_alpha = self.config.overlay_alpha
        logger.info(
            "Direct camera pipeline initialized (device=%s, fast_mode=%s)",
            self._pipeline.config.device,
            self.config.fast_mode,
        )

    def set_overlay_alpha(self, alpha: float):
        """Update overlay alpha for generated visualizations."""
        self.config.overlay_alpha = alpha
        if self._pipeline is not None:
            self._pipeline.config.overlay_alpha = alpha

    def process_frame(self, frame: np.ndarray) -> TerrainData:
        """Run inference on a raw RGB frame and return TerrainData."""
        if frame is None:
            raise ValueError("Cannot process an empty frame.")

        self._ensure_pipeline()
        result = self._pipeline.process(frame)

        if result.total_time_ms > 0:
            self._last_fps = float(1000.0 / result.total_time_ms)

        stats = result.stats or {}
        summary = result.hazard_summary or {}
        if not summary and stats.get("hazard_types"):
            summary = {
                "total_count": stats.get("num_hazards", len(result.hazards)),
                "by_type": stats.get("hazard_types", {}),
            }

        num_hazards = stats.get("num_hazards", len(result.hazards))
        overlay_image = result.overlay if result.overlay is not None else frame

        terrain_data = TerrainData(
            timestamp=result.timestamp,
            overlay_image=overlay_image,
            traversability_image=result.traversability_viz,
            hazard_image=result.hazard_viz,
            avg_traversability=stats.get("avg_traversability", 0.0),
            safe_area_ratio=stats.get("safe_area_ratio", 0.0),
            best_direction_deg=float(np.degrees(result.best_direction)),
            num_hazards=num_hazards,
            hazard_summary=summary,
            inference_time_ms=result.total_time_ms,
            fps=self._last_fps,
            terrain_distribution=stats.get("terrain_distribution", {}),
        )

        return terrain_data

    def get_last_fps(self) -> float:
        """Return the most recent FPS estimate."""
        return self._last_fps

