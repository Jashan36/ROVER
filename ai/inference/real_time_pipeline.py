"""
Real-time Inference Pipeline
Optimized end-to-end pipeline for real-time terrain analysis
"""

import numpy as np
import torch
import cv2
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
import time
import logging
from collections import deque
import threading

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.terrain_segmentation import TerrainSegmentationModel
from models.traversability import TraversabilityAnalyzer
from models.hazard_detector import HazardDetector, DetectedHazard

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for inference pipeline"""
    # Model paths
    segmentation_model_path: Optional[Path] = None
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Input processing
    input_size: tuple = (160, 160)
    
    # Performance
    enable_half_precision: bool = False  # FP16 for faster inference
    enable_model_caching: bool = True
    max_batch_size: int = 1
    
    # Traversability
    confidence_weight: float = 0.3
    safety_margin: int = 20
    min_traversability: float = 0.3
    
    # Hazard detection
    enable_hazard_detection: bool = True
    hazard_temporal_filtering: bool = True
    
    # Output
    generate_overlays: bool = True
    generate_costmap: bool = True
    overlay_alpha: float = 0.5


@dataclass
class InferenceResult:
    """Complete inference result"""
    # Timing
    timestamp: float
    total_time_ms: float
    segmentation_time_ms: float
    traversability_time_ms: float
    hazard_detection_time_ms: float
    
    # Segmentation outputs
    classes: np.ndarray
    confidence: np.ndarray
    segmentation_colored: np.ndarray
    
    # Traversability outputs
    traversability_map: np.ndarray
    best_direction: float
    direction_scores: Dict[str, float]
    costmap: Optional[np.ndarray] = None
    
    # Hazard detection outputs
    hazards: List[DetectedHazard] = field(default_factory=list)
    hazard_summary: Dict = field(default_factory=dict)
    
    # Visualizations
    overlay: Optional[np.ndarray] = None
    traversability_viz: Optional[np.ndarray] = None
    hazard_viz: Optional[np.ndarray] = None
    
    # Statistics
    stats: Dict = field(default_factory=dict)


class RealtimePipeline:
    """
    Optimized real-time inference pipeline
    Combines segmentation, traversability, and hazard detection
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize pipeline
        
        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()
        
        logger.info(f"Initializing RealtimePipeline on {self.config.device}")
        
        # Performance tracking (must be ready before warm-up)
        self.inference_times = deque(maxlen=100)
        self.frame_count = 0
        
        # Threading (used inside process())
        self.lock = threading.Lock()
        
        # Initialize models (may perform warm-up that calls process())
        self._init_models()
        
        logger.info("RealtimePipeline initialized")
        self._log_config()

    def _init_models(self):
        """Initialize all models"""
        # Segmentation model
        logger.info("Loading segmentation model...")
        self.segmentation_model = TerrainSegmentationModel(
            model_path=self.config.segmentation_model_path,
            device=self.config.device,
            input_size=self.config.input_size
        )
        
        # Enable optimizations
        if self.config.enable_half_precision and self.config.device == 'cuda':
            self.segmentation_model.model.half()
            logger.info("Enabled FP16 (half precision)")
        
        # Traversability analyzer
        logger.info("Initializing traversability analyzer...")
        self.traversability_analyzer = TraversabilityAnalyzer(
            confidence_weight=self.config.confidence_weight,
            safety_margin=self.config.safety_margin,
            min_traversability=self.config.min_traversability
        )
        
        # Hazard detector
        if self.config.enable_hazard_detection:
            logger.info("Initializing hazard detector...")
            self.hazard_detector = HazardDetector(
                image_height=self.config.input_size[0],
                image_width=self.config.input_size[1]
            )
        else:
            self.hazard_detector = None
        
        # Warm-up inference
        self._warmup()

    def _warmup(self, n_iterations: int = 3):
        """Warm-up models with dummy data"""
        logger.info("Warming up models...")
        dummy_image = np.random.randint(
            0, 255, 
            (self.config.input_size[0], self.config.input_size[1], 3),
            dtype=np.uint8
        )
        
        for i in range(n_iterations):
            _ = self.process(dummy_image, warmup=True)
        
        logger.info("Warm-up complete")

    def process(
        self,
        image: np.ndarray,
        warmup: bool = False
    ) -> InferenceResult:
        """
        Process single image through complete pipeline
        
        Args:
            image: RGB image (H, W, 3) in range [0, 255]
            warmup: Whether this is a warm-up run (skip logging)
            
        Returns:
            InferenceResult with all outputs
        """
        start_time = time.time()
        
        with self.lock:
            # Step 1: Terrain Segmentation
            seg_start = time.time()
            seg_results = self.segmentation_model.predict(
                image,
                return_confidence=True
            )
            seg_time = (time.time() - seg_start) * 1000
            
            classes = seg_results['classes']
            confidence = seg_results['confidence']
            segmentation_colored = seg_results['segmentation_colored']
            
            # Step 2: Traversability Analysis
            trav_start = time.time()
            trav_results = self.traversability_analyzer.analyze(
                classes,
                confidence
            )
            trav_time = (time.time() - trav_start) * 1000
            
            # Step 3: Hazard Detection
            hazard_time = 0.0
            hazards = []
            hazard_summary = {}
            
            if self.config.enable_hazard_detection and self.hazard_detector:
                hazard_start = time.time()
                hazards = self.hazard_detector.detect(
                    image,
                    classes,
                    confidence
                )
                hazard_summary = self.hazard_detector.get_hazard_summary(hazards)
                hazard_time = (time.time() - hazard_start) * 1000
            
            # Generate visualizations
            overlay = None
            traversability_viz = None
            hazard_viz = None
            
            if self.config.generate_overlays:
                overlay = self.segmentation_model.create_overlay(
                    image,
                    segmentation_colored,
                    alpha=self.config.overlay_alpha
                )
                
                traversability_viz = self.traversability_analyzer.visualize_traversability(
                    trav_results['traversability_map']
                )
                
                if self.hazard_detector and hazards:
                    hazard_viz = self.hazard_detector.visualize_hazards(
                        image,
                        hazards
                    )
            
            # Create result
            total_time = (time.time() - start_time) * 1000
            
            result = InferenceResult(
                timestamp=time.time(),
                total_time_ms=total_time,
                segmentation_time_ms=seg_time,
                traversability_time_ms=trav_time,
                hazard_detection_time_ms=hazard_time,
                classes=classes,
                confidence=confidence,
                segmentation_colored=segmentation_colored,
                traversability_map=trav_results['traversability_map'],
                best_direction=trav_results['best_direction'],
                direction_scores=trav_results['direction_scores'],
                costmap=trav_results['costmap'] if self.config.generate_costmap else None,
                hazards=hazards,
                hazard_summary=hazard_summary,
                overlay=overlay,
                traversability_viz=traversability_viz,
                hazard_viz=hazard_viz,
                stats=trav_results['stats']
            )
            
            # Update statistics
            if not warmup:
                self.inference_times.append(total_time)
                self.frame_count += 1
            
            return result

    def process_batch(
        self,
        images: List[np.ndarray]
    ) -> List[InferenceResult]:
        """
        Process batch of images
        
        Args:
            images: List of RGB images
            
        Returns:
            List of InferenceResult objects
        """
        results = []
        
        for image in images:
            result = self.process(image)
            results.append(result)
        
        return results

    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        if not self.inference_times:
            return {}
        
        times = list(self.inference_times)
        
        return {
            'frame_count': self.frame_count,
            'avg_time_ms': np.mean(times),
            'min_time_ms': np.min(times),
            'max_time_ms': np.max(times),
            'std_time_ms': np.std(times),
            'avg_fps': 1000.0 / np.mean(times),
            'device': self.config.device
        }

    def _log_config(self):
        """Log configuration"""
        logger.info("Pipeline Configuration:")
        logger.info(f"  Device: {self.config.device}")
        logger.info(f"  Input Size: {self.config.input_size}")
        logger.info(f"  Half Precision: {self.config.enable_half_precision}")
        logger.info(f"  Hazard Detection: {self.config.enable_hazard_detection}")
        logger.info(f"  Generate Overlays: {self.config.generate_overlays}")

    def reset_stats(self):
        """Reset performance statistics"""
        self.inference_times.clear()
        self.frame_count = 0
        logger.info("Performance statistics reset")


# Convenience function
def create_pipeline(
    model_path: Optional[str] = None,
    device: str = 'auto',
    fast_mode: bool = False
) -> RealtimePipeline:
    """
    Create inference pipeline with sensible defaults
    
    Args:
        model_path: Path to segmentation model weights
        device: 'cuda', 'cpu', or 'auto'
        fast_mode: Enable performance optimizations
        
    Returns:
        Initialized RealtimePipeline
    """
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    config = PipelineConfig(
        segmentation_model_path=Path(model_path) if model_path else None,
        device=device,
        enable_half_precision=fast_mode and device == 'cuda',
        input_size=(320, 320) if not fast_mode else (160, 160)
    )
    
    return RealtimePipeline(config)


# Testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Real-time Inference Pipeline Test")
    print("=" * 60)
    
    # Create pipeline
    pipeline = create_pipeline(device='cpu')
    
    # Create test image
    test_image = np.random.randint(0, 255, (960, 1280, 3), dtype=np.uint8)
    
    print("\nProcessing test image...")
    
    # Process
    result = pipeline.process(test_image)
    
    print("\nResults:")
    print(f"  Total Time: {result.total_time_ms:.1f}ms")
    print(f"  Segmentation: {result.segmentation_time_ms:.1f}ms")
    print(f"  Traversability: {result.traversability_time_ms:.1f}ms")
    print(f"  Hazard Detection: {result.hazard_detection_time_ms:.1f}ms")
    print(f"  FPS: {1000.0 / result.total_time_ms:.1f}")
    print(f"\n  Detected Hazards: {len(result.hazards)}")
    print(f"  Best Direction: {np.degrees(result.best_direction):.1f}°")
    print(f"  Avg Traversability: {result.stats['avg_traversability']:.3f}")
    
    # Performance stats
    print("\n" + "=" * 60)
    print("Processing 10 frames for statistics...")
    
    for i in range(10):
        _ = pipeline.process(test_image)
    
    stats = pipeline.get_performance_stats()
    print("\nPerformance Statistics:")
    print(f"  Frames Processed: {stats['frame_count']}")
    print(f"  Average Time: {stats['avg_time_ms']:.1f}ms")
    print(f"  Average FPS: {stats['avg_fps']:.1f}")
    print(f"  Min/Max Time: {stats['min_time_ms']:.1f}ms / {stats['max_time_ms']:.1f}ms")
