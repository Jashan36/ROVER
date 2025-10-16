"""
Perception Tests
Tests for AI perception models including segmentation, traversability, and hazard detection
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import cv2

# Import AI modules
from ai.models.terrain_segmentation import TerrainSegmentationModel, UNet
from ai.models.traversability import TraversabilityAnalyzer, TerrainClass
from ai.models.hazard_detector import HazardDetector, HazardType, HazardSeverity
from ai.inference.realtime_pipeline import RealtimePipeline, PipelineConfig


# ============================================
# FIXTURES
# ============================================

@pytest.fixture
def test_image():
    """Create test image"""
    return np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)


@pytest.fixture
def test_segmentation():
    """Create test segmentation"""
    return np.random.randint(0, 5, (512, 512), dtype=np.uint8)


@pytest.fixture
def test_confidence():
    """Create test confidence map"""
    return np.random.rand(512, 512).astype(np.float32)


@pytest.fixture
def segmentation_model():
    """Create segmentation model"""
    return TerrainSegmentationModel(device='cpu', input_size=(512, 512))


@pytest.fixture
def traversability_analyzer():
    """Create traversability analyzer"""
    return TraversabilityAnalyzer()


@pytest.fixture
def hazard_detector():
    """Create hazard detector"""
    return HazardDetector(image_height=512, image_width=512)


@pytest.fixture
def inference_pipeline():
    """Create inference pipeline"""
    config = PipelineConfig(
        device='cpu',
        input_size=(512, 512),
        enable_hazard_detection=True
    )
    return RealtimePipeline(config)


# ============================================
# SEGMENTATION MODEL TESTS
# ============================================

class TestTerrainSegmentation:
    """Test terrain segmentation model"""
    
    def test_unet_creation(self):
        """Test U-Net model creation"""
        model = UNet(n_channels=3, n_classes=5)
        
        assert model is not None
        assert model.n_channels == 3
        assert model.n_classes == 5
    
    def test_unet_forward_pass(self):
        """Test U-Net forward pass"""
        model = UNet(n_channels=3, n_classes=5)
        model.eval()
        
        # Create dummy input
        x = torch.randn(1, 3, 512, 512)
        
        # Forward pass
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (1, 5, 512, 512)
    
    def test_segmentation_model_predict(self, segmentation_model, test_image):
        """Test segmentation prediction"""
        result = segmentation_model.predict(test_image)
        
        assert 'classes' in result
        assert 'confidence' in result
        assert 'segmentation_colored' in result
        
        assert result['classes'].shape == (512, 512)
        assert result['confidence'].shape == (512, 512)
        assert result['segmentation_colored'].shape == (512, 512, 3)
    
    def test_segmentation_class_range(self, segmentation_model, test_image):
        """Test that predicted classes are in valid range"""
        result = segmentation_model.predict(test_image)
        classes = result['classes']
        
        assert classes.min() >= 0
        assert classes.max() < 5
    
    def test_confidence_range(self, segmentation_model, test_image):
        """Test that confidence is in [0, 1] range"""
        result = segmentation_model.predict(test_image)
        confidence = result['confidence']
        
        assert confidence.min() >= 0.0
        assert confidence.max() <= 1.0
    
    def test_class_stats(self, segmentation_model, test_image):
        """Test class statistics calculation"""
        result = segmentation_model.predict(test_image)
        stats = segmentation_model.get_class_stats(result['classes'])
        
        assert isinstance(stats, dict)
        assert sum(stats.values()) == pytest.approx(1.0, abs=0.01)
    
    def test_create_overlay(self, segmentation_model, test_image):
        """Test overlay creation"""
        result = segmentation_model.predict(test_image)
        overlay = segmentation_model.create_overlay(
            test_image,
            result['segmentation_colored'],
            alpha=0.5
        )
        
        assert overlay.shape == test_image.shape
        assert overlay.dtype == np.uint8


# ============================================
# TRAVERSABILITY TESTS
# ============================================

class TestTraversabilityAnalyzer:
    """Test traversability analysis"""
    
    def test_analyzer_creation(self, traversability_analyzer):
        """Test analyzer creation"""
        assert traversability_analyzer is not None
    
    def test_analyze(self, traversability_analyzer, test_segmentation, test_confidence):
        """Test complete analysis"""
        result = traversability_analyzer.analyze(test_segmentation, test_confidence)
        
        assert 'traversability_map' in result
        assert 'hazards' in result
        assert 'best_direction' in result
        assert 'direction_scores' in result
        assert 'costmap' in result
        assert 'stats' in result
    
    def test_traversability_range(self, traversability_analyzer, test_segmentation, test_confidence):
        """Test traversability is in [0, 1] range"""
        result = traversability_analyzer.analyze(test_segmentation, test_confidence)
        trav_map = result['traversability_map']
        
        assert trav_map.min() >= 0.0
        assert trav_map.max() <= 1.0
    
    def test_best_direction_range(self, traversability_analyzer, test_segmentation, test_confidence):
        """Test best direction is in valid range"""
        result = traversability_analyzer.analyze(test_segmentation, test_confidence)
        direction = result['best_direction']
        
        assert -np.pi <= direction <= np.pi
    
    def test_direction_scores(self, traversability_analyzer, test_segmentation, test_confidence):
        """Test direction scores"""
        result = traversability_analyzer.analyze(test_segmentation, test_confidence)
        scores = result['direction_scores']
        
        assert isinstance(scores, dict)
        assert len(scores) == 8  # 8 directions
        
        for score in scores.values():
            assert 0.0 <= score <= 1.0
    
    def test_costmap_generation(self, traversability_analyzer, test_segmentation, test_confidence):
        """Test costmap generation"""
        result = traversability_analyzer.analyze(test_segmentation, test_confidence)
        costmap = result['costmap']
        
        assert costmap.shape == test_segmentation.shape
        assert costmap.min() >= 0
        assert costmap.max() <= 100
    
    def test_statistics(self, traversability_analyzer, test_segmentation, test_confidence):
        """Test statistics calculation"""
        result = traversability_analyzer.analyze(test_segmentation, test_confidence)
        stats = result['stats']
        
        assert 'avg_traversability' in stats
        assert 'safe_area_ratio' in stats
        assert 'num_hazards' in stats
        assert 'terrain_distribution' in stats


# ============================================
# HAZARD DETECTION TESTS
# ============================================

class TestHazardDetector:
    """Test hazard detection"""
    
    def test_detector_creation(self, hazard_detector):
        """Test detector creation"""
        assert hazard_detector is not None
    
    def test_detect(self, hazard_detector, test_image, test_segmentation, test_confidence):
        """Test hazard detection"""
        hazards = hazard_detector.detect(test_image, test_segmentation, test_confidence)
        
        assert isinstance(hazards, list)
    
    def test_hazard_properties(self, hazard_detector, test_image, test_segmentation, test_confidence):
        """Test hazard properties"""
        # Create segmentation with known rocks
        seg = test_segmentation.copy()
        seg[200:250, 200:250] = 3  # Large rock
        
        hazards = hazard_detector.detect(test_image, seg, test_confidence)
        
        for hazard in hazards:
            assert hasattr(hazard, 'hazard_id')
            assert hasattr(hazard, 'type')
            assert hasattr(hazard, 'severity')
            assert hasattr(hazard, 'position')
            assert hasattr(hazard, 'distance')
            assert hasattr(hazard, 'confidence')
    
    def test_hazard_types(self):
        """Test hazard type enumeration"""
        assert HazardType.LARGE_ROCK
        assert HazardType.SAND_TRAP
        assert HazardType.STEEP_SLOPE
        assert HazardType.UNCERTAIN_TERRAIN
    
    def test_hazard_severity(self):
        """Test hazard severity enumeration"""
        assert HazardSeverity.LOW.value < HazardSeverity.MEDIUM.value
        assert HazardSeverity.MEDIUM.value < HazardSeverity.HIGH.value
        assert HazardSeverity.HIGH.value < HazardSeverity.CRITICAL.value
    
    def test_hazard_summary(self, hazard_detector, test_image, test_segmentation, test_confidence):
        """Test hazard summary generation"""
        hazards = hazard_detector.detect(test_image, test_segmentation, test_confidence)
        summary = hazard_detector.get_hazard_summary(hazards)
        
        assert isinstance(summary, dict)
        assert 'total_count' in summary
        assert 'by_type' in summary
        assert 'by_severity' in summary
    
    def test_visualize_hazards(self, hazard_detector, test_image, test_segmentation, test_confidence):
        """Test hazard visualization"""
        hazards = hazard_detector.detect(test_image, test_segmentation, test_confidence)
        viz = hazard_detector.visualize_hazards(test_image, hazards)
        
        assert viz.shape == test_image.shape
        assert viz.dtype == np.uint8


# ============================================
# INFERENCE PIPELINE TESTS
# ============================================

class TestInferencePipeline:
    """Test inference pipeline"""
    
    def test_pipeline_creation(self, inference_pipeline):
        """Test pipeline creation"""
        assert inference_pipeline is not None
    
    def test_process(self, inference_pipeline, test_image):
        """Test image processing"""
        result = inference_pipeline.process(test_image)
        
        assert result is not None
        assert result.total_time_ms > 0
        assert result.classes is not None
        assert result.confidence is not None
    
    def test_result_structure(self, inference_pipeline, test_image):
        """Test inference result structure"""
        result = inference_pipeline.process(test_image)
        
        # Timing
        assert hasattr(result, 'total_time_ms')
        assert hasattr(result, 'segmentation_time_ms')
        assert hasattr(result, 'traversability_time_ms')
        
        # Outputs
        assert hasattr(result, 'classes')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'traversability_map')
        assert hasattr(result, 'hazards')
        assert hasattr(result, 'stats')
    
    def test_performance_stats(self, inference_pipeline, test_image):
        """Test performance statistics"""
        # Process multiple images
        for _ in range(10):
            inference_pipeline.process(test_image)
        
        stats = inference_pipeline.get_performance_stats()
        
        assert 'frame_count' in stats
        assert 'avg_time_ms' in stats
        assert 'avg_fps' in stats
        assert stats['frame_count'] == 10
    
    def test_batch_processing(self, inference_pipeline, test_image):
        """Test batch processing"""
        images = [test_image for _ in range(5)]
        results = inference_pipeline.process_batch(images)
        
        assert len(results) == 5
        assert all(r is not None for r in results)


# ============================================
# INTEGRATION TESTS
# ============================================

class TestPerceptionIntegration:
    """Test integration between perception components"""
    
    def test_end_to_end_pipeline(self, test_image):
        """Test complete perception pipeline"""
        # Segmentation
        seg_model = TerrainSegmentationModel(device='cpu')
        seg_result = seg_model.predict(test_image)
        
        # Traversability
        trav_analyzer = TraversabilityAnalyzer()
        trav_result = trav_analyzer.analyze(
            seg_result['classes'],
            seg_result['confidence']
        )
        
        # Hazard detection
        hazard_detector = HazardDetector()
        hazards = hazard_detector.detect(
            test_image,
            seg_result['classes'],
            seg_result['confidence']
        )
        
        # Verify outputs
        assert trav_result['traversability_map'] is not None
        assert isinstance(hazards, list)
    
    def test_data_flow_consistency(self, test_image):
        """Test that data flows correctly between components"""
        pipeline = RealtimePipeline(PipelineConfig(device='cpu'))
        result = pipeline.process(test_image)
        
        # Check consistency
        assert result.classes.shape == result.confidence.shape
        assert result.traversability_map.shape == result.classes.shape
        
        # Check value ranges
        assert result.classes.min() >= 0
        assert 0.0 <= result.confidence.min() <= 1.0
        assert 0.0 <= result.traversability_map.min() <= 1.0


# ============================================
# PERFORMANCE TESTS
# ============================================

class TestPerformance:
    """Test performance requirements"""
    
    def test_inference_speed(self, inference_pipeline, test_image):
        """Test that inference meets speed requirements"""
        result = inference_pipeline.process(test_image)
        
        # Should process within 500ms on CPU
        assert result.total_time_ms < 500
    
    def test_memory_usage(self, inference_pipeline, test_image):
        """Test memory usage is reasonable"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process multiple images
        for _ in range(100):
            inference_pipeline.process(test_image)
        
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_increase = mem_after - mem_before
        
        # Should not leak significant memory
        assert mem_increase < 500  # Less than 500MB increase


# ============================================
# ERROR HANDLING TESTS
# ============================================

class TestErrorHandling:
    """Test error handling"""
    
    def test_invalid_image_shape(self, segmentation_model):
        """Test handling of invalid image shape"""
        invalid_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        
        # Should handle gracefully
        with pytest.raises((ValueError, Exception)):
            segmentation_model.predict(invalid_image)
    
    def test_invalid_dtype(self, segmentation_model):
        """Test handling of invalid data type"""
        invalid_image = np.random.rand(512, 512, 3)  # Float instead of uint8
        
        # Should still process
        result = segmentation_model.predict(invalid_image)
        assert result is not None
    
    def test_empty_image(self, segmentation_model):
        """Test handling of zero image"""
        empty_image = np.zeros((512, 512, 3), dtype=np.uint8)
        
        result = segmentation_model.predict(empty_image)
        assert result is not None


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])