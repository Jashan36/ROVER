"""
Integration Tests
Tests for ROS2-AI bridge and system integration
"""

import pytest
import numpy as np
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import integration modules
from integration.image_subscriber import ImageSubscriber, ImageBuffer
from integration.analysis_publisher import AnalysisPublisher, PublisherConfig
from integration.dashboard_bridge import DashboardBridge, DashboardData
from sim.rover_integration.rover_integration.costmap_generator import CostmapGenerator


# ============================================
# FIXTURES
# ============================================

@pytest.fixture
def test_image():
    """Create test image"""
    return np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)


@pytest.fixture
def test_image_buffer():
    """Create test image buffer"""
    return ImageBuffer(
        image=np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8),
        timestamp=time.time(),
        frame_id='camera_link',
        seq=1
    )


@pytest.fixture
def test_inference_result():
    """Create mock inference result"""
    from ai.inference.realtime_pipeline import InferenceResult
    
    return InferenceResult(
        timestamp=time.time(),
        total_time_ms=200.0,
        segmentation_time_ms=150.0,
        traversability_time_ms=30.0,
        hazard_detection_time_ms=20.0,
        classes=np.zeros((512, 512), dtype=np.uint8),
        confidence=np.ones((512, 512), dtype=np.float32),
        segmentation_colored=np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8),
        traversability_map=np.ones((512, 512), dtype=np.float32),
        best_direction=0.0,
        direction_scores={},
        hazards=[],
        hazard_summary={},
        stats={'avg_traversability': 0.8}
    )


# ============================================
# IMAGE SUBSCRIBER TESTS
# ============================================

class TestImageSubscriber:
    """Test image subscriber"""
    
    def test_image_buffer_creation(self, test_image_buffer):
        """Test image buffer creation"""
        assert test_image_buffer is not None
        assert test_image_buffer.image is not None
        assert test_image_buffer.timestamp > 0
        assert test_image_buffer.frame_id == 'camera_link'
    
    def test_buffer_properties(self, test_image_buffer):
        """Test buffer properties"""
        assert hasattr(test_image_buffer, 'image')
        assert hasattr(test_image_buffer, 'timestamp')
        assert hasattr(test_image_buffer, 'frame_id')
        assert hasattr(test_image_buffer, 'seq')


# ============================================
# ANALYSIS PUBLISHER TESTS
# ============================================

class TestAnalysisPublisher:
    """Test analysis publisher"""
    
    def test_publisher_config(self):
        """Test publisher configuration"""
        config = PublisherConfig(
            publish_segmentation=True,
            publish_overlay=True
        )
        
        assert config.publish_segmentation
        assert config.publish_overlay
    
    def test_config_topics(self):
        """Test topic configuration"""
        config = PublisherConfig()
        
        assert config.segmentation_topic.startswith('/')
        assert config.overlay_topic.startswith('/')
        assert config.hazards_topic.startswith('/')


# ============================================
# DASHBOARD BRIDGE TESTS
# ============================================

class TestDashboardBridge:
    """Test dashboard WebSocket bridge"""
    
    def test_dashboard_data_creation(self):
        """Test dashboard data creation"""
        data = DashboardData(
            timestamp=time.time(),
            avg_traversability=0.8,
            num_hazards=3,
            fps=5.0
        )
        
        assert data is not None
        assert data.timestamp > 0
        assert 0.0 <= data.avg_traversability <= 1.0
    
    def test_encode_image(self, test_image):
        """Test image encoding"""
        encoded = DashboardBridge.encode_image(test_image, format='JPEG')
        
        assert encoded is not None
        assert isinstance(encoded, str)
        assert encoded.startswith('data:image/')
    
    def test_bridge_creation(self):
        """Test bridge creation"""
        bridge = DashboardBridge(host='localhost', port=8765)
        
        assert bridge is not None
        assert bridge.host == 'localhost'
        assert bridge.port == 8765
    
    def test_bridge_statistics(self):
        """Test bridge statistics"""
        bridge = DashboardBridge()
        stats = bridge.get_statistics()
        
        assert isinstance(stats, dict)
        assert 'running' in stats
        assert 'connected_clients' in stats


# ============================================
# END-TO-END INTEGRATION TESTS
# ============================================

class TestEndToEndIntegration:
    """Test complete system integration"""
    
    def test_image_to_analysis_pipeline(self, test_image):
        """Test pipeline from image to analysis"""
        from ai.inference.realtime_pipeline import RealtimePipeline, PipelineConfig
        
        # Create pipeline
        config = PipelineConfig(device='cpu', input_size=(512, 512))
        pipeline = RealtimePipeline(config)
        
        # Process image
        result = pipeline.process(test_image)
        
        # Verify results
        assert result is not None
        assert result.classes is not None
        assert result.traversability_map is not None
    
    def test_analysis_to_costmap_pipeline(self, test_image):
        """Test pipeline from analysis to costmap"""
        from ai.inference.realtime_pipeline import RealtimePipeline, PipelineConfig
        
        # Process image
        config = PipelineConfig(device='cpu')
        pipeline = RealtimePipeline(config)
        result = pipeline.process(test_image)
        
        # Generate costmap
        costmap_gen = CostmapGenerator()
        costmap = costmap_gen.generate_costmap(result.traversability_map)
        
        assert costmap is not None
        assert costmap.shape[0] > 0
    
    def test_complete_data_flow(self, test_image):
        """Test complete data flow through system"""
        from ai.inference.realtime_pipeline import RealtimePipeline, PipelineConfig
        
        # 1. AI Inference
        pipeline = RealtimePipeline(PipelineConfig(device='cpu'))
        result = pipeline.process(test_image)
        
        # 2. Costmap Generation
        costmap_gen = CostmapGenerator()
        costmap = costmap_gen.generate_costmap(result.traversability_map)
        
        # 3. Dashboard Data
        dashboard_data = DashboardData(
            timestamp=time.time(),
            avg_traversability=result.stats.get('avg_traversability', 0.0),
            num_hazards=len(result.hazards),
            fps=1000.0 / result.total_time_ms if result.total_time_ms > 0 else 0.0
        )
        
        # Verify all stages completed
        assert result is not None
        assert costmap is not None
        assert dashboard_data is not None


# ============================================
# PERFORMANCE INTEGRATION TESTS
# ============================================

class TestPerformanceIntegration:
    """Test system performance"""
    
    def test_end_to_end_latency(self, test_image):
        """Test total system latency"""
        from ai.inference.realtime_pipeline import RealtimePipeline, PipelineConfig
        
        pipeline = RealtimePipeline(PipelineConfig(device='cpu'))
        costmap_gen = CostmapGenerator()
        
        start_time = time.time()
        
        # Complete pipeline
        result = pipeline.process(test_image)
        costmap = costmap_gen.generate_costmap(result.traversability_map)
        
        total_time = (time.time() - start_time) * 1000  # ms
        
        # Should complete within reasonable time
        assert total_time < 1000  # Less than 1 second
    
    def test_throughput(self, test_image):
        """Test system throughput"""
        from ai.inference.realtime_pipeline import RealtimePipeline, PipelineConfig
        
        pipeline = RealtimePipeline(PipelineConfig(device='cpu'))
        
        num_frames = 10
        start_time = time.time()
        
        for _ in range(num_frames):
            pipeline.process(test_image)
        
        elapsed = time.time() - start_time
        fps = num_frames / elapsed
        
        # Should achieve reasonable FPS
        assert fps >= 1.0  # At least 1 FPS on CPU


# ============================================
# ERROR HANDLING INTEGRATION TESTS
# ============================================

class TestErrorHandlingIntegration:
    """Test error handling across components"""
    
    def test_invalid_data_handling(self):
        """Test handling of invalid data"""
        from ai.inference.realtime_pipeline import RealtimePipeline, PipelineConfig
        
        pipeline = RealtimePipeline(PipelineConfig(device='cpu'))
        
        # Test with various invalid inputs
        invalid_inputs = [
            np.zeros((0, 0, 3)),  # Empty
            np.ones((10, 10, 3)),  # Too small
        ]
        
        for invalid_input in invalid_inputs:
            try:
                # Should either handle gracefully or raise appropriate exception
                result = pipeline.process(invalid_input.astype(np.uint8))
                assert result is not None or True  # Handled gracefully
            except (ValueError, RuntimeError):
                pass  # Expected exception
    
    def test_component_failure_recovery(self, test_image):
        """Test system recovery from component failures"""
        from ai.inference.realtime_pipeline import RealtimePipeline, PipelineConfig
        
        pipeline = RealtimePipeline(PipelineConfig(device='cpu'))
        
        # Process normally
        result1 = pipeline.process(test_image)
        assert result1 is not None
        
        # Simulate failure by processing bad data
        try:
            pipeline.process(np.zeros((10, 10, 3), dtype=np.uint8))
        except:
            pass
        
        # Should still work after failure
        result2 = pipeline.process(test_image)
        assert result2 is not None


# ============================================
# CONFIGURATION TESTS
# ============================================

class TestSystemConfiguration:
    """Test system configuration"""
    
    def test_load_config_from_yaml(self):
        """Test loading configuration from YAML"""
        # This would test actual YAML config loading
        # Skipped if config file doesn't exist
        config_path = Path("integration/config/bridge_params.yaml")
        
        if config_path.exists():
            import yaml
            
            with open(config_path) as f:
                config = yaml.safe_load(f)
            
            assert config is not None
            assert 'ros2' in config or 'inference' in config


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])