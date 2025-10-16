"""
Dashboard Tests
Tests for Streamlit dashboard components and utilities
"""

import pytest
import numpy as np
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import asyncio
import json

# Import dashboard modules
from dashboard.utils.ros_connector import ROSConnector, TerrainData
from dashboard.utils.data_processor import DataProcessor, ProcessedData
from dashboard.utils.visualizations import (
    create_terrain_plot,
    create_hazard_plot,
    create_performance_plot,
    create_direction_rose,
    create_terrain_pie_chart,
    create_hazard_bar_chart
)


# ============================================
# FIXTURES
# ============================================

@pytest.fixture
def test_terrain_data():
    """Create test terrain data"""
    return TerrainData(
        timestamp=time.time(),
        overlay_image=np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8),
        traversability_image=np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8),
        hazard_image=np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8),
        avg_traversability=0.75,
        safe_area_ratio=0.85,
        best_direction_deg=45.0,
        num_hazards=3,
        hazard_summary={'large_rock': 2, 'sand_trap': 1},
        inference_time_ms=200.0,
        fps=5.0,
        terrain_distribution={'soil': 0.6, 'bedrock': 0.3, 'sand': 0.1}
    )


@pytest.fixture
def data_processor():
    """Create data processor"""
    return DataProcessor(history_size=50)


# ============================================
# TERRAIN DATA TESTS
# ============================================

class TestTerrainData:
    """Test TerrainData class"""
    
    def test_terrain_data_creation(self, test_terrain_data):
        """Test terrain data creation"""
        assert test_terrain_data is not None
        assert test_terrain_data.timestamp > 0
    
    def test_terrain_data_properties(self, test_terrain_data):
        """Test terrain data properties"""
        assert hasattr(test_terrain_data, 'overlay_image')
        assert hasattr(test_terrain_data, 'traversability_image')
        assert hasattr(test_terrain_data, 'hazard_image')
        assert hasattr(test_terrain_data, 'avg_traversability')
        assert hasattr(test_terrain_data, 'num_hazards')
    
    def test_is_valid(self, test_terrain_data):
        """Test is_valid property"""
        assert test_terrain_data.is_valid
        
        # Test invalid data
        invalid_data = TerrainData(timestamp=time.time())
        assert not invalid_data.is_valid
    
    def test_image_shapes(self, test_terrain_data):
        """Test image shapes are correct"""
        if test_terrain_data.overlay_image is not None:
            assert test_terrain_data.overlay_image.shape == (512, 512, 3)
        
        if test_terrain_data.traversability_image is not None:
            assert test_terrain_data.traversability_image.shape == (512, 512, 3)
    
    def test_metric_ranges(self, test_terrain_data):
        """Test metrics are in valid ranges"""
        assert 0.0 <= test_terrain_data.avg_traversability <= 1.0
        assert 0.0 <= test_terrain_data.safe_area_ratio <= 1.0
        assert -180.0 <= test_terrain_data.best_direction_deg <= 180.0
        assert test_terrain_data.num_hazards >= 0
        assert test_terrain_data.fps >= 0.0


# ============================================
# ROS CONNECTOR TESTS
# ============================================

class TestROSConnector:
    """Test ROS connector"""
    
    def test_connector_creation(self):
        """Test connector creation"""
        connector = ROSConnector(websocket_uri="ws://localhost:8765")
        
        assert connector is not None
        assert connector.websocket_uri == "ws://localhost:8765"
    
    def test_connector_properties(self):
        """Test connector properties"""
        connector = ROSConnector()
        
        assert hasattr(connector, 'data_queue')
        assert hasattr(connector, 'connected')
        assert hasattr(connector, 'running')
    
    def test_get_statistics(self):
        """Test statistics retrieval"""
        connector = ROSConnector()
        stats = connector.get_statistics()
        
        assert isinstance(stats, dict)
        assert 'connected' in stats
        assert 'running' in stats
        assert 'messages_received' in stats
    
    def test_is_connected(self):
        """Test connection status"""
        connector = ROSConnector()
        
        # Should not be connected initially
        assert not connector.is_connected()
    
    @patch('websockets.connect')
    def test_decode_image(self, mock_connect):
        """Test image decoding from base64"""
        connector = ROSConnector()
        
        # Create test base64 image
        import base64
        import cv2
        
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', test_image)
        base64_str = base64.b64encode(buffer).decode('utf-8')
        full_str = f"data:image/jpeg;base64,{base64_str}"
        
        # Decode
        decoded = connector._decode_image(full_str)
        
        assert decoded is not None
        assert decoded.shape[2] == 3  # RGB


# ============================================
# DATA PROCESSOR TESTS
# ============================================

class TestDataProcessor:
    """Test data processor"""
    
    def test_processor_creation(self, data_processor):
        """Test processor creation"""
        assert data_processor is not None
        assert data_processor.history_size == 50
    
    def test_process_data(self, data_processor, test_terrain_data):
        """Test data processing"""
        processed = data_processor.process(test_terrain_data)
        
        assert isinstance(processed, ProcessedData)
        assert processed.current_traversability == test_terrain_data.avg_traversability
        assert processed.current_hazards == test_terrain_data.num_hazards
    
    def test_history_accumulation(self, data_processor, test_terrain_data):
        """Test history accumulation"""
        # Process multiple data points
        for i in range(10):
            data = test_terrain_data
            data.timestamp = time.time() + i
            data_processor.process(data)
        
        stats = data_processor.get_statistics()
        assert stats['data_points'] == 10
    
    def test_terrain_composition_averaging(self, data_processor, test_terrain_data):
        """Test terrain composition averaging"""
        # Process multiple times
        for _ in range(5):
            data_processor.process(test_terrain_data)
        
        processed = data_processor.process(test_terrain_data)
        
        assert processed.terrain_composition is not None
        assert 'soil' in processed.terrain_composition
    
    def test_hazard_breakdown(self, data_processor, test_terrain_data):
        """Test hazard breakdown accumulation"""
        # Process multiple times
        for _ in range(5):
            data_processor.process(test_terrain_data)
        
        processed = data_processor.process(test_terrain_data)
        
        assert processed.hazard_breakdown is not None
        assert 'large_rock' in processed.hazard_breakdown
    
    def test_reset(self, data_processor, test_terrain_data):
        """Test data reset"""
        # Process some data
        for _ in range(10):
            data_processor.process(test_terrain_data)
        
        # Reset
        data_processor.reset()
        
        stats = data_processor.get_statistics()
        assert stats['data_points'] == 0
    
    def test_statistics(self, data_processor, test_terrain_data):
        """Test statistics calculation"""
        for _ in range(5):
            data_processor.process(test_terrain_data)
        
        stats = data_processor.get_statistics()
        
        assert 'data_points' in stats
        assert 'max_history' in stats
        assert 'terrain_samples' in stats
    
    def test_history_size_limit(self, data_processor, test_terrain_data):
        """Test that history respects size limit"""
        # Process more than history size
        for _ in range(100):
            data_processor.process(test_terrain_data)
        
        stats = data_processor.get_statistics()
        assert stats['data_points'] <= data_processor.history_size


# ============================================
# VISUALIZATION TESTS
# ============================================

class TestVisualizations:
    """Test visualization functions"""
    
    def test_create_terrain_plot(self):
        """Test terrain plot creation"""
        time_hist = list(range(50))
        trav_hist = [0.7 + 0.1 * np.sin(t/5) for t in time_hist]
        
        fig = create_terrain_plot(time_hist, trav_hist)
        
        assert fig is not None
        assert hasattr(fig, 'data')
        assert len(fig.data) > 0
    
    def test_create_hazard_plot(self):
        """Test hazard plot creation"""
        time_hist = list(range(50))
        hazard_hist = [int(3 + 2 * np.sin(t/10)) for t in time_hist]
        
        fig = create_hazard_plot(time_hist, hazard_hist)
        
        assert fig is not None
        assert len(fig.data) > 0
    
    def test_create_performance_plot(self):
        """Test performance plot creation"""
        time_hist = list(range(50))
        fps_hist = [5.0 + np.random.rand() for _ in time_hist]
        
        fig = create_performance_plot(time_hist, fps_hist)
        
        assert fig is not None
        assert len(fig.data) > 0
    
    def test_create_direction_rose(self):
        """Test direction rose creation"""
        fig = create_direction_rose(45.0)
        
        assert fig is not None
        assert hasattr(fig, 'layout')
        assert 'polar' in fig.layout
    
    def test_create_terrain_pie_chart(self):
        """Test terrain pie chart creation"""
        terrain_comp = {
            'soil': 0.5,
            'bedrock': 0.3,
            'sand': 0.15,
            'big_rock': 0.05
        }
        
        fig = create_terrain_pie_chart(terrain_comp)
        
        assert fig is not None
        assert len(fig.data) > 0
    
    def test_create_hazard_bar_chart(self):
        """Test hazard bar chart creation"""
        hazard_breakdown = {
            'large_rock': 15,
            'sand_trap': 8,
            'uncertain': 5
        }
        
        fig = create_hazard_bar_chart(hazard_breakdown)
        
        assert fig is not None
        assert len(fig.data) > 0
    
    def test_empty_data_handling(self):
        """Test visualization with empty data"""
        # Should handle empty data gracefully
        fig = create_terrain_plot([], [])
        assert fig is not None
        
        fig = create_terrain_pie_chart({})
        assert fig is not None
    
    def test_plot_properties(self):
        """Test plot has correct properties"""
        time_hist = list(range(10))
        trav_hist = [0.7] * 10
        
        fig = create_terrain_plot(time_hist, trav_hist)
        
        assert fig.layout.template is not None
        y_range = fig.layout.yaxis.range
        assert list(y_range) == [0, 1]


# ============================================
# DASHBOARD COMPONENT TESTS
# ============================================

class TestDashboardComponents:
    """Test dashboard components (without Streamlit)"""
    
    def test_camera_view_data_requirements(self, test_terrain_data):
        """Test camera view data requirements"""
        # Component should handle data with images
        assert test_terrain_data.overlay_image is not None
        assert test_terrain_data.traversability_image is not None
    
    def test_metrics_calculation(self, test_terrain_data):
        """Test metrics calculations"""
        # Test that metrics are in valid ranges
        assert 0 <= test_terrain_data.avg_traversability <= 1
        assert test_terrain_data.num_hazards >= 0
        assert test_terrain_data.fps >= 0
    
    def test_hazard_panel_data(self, test_terrain_data):
        """Test hazard panel data"""
        assert test_terrain_data.hazard_summary is not None
        assert isinstance(test_terrain_data.hazard_summary, dict)
    
    def test_science_notes_generation(self, test_terrain_data):
        """Test that data is sufficient for science notes"""
        # Should have terrain distribution
        assert test_terrain_data.terrain_distribution is not None
        
        # Should have traversability data
        assert test_terrain_data.avg_traversability > 0


# ============================================
# INTEGRATION TESTS
# ============================================

class TestDashboardIntegration:
    """Test dashboard integration"""
    
    def test_connector_to_processor_flow(self, test_terrain_data):
        """Test data flow from connector to processor"""
        processor = DataProcessor()
        
        # Simulate receiving data
        processed = processor.process(test_terrain_data)
        
        assert processed is not None
        assert processed.current_traversability == test_terrain_data.avg_traversability
    
    def test_processor_to_visualization_flow(self, test_terrain_data):
        """Test data flow from processor to visualization"""
        processor = DataProcessor()
        
        # Process data
        for _ in range(10):
            processor.process(test_terrain_data)
        
        processed = processor.process(test_terrain_data)
        
        # Create visualization
        fig = create_terrain_plot(
            processed.time_history,
            processed.traversability_history
        )
        
        assert fig is not None
    
    def test_complete_dashboard_pipeline(self, test_terrain_data):
        """Test complete dashboard data pipeline"""
        # 1. Connector receives data (simulated)
        connector = ROSConnector()
        
        # 2. Processor processes data
        processor = DataProcessor()
        processed = processor.process(test_terrain_data)
        
        # 3. Create all visualizations
        terrain_plot = create_terrain_plot(
            processed.time_history,
            processed.traversability_history
        )
        
        hazard_plot = create_hazard_plot(
            processed.time_history,
            processed.hazard_history
        )
        
        direction_rose = create_direction_rose(processed.current_direction)
        
        # All should be created successfully
        assert terrain_plot is not None
        assert hazard_plot is not None
        assert direction_rose is not None


# ============================================
# PERFORMANCE TESTS
# ============================================

class TestDashboardPerformance:
    """Test dashboard performance"""
    
    def test_data_processing_speed(self, data_processor, test_terrain_data):
        """Test data processing speed"""
        start = time.time()
        
        for _ in range(100):
            data_processor.process(test_terrain_data)
        
        elapsed = time.time() - start
        
        # Should process 100 frames in reasonable time
        assert elapsed < 1.0  # Less than 1 second
    
    def test_visualization_creation_speed(self):
        """Test visualization creation speed"""
        time_hist = list(range(100))
        trav_hist = [0.7] * 100
        
        start = time.time()
        
        for _ in range(10):
            create_terrain_plot(time_hist, trav_hist)
        
        elapsed = time.time() - start
        
        # Should create 10 plots quickly
        assert elapsed < 2.0
    
    def test_memory_usage(self, data_processor, test_terrain_data):
        """Test memory usage doesn't grow unbounded"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process many frames
        for _ in range(1000):
            data_processor.process(test_terrain_data)
        
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_increase = mem_after - mem_before
        
        # Should not increase memory significantly
        assert mem_increase < 100  # Less than 100MB


# ============================================
# ERROR HANDLING TESTS
# ============================================

class TestDashboardErrorHandling:
    """Test dashboard error handling"""
    
    def test_missing_image_data(self):
        """Test handling of missing image data"""
        data = TerrainData(
            timestamp=time.time(),
            overlay_image=None,  # Missing
            avg_traversability=0.5
        )
        
        # Should still be processable
        processor = DataProcessor()
        processed = processor.process(data)
        
        assert processed is not None
    
    def test_invalid_metrics(self):
        """Test handling of invalid metrics"""
        data = TerrainData(
            timestamp=time.time(),
            avg_traversability=1.5,  # Invalid (> 1.0)
            num_hazards=-1  # Invalid (< 0)
        )
        
        # Should handle gracefully
        processor = DataProcessor()
        processed = processor.process(data)
        
        assert processed is not None
    
    def test_empty_terrain_distribution(self):
        """Test handling of empty terrain distribution"""
        data = TerrainData(
            timestamp=time.time(),
            terrain_distribution={}
        )
        
        processor = DataProcessor()
        processed = processor.process(data)
        
        assert processed is not None
        assert processed.terrain_composition is not None


# ============================================
# CONFIGURATION TESTS
# ============================================

class TestDashboardConfiguration:
    """Test dashboard configuration"""
    
    def test_load_config(self):
        """Test loading dashboard configuration"""
        config_path = Path("dashboard/config/dashboard_config.yaml")
        
        if config_path.exists():
            import yaml
            
            with open(config_path) as f:
                config = yaml.safe_load(f)
            
            assert config is not None
            assert 'dashboard' in config
    
    def test_config_structure(self):
        """Test configuration structure"""
        config_path = Path("dashboard/config/dashboard_config.yaml")
        
        if config_path.exists():
            import yaml
            
            with open(config_path) as f:
                config = yaml.safe_load(f)
            
            # Should have main sections
            expected_sections = ['connection', 'display', 'camera', 'map', 'hazards']
            
            for section in expected_sections:
                assert section in config


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
