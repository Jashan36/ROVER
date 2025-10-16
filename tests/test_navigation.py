"""
Navigation Tests
Tests for ROS2 navigation and costmap generation
"""

import pytest
import numpy as np
from pathlib import Path

# Import integration modules
from sim.rover_integration.rover_integration.costmap_generator import CostmapGenerator, CostmapConfig


# ============================================
# FIXTURES
# ============================================

@pytest.fixture
def costmap_generator():
    """Create costmap generator"""
    config = CostmapConfig(
        resolution=0.05,
        width=200,
        height=200
    )
    return CostmapGenerator(config)


@pytest.fixture
def test_traversability():
    """Create test traversability map"""
    trav = np.random.rand(512, 512).astype(np.float32)
    
    # Add some obstacles
    trav[200:250, 200:250] = 0.1
    trav[300:320, 300:320] = 0.0
    
    return trav


# ============================================
# COSTMAP GENERATOR TESTS
# ============================================

class TestCostmapGenerator:
    """Test costmap generation"""
    
    def test_generator_creation(self, costmap_generator):
        """Test generator creation"""
        assert costmap_generator is not None
    
    def test_generate_costmap(self, costmap_generator, test_traversability):
        """Test costmap generation"""
        costmap = costmap_generator.generate_costmap(test_traversability)
        
        assert costmap is not None
        assert costmap.shape == (200, 200)
        assert costmap.dtype == np.uint8
    
    def test_costmap_value_range(self, costmap_generator, test_traversability):
        """Test costmap values are in valid range"""
        costmap = costmap_generator.generate_costmap(test_traversability)
        
        assert costmap.min() >= 0
        assert costmap.max() <= 255
    
    def test_free_space_mapping(self, costmap_generator):
        """Test free space is correctly mapped"""
        # High traversability everywhere
        trav = np.ones((512, 512), dtype=np.float32) * 0.9
        
        costmap = costmap_generator.generate_costmap(trav)
        
        # Most should be free (cost = 0)
        free_ratio = (costmap == 0).sum() / costmap.size
        assert free_ratio > 0.8
    
    def test_obstacle_mapping(self, costmap_generator):
        """Test obstacles are correctly mapped"""
        # Low traversability everywhere
        trav = np.ones((512, 512), dtype=np.float32) * 0.1
        
        costmap = costmap_generator.generate_costmap(trav)
        
        # Most should be lethal (cost = 100)
        lethal_ratio = (costmap == 100).sum() / costmap.size
        assert lethal_ratio > 0.5
    
    def test_inflation(self, costmap_generator):
        """Test inflation around obstacles"""
        # Create single obstacle
        trav = np.ones((200, 200), dtype=np.float32) * 0.9
        trav[100, 100] = 0.0  # Single obstacle
        
        costmap = costmap_generator.generate_costmap(trav)
        
        # Check that area around obstacle is inflated
        inflated_area = costmap[95:105, 95:105]
        assert (inflated_area > 0).any()
    
    def test_layered_costmap(self, costmap_generator, test_traversability):
        """Test layered costmap generation"""
        # Create hazard map
        hazard_map = np.zeros((512, 512), dtype=np.float32)
        hazard_map[400:420, 400:420] = 1.0
        
        costmap = costmap_generator.create_layered_costmap(
            test_traversability,
            hazard_map=hazard_map
        )
        
        assert costmap is not None
        assert costmap.shape == (200, 200)
    
    def test_position_safety(self, costmap_generator, test_traversability):
        """Test position safety checking"""
        costmap = costmap_generator.generate_costmap(test_traversability)
        
        # Test safe position
        is_safe = costmap_generator.is_position_safe(costmap, 1.0, 1.0)
        assert isinstance(is_safe, bool)
    
    def test_cost_at_position(self, costmap_generator, test_traversability):
        """Test getting cost at specific position"""
        costmap = costmap_generator.generate_costmap(test_traversability)
        
        cost = costmap_generator.get_cost_at_position(costmap, 0.0, 0.0)
        assert 0 <= cost <= 255
    
    def test_visualize_costmap(self, costmap_generator, test_traversability):
        """Test costmap visualization"""
        costmap = costmap_generator.generate_costmap(test_traversability)
        viz = costmap_generator.visualize_costmap(costmap)
        
        assert viz.shape == (200, 200, 3)
        assert viz.dtype == np.uint8


# ============================================
# PATH FINDING TESTS
# ============================================

class TestPathFinding:
    """Test path finding functionality"""
    
    def test_find_safe_path(self, costmap_generator):
        """Test safe path finding"""
        # Create simple costmap
        trav = np.ones((200, 200), dtype=np.float32) * 0.9
        costmap = costmap_generator.generate_costmap(trav)
        
        # Find path
        start = (0.0, 0.0)
        goal = (5.0, 5.0)
        
        path = costmap_generator.find_safe_path(costmap, start, goal)
        
        if path is not None:
            assert len(path) > 0
            assert path.shape[1] == 2  # (x, y) pairs
    
    def test_path_avoids_obstacles(self, costmap_generator):
        """Test that path avoids obstacles"""
        # Create costmap with obstacle in middle
        trav = np.ones((200, 200), dtype=np.float32) * 0.9
        trav[95:105, 95:105] = 0.0  # Obstacle
        
        costmap = costmap_generator.generate_costmap(trav)
        
        start = (0.0, 0.0)
        goal = (10.0, 10.0)
        
        path = costmap_generator.find_safe_path(costmap, start, goal, max_cost=50)
        
        if path is not None:
            # Verify path doesn't go through obstacle
            for px, py in path:
                cost = costmap_generator.get_cost_at_position(costmap, px, py)
                assert cost <= 50


# ============================================
# INTEGRATION TESTS
# ============================================

class TestNavigationIntegration:
    """Test navigation integration with perception"""
    
    def test_perception_to_costmap_pipeline(self):
        """Test complete pipeline from perception to costmap"""
        from ai.models.terrain_segmentation import TerrainSegmentationModel
        from ai.models.traversability import TraversabilityAnalyzer
        
        # Create test image
        test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        # Segmentation
        seg_model = TerrainSegmentationModel(device='cpu')
        seg_result = seg_model.predict(test_image)
        
        # Traversability
        trav_analyzer = TraversabilityAnalyzer()
        trav_result = trav_analyzer.analyze(
            seg_result['classes'],
            seg_result['confidence']
        )
        
        # Costmap
        costmap_gen = CostmapGenerator()
        costmap = costmap_gen.generate_costmap(trav_result['traversability_map'])
        
        assert costmap is not None
        assert costmap.shape == (200, 200)


# ============================================
# CONFIGURATION TESTS
# ============================================

class TestCostmapConfig:
    """Test costmap configuration"""
    
    def test_config_creation(self):
        """Test config creation"""
        config = CostmapConfig(
            resolution=0.1,
            width=100,
            height=100
        )
        
        assert config.resolution == 0.1
        assert config.width == 100
        assert config.height == 100
    
    def test_config_defaults(self):
        """Test default configuration values"""
        config = CostmapConfig()
        
        assert config.resolution > 0
        assert config.width > 0
        assert config.height > 0
        assert config.free_threshold > config.occupied_threshold


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])