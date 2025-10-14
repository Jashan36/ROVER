"""
Costmap Generator
Converts AI traversability analysis to Nav2-compatible costmaps
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
import logging
import cv2

logger = logging.getLogger(__name__)


@dataclass
class CostmapConfig:
    """Configuration for costmap generation"""
    resolution: float = 0.05  # meters per pixel
    width: int = 200  # cells
    height: int = 200  # cells
    origin_x: float = -5.0  # meters
    origin_y: float = -5.0  # meters
    
    # Cost values
    free_threshold: float = 0.7  # Traversability above this = free
    occupied_threshold: float = 0.3  # Traversability below this = occupied
    
    # Inflation
    inflation_radius: float = 0.5  # meters
    cost_scaling_factor: float = 3.0
    
    # Update settings
    update_frequency: float = 5.0  # Hz
    publish_frequency: float = 2.0  # Hz


class CostmapGenerator:
    """
    Generates Nav2-compatible costmaps from traversability analysis
    """
    
    # Nav2 cost values
    FREE_SPACE = 0
    INSCRIBED_INFLATED_OBSTACLE = 99
    LETHAL_OBSTACLE = 100
    NO_INFORMATION = 255
    
    def __init__(self, config: Optional[CostmapConfig] = None):
        """
        Initialize costmap generator
        
        Args:
            config: Costmap configuration
        """
        self.config = config or CostmapConfig()
        
        logger.info("CostmapGenerator initialized")
        logger.info(f"  Resolution: {self.config.resolution}m/cell")
        logger.info(f"  Size: {self.config.width}x{self.config.height} cells")

    def generate_costmap(
        self,
        traversability_map: np.ndarray,
        image_resolution: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Generate Nav2 costmap from traversability map
        
        Args:
            traversability_map: Traversability scores (0-1)
            image_resolution: Original image resolution for scaling
            
        Returns:
            Costmap array (0-100, 255 for unknown)
        """
        # Resize to desired costmap dimensions
        if traversability_map.shape != (self.config.height, self.config.width):
            traversability_resized = cv2.resize(
                traversability_map,
                (self.config.width, self.config.height),
                interpolation=cv2.INTER_LINEAR
            )
        else:
            traversability_resized = traversability_map.copy()
        
        # Convert traversability to costs
        # High traversability (close to 1) = low cost (close to 0)
        # Low traversability (close to 0) = high cost (close to 100)
        costmap = self._traversability_to_cost(traversability_resized)
        
        # Apply inflation around obstacles
        costmap = self._apply_inflation(costmap)
        
        return costmap

    def _traversability_to_cost(self, traversability: np.ndarray) -> np.ndarray:
        """
        Convert traversability scores to cost values
        
        Args:
            traversability: Traversability scores (0-1)
            
        Returns:
            Cost map (0-100, 255)
        """
        costmap = np.zeros_like(traversability, dtype=np.uint8)
        
        # Free space: high traversability
        free_mask = traversability >= self.config.free_threshold
        costmap[free_mask] = self.FREE_SPACE
        
        # Lethal obstacles: very low traversability
        lethal_mask = traversability <= self.config.occupied_threshold
        costmap[lethal_mask] = self.LETHAL_OBSTACLE
        
        # Intermediate costs: scale between thresholds
        intermediate_mask = ~free_mask & ~lethal_mask
        if intermediate_mask.any():
            # Linear scaling between thresholds
            trav_intermediate = traversability[intermediate_mask]
            
            # Normalize to [0, 1] within threshold range
            normalized = (trav_intermediate - self.config.occupied_threshold) / \
                        (self.config.free_threshold - self.config.occupied_threshold)
            
            # Invert (high trav = low cost) and scale to cost range
            costs = ((1.0 - normalized) * self.INSCRIBED_INFLATED_OBSTACLE).astype(np.uint8)
            costmap[intermediate_mask] = costs
        
        return costmap

    def _apply_inflation(self, costmap: np.ndarray) -> np.ndarray:
        """
        Apply inflation around obstacles
        
        Args:
            costmap: Base costmap
            
        Returns:
            Inflated costmap
        """
        # Calculate inflation radius in cells
        inflation_radius_cells = int(self.config.inflation_radius / self.config.resolution)
        
        if inflation_radius_cells < 1:
            return costmap
        
        # Create inflated costmap
        inflated = costmap.copy()
        
        # Find lethal obstacles
        obstacles = (costmap >= self.INSCRIBED_INFLATED_OBSTACLE)
        
        # Apply distance transform
        distance = cv2.distanceTransform(
            (~obstacles).astype(np.uint8),
            cv2.DIST_L2,
            5
        )
        
        # Calculate inflation costs based on distance
        inflation_mask = (distance <= inflation_radius_cells) & (distance > 0)
        
        if inflation_mask.any():
            # Exponential decay from obstacle
            decay = np.exp(
                -self.config.cost_scaling_factor * distance[inflation_mask] / inflation_radius_cells
            )
            
            inflation_costs = (decay * self.INSCRIBED_INFLATED_OBSTACLE).astype(np.uint8)
            
            # Only inflate if higher than existing cost
            current_costs = inflated[inflation_mask]
            inflated[inflation_mask] = np.maximum(current_costs, inflation_costs)
        
        return inflated

    def create_layered_costmap(
        self,
        traversability_map: np.ndarray,
        hazard_map: Optional[np.ndarray] = None,
        static_map: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Create costmap with multiple layers
        
        Args:
            traversability_map: Traversability analysis
            hazard_map: Binary hazard map (1=hazard, 0=safe)
            static_map: Static obstacles from SLAM
            
        Returns:
            Combined costmap
        """
        # Base layer from traversability
        costmap = self.generate_costmap(traversability_map)
        
        # Add hazard layer
        if hazard_map is not None:
            hazard_resized = cv2.resize(
                hazard_map.astype(np.float32),
                (self.config.width, self.config.height),
                interpolation=cv2.INTER_NEAREST
            )
            
            # Mark hazards as lethal
            hazard_mask = hazard_resized > 0.5
            costmap[hazard_mask] = self.LETHAL_OBSTACLE
        
        # Add static obstacles
        if static_map is not None:
            static_resized = cv2.resize(
                static_map.astype(np.float32),
                (self.config.width, self.config.height),
                interpolation=cv2.INTER_NEAREST
            )
            
            # Combine with existing costmap (take maximum)
            costmap = np.maximum(costmap, static_resized.astype(np.uint8))
        
        return costmap

    def visualize_costmap(
        self,
        costmap: np.ndarray,
        colormap: str = 'jet'
    ) -> np.ndarray:
        """
        Create colored visualization of costmap
        
        Args:
            costmap: Costmap array
            colormap: Matplotlib colormap name
            
        Returns:
            RGB visualization
        """
        # Normalize costmap for visualization
        # Treat 255 (unknown) as separate
        vis_map = costmap.copy().astype(np.float32)
        vis_map[costmap == self.NO_INFORMATION] = np.nan
        
        # Normalize to [0, 1]
        valid_mask = ~np.isnan(vis_map)
        if valid_mask.any():
            vis_map[valid_mask] = vis_map[valid_mask] / 100.0
        
        # Apply colormap
        try:
            import matplotlib.cm as cm
            cmap = cm.get_cmap(colormap)
            
            colored = cmap(vis_map)
            rgb = (colored[:, :, :3] * 255).astype(np.uint8)
            
            # Mark unknown areas as gray
            unknown_mask = np.isnan(vis_map)
            rgb[unknown_mask] = [128, 128, 128]
            
            return rgb
            
        except ImportError:
            logger.warning("matplotlib not available, returning grayscale")
            gray = (vis_map * 255).astype(np.uint8)
            return np.stack([gray, gray, gray], axis=-1)

    def get_cost_at_position(
        self,
        costmap: np.ndarray,
        x: float,
        y: float
    ) -> int:
        """
        Get cost at world position
        
        Args:
            costmap: Costmap array
            x: X position in meters (map frame)
            y: Y position in meters (map frame)
            
        Returns:
            Cost value (0-100, 255)
        """
        # Convert world coordinates to grid coordinates
        grid_x = int((x - self.config.origin_x) / self.config.resolution)
        grid_y = int((y - self.config.origin_y) / self.config.resolution)
        
        # Check bounds
        if (0 <= grid_x < self.config.width and 
            0 <= grid_y < self.config.height):
            return int(costmap[grid_y, grid_x])
        
        return self.NO_INFORMATION

    def is_position_safe(
        self,
        costmap: np.ndarray,
        x: float,
        y: float,
        safety_threshold: int = 50
    ) -> bool:
        """
        Check if position is safe for navigation
        
        Args:
            costmap: Costmap array
            x: X position in meters
            y: Y position in meters
            safety_threshold: Maximum acceptable cost
            
        Returns:
            True if safe, False otherwise
        """
        cost = self.get_cost_at_position(costmap, x, y)
        
        if cost == self.NO_INFORMATION:
            return False  # Unknown is unsafe
        
        return cost <= safety_threshold

    def find_safe_path(
        self,
        costmap: np.ndarray,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        max_cost: int = 50
    ) -> Optional[np.ndarray]:
        """
        Simple path finding avoiding high-cost areas
        
        Args:
            costmap: Costmap array
            start: Start position (x, y) in meters
            goal: Goal position (x, y) in meters
            max_cost: Maximum acceptable cost
            
        Returns:
            Path as array of (x, y) positions or None
        """
        # Convert to grid coordinates
        start_grid = (
            int((start[0] - self.config.origin_x) / self.config.resolution),
            int((start[1] - self.config.origin_y) / self.config.resolution)
        )
        
        goal_grid = (
            int((goal[0] - self.config.origin_x) / self.config.resolution),
            int((goal[1] - self.config.origin_y) / self.config.resolution)
        )
        
        # Simple A* implementation
        # (For production, use Nav2's planners)
        try:
            path_grid = self._astar(costmap, start_grid, goal_grid, max_cost)
            
            if path_grid is None:
                return None
            
            # Convert back to world coordinates
            path_world = np.array([
                (
                    p[0] * self.config.resolution + self.config.origin_x,
                    p[1] * self.config.resolution + self.config.origin_y
                )
                for p in path_grid
            ])
            
            return path_world
            
        except Exception as e:
            logger.error(f"Path finding failed: {e}")
            return None

    def _astar(
        self,
        costmap: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        max_cost: int
    ) -> Optional[list]:
        """Simple A* pathfinding implementation"""
        from heapq import heappush, heappop
        
        # Heuristic: Euclidean distance
        def heuristic(a, b):
            return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
        
        # Check if start/goal are valid
        if not (0 <= start[0] < costmap.shape[1] and 0 <= start[1] < costmap.shape[0]):
            return None
        if not (0 <= goal[0] < costmap.shape[1] and 0 <= goal[1] < costmap.shape[0]):
            return None
        
        if costmap[start[1], start[0]] > max_cost or costmap[goal[1], goal[0]] > max_cost:
            return None
        
        # A* algorithm
        open_set = []
        heappush(open_set, (0, start))
        
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}
        
        # 8-connected grid
        neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), 
                    (0, 1), (1, -1), (1, 0), (1, 1)]
        
        max_iterations = 10000
        iterations = 0
        
        while open_set and iterations < max_iterations:
            iterations += 1
            
            current = heappop(open_set)[1]
            
            if current == goal:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]
            
            for dx, dy in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Check bounds
                if not (0 <= neighbor[0] < costmap.shape[1] and 
                       0 <= neighbor[1] < costmap.shape[0]):
                    continue
                
                # Check cost
                if costmap[neighbor[1], neighbor[0]] > max_cost:
                    continue
                
                # Calculate cost (Euclidean distance + cost penalty)
                move_cost = np.sqrt(dx**2 + dy**2)
                cost_penalty = costmap[neighbor[1], neighbor[0]] / 100.0
                tentative_g = g_score[current] + move_cost * (1.0 + cost_penalty)
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    heappush(open_set, (f_score[neighbor], neighbor))
        
        # No path found
        return None


# Testing
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Costmap Generator Test")
    print("=" * 60)
    
    # Create generator
    generator = CostmapGenerator()
    
    # Create test traversability map
    traversability = np.random.rand(512, 512).astype(np.float32)
    
    # Add some obstacles
    traversability[200:250, 200:250] = 0.1  # Low traversability
    traversability[300:320, 300:320] = 0.0  # Obstacle
    
    # Generate costmap
    print("\nGenerating costmap...")
    costmap = generator.generate_costmap(traversability)
    
    print(f"Costmap shape: {costmap.shape}")
    print(f"Costmap range: [{costmap.min()}, {costmap.max()}]")
    print(f"Free cells: {(costmap == 0).sum()}")
    print(f"Occupied cells: {(costmap >= 99).sum()}")
    
    # Test position safety
    print("\nTesting position safety...")
    test_positions = [
        (0.0, 0.0),
        (1.0, 1.0),
        (-2.0, -2.0)
    ]
    
    for x, y in test_positions:
        safe = generator.is_position_safe(costmap, x, y)
        cost = generator.get_cost_at_position(costmap, x, y)
        print(f"  Position ({x:.1f}, {y:.1f}): safe={safe}, cost={cost}")
    
    # Visualize
    print("\nGenerating visualization...")
    viz = generator.visualize_costmap(costmap)
    print(f"Visualization shape: {viz.shape}")