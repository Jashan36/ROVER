"""
Data Processor
Process and aggregate terrain analysis data for dashboard display
"""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class ProcessedData:
    """Processed data for dashboard display"""
    # Current values
    current_traversability: float
    current_hazards: int
    current_direction: float
    current_fps: float
    
    # Historical data (for plotting)
    traversability_history: List[float]
    hazard_history: List[int]
    fps_history: List[float]
    time_history: List[float]
    
    # Aggregated statistics
    avg_traversability: float
    max_hazards: int
    avg_fps: float
    
    # Terrain composition
    terrain_composition: Dict[str, float]
    
    # Hazard breakdown
    hazard_breakdown: Dict[str, int]


class DataProcessor:
    """
    Process and aggregate terrain data
    """
    
    def __init__(self, history_size: int = 100):
        """
        Initialize data processor
        
        Args:
            history_size: Number of historical data points to keep
        """
        self.history_size = history_size
        
        # Historical data buffers
        self.traversability_buffer = deque(maxlen=history_size)
        self.hazard_buffer = deque(maxlen=history_size)
        self.fps_buffer = deque(maxlen=history_size)
        self.time_buffer = deque(maxlen=history_size)
        
        # Current values
        self.current_data = None
        
        # Aggregated terrain composition
        self.terrain_accumulator = {}
        self.terrain_count = 0
        
        # Hazard accumulator
        self.hazard_accumulator = {}
        
        logger.info(f"DataProcessor initialized with history_size={history_size}")

    def process(self, terrain_data) -> ProcessedData:
        """
        Process new terrain data
        
        Args:
            terrain_data: TerrainData object
            
        Returns:
            ProcessedData for display
        """
        self.current_data = terrain_data
        
        # Add to history
        self.traversability_buffer.append(terrain_data.avg_traversability)
        self.hazard_buffer.append(terrain_data.num_hazards)
        self.fps_buffer.append(terrain_data.fps)
        self.time_buffer.append(terrain_data.timestamp)
        
        # Update terrain composition
        terrain_dist = terrain_data.terrain_distribution
        if isinstance(terrain_dist, dict) and terrain_dist:
            self._update_terrain_composition(terrain_dist)
        
        # Update hazard breakdown
        hazard_summary = terrain_data.hazard_summary
        if isinstance(hazard_summary, dict) and hazard_summary:
            self._update_hazard_breakdown(hazard_summary)
        
        # Calculate aggregated statistics
        avg_trav = np.mean(list(self.traversability_buffer)) if self.traversability_buffer else 0.0
        max_haz = max(self.hazard_buffer) if self.hazard_buffer else 0
        avg_fps = np.mean(list(self.fps_buffer)) if self.fps_buffer else 0.0
        
        # Create processed data
        processed = ProcessedData(
            current_traversability=terrain_data.avg_traversability,
            current_hazards=terrain_data.num_hazards,
            current_direction=terrain_data.best_direction_deg,
            current_fps=terrain_data.fps,
            traversability_history=list(self.traversability_buffer),
            hazard_history=list(self.hazard_buffer),
            fps_history=list(self.fps_buffer),
            time_history=list(self.time_buffer),
            avg_traversability=avg_trav,
            max_hazards=max_haz,
            avg_fps=avg_fps,
            terrain_composition=self._get_terrain_composition(),
            hazard_breakdown=self._get_hazard_breakdown()
        )
        
        return processed

    def _update_terrain_composition(self, distribution: Dict[str, float]):
        """Update terrain composition accumulator"""
        for terrain, ratio in distribution.items():
            if terrain not in self.terrain_accumulator:
                self.terrain_accumulator[terrain] = 0.0
            
            self.terrain_accumulator[terrain] += ratio
        
        self.terrain_count += 1

    def _get_terrain_composition(self) -> Dict[str, float]:
        """Get averaged terrain composition"""
        if self.terrain_count == 0:
            return {}
        
        composition = {
            terrain: total / self.terrain_count
            for terrain, total in self.terrain_accumulator.items()
        }
        
        return composition

    def _update_hazard_breakdown(self, summary: Dict):
        """Update hazard breakdown accumulator"""
        # Support both nested {'by_type': {...}} and flat {'type': count} formats
        src = summary.get('by_type') if isinstance(summary, dict) and 'by_type' in summary else summary
        if isinstance(src, dict):
            for hazard_type, count in src.items():
                if hazard_type not in self.hazard_accumulator:
                    self.hazard_accumulator[hazard_type] = 0
                try:
                    inc = int(count)
                except Exception:
                    inc = 0
                self.hazard_accumulator[hazard_type] += inc

    def _get_hazard_breakdown(self) -> Dict[str, int]:
        """Get total hazard breakdown"""
        return dict(self.hazard_accumulator)

    def reset(self):
        """Reset all buffers and accumulators"""
        self.traversability_buffer.clear()
        self.hazard_buffer.clear()
        self.fps_buffer.clear()
        self.time_buffer.clear()
        
        self.terrain_accumulator.clear()
        self.terrain_count = 0
        
        self.hazard_accumulator.clear()
        
        self.current_data = None
        
        logger.info("Data processor reset")

    def get_statistics(self) -> Dict:
        """Get processor statistics"""
        return {
            'data_points': len(self.traversability_buffer),
            'max_history': self.history_size,
            'terrain_samples': self.terrain_count,
            'total_hazards': sum(self.hazard_accumulator.values())
        }


# Testing
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    processor = DataProcessor(history_size=50)
    
    # Simulate data
    from dashboard.utils.ros_connector import TerrainData
    import time
    
    for i in range(10):
        data = TerrainData(
            timestamp=time.time(),
            avg_traversability=0.7 + np.random.rand() * 0.2,
            num_hazards=np.random.randint(0, 5),
            fps=5.0 + np.random.rand(),
            terrain_distribution={
                'soil': 0.6,
                'bedrock': 0.2,
                'sand': 0.15,
                'big_rock': 0.05
            },
            hazard_summary={
                'by_type': {
                    'large_rock': 2,
                    'sand_trap': 1
                }
            }
        )
        
        processed = processor.process(data)
        
        print(f"\nIteration {i+1}:")
        print(f"  Current trav: {processed.current_traversability:.3f}")
        print(f"  Avg trav: {processed.avg_traversability:.3f}")
        print(f"  Current hazards: {processed.current_hazards}")
        print(f"  Max hazards: {processed.max_hazards}")
        print(f"  Terrain: {processed.terrain_composition}")
        
        time.sleep(0.5)
    
    print(f"\nStatistics: {processor.get_statistics()}")
