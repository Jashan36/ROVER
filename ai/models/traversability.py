"""
Traversability Analysis Module
Analyzes terrain safety for rover navigation with hazard detection
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from enum import IntEnum

logger = logging.getLogger(__name__)


class TerrainClass(IntEnum):
    """Terrain class enumeration"""
    SOIL = 0
    BEDROCK = 1
    SAND = 2
    BIG_ROCK = 3
    BACKGROUND = 4


@dataclass
class Hazard:
    """Detected hazard information"""
    type: str
    position: Tuple[int, int]  # (y, x) in image coordinates
    severity: float  # 0-1
    area: int  # pixels
    confidence: float  # 0-1


class TraversabilityAnalyzer:
    """
    Analyzes terrain traversability for safe rover navigation
    """
    
    # Base traversability scores per terrain class
    BASE_SCORES = {
        TerrainClass.SOIL: 0.9,      # Highly traversable
        TerrainClass.BEDROCK: 0.6,   # Moderately traversable
        TerrainClass.SAND: 0.4,      # Risk of wheel slip
        TerrainClass.BIG_ROCK: 0.1,  # Dangerous obstacle
        TerrainClass.BACKGROUND: 0.0  # Unknown/unsafe
    }
    
    # Hazard thresholds
    MIN_ROCK_AREA = 50  # pixels
    MAX_SAFE_SAND_RATIO = 0.3
    LOW_CONFIDENCE_THRESHOLD = 0.6
    
    def __init__(
        self,
        confidence_weight: float = 0.3,
        safety_margin: int = 20,  # pixels
        min_traversability: float = 0.3
    ):
        """
        Initialize traversability analyzer
        
        Args:
            confidence_weight: How much confidence affects traversability (0-1)
            safety_margin: Safety buffer around hazards (pixels)
            min_traversability: Minimum acceptable traversability score
        """
        self.confidence_weight = confidence_weight
        self.safety_margin = safety_margin
        self.min_traversability = min_traversability
        
        logger.info("TraversabilityAnalyzer initialized")

    def analyze(
        self,
        pred_classes: np.ndarray,
        confidence: np.ndarray
    ) -> Dict:
        """
        Complete traversability analysis
        
        Args:
            pred_classes: Predicted terrain classes (H, W)
            confidence: Confidence scores (H, W)
            
        Returns:
            Dictionary with analysis results
        """
        # Calculate base traversability
        traversability_map = self._calculate_base_traversability(pred_classes)
        
        # Modulate by confidence
        traversability_map = self._apply_confidence_modulation(
            traversability_map, confidence
        )
        
        # Detect hazards
        hazards = self._detect_hazards(pred_classes, confidence)
        
        # Apply safety margins around hazards
        traversability_map = self._apply_safety_margins(
            traversability_map, hazards
        )
        
        # Find best navigation direction
        best_direction, direction_scores = self._find_best_direction(traversability_map)
        
        # Generate costmap for Nav2
        costmap = self._generate_costmap(traversability_map)
        
        # Calculate statistics
        stats = self._calculate_statistics(
            pred_classes, traversability_map, hazards
        )
        
        return {
            'traversability_map': traversability_map,
            'hazards': hazards,
            'best_direction': best_direction,
            'direction_scores': direction_scores,
            'costmap': costmap,
            'stats': stats
        }

    def _calculate_base_traversability(
        self,
        pred_classes: np.ndarray
    ) -> np.ndarray:
        """Calculate base traversability from terrain classes"""
        traversability = np.zeros_like(pred_classes, dtype=np.float32)
        
        for terrain_class, score in self.BASE_SCORES.items():
            mask = (pred_classes == terrain_class)
            traversability[mask] = score
        
        return traversability

    def _apply_confidence_modulation(
        self,
        traversability: np.ndarray,
        confidence: np.ndarray
    ) -> np.ndarray:
        """
        Modulate traversability by prediction confidence
        Low confidence = lower traversability (more cautious)
        """
        # Scale confidence to [0, 1] range if needed
        conf_normalized = np.clip(confidence, 0, 1)
        
        # Apply modulation: high confidence preserves score, low confidence reduces it
        modulated = traversability * (
            (1 - self.confidence_weight) + 
            self.confidence_weight * conf_normalized
        )
        
        return modulated

    def _detect_hazards(
        self,
        pred_classes: np.ndarray,
        confidence: np.ndarray
    ) -> List[Hazard]:
        """Detect various hazards in the terrain"""
        hazards = []
        
        # Detect large rocks
        rock_hazards = self._detect_rocks(pred_classes, confidence)
        hazards.extend(rock_hazards)
        
        # Detect sand patches
        sand_hazards = self._detect_sand_patches(pred_classes, confidence)
        hazards.extend(sand_hazards)
        
        # Detect low-confidence regions
        uncertainty_hazards = self._detect_uncertainty_regions(confidence)
        hazards.extend(uncertainty_hazards)
        
        logger.debug(f"Detected {len(hazards)} hazards")
        return hazards

    def _detect_rocks(
        self,
        pred_classes: np.ndarray,
        confidence: np.ndarray
    ) -> List[Hazard]:
        """Detect rock hazards using connected component analysis"""
        import cv2
        
        hazards = []
        rock_mask = (pred_classes == TerrainClass.BIG_ROCK).astype(np.uint8)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            rock_mask, connectivity=8
        )
        
        for i in range(1, num_labels):  # Skip background (0)
            area = stats[i, cv2.CC_STAT_AREA]
            
            if area >= self.MIN_ROCK_AREA:
                cy, cx = int(centroids[i][1]), int(centroids[i][0])
                
                # Calculate average confidence in this region
                component_mask = (labels == i)
                avg_confidence = confidence[component_mask].mean()
                
                # Severity based on size
                severity = min(1.0, area / (self.MIN_ROCK_AREA * 10))
                
                hazards.append(Hazard(
                    type='rock',
                    position=(cy, cx),
                    severity=severity,
                    area=int(area),
                    confidence=float(avg_confidence)
                ))
        
        return hazards

    def _detect_sand_patches(
        self,
        pred_classes: np.ndarray,
        confidence: np.ndarray
    ) -> List[Hazard]:
        """Detect large sand patches that may cause wheel slip"""
        import cv2
        
        hazards = []
        sand_mask = (pred_classes == TerrainClass.SAND).astype(np.uint8)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            sand_mask, connectivity=8
        )
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            
            # Large sand patches are hazardous
            if area >= 200:  # Threshold for concerning sand patches
                cy, cx = int(centroids[i][1]), int(centroids[i][0])
                
                component_mask = (labels == i)
                avg_confidence = confidence[component_mask].mean()
                
                severity = min(0.7, area / 1000)  # Max severity 0.7 for sand
                
                hazards.append(Hazard(
                    type='sand',
                    position=(cy, cx),
                    severity=severity,
                    area=int(area),
                    confidence=float(avg_confidence)
                ))
        
        return hazards

    def _detect_uncertainty_regions(
        self,
        confidence: np.ndarray
    ) -> List[Hazard]:
        """Detect regions with low prediction confidence"""
        import cv2
        
        hazards = []
        low_conf_mask = (confidence < self.LOW_CONFIDENCE_THRESHOLD).astype(np.uint8)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            low_conf_mask, connectivity=8
        )
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            
            if area >= 100:  # Significant uncertain region
                cy, cx = int(centroids[i][1]), int(centroids[i][0])
                
                component_mask = (labels == i)
                avg_confidence = confidence[component_mask].mean()
                
                severity = 1.0 - avg_confidence  # Lower confidence = higher severity
                
                hazards.append(Hazard(
                    type='uncertain',
                    position=(cy, cx),
                    severity=float(severity),
                    area=int(area),
                    confidence=float(avg_confidence)
                ))
        
        return hazards

    def _apply_safety_margins(
        self,
        traversability: np.ndarray,
        hazards: List[Hazard]
    ) -> np.ndarray:
        """Apply safety margins around detected hazards"""
        import cv2
        
        result = traversability.copy()
        
        for hazard in hazards:
            y, x = hazard.position
            
            # Create circular safety zone
            cv2.circle(
                result,
                (x, y),
                self.safety_margin,
                0.0,  # Zero traversability
                thickness=-1  # Filled circle
            )
        
        return result

    def _find_best_direction(
        self,
        traversability: np.ndarray
    ) -> Tuple[float, Dict[str, float]]:
        """
        Find best navigation direction based on traversability
        Returns angle in radians (0 = forward, positive = left)
        """
        height, width = traversability.shape
        
        # Define angular sectors (8 directions)
        n_sectors = 8
        angles = np.linspace(-np.pi, np.pi, n_sectors, endpoint=False)
        sector_scores = {}
        
        # Analyze each sector
        for angle in angles:
            direction_name = self._angle_to_direction_name(angle)
            
            # Sample traversability in this direction
            score = self._sample_direction_traversability(
                traversability, angle, width, height
            )
            
            sector_scores[direction_name] = float(score)
        
        # Find best direction
        best_direction_name = max(sector_scores, key=sector_scores.get)
        best_angle = self._direction_name_to_angle(best_direction_name)
        
        return best_angle, sector_scores

    def _sample_direction_traversability(
        self,
        traversability: np.ndarray,
        angle: float,
        width: int,
        height: int
    ) -> float:
        """Sample traversability along a direction"""
        # Start from bottom center (rover position)
        start_y, start_x = height - 1, width // 2
        
        # Sample points along this direction
        max_distance = min(height, width) // 2
        num_samples = 20
        
        scores = []
        for dist in np.linspace(0, max_distance, num_samples):
            x = int(start_x + dist * np.sin(angle))
            y = int(start_y - dist * np.cos(angle))  # Negative because y increases downward
            
            # Check bounds
            if 0 <= y < height and 0 <= x < width:
                scores.append(traversability[y, x])
        
        return np.mean(scores) if scores else 0.0

    @staticmethod
    def _angle_to_direction_name(angle: float) -> str:
        """Convert angle to direction name"""
        # Normalize angle to [0, 2π)
        angle = angle % (2 * np.pi)
        
        directions = [
            'forward', 'forward_left', 'left', 'back_left',
            'back', 'back_right', 'right', 'forward_right'
        ]
        
        idx = int((angle + np.pi / 8) / (np.pi / 4)) % 8
        return directions[idx]

    @staticmethod
    def _direction_name_to_angle(direction: str) -> float:
        """Convert direction name to angle"""
        direction_angles = {
            'forward': 0.0,
            'forward_left': np.pi / 4,
            'left': np.pi / 2,
            'back_left': 3 * np.pi / 4,
            'back': np.pi,
            'back_right': -3 * np.pi / 4,
            'right': -np.pi / 2,
            'forward_right': -np.pi / 4
        }
        return direction_angles.get(direction, 0.0)

    def _generate_costmap(self, traversability: np.ndarray) -> np.ndarray:
        """
        Generate Nav2-compatible costmap
        0 = free, 100 = lethal obstacle, 255 = unknown
        """
        # Invert traversability: high traversability = low cost
        cost = (1.0 - traversability) * 100
        
        # Clip to valid range
        costmap = np.clip(cost, 0, 100).astype(np.uint8)
        
        return costmap

    def _calculate_statistics(
        self,
        pred_classes: np.ndarray,
        traversability: np.ndarray,
        hazards: List[Hazard]
    ) -> Dict:
        """Calculate summary statistics"""
        total_pixels = pred_classes.size
        
        # Terrain distribution
        unique, counts = np.unique(pred_classes, return_counts=True)
        terrain_dist = {
            TerrainClass(cls).name: float(count / total_pixels)
            for cls, count in zip(unique, counts)
        }
        
        # Traversability statistics
        avg_traversability = float(traversability.mean())
        safe_area_ratio = float((traversability > self.min_traversability).sum() / total_pixels)
        
        # Hazard statistics
        hazard_counts = {}
        for hazard in hazards:
            hazard_counts[hazard.type] = hazard_counts.get(hazard.type, 0) + 1
        
        return {
            'terrain_distribution': terrain_dist,
            'avg_traversability': avg_traversability,
            'safe_area_ratio': safe_area_ratio,
            'num_hazards': len(hazards),
            'hazard_types': hazard_counts
        }

    def visualize_traversability(
        self,
        traversability: np.ndarray,
        colormap: str = 'RdYlGn'
    ) -> np.ndarray:
        """Create colored visualization of traversability map"""
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        
        # Apply colormap
        cmap = cm.get_cmap(colormap)
        colored = cmap(traversability)
        
        # Convert to uint8 RGB
        colored_rgb = (colored[:, :, :3] * 255).astype(np.uint8)
        
        return colored_rgb


# Testing
if __name__ == "__main__":
    analyzer = TraversabilityAnalyzer()
    
    # Create dummy data
    pred_classes = np.random.randint(0, 5, (512, 512), dtype=np.uint8)
    confidence = np.random.rand(512, 512).astype(np.float32)
    
    # Run analysis
    import time
    start = time.time()
    results = analyzer.analyze(pred_classes, confidence)
    end = time.time()
    
    logger.info(f"Analysis time: {(end-start)*1000:.1f}ms")
    logger.info(f"Detected {len(results['hazards'])} hazards")
    logger.info(f"Best direction: {results['best_direction']:.2f} rad")
    logger.info(f"Stats: {results['stats']}")