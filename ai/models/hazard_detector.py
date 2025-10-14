"""
Dedicated Real-time Hazard Detection Module
Specialized detector for critical obstacles using multiple detection strategies
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import deque

logger = logging.getLogger(__name__)


class HazardType(Enum):
    """Types of detected hazards"""
    LARGE_ROCK = "large_rock"
    ROCK_CLUSTER = "rock_cluster"
    STEEP_SLOPE = "steep_slope"
    SAND_TRAP = "sand_trap"
    CLIFF_EDGE = "cliff_edge"
    UNCERTAIN_TERRAIN = "uncertain_terrain"
    SHADOW_REGION = "shadow_region"


class HazardSeverity(Enum):
    """Hazard severity levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class DetectedHazard:
    """Complete hazard information"""
    hazard_id: int
    type: HazardType
    severity: HazardSeverity
    position: Tuple[int, int]  # (y, x) in image coordinates
    position_3d: Optional[Tuple[float, float, float]] = None  # (x, y, z) in meters
    bounding_box: Tuple[int, int, int, int] = (0, 0, 0, 0)  # (x, y, w, h)
    area: int = 0  # pixels
    confidence: float = 0.0  # 0-1
    distance: float = 0.0  # estimated distance in meters
    bearing: float = 0.0  # angle from forward direction (radians)
    description: str = ""
    detection_time: float = 0.0
    metadata: Dict = field(default_factory=dict)


class HazardDetector:
    """
    Multi-strategy hazard detection system
    Combines semantic segmentation, edge detection, and statistical analysis
    """
    
    def __init__(
        self,
        image_height: int = 512,
        image_width: int = 512,
        camera_fov: float = 1.396,  # 80 degrees in radians
        camera_height: float = 0.4,  # meters
        camera_tilt: float = 0.1,    # radians
        detection_history_size: int = 30
    ):
        """
        Initialize hazard detector
        
        Args:
            image_height: Input image height
            image_width: Input image width
            camera_fov: Horizontal field of view in radians
            camera_height: Camera mounting height in meters
            camera_tilt: Camera tilt angle (positive = down)
            detection_history_size: Number of frames to keep in history
        """
        self.image_height = image_height
        self.image_width = image_width
        self.camera_fov = camera_fov
        self.camera_height = camera_height
        self.camera_tilt = camera_tilt
        
        # Detection history for temporal filtering
        self.detection_history = deque(maxlen=detection_history_size)
        self.hazard_id_counter = 0
        
        # Calibration parameters
        self.focal_length = (image_width / 2) / np.tan(camera_fov / 2)
        
        logger.info(f"HazardDetector initialized: focal_length={self.focal_length:.1f}px")

    def detect(
        self,
        image: np.ndarray,
        segmentation: np.ndarray,
        confidence: np.ndarray,
        depth_map: Optional[np.ndarray] = None
    ) -> List[DetectedHazard]:
        """
        Main detection pipeline
        
        Args:
            image: Original RGB image (H, W, 3)
            segmentation: Terrain class predictions (H, W)
            confidence: Prediction confidence (H, W)
            depth_map: Optional depth information (H, W)
            
        Returns:
            List of detected hazards
        """
        hazards = []
        
        # Strategy 1: Semantic segmentation-based detection
        seg_hazards = self._detect_from_segmentation(segmentation, confidence)
        hazards.extend(seg_hazards)
        
        # Strategy 2: Edge-based detection (for rocks)
        edge_hazards = self._detect_from_edges(image, segmentation)
        hazards.extend(edge_hazards)
        
        # Strategy 3: Texture analysis (for sand traps)
        texture_hazards = self._detect_from_texture(image, segmentation)
        hazards.extend(texture_hazards)
        
        # Strategy 4: Gradient analysis (for slopes)
        if depth_map is not None:
            slope_hazards = self._detect_slopes_from_depth(depth_map)
            hazards.extend(slope_hazards)
        
        # Strategy 5: Shadow detection
        shadow_hazards = self._detect_shadows(image)
        hazards.extend(shadow_hazards)
        
        # Remove duplicate detections
        hazards = self._merge_overlapping_hazards(hazards)
        
        # Estimate distances and bearings
        hazards = self._estimate_spatial_properties(hazards)
        
        # Filter based on temporal consistency
        hazards = self._temporal_filter(hazards)
        
        # Sort by severity and distance
        hazards.sort(key=lambda h: (h.severity.value, h.distance), reverse=True)
        
        logger.debug(f"Detected {len(hazards)} hazards")
        
        return hazards

    def _detect_from_segmentation(
        self,
        segmentation: np.ndarray,
        confidence: np.ndarray
    ) -> List[DetectedHazard]:
        """Detect hazards from semantic segmentation"""
        hazards = []
        
        # Detect large rocks (class 3)
        rock_mask = (segmentation == 3).astype(np.uint8)
        rock_hazards = self._find_connected_regions(
            rock_mask,
            confidence,
            HazardType.LARGE_ROCK,
            min_area=50,
            severity_func=lambda area: HazardSeverity.HIGH if area > 500 else HazardSeverity.MEDIUM
        )
        hazards.extend(rock_hazards)
        
        # Detect sand regions (class 2)
        sand_mask = (segmentation == 2).astype(np.uint8)
        sand_hazards = self._find_connected_regions(
            sand_mask,
            confidence,
            HazardType.SAND_TRAP,
            min_area=200,
            severity_func=lambda area: HazardSeverity.MEDIUM if area > 1000 else HazardSeverity.LOW
        )
        hazards.extend(sand_hazards)
        
        # Detect uncertain regions (low confidence)
        uncertain_mask = (confidence < 0.6).astype(np.uint8)
        uncertain_hazards = self._find_connected_regions(
            uncertain_mask,
            confidence,
            HazardType.UNCERTAIN_TERRAIN,
            min_area=150,
            severity_func=lambda area: HazardSeverity.LOW
        )
        hazards.extend(uncertain_hazards)
        
        return hazards

    def _detect_from_edges(
        self,
        image: np.ndarray,
        segmentation: np.ndarray
    ) -> List[DetectedHazard]:
        """Detect rocks using edge detection"""
        hazards = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate edges to connect nearby contours
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edges_dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(
            edges_dilated,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter small contours
            if area < 100:
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate aspect ratio
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # Rocks tend to be roughly circular (aspect ratio near 1)
            if 0.5 < aspect_ratio < 2.0:
                # Calculate moments for centroid
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Check if this region is not already classified as rock
                    if cy < segmentation.shape[0] and cx < segmentation.shape[1]:
                        if segmentation[cy, cx] != 3:  # Not already classified as rock
                            
                            # Determine severity based on size and position
                            if cy > segmentation.shape[0] * 0.7:  # Close to rover
                                severity = HazardSeverity.HIGH
                            else:
                                severity = HazardSeverity.MEDIUM
                            
                            hazard = DetectedHazard(
                                hazard_id=self._get_next_id(),
                                type=HazardType.LARGE_ROCK,
                                severity=severity,
                                position=(cy, cx),
                                bounding_box=(x, y, w, h),
                                area=int(area),
                                confidence=0.7,  # Edge-based confidence
                                description=f"Edge-detected rock ({w}x{h}px)"
                            )
                            hazards.append(hazard)
        
        return hazards

    def _detect_from_texture(
        self,
        image: np.ndarray,
        segmentation: np.ndarray
    ) -> List[DetectedHazard]:
        """Detect sand traps using texture analysis"""
        hazards = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Calculate local standard deviation (texture measure)
        kernel_size = 15
        mean = cv2.blur(gray.astype(np.float32), (kernel_size, kernel_size))
        sqr_mean = cv2.blur((gray.astype(np.float32) ** 2), (kernel_size, kernel_size))
        std_dev = np.sqrt(np.maximum(sqr_mean - mean ** 2, 0))
        
        # Low texture indicates smooth sand
        low_texture_mask = (std_dev < 20).astype(np.uint8)
        
        # Also check if it's classified as sand
        sand_mask = (segmentation == 2).astype(np.uint8)
        
        # Combine both conditions
        sand_trap_mask = cv2.bitwise_and(low_texture_mask, sand_mask)
        
        # Find connected regions
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            sand_trap_mask, connectivity=8
        )
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            
            if area > 300:  # Significant sand trap
                cx, cy = int(centroids[i][0]), int(centroids[i][1])
                x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
                w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
                
                severity = HazardSeverity.HIGH if area > 1500 else HazardSeverity.MEDIUM
                
                hazard = DetectedHazard(
                    hazard_id=self._get_next_id(),
                    type=HazardType.SAND_TRAP,
                    severity=severity,
                    position=(cy, cx),
                    bounding_box=(x, y, w, h),
                    area=int(area),
                    confidence=0.75,
                    description=f"Low-texture sand trap ({area}px)",
                    metadata={'avg_texture': float(std_dev[labels == i].mean())}
                )
                hazards.append(hazard)
        
        return hazards

    def _detect_slopes_from_depth(
        self,
        depth_map: np.ndarray
    ) -> List[DetectedHazard]:
        """Detect steep slopes from depth information"""
        hazards = []
        
        # Calculate gradients
        grad_y, grad_x = np.gradient(depth_map)
        
        # Calculate slope magnitude
        slope_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Threshold for steep slopes (tune based on rover capabilities)
        steep_threshold = 0.5  # ~26 degrees
        steep_mask = (slope_magnitude > steep_threshold).astype(np.uint8)
        
        # Find connected regions
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            steep_mask, connectivity=8
        )
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            
            if area > 100:
                cx, cy = int(centroids[i][0]), int(centroids[i][1])
                x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
                w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
                
                # Calculate average slope in region
                region_mask = (labels == i)
                avg_slope = slope_magnitude[region_mask].mean()
                max_slope = slope_magnitude[region_mask].max()
                
                # Determine severity
                if max_slope > 1.0:  # ~45 degrees
                    severity = HazardSeverity.CRITICAL
                elif max_slope > 0.7:  # ~35 degrees
                    severity = HazardSeverity.HIGH
                else:
                    severity = HazardSeverity.MEDIUM
                
                hazard = DetectedHazard(
                    hazard_id=self._get_next_id(),
                    type=HazardType.STEEP_SLOPE,
                    severity=severity,
                    position=(cy, cx),
                    bounding_box=(x, y, w, h),
                    area=int(area),
                    confidence=0.85,
                    description=f"Steep slope (avg={np.degrees(np.arctan(avg_slope)):.1f}°)",
                    metadata={
                        'avg_slope_rad': float(avg_slope),
                        'max_slope_rad': float(max_slope),
                        'avg_slope_deg': float(np.degrees(np.arctan(avg_slope))),
                        'max_slope_deg': float(np.degrees(np.arctan(max_slope)))
                    }
                )
                hazards.append(hazard)
        
        return hazards

    def _detect_shadows(self, image: np.ndarray) -> List[DetectedHazard]:
        """Detect shadow regions that may hide hazards"""
        hazards = []
        
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Shadow detection: low value (darkness)
        value = hsv[:, :, 2]
        shadow_mask = (value < 60).astype(np.uint8)
        
        # Morphological operations to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel)
        
        # Find connected regions
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            shadow_mask, connectivity=8
        )
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            
            if area > 200:  # Significant shadow
                cx, cy = int(centroids[i][0]), int(centroids[i][1])
                x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
                w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
                
                hazard = DetectedHazard(
                    hazard_id=self._get_next_id(),
                    type=HazardType.SHADOW_REGION,
                    severity=HazardSeverity.LOW,
                    position=(cy, cx),
                    bounding_box=(x, y, w, h),
                    area=int(area),
                    confidence=0.6,
                    description=f"Shadow region ({area}px)",
                    metadata={'avg_brightness': float(value[labels == i].mean())}
                )
                hazards.append(hazard)
        
        return hazards

    def _find_connected_regions(
        self,
        mask: np.ndarray,
        confidence: np.ndarray,
        hazard_type: HazardType,
        min_area: int,
        severity_func
    ) -> List[DetectedHazard]:
        """Helper to find connected regions in a binary mask"""
        hazards = []
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            
            if area >= min_area:
                cx, cy = int(centroids[i][0]), int(centroids[i][1])
                x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
                w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
                
                # Calculate average confidence
                region_mask = (labels == i)
                avg_confidence = confidence[region_mask].mean()
                
                severity = severity_func(area)
                
                hazard = DetectedHazard(
                    hazard_id=self._get_next_id(),
                    type=hazard_type,
                    severity=severity,
                    position=(cy, cx),
                    bounding_box=(x, y, w, h),
                    area=int(area),
                    confidence=float(avg_confidence),
                    description=f"{hazard_type.value} ({area}px)"
                )
                hazards.append(hazard)
        
        return hazards

    def _merge_overlapping_hazards(
        self,
        hazards: List[DetectedHazard]
    ) -> List[DetectedHazard]:
        """Merge overlapping hazard detections"""
        if not hazards:
            return []
        
        # Sort by area (larger first)
        hazards.sort(key=lambda h: h.area, reverse=True)
        
        merged = []
        used = set()
        
        for i, h1 in enumerate(hazards):
            if i in used:
                continue
            
            # Find overlapping hazards
            overlapping = [h1]
            
            for j, h2 in enumerate(hazards[i+1:], start=i+1):
                if j in used:
                    continue
                
                # Check if bounding boxes overlap
                if self._boxes_overlap(h1.bounding_box, h2.bounding_box):
                    overlapping.append(h2)
                    used.add(j)
            
            # Merge if multiple detections
            if len(overlapping) > 1:
                merged_hazard = self._merge_hazard_group(overlapping)
                merged.append(merged_hazard)
            else:
                merged.append(h1)
        
        return merged

    def _boxes_overlap(
        self,
        box1: Tuple[int, int, int, int],
        box2: Tuple[int, int, int, int],
        threshold: float = 0.3
    ) -> bool:
        """Check if two bounding boxes overlap significantly"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return False
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = w1 * h1
        box2_area = w2 * h2
        
        # IoU (Intersection over Union)
        iou = intersection_area / float(box1_area + box2_area - intersection_area)
        
        return iou > threshold

    def _merge_hazard_group(
        self,
        hazards: List[DetectedHazard]
    ) -> DetectedHazard:
        """Merge multiple overlapping hazards into one"""
        # Take highest severity
        max_severity = max(h.severity for h in hazards)
        
        # Average position
        avg_y = int(np.mean([h.position[0] for h in hazards]))
        avg_x = int(np.mean([h.position[1] for h in hazards]))
        
        # Union bounding box
        min_x = min(h.bounding_box[0] for h in hazards)
        min_y = min(h.bounding_box[1] for h in hazards)
        max_x = max(h.bounding_box[0] + h.bounding_box[2] for h in hazards)
        max_y = max(h.bounding_box[1] + h.bounding_box[3] for h in hazards)
        
        merged_box = (min_x, min_y, max_x - min_x, max_y - min_y)
        
        # Sum areas
        total_area = sum(h.area for h in hazards)
        
        # Average confidence
        avg_confidence = np.mean([h.confidence for h in hazards])
        
        # Combine types
        types = set(h.type for h in hazards)
        primary_type = hazards[0].type  # Use first (largest) detection's type
        
        description = f"Merged: {', '.join(t.value for t in types)}"
        
        return DetectedHazard(
            hazard_id=self._get_next_id(),
            type=primary_type,
            severity=max_severity,
            position=(avg_y, avg_x),
            bounding_box=merged_box,
            area=total_area,
            confidence=float(avg_confidence),
            description=description,
            metadata={'merged_from': len(hazards)}
        )

    def _estimate_spatial_properties(
        self,
        hazards: List[DetectedHazard]
    ) -> List[DetectedHazard]:
        """Estimate distance and bearing for each hazard"""
        for hazard in hazards:
            y, x = hazard.position
            
            # Estimate distance using pinhole camera model
            # This is a rough approximation
            pixel_from_center = y - (self.image_height / 2)
            
            # Account for camera tilt
            elevation_angle = (pixel_from_center / self.focal_length) + self.camera_tilt
            
            # Distance estimation (assuming flat ground)
            if np.abs(elevation_angle) > 0.01:
                distance = self.camera_height / np.tan(elevation_angle)
                distance = max(0.5, min(distance, 50.0))  # Clamp to reasonable range
            else:
                distance = 10.0  # Default for far objects
            
            hazard.distance = float(distance)
            
            # Calculate bearing (angle from forward direction)
            pixel_from_center_x = x - (self.image_width / 2)
            bearing = np.arctan(pixel_from_center_x / self.focal_length)
            hazard.bearing = float(bearing)
        
        return hazards

    def _temporal_filter(
        self,
        current_hazards: List[DetectedHazard]
    ) -> List[DetectedHazard]:
        """Filter hazards based on temporal consistency"""
        # Add current detections to history
        self.detection_history.append(current_hazards)
        
        # If not enough history, return current
        if len(self.detection_history) < 3:
            return current_hazards
        
        # Find persistent hazards (detected in multiple frames)
        filtered = []
        
        for hazard in current_hazards:
            # Count how many times similar hazards appear in history
            match_count = 0
            
            for past_frame in list(self.detection_history)[-5:]:  # Last 5 frames
                for past_hazard in past_frame:
                    if self._hazards_match(hazard, past_hazard):
                        match_count += 1
                        break
            
            # Keep hazards that appear consistently OR are critical
            if match_count >= 2 or hazard.severity == HazardSeverity.CRITICAL:
                filtered.append(hazard)
        
        return filtered

    def _hazards_match(
        self,
        h1: DetectedHazard,
        h2: DetectedHazard,
        distance_threshold: float = 50.0
    ) -> bool:
        """Check if two hazards are likely the same object"""
        # Same type
        if h1.type != h2.type:
            return False
        
        # Similar position (Euclidean distance)
        y1, x1 = h1.position
        y2, x2 = h2.position
        distance = np.sqrt((y1 - y2)**2 + (x1 - x2)**2)
        
        return distance < distance_threshold

    def _get_next_id(self) -> int:
        """Get next unique hazard ID"""
        self.hazard_id_counter += 1
        return self.hazard_id_counter

    def visualize_hazards(
        self,
        image: np.ndarray,
        hazards: List[DetectedHazard],
        show_labels: bool = True
    ) -> np.ndarray:
        """
        Create visualization with hazard overlays
        
        Args:
            image: Original RGB image
            hazards: List of detected hazards
            show_labels: Whether to show text labels
            
        Returns:
            Annotated image
        """
        result = image.copy()
        
        # Severity colors
        severity_colors = {
            HazardSeverity.LOW: (255, 255, 0),      # Yellow
            HazardSeverity.MEDIUM: (255, 165, 0),   # Orange
            HazardSeverity.HIGH: (255, 69, 0),      # Red-orange
            HazardSeverity.CRITICAL: (255, 0, 0)    # Red
        }
        
        for hazard in hazards:
            color = severity_colors[hazard.severity]
            
            # Draw bounding box
            x, y, w, h = hazard.bounding_box
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
            
            # Draw centroid
            cy, cx = hazard.position
            cv2.circle(result, (cx, cy), 5, color, -1)
            
            # Draw label
            if show_labels:
                label = f"{hazard.type.value[:4]} {hazard.severity.name}"
                if hazard.distance > 0:
                    label += f" {hazard.distance:.1f}m"
                
                # Background for text
                (text_w, text_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    result,
                    (x, y - text_h - 5),
                    (x + text_w, y),
                    color,
                    -1
                )
                
                cv2.putText(
                    result,
                    label,
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )
        
        return result

    def get_hazard_summary(
        self,
        hazards: List[DetectedHazard]
    ) -> Dict:
        """Generate summary statistics for detected hazards"""
        if not hazards:
            return {
                'total_count': 0,
                'by_type': {},
                'by_severity': {},
                'closest_hazard': None,
                'most_severe': None
            }
        
        # Count by type
        by_type = {}
        for hazard in hazards:
            type_name = hazard.type.value
            by_type[type_name] = by_type.get(type_name, 0) + 1
        
        # Count by severity
        by_severity = {}
        for hazard in hazards:
            sev_name = hazard.severity.name
            by_severity[sev_name] = by_severity.get(sev_name, 0) + 1
        
        # Find closest and most severe
        closest = min(hazards, key=lambda h: h.distance)
        most_severe = max(hazards, key=lambda h: h.severity.value)
        
        return {
            'total_count': len(hazards),
            'by_type': by_type,
            'by_severity': by_severity,
            'closest_hazard': {
                'type': closest.type.value,
                'distance': closest.distance,
                'severity': closest.severity.name
            },
            'most_severe': {
                'type': most_severe.type.value,
                'severity': most_severe.severity.name,
                'position': most_severe.position
            }
        }


# Testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    detector = HazardDetector(image_height=512, image_width=512)
    
    # Create test data
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    test_segmentation = np.random.randint(0, 5, (512, 512), dtype=np.uint8)
    test_confidence = np.random.rand(512, 512).astype(np.float32)
    
    # Run detection
    import time
    start = time.time()
    hazards = detector.detect(test_image, test_segmentation, test_confidence)
    end = time.time()
    
    logger.info(f"Detection time: {(end-start)*1000:.1f}ms")
    logger.info(f"Detected {len(hazards)} hazards")
    
    # Print summary
    summary = detector.get_hazard_summary(hazards)
    logger.info(f"Summary: {summary}")
    
    # Visualize
    viz = detector.visualize_hazards(test_image, hazards)
    logger.info(f"Visualization created: {viz.shape}")