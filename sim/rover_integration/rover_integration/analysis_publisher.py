"""
Analysis Publisher
Publishes AI analysis results to ROS2 topics

Note: ROS2 imports are guarded to allow import in non-ROS environments.
"""

import numpy as np
import json
from typing import List, Optional
from dataclasses import dataclass
import logging

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from nav_msgs.msg import OccupancyGrid
    from std_msgs.msg import String
    from visualization_msgs.msg import Marker, MarkerArray
    from geometry_msgs.msg import Point
    from cv_bridge import CvBridge
    _ROS_AVAILABLE = True
except Exception:
    rclpy = None
    _ROS_AVAILABLE = False

    class Node:  # type: ignore
        pass

    class Image:  # type: ignore
        pass

    class OccupancyGrid:  # type: ignore
        pass

    class String:  # type: ignore
        pass

    class Marker:  # type: ignore
        pass

    class MarkerArray:  # type: ignore
        pass

    class Point:  # type: ignore
        pass

    class CvBridge:  # type: ignore
        def cv2_to_imgmsg(self, *args, **kwargs):
            raise RuntimeError("CvBridge unavailable: ROS2 not installed")

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from ai.inference.real_time_pipeline import InferenceResult
from ai.models.hazard_detector import DetectedHazard

logger = logging.getLogger(__name__)


@dataclass
class PublisherConfig:
    """Configuration for publishers"""
    publish_segmentation: bool = True
    publish_overlay: bool = True
    publish_traversability: bool = True
    publish_hazards: bool = True
    publish_costmap: bool = True
    publish_markers: bool = True
    
    # Topic names
    segmentation_topic: str = '/terrain/segmentation'
    overlay_topic: str = '/terrain/overlay'
    traversability_topic: str = '/terrain/traversability_map'
    hazards_topic: str = '/terrain/hazards'
    costmap_topic: str = '/terrain/costmap'
    markers_topic: str = '/terrain/hazard_markers'
    
    # QoS
    qos_depth: int = 2


class AnalysisPublisher(Node):
    """
    Publisher for terrain analysis results
    """
    
    def __init__(
        self,
        node_name: str = 'analysis_publisher',
        config: Optional[PublisherConfig] = None
    ):
        """
        Initialize analysis publisher
        
        Args:
            node_name: ROS2 node name
            config: Publisher configuration
        """
        super().__init__(node_name)
        
        self.config = config or PublisherConfig()
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Create publishers
        self._create_publishers()
        
        # Statistics
        self.publish_count = 0
        
        self.get_logger().info("AnalysisPublisher initialized")

    def _create_publishers(self):
        """Create all ROS2 publishers"""
        qos = self.config.qos_depth
        
        if self.config.publish_segmentation:
            self.segmentation_pub = self.create_publisher(
                Image,
                self.config.segmentation_topic,
                qos
            )
        
        if self.config.publish_overlay:
            self.overlay_pub = self.create_publisher(
                Image,
                self.config.overlay_topic,
                qos
            )
        
        if self.config.publish_traversability:
            self.traversability_pub = self.create_publisher(
                Image,
                self.config.traversability_topic,
                qos
            )
        
        if self.config.publish_hazards:
            self.hazards_pub = self.create_publisher(
                String,
                self.config.hazards_topic,
                10  # Higher depth for hazards
            )
        
        if self.config.publish_costmap:
            self.costmap_pub = self.create_publisher(
                OccupancyGrid,
                self.config.costmap_topic,
                qos
            )
        
        if self.config.publish_markers:
            self.markers_pub = self.create_publisher(
                MarkerArray,
                self.config.markers_topic,
                10
            )

    def publish_results(
        self,
        result: InferenceResult,
        frame_id: str = 'camera_link'
    ):
        """
        Publish complete analysis results
        
        Args:
            result: InferenceResult from AI pipeline
            frame_id: TF frame ID
        """
        try:
            timestamp = self.get_clock().now().to_msg()
            
            # Publish segmentation
            if self.config.publish_segmentation:
                self._publish_segmentation(
                    result.segmentation_colored,
                    frame_id,
                    timestamp
                )
            
            # Publish overlay
            if self.config.publish_overlay and result.overlay is not None:
                self._publish_overlay(
                    result.overlay,
                    frame_id,
                    timestamp
                )
            
            # Publish traversability
            if self.config.publish_traversability and result.traversability_viz is not None:
                self._publish_traversability(
                    result.traversability_viz,
                    frame_id,
                    timestamp
                )
            
            # Publish hazards
            if self.config.publish_hazards:
                self._publish_hazards(result.hazards, result.hazard_summary)
            
            # Publish costmap
            if self.config.publish_costmap and result.costmap is not None:
                self._publish_costmap(
                    result.costmap,
                    frame_id,
                    timestamp
                )
            
            # Publish markers
            if self.config.publish_markers and result.hazards:
                self._publish_markers(
                    result.hazards,
                    frame_id,
                    timestamp
                )
            
            self.publish_count += 1
            
        except Exception as e:
            self.get_logger().error(f"Failed to publish results: {e}")

    def _publish_segmentation(
        self,
        segmentation: np.ndarray,
        frame_id: str,
        timestamp
    ):
        """Publish segmentation image"""
        try:
            # Convert RGB to BGR for ROS
            bgr_image = segmentation[:, :, ::-1]
            
            msg = self.bridge.cv2_to_imgmsg(bgr_image, encoding='bgr8')
            msg.header.frame_id = frame_id
            msg.header.stamp = timestamp
            
            self.segmentation_pub.publish(msg)
            
        except Exception as e:
            self.get_logger().error(f"Failed to publish segmentation: {e}")

    def _publish_overlay(
        self,
        overlay: np.ndarray,
        frame_id: str,
        timestamp
    ):
        """Publish overlay image"""
        try:
            bgr_image = overlay[:, :, ::-1]
            
            msg = self.bridge.cv2_to_imgmsg(bgr_image, encoding='bgr8')
            msg.header.frame_id = frame_id
            msg.header.stamp = timestamp
            
            self.overlay_pub.publish(msg)
            
        except Exception as e:
            self.get_logger().error(f"Failed to publish overlay: {e}")

    def _publish_traversability(
        self,
        traversability_viz: np.ndarray,
        frame_id: str,
        timestamp
    ):
        """Publish traversability visualization"""
        try:
            bgr_image = traversability_viz[:, :, ::-1]
            
            msg = self.bridge.cv2_to_imgmsg(bgr_image, encoding='bgr8')
            msg.header.frame_id = frame_id
            msg.header.stamp = timestamp
            
            self.traversability_pub.publish(msg)
            
        except Exception as e:
            self.get_logger().error(f"Failed to publish traversability: {e}")

    def _publish_hazards(
        self,
        hazards: List[DetectedHazard],
        summary: dict
    ):
        """Publish hazard information as JSON"""
        try:
            # Convert hazards to serializable format
            hazards_data = []
            for hazard in hazards:
                hazard_dict = {
                    'id': hazard.hazard_id,
                    'type': hazard.type.value,
                    'severity': hazard.severity.name,
                    'position': {
                        'y': int(hazard.position[0]),
                        'x': int(hazard.position[1])
                    },
                    'distance': float(hazard.distance),
                    'bearing': float(hazard.bearing),
                    'area': int(hazard.area),
                    'confidence': float(hazard.confidence),
                    'description': hazard.description
                }
                hazards_data.append(hazard_dict)
            
            # Create JSON message
            data = {
                'hazards': hazards_data,
                'summary': summary,
                'count': len(hazards)
            }
            
            msg = String()
            msg.data = json.dumps(data)
            
            self.hazards_pub.publish(msg)
            
        except Exception as e:
            self.get_logger().error(f"Failed to publish hazards: {e}")

    def _publish_costmap(
        self,
        costmap: np.ndarray,
        frame_id: str,
        timestamp
    ):
        """Publish costmap as OccupancyGrid"""
        try:
            msg = OccupancyGrid()
            msg.header.frame_id = frame_id
            msg.header.stamp = timestamp
            
            # Costmap info
            msg.info.resolution = 0.05  # 5cm per cell
            msg.info.width = costmap.shape[1]
            msg.info.height = costmap.shape[0]
            
            # Origin at center-bottom
            msg.info.origin.position.x = -(msg.info.width * msg.info.resolution) / 2.0
            msg.info.origin.position.y = 0.0
            msg.info.origin.position.z = 0.0
            msg.info.origin.orientation.w = 1.0
            
            # Convert costmap to OccupancyGrid format
            # OccupancyGrid: 0=free, 100=occupied, -1=unknown
            # Our costmap: 0=free, 100=occupied, 255=unknown
            grid_data = costmap.flatten().astype(np.int8)
            grid_data[grid_data == 255] = -1  # Unknown
            
            msg.data = grid_data.tolist()
            
            self.costmap_pub.publish(msg)
            
        except Exception as e:
            self.get_logger().error(f"Failed to publish costmap: {e}")

    def _publish_markers(
        self,
        hazards: List[DetectedHazard],
        frame_id: str,
        timestamp
    ):
        """Publish hazard markers for RViz visualization"""
        try:
            marker_array = MarkerArray()
            
            for i, hazard in enumerate(hazards):
                # Create marker
                marker = Marker()
                marker.header.frame_id = frame_id
                marker.header.stamp = timestamp
                marker.ns = "hazards"
                marker.id = i
                marker.type = Marker.CYLINDER
                marker.action = Marker.ADD
                
                # Position (convert from image coordinates to world)
                # This is a simplified conversion - adjust based on camera calibration
                y, x = hazard.position
                marker.pose.position.x = hazard.distance
                marker.pose.position.y = hazard.distance * np.tan(hazard.bearing)
                marker.pose.position.z = 0.0
                
                marker.pose.orientation.w = 1.0
                
                # Scale based on severity
                scale = 0.2 + (hazard.severity.value * 0.1)
                marker.scale.x = scale
                marker.scale.y = scale
                marker.scale.z = 0.5
                
                # Color based on severity
                severity_colors = {
                    1: (1.0, 1.0, 0.0),    # Low: Yellow
                    2: (1.0, 0.65, 0.0),   # Medium: Orange
                    3: (1.0, 0.27, 0.0),   # High: Red-orange
                    4: (1.0, 0.0, 0.0)     # Critical: Red
                }
                
                r, g, b = severity_colors.get(hazard.severity.value, (1.0, 1.0, 1.0))
                marker.color.r = r
                marker.color.g = g
                marker.color.b = b
                marker.color.a = 0.8
                
                marker.lifetime.sec = 1  # Markers last 1 second
                
                marker_array.markers.append(marker)
                
                # Add text label
                text_marker = Marker()
                text_marker.header = marker.header
                text_marker.ns = "hazard_labels"
                text_marker.id = i + 1000
                text_marker.type = Marker.TEXT_VIEW_FACING
                text_marker.action = Marker.ADD
                
                text_marker.pose.position.x = marker.pose.position.x
                text_marker.pose.position.y = marker.pose.position.y
                text_marker.pose.position.z = 0.5
                
                text_marker.scale.z = 0.2
                
                text_marker.color.r = 1.0
                text_marker.color.g = 1.0
                text_marker.color.b = 1.0
                text_marker.color.a = 1.0
                
                text_marker.text = f"{hazard.type.value}\n{hazard.distance:.1f}m"
                text_marker.lifetime.sec = 1
                
                marker_array.markers.append(text_marker)
            
            self.markers_pub.publish(marker_array)
            
        except Exception as e:
            self.get_logger().error(f"Failed to publish markers: {e}")

    def get_statistics(self) -> dict:
        """Get publisher statistics"""
        return {
            'publish_count': self.publish_count,
            'topics': {
                'segmentation': self.config.segmentation_topic if self.config.publish_segmentation else None,
                'overlay': self.config.overlay_topic if self.config.publish_overlay else None,
                'traversability': self.config.traversability_topic if self.config.publish_traversability else None,
                'hazards': self.config.hazards_topic if self.config.publish_hazards else None,
                'costmap': self.config.costmap_topic if self.config.publish_costmap else None,
                'markers': self.config.markers_topic if self.config.publish_markers else None,
            }
        }


# Standalone test
def main(args=None):
    rclpy.init(args=args)
    
    publisher = AnalysisPublisher()
    
    # Create dummy result for testing
    from ai.inference.realtime_pipeline import InferenceResult
    
    dummy_result = InferenceResult(
        timestamp=0.0,
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
        stats={}
    )
    
    publisher.publish_results(dummy_result)
    
    print("Published dummy result")
    print(f"Statistics: {publisher.get_statistics()}")
    
    publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
