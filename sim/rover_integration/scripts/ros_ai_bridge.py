#!/usr/bin/env python3
"""
ROS2 AI Bridge - Main coordination node for Mars Rover AI pipeline
Connects camera feed to AI models and publishes analysis results
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import cv2
import numpy as np
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Header, Float32MultiArray, String
from geometry_msgs.msg import PoseStamped, Twist
import json
import time
import threading
from typing import Dict, Any, Optional

# Import AI modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from ai.models.terrain_segmentation import TerrainSegmentation
from ai.models.hazard_detector import HazardDetector, DetectedHazard
from ai.models.traversability import score_traversability


class MarsRoverAIBridge(Node):
    """
    Main ROS2 node that bridges camera data to AI inference pipeline
    """
    
    def __init__(self):
        super().__init__('mars_rover_ai_bridge')
        
        # Configuration
        self.image_height = 512
        self.image_width = 512
        self.inference_rate = 5.0  # Hz
        self.camera_topic = '/camera/image_raw'
        self.model_weights_path = 'ai/models/weights/latest.pth'
        
        # QoS profile for image data
        self.image_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Initialize AI models
        self._init_ai_models()
        
        # ROS2 Publishers
        self._init_publishers()
        
        # ROS2 Subscribers  
        self._init_subscribers()
        
        # Processing state
        self.latest_image = None
        self.processing_lock = threading.Lock()
        self.last_inference_time = 0.0
        
        # Create inference timer
        self.inference_timer = self.create_timer(
            1.0 / self.inference_rate, 
            self._inference_callback
        )
        
        self.get_logger().info('Mars Rover AI Bridge initialized')
        self.get_logger().info(f'Inference rate: {self.inference_rate} Hz')
        self.get_logger().info(f'Image size: {self.image_width}x{self.image_height}')
        
    def _init_ai_models(self):
        """Initialize AI models"""
        try:
            # Terrain segmentation model
            self.terrain_model = TerrainSegmentation(
                weights_path=self.model_weights_path,
                device='cpu'  # Use GPU if available: 'cuda'
            )
            self.get_logger().info('Terrain segmentation model loaded')
            
            # Hazard detection model
            self.hazard_detector = HazardDetector(
                image_height=self.image_height,
                image_width=self.image_width,
                camera_fov=1.396,  # 80 degrees
                camera_height=0.4,  # meters
                camera_tilt=0.1     # radians
            )
            self.get_logger().info('Hazard detector initialized')
            
        except Exception as e:
            self.get_logger().error(f'Failed to initialize AI models: {e}')
            self.terrain_model = None
            self.hazard_detector = None
    
    def _init_publishers(self):
        """Initialize ROS2 publishers"""
        # Terrain segmentation results
        self.segmentation_pub = self.create_publisher(
            Image,
            '/rover/ai/terrain_segmentation',
            qos_profile=self.image_qos
        )
        
        # Hazard detection results
        self.hazards_pub = self.create_publisher(
            String,
            '/rover/ai/hazards',
            qos_profile=10
        )
        
        # Traversability scores
        self.traversability_pub = self.create_publisher(
            Float32MultiArray,
            '/rover/ai/traversability',
            qos_profile=10
        )
        
        # AI analysis summary
        self.analysis_pub = self.create_publisher(
            String,
            '/rover/ai/analysis',
            qos_profile=10
        )
        
        # Visualization images
        self.viz_pub = self.create_publisher(
            Image,
            '/rover/ai/visualization',
            qos_profile=self.image_qos
        )
        
    def _init_subscribers(self):
        """Initialize ROS2 subscribers"""
        # Camera image subscriber
        self.image_sub = self.create_subscription(
            Image,
            self.camera_topic,
            self._image_callback,
            qos_profile=self.image_qos
        )
        
        # Compressed image subscriber (alternative)
        self.compressed_image_sub = self.create_subscription(
            CompressedImage,
            self.camera_topic + '/compressed',
            self._compressed_image_callback,
            qos_profile=self.image_qos
        )
    
    def _image_callback(self, msg: Image):
        """Handle incoming camera images"""
        try:
            # Convert ROS Image to OpenCV format
            cv_image = self._ros_image_to_cv2(msg)
            
            # Resize if needed
            if cv_image.shape[:2] != (self.image_height, self.image_width):
                cv_image = cv2.resize(cv_image, (self.image_width, self.image_height))
            
            with self.processing_lock:
                self.latest_image = cv_image
                
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')
    
    def _compressed_image_callback(self, msg: CompressedImage):
        """Handle compressed camera images"""
        try:
            # Convert compressed image to OpenCV format
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            # Resize if needed
            if cv_image.shape[:2] != (self.image_height, self.image_width):
                cv_image = cv2.resize(cv_image, (self.image_width, self.image_height))
            
            with self.processing_lock:
                self.latest_image = cv_image
                
        except Exception as e:
            self.get_logger().error(f'Error processing compressed image: {e}')
    
    def _inference_callback(self):
        """Main inference callback - runs at specified rate"""
        if self.terrain_model is None or self.hazard_detector is None:
            return
            
        with self.processing_lock:
            if self.latest_image is None:
                return
            current_image = self.latest_image.copy()
        
        try:
            # Run AI inference pipeline
            start_time = time.time()
            
            # 1. Terrain Segmentation
            segmentation_result = self._run_terrain_segmentation(current_image)
            
            # 2. Hazard Detection
            hazards = self._run_hazard_detection(current_image, segmentation_result)
            
            # 3. Traversability Analysis
            traversability_score = self._calculate_traversability(segmentation_result)
            
            # 4. Publish results
            self._publish_results(current_image, segmentation_result, hazards, traversability_score)
            
            # 5. Performance logging
            inference_time = (time.time() - start_time) * 1000  # ms
            self.get_logger().debug(f'Inference time: {inference_time:.1f}ms')
            
            # Update timing
            self.last_inference_time = time.time()
            
        except Exception as e:
            self.get_logger().error(f'Inference error: {e}')
    
    def _run_terrain_segmentation(self, image: np.ndarray) -> Dict[str, Any]:
        """Run terrain segmentation on the image"""
        try:
            # Convert to tensor format
            import torch
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
            
            # Run inference
            with torch.no_grad():
                segmentation_output = self.terrain_model.predict(image_tensor)
                segmentation_mask = torch.sigmoid(segmentation_output).squeeze().cpu().numpy()
                
                # Convert to class labels (0-4: soil, bedrock, sand, rock, big rock)
                class_labels = (segmentation_mask * 4).astype(np.uint8)
                
                # Calculate confidence
                confidence = np.abs(segmentation_mask - 0.5) * 2  # Distance from decision boundary
            
            return {
                'segmentation': class_labels,
                'confidence': confidence,
                'raw_output': segmentation_output.cpu().numpy()
            }
            
        except Exception as e:
            self.get_logger().error(f'Segmentation error: {e}')
            # Return dummy data
            return {
                'segmentation': np.zeros((self.image_height, self.image_width), dtype=np.uint8),
                'confidence': np.zeros((self.image_height, self.image_width), dtype=np.float32),
                'raw_output': None
            }
    
    def _run_hazard_detection(self, image: np.ndarray, segmentation_result: Dict) -> list:
        """Run hazard detection on the image and segmentation"""
        try:
            hazards = self.hazard_detector.detect(
                image=image,
                segmentation=segmentation_result['segmentation'],
                confidence=segmentation_result['confidence']
            )
            
            return hazards
            
        except Exception as e:
            self.get_logger().error(f'Hazard detection error: {e}')
            return []
    
    def _calculate_traversability(self, segmentation_result: Dict) -> float:
        """Calculate traversability score"""
        try:
            # Simple traversability based on terrain classes
            seg_mask = segmentation_result['segmentation']
            
            # Penalize obstacles (class 3, 4: rocks)
            obstacle_mask = (seg_mask >= 3).astype(np.float32)
            obstacle_penalty = np.mean(obstacle_mask) * 50.0
            
            # Base score
            base_score = 100.0
            
            # Calculate final score
            traversability = max(0.0, base_score - obstacle_penalty)
            
            return float(traversability)
            
        except Exception as e:
            self.get_logger().error(f'Traversability calculation error: {e}')
            return 50.0  # Default moderate score
    
    def _publish_results(self, image: np.ndarray, segmentation_result: Dict, 
                        hazards: list, traversability_score: float):
        """Publish all analysis results"""
        
        # 1. Publish segmentation mask as Image
        self._publish_segmentation(segmentation_result['segmentation'])
        
        # 2. Publish hazards as JSON
        self._publish_hazards(hazards)
        
        # 3. Publish traversability score
        self._publish_traversability(traversability_score)
        
        # 4. Create and publish visualization
        viz_image = self._create_visualization(image, segmentation_result, hazards)
        self._publish_visualization(viz_image)
        
        # 5. Publish analysis summary
        self._publish_analysis_summary(hazards, traversability_score)
    
    def _publish_segmentation(self, segmentation: np.ndarray):
        """Publish segmentation mask"""
        try:
            # Convert segmentation to color image for visualization
            color_seg = self._segmentation_to_color(segmentation)
            
            # Convert to ROS Image message
            msg = self._cv2_to_ros_image(color_seg)
            self.segmentation_pub.publish(msg)
            
        except Exception as e:
            self.get_logger().error(f'Error publishing segmentation: {e}')
    
    def _publish_hazards(self, hazards: list):
        """Publish hazard detection results"""
        try:
            # Convert hazards to JSON-serializable format
            hazards_data = []
            for hazard in hazards:
                hazard_dict = {
                    'id': hazard.hazard_id,
                    'type': hazard.type.value,
                    'severity': hazard.severity.name,
                    'position': hazard.position,
                    'bounding_box': hazard.bounding_box,
                    'area': hazard.area,
                    'confidence': hazard.confidence,
                    'distance': hazard.distance,
                    'bearing': hazard.bearing,
                    'description': hazard.description
                }
                hazards_data.append(hazard_dict)
            
            # Create and publish message
            msg = String()
            msg.data = json.dumps(hazards_data)
            self.hazards_pub.publish(msg)
            
        except Exception as e:
            self.get_logger().error(f'Error publishing hazards: {e}')
    
    def _publish_traversability(self, score: float):
        """Publish traversability score"""
        try:
            msg = Float32MultiArray()
            msg.data = [score]
            self.traversability_pub.publish(msg)
            
        except Exception as e:
            self.get_logger().error(f'Error publishing traversability: {e}')
    
    def _publish_visualization(self, viz_image: np.ndarray):
        """Publish visualization image"""
        try:
            msg = self._cv2_to_ros_image(viz_image)
            self.viz_pub.publish(msg)
            
        except Exception as e:
            self.get_logger().error(f'Error publishing visualization: {e}')
    
    def _publish_analysis_summary(self, hazards: list, traversability_score: float):
        """Publish analysis summary"""
        try:
            summary = {
                'timestamp': time.time(),
                'hazard_count': len(hazards),
                'traversability_score': traversability_score,
                'critical_hazards': len([h for h in hazards if h.severity.value >= 3]),
                'closest_hazard_distance': min([h.distance for h in hazards], default=float('inf')),
                'processing_rate': self.inference_rate
            }
            
            msg = String()
            msg.data = json.dumps(summary)
            self.analysis_pub.publish(msg)
            
        except Exception as e:
            self.get_logger().error(f'Error publishing analysis summary: {e}')
    
    def _create_visualization(self, image: np.ndarray, segmentation_result: Dict, 
                            hazards: list) -> np.ndarray:
        """Create comprehensive visualization"""
        try:
            # Start with original image
            viz = image.copy()
            
            # Overlay segmentation (semi-transparent)
            seg_color = self._segmentation_to_color(segmentation_result['segmentation'])
            viz = cv2.addWeighted(viz, 0.7, seg_color, 0.3, 0)
            
            # Add hazard overlays
            if self.hazard_detector:
                viz = self.hazard_detector.visualize_hazards(viz, hazards, show_labels=True)
            
            # Add text overlays
            viz = self._add_text_overlays(viz, segmentation_result, hazards)
            
            return viz
            
        except Exception as e:
            self.get_logger().error(f'Error creating visualization: {e}')
            return image
    
    def _add_text_overlays(self, image: np.ndarray, segmentation_result: Dict, 
                          hazards: list) -> np.ndarray:
        """Add text overlays to visualization"""
        try:
            # Add frame info
            cv2.putText(image, f"Mars Rover AI - {time.strftime('%H:%M:%S')}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add hazard count
            cv2.putText(image, f"Hazards: {len(hazards)}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add traversability score
            traversability = self._calculate_traversability(segmentation_result)
            color = (0, 255, 0) if traversability > 70 else (0, 255, 255) if traversability > 40 else (0, 0, 255)
            cv2.putText(image, f"Traversability: {traversability:.1f}%", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            return image
            
        except Exception as e:
            self.get_logger().error(f'Error adding text overlays: {e}')
            return image
    
    def _segmentation_to_color(self, segmentation: np.ndarray) -> np.ndarray:
        """Convert segmentation mask to color image"""
        # Define colors for each terrain class
        colors = np.array([
            [139, 69, 19],    # 0: Soil - Brown
            [105, 105, 105],  # 1: Bedrock - Gray  
            [238, 203, 173],  # 2: Sand - Beige
            [101, 67, 33],    # 3: Rock - Dark Brown
            [64, 64, 64]      # 4: Big Rock - Dark Gray
        ], dtype=np.uint8)
        
        # Create color image
        color_seg = colors[segmentation]
        return color_seg
    
    def _ros_image_to_cv2(self, ros_image: Image) -> np.ndarray:
        """Convert ROS Image message to OpenCV format"""
        if ros_image.encoding == 'rgb8':
            return np.frombuffer(ros_image.data, dtype=np.uint8).reshape(
                (ros_image.height, ros_image.width, 3)
            )
        elif ros_image.encoding == 'bgr8':
            image = np.frombuffer(ros_image.data, dtype=np.uint8).reshape(
                (ros_image.height, ros_image.width, 3)
            )
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Unsupported image encoding: {ros_image.encoding}")
    
    def _cv2_to_ros_image(self, cv_image: np.ndarray) -> Image:
        """Convert OpenCV image to ROS Image message"""
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_link'
        msg.height = cv_image.shape[0]
        msg.width = cv_image.shape[1]
        msg.encoding = 'rgb8'
        msg.is_bigendian = False
        msg.step = msg.width * 3
        msg.data = cv_image.tobytes()
        return msg


def main(args=None):
    """Main function"""
    rclpy.init(args=args)
    
    try:
        node = MarsRoverAIBridge()
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
