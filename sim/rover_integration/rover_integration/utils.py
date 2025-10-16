#!/usr/bin/env python3

"""
Utility functions for ROS2-AI integration
Image conversion, message formatting, and helper functions
"""

import rclpy
from rclpy.node import Node
import numpy as np
import yaml
import os
from cv_bridge import CvBridge
import cv2

from sensor_msgs.msg import Image
from nav_msgs.msg import OccupancyGrid, MapMetaData
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3
from std_msgs.msg import Header, ColorRGBA


# Global CV bridge for image conversion
cv_bridge = CvBridge()


def image_to_cv2(ros_image):
    """
    Convert ROS Image message to OpenCV format
    """
    try:
        cv_image = cv_bridge.imgmsg_to_cv2(ros_image, desired_encoding='bgr8')
        return cv_image
    except Exception as e:
        raise ValueError(f"Error converting ROS image to CV2: {e}")


def cv2_to_image(cv_image, timestamp=None):
    """
    Convert OpenCV image to ROS Image message
    """
    try:
        ros_image = cv_bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
        if timestamp:
            ros_image.header.stamp = timestamp
        return ros_image
    except Exception as e:
        raise ValueError(f"Error converting CV2 to ROS image: {e}")


def create_costmap(traversability_map, config):
    """
    Create Nav2 costmap from traversability analysis
    """
    costmap = OccupancyGrid()
    
    # Set header
    costmap.header = Header()
    costmap.header.stamp = Node().get_clock().now().to_msg()
    costmap.header.frame_id = 'base_link'
    
    # Set map metadata
    resolution = config.get('costmap_resolution', 0.1)
    width = traversability_map.shape[1]
    height = traversability_map.shape[0]
    
    costmap.info = MapMetaData()
    costmap.info.resolution = resolution
    costmap.info.width = width
    costmap.info.height = height
    costmap.info.origin.position.x = -width * resolution / 2
    costmap.info.origin.position.y = -height * resolution / 2
    costmap.info.origin.position.z = 0.0
    costmap.info.origin.orientation.w = 1.0
    
    # Convert traversability to cost (0-100)
    # Low traversability = high cost, high traversability = low cost
    cost_data = (100 * (1 - traversability_map)).flatten().astype(np.int8)
    costmap.data = cost_data.tolist()
    
    return costmap


def create_hazard_markers(hazards, timestamp):
    """
    Create visualization markers for hazards
    """
    marker_array = MarkerArray()
    
    for i, hazard in enumerate(hazards):
        marker = Marker()
        
        marker.header.stamp = timestamp
        marker.header.frame_id = 'base_link'
        marker.ns = 'hazards'
        marker.id = i
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        
        # Position
        marker.pose.position.x = float(hazard['position'][0])
        marker.pose.position.y = float(hazard['position'][1])
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        
        # Scale (radius, height)
        radius = hazard.get('radius', 1.0)
        marker.scale.x = radius * 2
        marker.scale.y = radius * 2
        marker.scale.z = 0.2  # Flat cylinder
        
        # Color based on severity
        severity = hazard.get('severity', 'medium')
        color_map = {
            'low': ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.5),      # Green
            'medium': ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.5),   # Yellow
            'high': ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.5)      # Red
        }
        marker.color = color_map.get(severity, color_map['medium'])
        
        # Text marker for hazard type
        text_marker = Marker()
        text_marker.header = marker.header
        text_marker.ns = 'hazard_labels'
        text_marker.id = i
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        
        text_marker.pose.position.x = marker.pose.position.x
        text_marker.pose.position.y = marker.pose.position.y
        text_marker.pose.position.z = 0.5
        text_marker.pose.orientation.w = 1.0
        
        text_marker.scale.z = 0.3  # Text height
        text_marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        text_marker.text = f"{hazard['type']}\n({severity})"
        
        marker_array.markers.append(marker)
        marker_array.markers.append(text_marker)
    
    return marker_array


def load_model_config():
    """
    Load AI model configuration
    """
    config_path = os.path.join(
        os.path.dirname(__file__),
        'config',
        'model_config.yaml'
    )
    
    default_config = {
        'model_path': 'ai/models/weights/terrain_unet_best.pth',
        'input_size': [512, 512],
        'num_classes': 4,
        'class_names': ['soil', 'bedrock', 'sand', 'big_rock'],
        'class_colors': [
            [100, 100, 50],    # soil - brown
            [150, 150, 150],   # bedrock - gray
            [255, 255, 100],   # sand - yellow
            [50, 50, 50]       # big_rock - dark gray
        ]
    }
    
    try:
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f)
            default_config.update(user_config)
    except FileNotFoundError:
        print(f"⚠️ Model config not found at {config_path}, using defaults")
    except Exception as e:
        print(f"❌ Error loading model config: {e}")
    
    return default_config


def calculate_traversability_score(segmentation, confidence, class_weights):
    """
    Calculate overall traversability score from segmentation
    """
    # Class weights: higher = more traversable
    # soil: 0.9, bedrock: 0.7, sand: 0.3, big_rock: 0.1
    
    total_pixels = segmentation.size
    weighted_sum = 0.0
    
    for class_id, weight in class_weights.items():
        class_mask = segmentation == class_id
        class_pixels = np.sum(class_mask)
        
        # Weight by confidence
        class_confidence = np.mean(confidence[class_mask]) if class_pixels > 0 else 0.5
        confidence_weight = 0.5 + 0.5 * class_confidence
        
        weighted_sum += class_pixels * weight * confidence_weight
    
    return weighted_sum / total_pixels


def normalize_image(image):
    """
    Normalize image for AI model input
    """
    image_float = image.astype(np.float32) / 255.0
    
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    normalized = (image_float - mean) / std
    return normalized


def denormalize_image(normalized_image):
    """
    Denormalize image for display
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    denormalized = normalized_image * std + mean
    denormalized = np.clip(denormalized * 255, 0, 255).astype(np.uint8)
    return denormalized


def create_status_message(node_name, status, details=None):
    """
    Create standardized status message
    """
    status_data = {
        'node': node_name,
        'status': status,
        'timestamp': time.time(),
        'details': details or {}
    }
    return str(status_data)


def get_timestamp():
    """
    Get current timestamp in ROS2 format
    """
    return Node().get_clock().now().to_msg()