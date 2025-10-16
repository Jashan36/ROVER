#!/usr/bin/env python3
"""
ROS2 Image Subscriber - Dedicated camera data handler for Mars Rover
Handles camera feed processing, buffering, and forwarding to AI pipeline
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import cv2
import numpy as np
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from std_msgs.msg import Header
import time
import threading
from collections import deque
from typing import Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarsRoverImageSubscriber(Node):
    """
    Dedicated ROS2 node for handling camera image streams
    Provides buffering, preprocessing, and multiple output formats
    """
    
    def __init__(self):
        super().__init__('mars_rover_image_subscriber')
        
        # Configuration
        self.target_height = 512
        self.target_width = 512
        self.buffer_size = 10  # Number of frames to buffer
        self.camera_topic = '/camera/image_raw'
        self.camera_info_topic = '/camera/camera_info'
        
        # Image buffer and processing
        self.image_buffer = deque(maxlen=self.buffer_size)
        self.latest_image = None
        self.latest_timestamp = None
        self.processing_lock = threading.Lock()
        
        # Camera calibration data
        self.camera_info = None
        self.camera_matrix = None
        self.distortion_coeffs = None
        
        # Statistics
        self.frame_count = 0
        self.dropped_frames = 0
        self.last_stats_time = time.time()
        
        # QoS profiles
        self.image_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        self.reliable_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Initialize subscribers and publishers
        self._init_subscribers()
        self._init_publishers()
        
        # Create processing timer
        self.processing_timer = self.create_timer(0.1, self._process_buffer)  # 10 Hz
        
        # Create stats timer
        self.stats_timer = self.create_timer(5.0, self._log_statistics)  # Every 5 seconds
        
        logger.info('Mars Rover Image Subscriber initialized')
        logger.info(f'Target image size: {self.target_width}x{self.target_height}')
        logger.info(f'Buffer size: {self.buffer_size} frames')
        
    def _init_subscribers(self):
        """Initialize ROS2 subscribers"""
        # Primary camera image subscriber
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
        
        # Camera info subscriber
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self._camera_info_callback,
            qos_profile=self.reliable_qos
        )
        
        logger.info(f'Subscribed to camera topics:')
        logger.info(f'  - {self.camera_topic}')
        logger.info(f'  - {self.camera_topic}/compressed')
        logger.info(f'  - {self.camera_info_topic}')
    
    def _init_publishers(self):
        """Initialize ROS2 publishers for processed images"""
        # Processed image (resized, corrected)
        self.processed_image_pub = self.create_publisher(
            Image,
            '/rover/camera/processed',
            qos_profile=self.image_qos
        )
        
        # Undistorted image (if camera calibration available)
        self.undistorted_image_pub = self.create_publisher(
            Image,
            '/rover/camera/undistorted',
            qos_profile=self.image_qos
        )
        
        # Cropped region of interest
        self.roi_image_pub = self.create_publisher(
            Image,
            '/rover/camera/roi',
            qos_profile=self.image_qos
        )
        
        # Downsampled image for AI processing
        self.ai_ready_image_pub = self.create_publisher(
            Image,
            '/rover/camera/ai_ready',
            qos_profile=self.image_qos
        )
        
        logger.info('Initialized image publishers')
    
    def _image_callback(self, msg: Image):
        """Handle incoming camera images"""
        try:
            # Convert ROS Image to OpenCV format
            cv_image = self._ros_image_to_cv2(msg)
            
            # Process and buffer the image
            self._process_and_buffer_image(cv_image, msg.header.stamp)
            
        except Exception as e:
            logger.error(f'Error processing image: {e}')
            self.dropped_frames += 1
    
    def _compressed_image_callback(self, msg: CompressedImage):
        """Handle compressed camera images"""
        try:
            # Convert compressed image to OpenCV format
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            # Process and buffer the image
            self._process_and_buffer_image(cv_image, msg.header.stamp)
            
        except Exception as e:
            logger.error(f'Error processing compressed image: {e}')
            self.dropped_frames += 1
    
    def _camera_info_callback(self, msg: CameraInfo):
        """Handle camera calibration information"""
        try:
            self.camera_info = msg
            
            # Extract camera matrix
            if len(msg.k) == 9:
                self.camera_matrix = np.array(msg.k).reshape(3, 3)
            
            # Extract distortion coefficients
            if len(msg.d) > 0:
                self.distortion_coeffs = np.array(msg.d)
            
            logger.info('Camera calibration data received')
            logger.info(f'Camera matrix: {self.camera_matrix is not None}')
            logger.info(f'Distortion coeffs: {self.distortion_coeffs is not None}')
            
        except Exception as e:
            logger.error(f'Error processing camera info: {e}')
    
    def _process_and_buffer_image(self, image: np.ndarray, timestamp):
        """Process and buffer incoming image"""
        try:
            # Apply image corrections
            processed_image = self._apply_image_corrections(image)
            
            # Store in buffer
            with self.processing_lock:
                self.image_buffer.append({
                    'image': processed_image,
                    'timestamp': timestamp,
                    'original_size': image.shape[:2]
                })
                
                # Update latest image
                self.latest_image = processed_image
                self.latest_timestamp = timestamp
                
            self.frame_count += 1
            
        except Exception as e:
            logger.error(f'Error processing and buffering image: {e}')
            self.dropped_frames += 1
    
    def _apply_image_corrections(self, image: np.ndarray) -> np.ndarray:
        """Apply image corrections (undistortion, etc.)"""
        try:
            corrected_image = image.copy()
            
            # Apply undistortion if camera calibration is available
            if self.camera_matrix is not None and self.distortion_coeffs is not None:
                corrected_image = cv2.undistort(
                    corrected_image,
                    self.camera_matrix,
                    self.distortion_coeffs
                )
            
            return corrected_image
            
        except Exception as e:
            logger.error(f'Error applying image corrections: {e}')
            return image
    
    def _process_buffer(self):
        """Process buffered images and publish results"""
        if self.latest_image is None:
            return
            
        try:
            with self.processing_lock:
                current_image = self.latest_image.copy()
                current_timestamp = self.latest_timestamp
            
            # Create different processed versions
            processed_versions = self._create_processed_versions(current_image)
            
            # Publish all versions
            self._publish_processed_images(processed_versions, current_timestamp)
            
        except Exception as e:
            logger.error(f'Error processing buffer: {e}')
    
    def _create_processed_versions(self, image: np.ndarray) -> dict:
        """Create different processed versions of the image"""
        versions = {}
        
        try:
            # 1. Basic processed (resized)
            versions['processed'] = cv2.resize(
                image, 
                (self.target_width, self.target_height),
                interpolation=cv2.INTER_LINEAR
            )
            
            # 2. AI-ready version (optimized for AI processing)
            versions['ai_ready'] = self._prepare_ai_ready_image(versions['processed'])
            
            # 3. Region of Interest (center crop)
            versions['roi'] = self._extract_roi(versions['processed'])
            
            # 4. Undistorted version (if calibration available)
            if self.camera_matrix is not None:
                versions['undistorted'] = versions['processed']  # Already undistorted
            
            return versions
            
        except Exception as e:
            logger.error(f'Error creating processed versions: {e}')
            return {'processed': image}
    
    def _prepare_ai_ready_image(self, image: np.ndarray) -> np.ndarray:
        """Prepare image specifically for AI processing"""
        try:
            # Ensure exact dimensions for AI models
            ai_image = cv2.resize(
                image, 
                (self.target_width, self.target_height),
                interpolation=cv2.INTER_AREA  # Better for downsampling
            )
            
            # Apply slight sharpening for better feature detection
            kernel = np.array([[-1,-1,-1],
                              [-1, 9,-1],
                              [-1,-1,-1]])
            ai_image = cv2.filter2D(ai_image, -1, kernel)
            ai_image = np.clip(ai_image, 0, 255).astype(np.uint8)
            
            return ai_image
            
        except Exception as e:
            logger.error(f'Error preparing AI-ready image: {e}')
            return image
    
    def _extract_roi(self, image: np.ndarray) -> np.ndarray:
        """Extract region of interest (center crop)"""
        try:
            h, w = image.shape[:2]
            
            # Define ROI as center 70% of the image
            roi_h = int(h * 0.7)
            roi_w = int(w * 0.7)
            
            start_h = (h - roi_h) // 2
            start_w = (w - roi_w) // 2
            
            roi_image = image[start_h:start_h + roi_h, start_w:start_w + roi_w]
            
            # Resize ROI back to target size
            roi_image = cv2.resize(roi_image, (self.target_width, self.target_height))
            
            return roi_image
            
        except Exception as e:
            logger.error(f'Error extracting ROI: {e}')
            return image
    
    def _publish_processed_images(self, versions: dict, timestamp):
        """Publish all processed image versions"""
        try:
            # Publish processed image
            if 'processed' in versions:
                msg = self._cv2_to_ros_image(versions['processed'], timestamp)
                self.processed_image_pub.publish(msg)
            
            # Publish AI-ready image
            if 'ai_ready' in versions:
                msg = self._cv2_to_ros_image(versions['ai_ready'], timestamp)
                self.ai_ready_image_pub.publish(msg)
            
            # Publish ROI image
            if 'roi' in versions:
                msg = self._cv2_to_ros_image(versions['roi'], timestamp)
                self.roi_image_pub.publish(msg)
            
            # Publish undistorted image
            if 'undistorted' in versions:
                msg = self._cv2_to_ros_image(versions['undistorted'], timestamp)
                self.undistorted_image_pub.publish(msg)
                
        except Exception as e:
            logger.error(f'Error publishing processed images: {e}')
    
    def _log_statistics(self):
        """Log processing statistics"""
        current_time = time.time()
        time_diff = current_time - self.last_stats_time
        
        if time_diff > 0:
            fps = self.frame_count / time_diff
            drop_rate = self.dropped_frames / max(1, self.frame_count + self.dropped_frames)
            
            logger.info(f'Image Processing Stats:')
            logger.info(f'  FPS: {fps:.1f}')
            logger.info(f'  Total frames: {self.frame_count}')
            logger.info(f'  Dropped frames: {self.dropped_frames}')
            logger.info(f'  Drop rate: {drop_rate:.1%}')
            logger.info(f'  Buffer size: {len(self.image_buffer)}')
            
            # Reset counters
            self.frame_count = 0
            self.dropped_frames = 0
            self.last_stats_time = current_time
    
    def get_latest_image(self) -> Optional[Tuple[np.ndarray, float]]:
        """Get the latest processed image and timestamp"""
        with self.processing_lock:
            if self.latest_image is not None:
                return self.latest_image.copy(), self.latest_timestamp.sec + self.latest_timestamp.nanosec / 1e9
            return None, None
    
    def get_image_buffer(self) -> list:
        """Get a copy of the current image buffer"""
        with self.processing_lock:
            return list(self.image_buffer)
    
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
        elif ros_image.encoding == 'mono8':
            image = np.frombuffer(ros_image.data, dtype=np.uint8).reshape(
                (ros_image.height, ros_image.width)
            )
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            raise ValueError(f"Unsupported image encoding: {ros_image.encoding}")
    
    def _cv2_to_ros_image(self, cv_image: np.ndarray, timestamp=None) -> Image:
        """Convert OpenCV image to ROS Image message"""
        msg = Image()
        
        if timestamp is not None:
            msg.header.stamp = timestamp
        else:
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
        node = MarsRoverImageSubscriber()
        
        logger.info('Starting Mars Rover Image Subscriber...')
        logger.info('Press Ctrl+C to exit')
        
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        logger.info('Shutting down...')
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
