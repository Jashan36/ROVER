"""
Image Subscriber
Handles camera stream from ROS2 with efficient buffering

Note: ROS2 imports are guarded to allow import in non-ROS environments.
"""

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge
    _ROS_AVAILABLE = True
except Exception:
    # Provide minimal fallbacks so module import succeeds during tests
    rclpy = None
    _ROS_AVAILABLE = False

    class Node:  # type: ignore
        pass

    class Image:  # type: ignore
        pass

    class CvBridge:  # type: ignore
        def imgmsg_to_cv2(self, *args, **kwargs):
            raise RuntimeError("CvBridge unavailable: ROS2 not installed")
import numpy as np
from collections import deque
from typing import Optional, Callable
import threading
import logging
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class ImageBuffer:
    """Thread-safe image buffer"""
    image: np.ndarray
    timestamp: float
    frame_id: str
    seq: int


class ImageSubscriber(Node):
    """
    ROS2 subscriber for camera images with efficient buffering
    """
    
    def __init__(
        self,
        node_name: str = 'image_subscriber',
        topic: str = '/camera/image_raw',
        buffer_size: int = 5,
        qos_depth: int = 10
    ):
        """
        Initialize image subscriber
        
        Args:
            node_name: ROS2 node name
            topic: Camera topic to subscribe to
            buffer_size: Maximum buffer size
            qos_depth: QoS depth for subscriber
        """
        super().__init__(node_name)
        
        self.topic = topic
        self.buffer_size = buffer_size
        
        # CV Bridge for ROS <-> OpenCV conversion
        self.bridge = CvBridge()
        
        # Thread-safe buffer
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_lock = threading.Lock()
        
        # Statistics
        self.frame_count = 0
        self.last_frame_time = time.time()
        self.fps = 0.0
        
        # Callback
        self.callback = None
        
        # Create subscription
        self.subscription = self.create_subscription(
            Image,
            topic,
            self.image_callback,
            qos_depth
        )
        
        self.get_logger().info(f"ImageSubscriber initialized on {topic}")

    def image_callback(self, msg: Image):
        """
        Callback for incoming images
        
        Args:
            msg: ROS Image message
        """
        try:
            # Convert ROS Image to OpenCV format (BGR)
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Convert BGR to RGB
            rgb_image = cv_image[:, :, ::-1]
            
            # Create buffer entry
            buffer_entry = ImageBuffer(
                image=rgb_image,
                timestamp=msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                frame_id=msg.header.frame_id,
                seq=self.frame_count
            )
            
            # Add to buffer
            with self.buffer_lock:
                self.buffer.append(buffer_entry)
            
            # Update statistics
            self.frame_count += 1
            current_time = time.time()
            elapsed = current_time - self.last_frame_time
            if elapsed > 0:
                self.fps = 1.0 / elapsed
            self.last_frame_time = current_time
            
            # Call user callback if set
            if self.callback:
                self.callback(buffer_entry)
                
        except Exception as e:
            self.get_logger().error(f"Failed to process image: {e}")

    def get_latest_image(self) -> Optional[ImageBuffer]:
        """
        Get most recent image from buffer
        
        Returns:
            Latest ImageBuffer or None if buffer is empty
        """
        with self.buffer_lock:
            if self.buffer:
                return self.buffer[-1]
            return None

    def get_all_images(self) -> list:
        """
        Get all images in buffer
        
        Returns:
            List of ImageBuffer objects
        """
        with self.buffer_lock:
            return list(self.buffer)

    def clear_buffer(self):
        """Clear image buffer"""
        with self.buffer_lock:
            self.buffer.clear()
        self.get_logger().info("Image buffer cleared")

    def set_callback(self, callback: Callable[[ImageBuffer], None]):
        """
        Set callback function for new images
        
        Args:
            callback: Function to call with new ImageBuffer
        """
        self.callback = callback

    def get_statistics(self) -> dict:
        """Get subscriber statistics"""
        with self.buffer_lock:
            buffer_size = len(self.buffer)
        
        return {
            'topic': self.topic,
            'frame_count': self.frame_count,
            'fps': self.fps,
            'buffer_size': buffer_size,
            'buffer_max': self.buffer_size
        }


# Standalone test
def main(args=None):
    rclpy.init(args=args)
    
    subscriber = ImageSubscriber()
    
    def print_stats(buffer: ImageBuffer):
        stats = subscriber.get_statistics()
        print(f"Frame {stats['frame_count']}: {stats['fps']:.1f} FPS, "
              f"Buffer: {stats['buffer_size']}/{stats['buffer_max']}, "
              f"Shape: {buffer.image.shape}")
    
    subscriber.set_callback(print_stats)
    
    try:
        rclpy.spin(subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        subscriber.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
