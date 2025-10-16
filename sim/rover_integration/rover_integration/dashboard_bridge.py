#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import json
import threading
import time
from queue import Queue

from sensor_msgs.msg import Image
from std_msgs.msg import String
from nav_msgs.msg import OccupancyGrid

from .utils import image_to_cv2
import cv2


class DashboardBridge(Node):
    """
    Bridges ROS2 topics to Streamlit dashboard
    Handles data collection and formatting for web display
    """
    
    def __init__(self):
        super().__init__('dashboard_bridge')
        
        # Data buffers for dashboard
        self.overlay_buffer = Queue(maxsize=2)
        self.traversability_buffer = Queue(maxsize=2)
        self.hazards_buffer = Queue(maxsize=2)
        self.status_buffer = Queue(maxsize=10)
        
        # Performance metrics
        self.metrics = {
            'fps': 0,
            'last_update': time.time(),
            'message_count': 0
        }
        
        # Initialize subscribers
        self.setup_subscribers()
        
        # Data publishing thread (simulates Streamlit updates)
        self.publish_thread = threading.Thread(target=self.publish_to_dashboard)
        self.publish_thread.daemon = True
        self.publish_thread.start()
        
        self.get_logger().info("🌐 Dashboard bridge initialized")
    
    def setup_subscribers(self):
        """Setup ROS2 subscribers for dashboard data"""
        # QoS profile for dashboard data
        best_effort_qos = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            depth=10
        )
        
        # Subscribe to AI output topics
        self.overlay_sub = self.create_subscription(
            Image,
            '/terrain/overlay',
            self.overlay_callback,
            best_effort_qos
        )
        
        self.traversability_sub = self.create_subscription(
            Image,
            '/terrain/traversability_map',
            self.traversability_callback,
            best_effort_qos
        )
        
        self.hazards_sub = self.create_subscription(
            String,
            '/terrain/hazards',
            self.hazards_callback,
            best_effort_qos
        )
        
        self.costmap_sub = self.create_subscription(
            OccupancyGrid,
            '/terrain/costmap',
            self.costmap_callback,
            best_effort_qos
        )
        
        self.get_logger().info("📥 Dashboard subscribers setup complete")
    
    def overlay_callback(self, msg):
        """Handle new overlay image data"""
        try:
            if self.overlay_buffer.full():
                self.overlay_buffer.get_nowait()
            
            # Convert ROS Image to OpenCV for processing
            cv_image = image_to_cv2(msg)
            self.overlay_buffer.put(cv_image, block=False)
            
            self.update_metrics('overlay')
            
        except Exception as e:
            self.get_logger().error(f"❌ Overlay callback error: {e}")
    
    def traversability_callback(self, msg):
        """Handle new traversability map data"""
        try:
            if self.traversability_buffer.full():
                self.traversability_buffer.get_nowait()
            
            cv_image = image_to_cv2(msg)
            self.traversability_buffer.put(cv_image, block=False)
            
            self.update_metrics('traversability')
            
        except Exception as e:
            self.get_logger().error(f"❌ Traversability callback error: {e}")
    
    def hazards_callback(self, msg):
        """Handle new hazard detection data"""
        try:
            if self.hazards_buffer.full():
                self.hazards_buffer.get_nowait()
            
            # Parse hazards data
            hazards_data = json.loads(msg.data)
            self.hazards_buffer.put(hazards_data, block=False)
            
            self.update_metrics('hazards')
            
            # Log significant hazards
            if hazards_data.get('hazards_detected', 0) > 0:
                self.get_logger().warning(
                    f"⚠️ Detected {hazards_data['hazards_detected']} hazards"
                )
                
        except Exception as e:
            self.get_logger().error(f"❌ Hazards callback error: {e}")
    
    def costmap_callback(self, msg):
        """Handle new costmap data"""
        try:
            # Process costmap for dashboard display
            costmap_data = {
                'resolution': msg.info.resolution,
                'width': msg.info.width,
                'height': msg.info.height,
                'origin': {
                    'x': msg.info.origin.position.x,
                    'y': msg.info.origin.position.y
                },
                'data': list(msg.data)  # Convert to list for JSON
            }
            
            self.update_metrics('costmap')
            
        except Exception as e:
            self.get_logger().error(f"❌ Costmap callback error: {e}")
    
    def update_metrics(self, data_type):
        """Update performance metrics"""
        current_time = time.time()
        self.metrics['message_count'] += 1
        
        # Calculate FPS every second
        if current_time - self.metrics['last_update'] >= 1.0:
            time_span = current_time - self.metrics['last_update']
            self.metrics['fps'] = self.metrics['message_count'] / time_span
            self.metrics['message_count'] = 0
            self.metrics['last_update'] = current_time
            
            # Log performance periodically
            if int(current_time) % 10 == 0:
                self.get_logger().info(
                    f"📊 Dashboard FPS: {self.metrics['fps']:.1f}, "
                    f"Buffer sizes - Overlay: {self.overlay_buffer.qsize()}, "
                    f"Traversability: {self.traversability_buffer.qsize()}"
                )
    
    def publish_to_dashboard(self):
        """
        Simulates publishing data to Streamlit dashboard
        In real implementation, this would use WebSocket or shared memory
        """
        refresh_rate = 5  # Hz (as per spec)
        
        while rclpy.ok():
            try:
                # Collect current data state
                dashboard_data = self.collect_dashboard_data()
                
                # In real implementation, send to Streamlit via:
                # - WebSocket connection
                # - Shared memory
                # - File-based communication
                # - ROS2-web bridge
                
                # For now, just log the data flow
                if int(time.time()) % 5 == 0:  # Log every 5 seconds
                    self.get_logger().info(
                        f"🌐 Dashboard update - "
                        f"Hazards: {len(dashboard_data.get('hazards', []))}, "
                        f"FPS: {self.metrics['fps']:.1f}"
                    )
                
                time.sleep(1.0 / refresh_rate)
                
            except Exception as e:
                self.get_logger().error(f"❌ Dashboard publishing error: {e}")
                time.sleep(1.0)
    
    def collect_dashboard_data(self):
        """Collect all current data for dashboard"""
        data = {
            'timestamp': time.time(),
            'metrics': self.metrics.copy(),
            'overlay_available': not self.overlay_buffer.empty(),
            'traversability_available': not self.traversability_buffer.empty(),
            'hazards_available': not self.hazards_buffer.empty(),
        }
        
        # Get latest hazards data
        try:
            if not self.hazards_buffer.empty():
                hazards_data = self.hazards_buffer.queue[-1]  # Peek at latest
                data['hazards'] = hazards_data.get('hazards', [])
                data['hazards_detected'] = hazards_data.get('hazards_detected', 0)
        except:
            data['hazards'] = []
            data['hazards_detected'] = 0
        
        # Get system status
        data['system_status'] = self.get_system_status()
        
        return data
    
    def get_system_status(self):
        """Get overall system status for dashboard"""
        return {
            'ros_connected': True,
            'ai_model_loaded': True,  # This would be dynamic
            'navigation_active': True,
            'camera_streaming': not self.overlay_buffer.empty(),
            'last_update': time.time(),
            'performance': {
                'fps': self.metrics['fps'],
                'buffer_health': self.get_buffer_health()
            }
        }
    
    def get_buffer_health(self):
        """Calculate buffer health metrics"""
        buffers = [
            ('overlay', self.overlay_buffer),
            ('traversability', self.traversability_buffer),
            ('hazards', self.hazards_buffer)
        ]
        
        health = {}
        for name, buffer in buffers:
            capacity = buffer.maxsize
            current = buffer.qsize()
            health[name] = {
                'current': current,
                'capacity': capacity,
                'percentage': (current / capacity) * 100 if capacity > 0 else 0
            }
        
        return health
    
    def save_data_for_dashboard(self, data):
        """
        Save data to location accessible by Streamlit
        This is a simplified implementation
        """
        try:
            import json
            import os
            
            # Create dashboard data directory
            os.makedirs('/tmp/mars_rover_dashboard', exist_ok=True)
            
            # Save JSON data
            with open('/tmp/mars_rover_dashboard/latest_data.json', 'w') as f:
                json.dump(data, f, indent=2)
            
            # Save images if available
            if not self.overlay_buffer.empty():
                overlay_img = self.overlay_buffer.queue[-1]
                cv2.imwrite('/tmp/mars_rover_dashboard/overlay.jpg', overlay_img)
                
        except Exception as e:
            self.get_logger().error(f"❌ Data save error: {e}")


def main(args=None):
    rclpy.init(args=args)
    dashboard_bridge = DashboardBridge()
    
    try:
        rclpy.spin(dashboard_bridge)
    except KeyboardInterrupt:
        pass
    finally:
        dashboard_bridge.get_logger().info("🛑 Shutting down Dashboard Bridge")
        dashboard_bridge.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()