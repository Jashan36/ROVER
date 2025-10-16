#!/usr/bin/env python3
"""
ROS2 Dashboard Bridge - Stream Mars Rover data to Streamlit dashboard
Bridges ROS2 topics to web interface for real-time monitoring and control
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float32MultiArray
from geometry_msgs.msg import PoseStamped, Twist
import json
import time
import threading
import asyncio
import websockets
import cv2
import numpy as np
from typing import Dict, Any, Optional, List
import logging
import base64
from collections import deque
import queue

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarsRoverDashboardBridge(Node):
    """
    ROS2 node that bridges rover data to web dashboard
    Handles WebSocket connections and data streaming to Streamlit
    """
    
    def __init__(self):
        super().__init__('mars_rover_dashboard_bridge')
        
        # Configuration
        self.websocket_port = 8765
        self.max_clients = 5
        self.update_rate = 10.0  # Hz
        
        # Data storage
        self.latest_data = {
            'camera_image': None,
            'segmentation': None,
            'hazards': [],
            'traversability': 0.0,
            'analysis': {},
            'rover_pose': None,
            'rover_velocity': None,
            'system_status': 'disconnected'
        }
        
        # Image processing
        self.image_buffer = deque(maxlen=3)  # Keep last 3 frames
        self.image_quality = 85  # JPEG quality for compression
        
        # WebSocket management
        self.websocket_clients = set()
        self.websocket_server = None
        self.data_lock = threading.Lock()
        
        # Statistics
        self.bytes_sent = 0
        self.messages_sent = 0
        self.client_count = 0
        
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
        
        # Initialize ROS2 subscribers
        self._init_subscribers()
        
        # Create update timer
        self.update_timer = self.create_timer(
            1.0 / self.update_rate,
            self._update_dashboard_data
        )
        
        # Start WebSocket server in a separate thread
        self.websocket_thread = threading.Thread(
            target=self._start_websocket_server,
            daemon=True
        )
        self.websocket_thread.start()
        
        logger.info('Mars Rover Dashboard Bridge initialized')
        logger.info(f'WebSocket server starting on port {self.websocket_port}')
        logger.info(f'Update rate: {self.update_rate} Hz')
        
    def _init_subscribers(self):
        """Initialize ROS2 subscribers for rover data"""
        # Camera and AI data
        self.camera_sub = self.create_subscription(
            Image,
            '/rover/camera/processed',
            self._camera_callback,
            qos_profile=self.image_qos
        )
        
        self.segmentation_sub = self.create_subscription(
            Image,
            '/rover/ai/terrain_segmentation',
            self._segmentation_callback,
            qos_profile=self.image_qos
        )
        
        self.hazards_sub = self.create_subscription(
            String,
            '/rover/ai/hazards',
            self._hazards_callback,
            qos_profile=10
        )
        
        self.traversability_sub = self.create_subscription(
            Float32MultiArray,
            '/rover/ai/traversability',
            self._traversability_callback,
            qos_profile=10
        )
        
        self.analysis_sub = self.create_subscription(
            String,
            '/rover/ai/analysis',
            self._analysis_callback,
            qos_profile=10
        )
        
        # Rover navigation data
        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/rover/pose',
            self._pose_callback,
            qos_profile=self.reliable_qos
        )
        
        self.velocity_sub = self.create_subscription(
            Twist,
            '/rover/cmd_vel',
            self._velocity_callback,
            qos_profile=self.reliable_qos
        )
        
        logger.info('Initialized ROS2 subscribers')
    
    def _camera_callback(self, msg: Image):
        """Handle camera image data"""
        try:
            # Convert ROS Image to OpenCV format
            cv_image = self._ros_image_to_cv2(msg)
            
            # Compress image for web transmission
            compressed_image = self._compress_image(cv_image)
            
            with self.data_lock:
                self.latest_data['camera_image'] = compressed_image
                self.image_buffer.append({
                    'image': compressed_image,
                    'timestamp': time.time()
                })
                
        except Exception as e:
            logger.error(f'Error processing camera image: {e}')
    
    def _segmentation_callback(self, msg: Image):
        """Handle terrain segmentation data"""
        try:
            # Convert ROS Image to OpenCV format
            cv_image = self._ros_image_to_cv2(msg)
            
            # Compress segmentation image
            compressed_seg = self._compress_image(cv_image)
            
            with self.data_lock:
                self.latest_data['segmentation'] = compressed_seg
                
        except Exception as e:
            logger.error(f'Error processing segmentation: {e}')
    
    def _hazards_callback(self, msg: String):
        """Handle hazard detection data"""
        try:
            hazards_data = json.loads(msg.data)
            
            with self.data_lock:
                self.latest_data['hazards'] = hazards_data
                
        except Exception as e:
            logger.error(f'Error processing hazards: {e}')
    
    def _traversability_callback(self, msg: Float32MultiArray):
        """Handle traversability data"""
        try:
            traversability_score = msg.data[0] if len(msg.data) > 0 else 0.0
            
            with self.data_lock:
                self.latest_data['traversability'] = traversability_score
                
        except Exception as e:
            logger.error(f'Error processing traversability: {e}')
    
    def _analysis_callback(self, msg: String):
        """Handle AI analysis summary"""
        try:
            analysis_data = json.loads(msg.data)
            
            with self.data_lock:
                self.latest_data['analysis'] = analysis_data
                
        except Exception as e:
            logger.error(f'Error processing analysis: {e}')
    
    def _pose_callback(self, msg: PoseStamped):
        """Handle rover pose data"""
        try:
            pose_data = {
                'position': {
                    'x': msg.pose.position.x,
                    'y': msg.pose.position.y,
                    'z': msg.pose.position.z
                },
                'orientation': {
                    'x': msg.pose.orientation.x,
                    'y': msg.pose.orientation.y,
                    'z': msg.pose.orientation.z,
                    'w': msg.pose.orientation.w
                },
                'timestamp': time.time()
            }
            
            with self.data_lock:
                self.latest_data['rover_pose'] = pose_data
                
        except Exception as e:
            logger.error(f'Error processing pose: {e}')
    
    def _velocity_callback(self, msg: Twist):
        """Handle rover velocity data"""
        try:
            velocity_data = {
                'linear': {
                    'x': msg.linear.x,
                    'y': msg.linear.y,
                    'z': msg.linear.z
                },
                'angular': {
                    'x': msg.angular.x,
                    'y': msg.angular.y,
                    'z': msg.angular.z
                },
                'timestamp': time.time()
            }
            
            with self.data_lock:
                self.latest_data['rover_velocity'] = velocity_data
                
        except Exception as e:
            logger.error(f'Error processing velocity: {e}')
    
    def _compress_image(self, cv_image: np.ndarray) -> str:
        """Compress image to base64 string for web transmission"""
        try:
            # Resize for web display (reduce bandwidth)
            height, width = cv_image.shape[:2]
            if width > 640:
                scale = 640 / width
                new_width = 640
                new_height = int(height * scale)
                cv_image = cv2.resize(cv_image, (new_width, new_height))
            
            # Encode as JPEG
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.image_quality]
            _, buffer = cv2.imencode('.jpg', cv_image, encode_param)
            
            # Convert to base64
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return img_base64
            
        except Exception as e:
            logger.error(f'Error compressing image: {e}')
            return ""
    
    def _update_dashboard_data(self):
        """Update dashboard data and broadcast to clients"""
        try:
            # Prepare data package
            with self.data_lock:
                data_package = self._prepare_data_package()
            
            # Broadcast to all connected clients
            if self.websocket_clients:
                self._broadcast_to_clients(data_package)
                
        except Exception as e:
            logger.error(f'Error updating dashboard data: {e}')
    
    def _prepare_data_package(self) -> Dict[str, Any]:
        """Prepare complete data package for dashboard"""
        try:
            # Calculate statistics
            stats = {
                'timestamp': time.time(),
                'client_count': len(self.websocket_clients),
                'bytes_sent': self.bytes_sent,
                'messages_sent': self.messages_sent,
                'update_rate': self.update_rate
            }
            
            # Prepare rover status
            rover_status = self._get_rover_status()
            
            # Create data package
            data_package = {
                'type': 'dashboard_update',
                'timestamp': time.time(),
                'data': {
                    'camera': self.latest_data['camera_image'],
                    'segmentation': self.latest_data['segmentation'],
                    'hazards': self.latest_data['hazards'],
                    'traversability': self.latest_data['traversability'],
                    'analysis': self.latest_data['analysis'],
                    'rover_pose': self.latest_data['rover_pose'],
                    'rover_velocity': self.latest_data['rover_velocity'],
                    'rover_status': rover_status,
                    'system_stats': stats
                }
            }
            
            return data_package
            
        except Exception as e:
            logger.error(f'Error preparing data package: {e}')
            return {'type': 'error', 'message': str(e)}
    
    def _get_rover_status(self) -> Dict[str, Any]:
        """Get current rover system status"""
        try:
            status = {
                'connected': True,
                'ai_active': len(self.latest_data['hazards']) > 0 or self.latest_data['traversability'] > 0,
                'camera_active': self.latest_data['camera_image'] is not None,
                'navigation_active': self.latest_data['rover_pose'] is not None,
                'last_update': time.time()
            }
            
            # Determine overall status
            if status['ai_active'] and status['camera_active']:
                status['overall'] = 'operational'
            elif status['camera_active']:
                status['overall'] = 'limited'
            else:
                status['overall'] = 'disconnected'
                
            return status
            
        except Exception as e:
            logger.error(f'Error getting rover status: {e}')
            return {'overall': 'error', 'message': str(e)}
    
    def _start_websocket_server(self):
        """Start WebSocket server in separate thread"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            start_server = websockets.serve(
                self._handle_websocket_client,
                "localhost",
                self.websocket_port,
                max_size=10 * 1024 * 1024  # 10MB max message size
            )
            
            logger.info(f'WebSocket server started on ws://localhost:{self.websocket_port}')
            loop.run_until_complete(start_server)
            loop.run_forever()
            
        except Exception as e:
            logger.error(f'Error starting WebSocket server: {e}')
    
    async def _handle_websocket_client(self, websocket, path):
        """Handle individual WebSocket client connections"""
        client_id = f"client_{len(self.websocket_clients)}"
        logger.info(f'New WebSocket client connected: {client_id}')
        
        try:
            self.websocket_clients.add(websocket)
            self.client_count += 1
            
            # Send welcome message
            welcome_msg = {
                'type': 'welcome',
                'client_id': client_id,
                'timestamp': time.time(),
                'message': 'Connected to Mars Rover Dashboard'
            }
            await websocket.send(json.dumps(welcome_msg))
            
            # Handle incoming messages
            async for message in websocket:
                await self._handle_client_message(websocket, message)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info(f'WebSocket client disconnected: {client_id}')
        except Exception as e:
            logger.error(f'Error handling WebSocket client {client_id}: {e}')
        finally:
            self.websocket_clients.discard(websocket)
    
    async def _handle_client_message(self, websocket, message):
        """Handle messages from WebSocket clients"""
        try:
            data = json.loads(message)
            msg_type = data.get('type', 'unknown')
            
            if msg_type == 'ping':
                # Respond to ping
                pong_msg = {
                    'type': 'pong',
                    'timestamp': time.time()
                }
                await websocket.send(json.dumps(pong_msg))
                
            elif msg_type == 'request_data':
                # Send current data immediately
                with self.data_lock:
                    data_package = self._prepare_data_package()
                await websocket.send(json.dumps(data_package))
                
            elif msg_type == 'set_config':
                # Handle configuration changes
                await self._handle_config_change(data.get('config', {}))
                
            else:
                logger.warning(f'Unknown message type from client: {msg_type}')
                
        except Exception as e:
            logger.error(f'Error handling client message: {e}')
    
    async def _handle_config_change(self, config: Dict[str, Any]):
        """Handle configuration changes from dashboard"""
        try:
            if 'image_quality' in config:
                self.image_quality = max(10, min(100, config['image_quality']))
                logger.info(f'Image quality updated to: {self.image_quality}')
                
            if 'update_rate' in config:
                new_rate = max(1.0, min(30.0, config['update_rate']))
                self.update_rate = new_rate
                self.update_timer.cancel()
                self.update_timer = self.create_timer(
                    1.0 / self.update_rate,
                    self._update_dashboard_data
                )
                logger.info(f'Update rate changed to: {self.update_rate} Hz')
                
        except Exception as e:
            logger.error(f'Error handling config change: {e}')
    
    def _broadcast_to_clients(self, data_package: Dict[str, Any]):
        """Broadcast data to all connected WebSocket clients"""
        if not self.websocket_clients:
            return
            
        try:
            message = json.dumps(data_package)
            message_bytes = len(message.encode('utf-8'))
            
            # Create task for broadcasting
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._broadcast_async(message))
            
            # Update statistics
            self.bytes_sent += message_bytes
            self.messages_sent += 1
            
        except Exception as e:
            logger.error(f'Error broadcasting to clients: {e}')
    
    async def _broadcast_async(self, message: str):
        """Async broadcast to all clients"""
        if not self.websocket_clients:
            return
            
        # Send to all connected clients
        disconnected_clients = set()
        
        for websocket in self.websocket_clients:
            try:
                await websocket.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(websocket)
            except Exception as e:
                logger.error(f'Error sending to client: {e}')
                disconnected_clients.add(websocket)
        
        # Remove disconnected clients
        self.websocket_clients -= disconnected_clients
    
    def get_dashboard_stats(self) -> Dict[str, Any]:
        """Get dashboard statistics"""
        return {
            'connected_clients': len(self.websocket_clients),
            'total_clients': self.client_count,
            'bytes_sent': self.bytes_sent,
            'messages_sent': self.messages_sent,
            'update_rate': self.update_rate,
            'image_quality': self.image_quality,
            'websocket_port': self.websocket_port
        }
    
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


def main(args=None):
    """Main function"""
    rclpy.init(args=args)
    
    try:
        node = MarsRoverDashboardBridge()
        
        logger.info('Starting Mars Rover Dashboard Bridge...')
        logger.info('WebSocket server available at ws://localhost:8765')
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
