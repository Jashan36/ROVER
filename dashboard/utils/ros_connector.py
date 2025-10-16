"""
ROS2 Connector
Connects to ROS2 topics and WebSocket bridge to collect terrain data
"""

import asyncio
try:
    import websockets  # type: ignore
except ImportError:  # pragma: no cover - optional dependency for tests
    import types

    class _MissingWebsockets(types.ModuleType):
        def __getattr__(self, name):
            raise RuntimeError(
                "ROSConnector requires the 'websockets' package. Install it with "
                "'pip install websockets' or run the dashboard in mock mode."
            )

    websockets = _MissingWebsockets("websockets")
    websockets._missing = True  # type: ignore[attr-defined]
import json
import threading
import queue
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
import logging
import time
import base64
import numpy as np
import cv2

logger = logging.getLogger(__name__)


@dataclass
class TerrainData:
    """Container for terrain analysis data"""
    timestamp: float
    
    # Images (numpy arrays)
    overlay_image: Optional[np.ndarray] = None
    traversability_image: Optional[np.ndarray] = None
    hazard_image: Optional[np.ndarray] = None
    
    # Metrics
    avg_traversability: float = 0.0
    safe_area_ratio: float = 0.0
    best_direction_deg: float = 0.0
    
    # Hazards
    num_hazards: int = 0
    hazard_summary: Dict = field(default_factory=dict)
    
    # Performance
    inference_time_ms: float = 0.0
    fps: float = 0.0
    
    # Terrain distribution
    terrain_distribution: Dict = field(default_factory=dict)
    
    @property
    def is_valid(self) -> bool:
        """Check if data is valid"""
        return self.overlay_image is not None or self.traversability_image is not None


class ROSConnector:
    """
    Connector for ROS2 data via WebSocket bridge
    """
    
    def __init__(
        self,
        websocket_uri: str = "ws://localhost:8765",
        max_queue_size: int = 5,
        reconnect_interval: float = 5.0
    ):
        """
        Initialize ROS connector
        
        Args:
            websocket_uri: WebSocket server URI
            max_queue_size: Maximum queue size
            reconnect_interval: Reconnection interval in seconds
        """
        self.websocket_uri = websocket_uri
        self.max_queue_size = max_queue_size
        self.reconnect_interval = reconnect_interval
        
        # Data queue
        self.data_queue = queue.Queue(maxsize=max_queue_size)
        
        # Connection state
        self.connected = False
        self.running = False
        self.client_thread = None
        
        # Statistics
        self.messages_received = 0
        self.last_message_time = 0.0
        self.connection_errors = 0
        
        if getattr(websockets, "_missing", False):
            logger.warning(
                "websockets package not installed; ROSConnector can be instantiated "
                "but live WebSocket connections are disabled."
            )
        else:
            logger.info(f"ROSConnector initialized for {websocket_uri}")

    def start(self):
        """Start WebSocket client"""
        if self.running:
            logger.warning("Connector already running")
            return

        if getattr(websockets, "_missing", False):
            raise RuntimeError(
                "Cannot start ROSConnector: the 'websockets' package is not installed. "
                "Install it with 'pip install websockets' to enable live connections."
            )

        self.running = True
        
        # Start client in background thread
        self.client_thread = threading.Thread(
            target=self._run_client,
            daemon=True
        )
        self.client_thread.start()
        
        logger.info("ROS connector started")

    def stop(self):
        """Stop WebSocket client"""
        self.running = False
        
        if self.client_thread:
            self.client_thread.join(timeout=5.0)
        
        logger.info("ROS connector stopped")

    def _run_client(self):
        """Run WebSocket client (in separate thread)"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        while self.running:
            try:
                loop.run_until_complete(self._connect_and_receive())
            except Exception as e:
                logger.error(f"Client error: {e}")
                self.connection_errors += 1
                self.connected = False
                
                if self.running:
                    logger.info(f"Reconnecting in {self.reconnect_interval}s...")
                    time.sleep(self.reconnect_interval)
        
        loop.close()

    async def _connect_and_receive(self):
        """Connect to WebSocket and receive data"""
        try:
            async with websockets.connect(self.websocket_uri) as websocket:
                logger.info(f"Connected to {self.websocket_uri}")
                self.connected = True
                
                while self.running:
                    try:
                        # Receive message
                        message = await asyncio.wait_for(
                            websocket.recv(),
                            timeout=10.0
                        )
                        
                        # Parse data
                        data = self._parse_message(message)
                        
                        if data:
                            # Add to queue
                            try:
                                self.data_queue.put_nowait(data)
                            except queue.Full:
                                # Drop oldest
                                try:
                                    self.data_queue.get_nowait()
                                    self.data_queue.put_nowait(data)
                                except:
                                    pass
                            
                            self.messages_received += 1
                            self.last_message_time = time.time()
                        
                    except asyncio.TimeoutError:
                        # No data for a while, check connection
                        await websocket.ping()
                        
        except Exception as e:
            logger.error(f"Connection error: {e}")
            raise

    def _parse_message(self, message: str) -> Optional[TerrainData]:
        """Parse JSON message to TerrainData"""
        try:
            data_dict = json.loads(message)
            
            # Decode images
            overlay_img = self._decode_image(data_dict.get('overlay_image'))
            traversability_img = self._decode_image(data_dict.get('traversability_image'))
            hazard_img = self._decode_image(data_dict.get('hazard_image'))
            
            # Create TerrainData
            data = TerrainData(
                timestamp=data_dict.get('timestamp', time.time()),
                overlay_image=overlay_img,
                traversability_image=traversability_img,
                hazard_image=hazard_img,
                avg_traversability=data_dict.get('avg_traversability', 0.0),
                safe_area_ratio=data_dict.get('safe_area_ratio', 0.0),
                best_direction_deg=data_dict.get('best_direction_deg', 0.0),
                num_hazards=data_dict.get('num_hazards', 0),
                hazard_summary=data_dict.get('hazard_summary', {}),
                inference_time_ms=data_dict.get('inference_time_ms', 0.0),
                fps=data_dict.get('fps', 0.0),
                terrain_distribution=data_dict.get('terrain_distribution', {})
            )
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to parse message: {e}")
            return None

    def _decode_image(self, base64_str: Optional[str]) -> Optional[np.ndarray]:
        """Decode base64 image to numpy array"""
        if not base64_str:
            return None
        
        try:
            # Remove data URL prefix if present
            if base64_str.startswith('data:image'):
                base64_str = base64_str.split(',')[1]
            
            # Decode base64
            img_data = base64.b64decode(base64_str)
            
            # Convert to numpy array
            nparr = np.frombuffer(img_data, np.uint8)
            
            # Decode image
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is not None:
                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            return img
            
        except Exception as e:
            logger.error(f"Failed to decode image: {e}")
            return None

    def get_latest_data(self) -> Optional[TerrainData]:
        """Get most recent data"""
        try:
            return self.data_queue.get_nowait()
        except queue.Empty:
            return None

    def get_all_data(self) -> list:
        """Get all available data"""
        data_list = []
        
        while True:
            try:
                data = self.data_queue.get_nowait()
                data_list.append(data)
            except queue.Empty:
                break
        
        return data_list

    def is_connected(self) -> bool:
        """Check if connected"""
        return self.connected

    def get_statistics(self) -> Dict[str, Any]:
        """Get connector statistics"""
        time_since_last = time.time() - self.last_message_time if self.last_message_time > 0 else 0
        
        return {
            'connected': self.connected,
            'running': self.running,
            'messages_received': self.messages_received,
            'queue_size': self.data_queue.qsize(),
            'time_since_last_message': time_since_last,
            'connection_errors': self.connection_errors
        }


# Testing
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    connector = ROSConnector()
    connector.start()
    
    print("ROS Connector running...")
    print("Waiting for data from ws://localhost:8765")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            data = connector.get_latest_data()
            
            if data:
                print(f"\nReceived data:")
                print(f"  Timestamp: {data.timestamp}")
                print(f"  Traversability: {data.avg_traversability:.3f}")
                print(f"  Hazards: {data.num_hazards}")
                print(f"  FPS: {data.fps:.1f}")
                
                if data.overlay_image is not None:
                    print(f"  Overlay shape: {data.overlay_image.shape}")
            
            stats = connector.get_statistics()
            if stats['messages_received'] % 10 == 0 and stats['messages_received'] > 0:
                print(f"\nStats: {stats}")
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nStopping...")
        connector.stop()
