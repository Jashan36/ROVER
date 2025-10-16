"""
ROS2-AI Bridge
Main integration node connecting camera, AI inference, and navigation
"""

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
import threading
import time
from pathlib import Path
from typing import Optional
import logging
import yaml

import sys
sys.path.append(str(Path(__file__).parent.parent))

from ai.inference.realtime_pipeline import RealtimePipeline, PipelineConfig
from ai.inference.realtime_pipeline import InferenceResult

from integration.image_subscriber import ImageSubscriber, ImageBuffer
from integration.analysis_publisher import AnalysisPublisher, PublisherConfig
from integration.dashboard_bridge import DashboardBridge, DashboardData
from sim.rover_integration.rover_integration.costmap_generator import CostmapGenerator, CostmapConfig

logger = logging.getLogger(__name__)


class BridgeConfig:
    """Configuration for ROS-AI bridge"""
    def __init__(self):
        # Node
        self.node_name = 'terrain_analyzer'
        
        # Topics
        self.camera_topic = '/camera/image_raw'
        
        # AI Pipeline
        self.model_path = 'ai/models/weights/terrain_unet_best.pth'
        self.device = 'auto'
        self.input_size = (512, 512)
        
        # Processing
        self.inference_rate_hz = 5.0
        self.enable_threading = True
        
        # Dashboard
        self.enable_dashboard = True
        self.dashboard_host = 'localhost'
        self.dashboard_port = 8765
        
        # Costmap
        self.enable_costmap = True
        
    @classmethod
    def from_yaml(cls, config_path: str):
        """Load configuration from YAML file"""
        config = cls()
        
        try:
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
            
            # Parse YAML and update config
            if 'ros2' in yaml_config:
                ros_config = yaml_config['ros2']
                config.node_name = ros_config.get('node_name', config.node_name)
                
                if 'subscribers' in ros_config and 'camera' in ros_config['subscribers']:
                    config.camera_topic = ros_config['subscribers']['camera'].get('topic', config.camera_topic)
                
                if 'processing' in ros_config:
                    config.inference_rate_hz = ros_config['processing'].get('inference_rate_hz', config.inference_rate_hz)
            
            if 'inference' in yaml_config:
                inf_config = yaml_config['inference']
                if 'model' in inf_config:
                    config.model_path = inf_config['model'].get('path', config.model_path)
                    config.device = inf_config['model'].get('device', config.device)
                
                if 'input' in inf_config:
                    size = inf_config['input'].get('size', [512, 512])
                    config.input_size = tuple(size)
            
            logger.info(f"Loaded configuration from {config_path}")
            
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            logger.info("Using default configuration")
        
        return config


class ROSAIBridge(Node):
    """
    Main ROS2-AI integration node
    Connects camera stream to AI inference and publishes results
    """
    
    def __init__(
        self,
        config: Optional[BridgeConfig] = None,
        config_path: Optional[str] = None
    ):
        """
        Initialize ROS-AI bridge
        
        Args:
            config: Bridge configuration
            config_path: Path to YAML config file
        """
        # Load config
        if config_path:
            self.config = BridgeConfig.from_yaml(config_path)
        else:
            self.config = config or BridgeConfig()
        
        # Initialize ROS2 node
        super().__init__(self.config.node_name)
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("Initializing ROS-AI Bridge")
        self.get_logger().info("=" * 60)
        
        # Initialize AI pipeline
        self.get_logger().info("Loading AI pipeline...")
        pipeline_config = PipelineConfig(
            segmentation_model_path=Path(self.config.model_path) if self.config.model_path else None,
            device=self.config.device,
            input_size=self.config.input_size,
            enable_hazard_detection=True,
            generate_overlays=True
        )
        self.pipeline = RealtimePipeline(pipeline_config)
        
        # Initialize image subscriber
        self.get_logger().info(f"Subscribing to {self.config.camera_topic}...")
        self.image_subscriber = ImageSubscriber(
            node_name='bridge_image_subscriber',
            topic=self.config.camera_topic
        )
        
        # Initialize analysis publisher
        self.get_logger().info("Setting up publishers...")
        self.analysis_publisher = AnalysisPublisher(
            node_name='bridge_analysis_publisher'
        )
        
        # Initialize costmap generator
        if self.config.enable_costmap:
            self.get_logger().info("Initializing costmap generator...")
            self.costmap_generator = CostmapGenerator()
        else:
            self.costmap_generator = None
        
        # Initialize dashboard bridge
        self.dashboard_bridge = None
        if self.config.enable_dashboard:
            self.get_logger().info("Starting dashboard bridge...")
            self.dashboard_bridge = DashboardBridge(
                host=self.config.dashboard_host,
                port=self.config.dashboard_port
            )
            self.dashboard_bridge.start()
        
        # Processing state
        self.running = False
        self.processing_thread = None
        self.last_inference_time = 0.0
        self.inference_interval = 1.0 / self.config.inference_rate_hz
        
        # Statistics
        self.frame_count = 0
        self.inference_count = 0
        self.start_time = time.time()
        
        # Create timer for statistics logging
        self.stats_timer = self.create_timer(10.0, self._log_statistics)
        
        self.get_logger().info("ROS-AI Bridge initialized successfully")
        self.get_logger().info("=" * 60)

    def start_processing(self):
        """Start processing pipeline"""
        if self.running:
            self.get_logger().warning("Processing already running")
            return
        
        self.running = True
        
        if self.config.enable_threading:
            # Run in separate thread
            self.processing_thread = threading.Thread(
                target=self._processing_loop,
                daemon=True
            )
            self.processing_thread.start()
            self.get_logger().info("Started processing in background thread")
        else:
            # Create timer for processing
            self.processing_timer = self.create_timer(
                self.inference_interval,
                self._process_latest_image
            )
            self.get_logger().info("Started processing with timer")

    def stop_processing(self):
        """Stop processing pipeline"""
        self.running = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
        
        if self.dashboard_bridge:
            self.dashboard_bridge.stop()
        
        self.get_logger().info("Processing stopped")

    def _processing_loop(self):
        """Main processing loop (runs in separate thread)"""
        self.get_logger().info("Processing loop started")
        
        while self.running:
            try:
                # Rate limiting
                current_time = time.time()
                elapsed = current_time - self.last_inference_time
                
                if elapsed < self.inference_interval:
                    time.sleep(self.inference_interval - elapsed)
                    continue
                
                # Process latest image
                self._process_latest_image()
                
                self.last_inference_time = time.time()
                
            except Exception as e:
                self.get_logger().error(f"Error in processing loop: {e}")
                time.sleep(1.0)

    def _process_latest_image(self):
        """Process latest image from camera"""
        try:
            # Get latest image
            image_buffer = self.image_subscriber.get_latest_image()
            
            if image_buffer is None:
                return
            
            self.frame_count += 1
            
            # Run AI inference
            start_time = time.time()
            result = self.pipeline.process(image_buffer.image)
            inference_time = (time.time() - start_time) * 1000
            
            self.inference_count += 1
            
            # Publish results to ROS
            self.analysis_publisher.publish_results(
                result,
                frame_id=image_buffer.frame_id
            )
            
            # Publish to dashboard
            if self.dashboard_bridge:
                self._publish_to_dashboard(result, inference_time)
            
            # Log performance
            if self.inference_count % 10 == 0:
                self.get_logger().debug(
                    f"Inference {self.inference_count}: {inference_time:.1f}ms, "
                    f"{len(result.hazards)} hazards"
                )
            
        except Exception as e:
            self.get_logger().error(f"Failed to process image: {e}")

    def _publish_to_dashboard(self, result: InferenceResult, inference_time: float):
        """Publish data to dashboard"""
        try:
            # Encode images
            overlay_encoded = None
            traversability_encoded = None
            hazard_encoded = None
            
            if result.overlay is not None:
                overlay_encoded = DashboardBridge.encode_image(result.overlay)
            
            if result.traversability_viz is not None:
                traversability_encoded = DashboardBridge.encode_image(result.traversability_viz)
            
            if result.hazard_viz is not None:
                hazard_encoded = DashboardBridge.encode_image(result.hazard_viz)
            
            # Create dashboard data
            data = DashboardData(
                timestamp=time.time(),
                overlay_image=overlay_encoded,
                traversability_image=traversability_encoded,
                hazard_image=hazard_encoded,
                avg_traversability=result.stats.get('avg_traversability', 0.0),
                safe_area_ratio=result.stats.get('safe_area_ratio', 0.0),
                best_direction_deg=float(result.best_direction * 180.0 / 3.14159),
                num_hazards=len(result.hazards),
                hazard_summary=result.hazard_summary,
                inference_time_ms=inference_time,
                fps=self.inference_count / (time.time() - self.start_time),
                terrain_distribution=result.stats.get('terrain_distribution', {})
            )
            
            self.dashboard_bridge.publish_data(data)
            
        except Exception as e:
            self.get_logger().error(f"Failed to publish to dashboard: {e}")

    def _log_statistics(self):
        """Log periodic statistics"""
        uptime = time.time() - self.start_time
        
        img_stats = self.image_subscriber.get_statistics()
        pub_stats = self.analysis_publisher.get_statistics()
        perf_stats = self.pipeline.get_performance_stats()
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("Statistics:")
        self.get_logger().info(f"  Uptime: {uptime:.1f}s")
        self.get_logger().info(f"  Camera FPS: {img_stats['fps']:.1f}")
        self.get_logger().info(f"  Frames received: {img_stats['frame_count']}")
        self.get_logger().info(f"  Inferences: {self.inference_count}")
        
        if perf_stats:
            self.get_logger().info(f"  Avg inference time: {perf_stats['avg_time_ms']:.1f}ms")
            self.get_logger().info(f"  Avg FPS: {perf_stats['avg_fps']:.1f}")
        
        self.get_logger().info(f"  Published: {pub_stats['publish_count']}")
        
        if self.dashboard_bridge:
            dash_stats = self.dashboard_bridge.get_statistics()
            self.get_logger().info(f"  Dashboard clients: {dash_stats['connected_clients']}")
        
        self.get_logger().info("=" * 60)


# Main execution
def main(args=None):
    """Main entry point"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    rclpy.init(args=args)
    
    # Create bridge
    config_path = 'integration/config/bridge_params.yaml'
    if not Path(config_path).exists():
        config_path = None
    
    bridge = ROSAIBridge(config_path=config_path)
    
    # Create executor
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(bridge)
    executor.add_node(bridge.image_subscriber)
    executor.add_node(bridge.analysis_publisher)
    
    # Start processing
    bridge.start_processing()
    
    try:
        # Spin
        bridge.get_logger().info("ROS-AI Bridge running... Press Ctrl+C to stop")
        executor.spin()
    except KeyboardInterrupt:
        bridge.get_logger().info("Shutting down...")
    finally:
        bridge.stop_processing()
        bridge.destroy_node()
        bridge.image_subscriber.destroy_node()
        bridge.analysis_publisher.destroy_node()
        executor.shutdown()
        rclpy.shutdown()


if __name__ == '__main__':
    main()