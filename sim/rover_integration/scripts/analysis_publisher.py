#!/usr/bin/env python3
"""
ROS2 Analysis Publisher - Publish comprehensive AI analysis results
Aggregates and publishes scientific insights, mission planning data, and rover recommendations
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from std_msgs.msg import String, Float32MultiArray, Header
from geometry_msgs.msg import PoseStamped, Twist, Point, Quaternion
from sensor_msgs.msg import Image, PointCloud2, PointField
from nav_msgs.msg import OccupancyGrid, Path
import json
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from collections import deque
import threading
from dataclasses import dataclass, asdict
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ScienceObservation:
    """Scientific observation data structure"""
    observation_id: str
    timestamp: float
    location: Tuple[float, float, float]  # x, y, z
    observation_type: str  # 'geology', 'atmosphere', 'biology', 'physics'
    confidence: float
    description: str
    measurements: Dict[str, float]
    image_reference: Optional[str] = None
    priority: int = 1  # 1=low, 2=medium, 3=high, 4=critical


@dataclass
class MissionRecommendation:
    """Mission planning recommendation"""
    recommendation_id: str
    timestamp: float
    priority: int  # 1-5 scale
    category: str  # 'navigation', 'science', 'safety', 'efficiency'
    action: str
    reasoning: str
    estimated_duration: float  # seconds
    risk_level: str  # 'low', 'medium', 'high', 'critical'
    prerequisites: List[str] = None


@dataclass
class TerrainAnalysis:
    """Comprehensive terrain analysis"""
    timestamp: float
    traversability_score: float
    hazard_count: int
    critical_hazards: int
    dominant_terrain_type: str
    slope_analysis: Dict[str, float]
    surface_roughness: float
    recommended_path: List[Tuple[float, float]]
    alternative_paths: List[List[Tuple[float, float]]]


class MarsRoverAnalysisPublisher(Node):
    """
    ROS2 node that publishes comprehensive AI analysis results
    Provides scientific insights, mission planning, and rover recommendations
    """
    
    def __init__(self):
        super().__init__('mars_rover_analysis_publisher')
        
        # Configuration
        self.analysis_rate = 2.0  # Hz - Lower rate for complex analysis
        self.history_size = 100
        self.science_observation_id = 0
        self.recommendation_id = 0
        
        # Data storage
        self.latest_analysis = {
            'hazards': [],
            'traversability': 0.0,
            'rover_pose': None,
            'rover_velocity': None,
            'segmentation': None,
            'timestamp': time.time()
        }
        
        # Analysis history
        self.terrain_history = deque(maxlen=self.history_size)
        self.science_observations = deque(maxlen=self.history_size)
        self.mission_recommendations = deque(maxlen=self.history_size)
        self.data_lock = threading.Lock()
        
        # Mission state
        self.current_mission_phase = 'exploration'
        self.target_locations = []
        self.visited_locations = []
        self.mission_start_time = time.time()
        
        # QoS profiles
        self.reliable_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        self.best_effort_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5
        )
        
        # Initialize subscribers and publishers
        self._init_subscribers()
        self._init_publishers()
        
        # Create analysis timer
        self.analysis_timer = self.create_timer(
            1.0 / self.analysis_rate,
            self._perform_comprehensive_analysis
        )
        
        logger.info('Mars Rover Analysis Publisher initialized')
        logger.info(f'Analysis rate: {self.analysis_rate} Hz')
        logger.info(f'Mission phase: {self.current_mission_phase}')
        
    def _init_subscribers(self):
        """Initialize ROS2 subscribers"""
        # AI analysis data
        self.hazards_sub = self.create_subscription(
            String,
            '/rover/ai/hazards',
            self._hazards_callback,
            qos_profile=self.best_effort_qos
        )
        
        self.traversability_sub = self.create_subscription(
            Float32MultiArray,
            '/rover/ai/traversability',
            self._traversability_callback,
            qos_profile=self.best_effort_qos
        )
        
        self.segmentation_sub = self.create_subscription(
            Image,
            '/rover/ai/terrain_segmentation',
            self._segmentation_callback,
            qos_profile=self.best_effort_qos
        )
        
        # Rover state data
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
        
        logger.info('Initialized analysis subscribers')
    
    def _init_publishers(self):
        """Initialize ROS2 publishers for analysis results"""
        # Scientific observations
        self.science_obs_pub = self.create_publisher(
            String,
            '/rover/science/observations',
            qos_profile=self.reliable_qos
        )
        
        # Mission recommendations
        self.mission_rec_pub = self.create_publisher(
            String,
            '/rover/mission/recommendations',
            qos_profile=self.reliable_qos
        )
        
        # Terrain analysis
        self.terrain_analysis_pub = self.create_publisher(
            String,
            '/rover/analysis/terrain',
            qos_profile=self.reliable_qos
        )
        
        # Navigation recommendations
        self.nav_rec_pub = self.create_publisher(
            Path,
            '/rover/navigation/recommended_path',
            qos_profile=self.reliable_qos
        )
        
        # Mission status
        self.mission_status_pub = self.create_publisher(
            String,
            '/rover/mission/status',
            qos_profile=self.reliable_qos
        )
        
        # Science targets
        self.science_targets_pub = self.create_publisher(
            String,
            '/rover/science/targets',
            qos_profile=self.reliable_qos
        )
        
        # Performance metrics
        self.performance_pub = self.create_publisher(
            String,
            '/rover/analysis/performance',
            qos_profile=self.reliable_qos
        )
        
        logger.info('Initialized analysis publishers')
    
    def _hazards_callback(self, msg: String):
        """Handle hazard detection data"""
        try:
            hazards_data = json.loads(msg.data)
            
            with self.data_lock:
                self.latest_analysis['hazards'] = hazards_data
                self.latest_analysis['timestamp'] = time.time()
                
        except Exception as e:
            logger.error(f'Error processing hazards: {e}')
    
    def _traversability_callback(self, msg: Float32MultiArray):
        """Handle traversability data"""
        try:
            traversability_score = msg.data[0] if len(msg.data) > 0 else 0.0
            
            with self.data_lock:
                self.latest_analysis['traversability'] = traversability_score
                
        except Exception as e:
            logger.error(f'Error processing traversability: {e}')
    
    def _segmentation_callback(self, msg: Image):
        """Handle terrain segmentation data"""
        try:
            # Store segmentation data for analysis
            with self.data_lock:
                self.latest_analysis['segmentation'] = msg
                
        except Exception as e:
            logger.error(f'Error processing segmentation: {e}')
    
    def _pose_callback(self, msg: PoseStamped):
        """Handle rover pose data"""
        try:
            pose_data = {
                'position': [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
                'orientation': [msg.pose.orientation.x, msg.pose.orientation.y, 
                               msg.pose.orientation.z, msg.pose.orientation.w],
                'timestamp': time.time()
            }
            
            with self.data_lock:
                self.latest_analysis['rover_pose'] = pose_data
                
        except Exception as e:
            logger.error(f'Error processing pose: {e}')
    
    def _velocity_callback(self, msg: Twist):
        """Handle rover velocity data"""
        try:
            velocity_data = {
                'linear': [msg.linear.x, msg.linear.y, msg.linear.z],
                'angular': [msg.angular.x, msg.angular.y, msg.angular.z],
                'timestamp': time.time()
            }
            
            with self.data_lock:
                self.latest_analysis['rover_velocity'] = velocity_data
                
        except Exception as e:
            logger.error(f'Error processing velocity: {e}')
    
    def _perform_comprehensive_analysis(self):
        """Main analysis callback - performs comprehensive analysis"""
        try:
            with self.data_lock:
                current_analysis = self.latest_analysis.copy()
            
            if not current_analysis['hazards'] and current_analysis['traversability'] == 0:
                return  # No data to analyze
            
            # Perform different types of analysis
            self._analyze_terrain_features(current_analysis)
            self._generate_science_observations(current_analysis)
            self._generate_mission_recommendations(current_analysis)
            self._analyze_navigation_options(current_analysis)
            self._update_mission_status()
            
        except Exception as e:
            logger.error(f'Error in comprehensive analysis: {e}')
    
    def _analyze_terrain_features(self, analysis: Dict[str, Any]):
        """Analyze terrain features and generate terrain analysis"""
        try:
            hazards = analysis.get('hazards', [])
            traversability = analysis.get('traversability', 0.0)
            pose = analysis.get('rover_pose', {})
            
            # Calculate terrain statistics
            hazard_count = len(hazards)
            critical_hazards = len([h for h in hazards if h.get('severity') in ['HIGH', 'CRITICAL']])
            
            # Determine dominant terrain type from hazards
            terrain_types = [h.get('type', 'unknown') for h in hazards]
            dominant_terrain = self._get_dominant_terrain_type(terrain_types)
            
            # Calculate slope analysis
            slope_analysis = self._analyze_slopes(hazards)
            
            # Estimate surface roughness
            surface_roughness = self._estimate_surface_roughness(hazards)
            
            # Generate recommended path
            recommended_path = self._generate_recommended_path(pose, hazards, traversability)
            alternative_paths = self._generate_alternative_paths(pose, hazards)
            
            # Create terrain analysis
            terrain_analysis = TerrainAnalysis(
                timestamp=time.time(),
                traversability_score=traversability,
                hazard_count=hazard_count,
                critical_hazards=critical_hazards,
                dominant_terrain_type=dominant_terrain,
                slope_analysis=slope_analysis,
                surface_roughness=surface_roughness,
                recommended_path=recommended_path,
                alternative_paths=alternative_paths
            )
            
            # Store and publish
            self.terrain_history.append(terrain_analysis)
            self._publish_terrain_analysis(terrain_analysis)
            
        except Exception as e:
            logger.error(f'Error analyzing terrain features: {e}')
    
    def _generate_science_observations(self, analysis: Dict[str, Any]):
        """Generate scientific observations from current data"""
        try:
            hazards = analysis.get('hazards', [])
            pose = analysis.get('rover_pose', {})
            traversability = analysis.get('traversability', 0.0)
            
            observations = []
            
            # Analyze geological features
            geological_obs = self._analyze_geological_features(hazards, pose)
            if geological_obs:
                observations.append(geological_obs)
            
            # Analyze surface composition
            composition_obs = self._analyze_surface_composition(hazards, traversability)
            if composition_obs:
                observations.append(composition_obs)
            
            # Analyze terrain stability
            stability_obs = self._analyze_terrain_stability(hazards, traversability)
            if stability_obs:
                observations.append(stability_obs)
            
            # Publish observations
            for obs in observations:
                self.science_observations.append(obs)
                self._publish_science_observation(obs)
                
        except Exception as e:
            logger.error(f'Error generating science observations: {e}')
    
    def _generate_mission_recommendations(self, analysis: Dict[str, Any]):
        """Generate mission planning recommendations"""
        try:
            hazards = analysis.get('hazards', [])
            traversability = analysis.get('traversability', 0.0)
            pose = analysis.get('rover_pose', {})
            
            recommendations = []
            
            # Navigation recommendations
            nav_rec = self._generate_navigation_recommendation(hazards, traversability)
            if nav_rec:
                recommendations.append(nav_rec)
            
            # Science recommendations
            science_rec = self._generate_science_recommendation(hazards, pose)
            if science_rec:
                recommendations.append(science_rec)
            
            # Safety recommendations
            safety_rec = self._generate_safety_recommendation(hazards, traversability)
            if safety_rec:
                recommendations.append(safety_rec)
            
            # Efficiency recommendations
            efficiency_rec = self._generate_efficiency_recommendation(analysis)
            if efficiency_rec:
                recommendations.append(efficiency_rec)
            
            # Publish recommendations
            for rec in recommendations:
                self.mission_recommendations.append(rec)
                self._publish_mission_recommendation(rec)
                
        except Exception as e:
            logger.error(f'Error generating mission recommendations: {e}')
    
    def _analyze_navigation_options(self, analysis: Dict[str, Any]):
        """Analyze navigation options and publish recommended path"""
        try:
            pose = analysis.get('rover_pose', {})
            hazards = analysis.get('hazards', [])
            traversability = analysis.get('traversability', 0.0)
            
            # Generate optimal path
            optimal_path = self._calculate_optimal_path(pose, hazards, traversability)
            
            # Publish as ROS Path message
            if optimal_path:
                self._publish_navigation_path(optimal_path)
                
        except Exception as e:
            logger.error(f'Error analyzing navigation options: {e}')
    
    def _update_mission_status(self):
        """Update and publish current mission status"""
        try:
            mission_duration = time.time() - self.mission_start_time
            
            status = {
                'mission_phase': self.current_mission_phase,
                'mission_duration': mission_duration,
                'targets_visited': len(self.visited_locations),
                'targets_remaining': len(self.target_locations),
                'current_observations': len(self.science_observations),
                'active_recommendations': len(self.mission_recommendations),
                'terrain_analyses': len(self.terrain_history),
                'timestamp': time.time()
            }
            
            self._publish_mission_status(status)
            
        except Exception as e:
            logger.error(f'Error updating mission status: {e}')
    
    # Helper methods for analysis
    def _get_dominant_terrain_type(self, terrain_types: List[str]) -> str:
        """Determine dominant terrain type"""
        if not terrain_types:
            return 'unknown'
        
        type_counts = {}
        for terrain_type in terrain_types:
            type_counts[terrain_type] = type_counts.get(terrain_type, 0) + 1
        
        return max(type_counts, key=type_counts.get)
    
    def _analyze_slopes(self, hazards: List[Dict]) -> Dict[str, float]:
        """Analyze slope characteristics"""
        slope_hazards = [h for h in hazards if h.get('type') == 'steep_slope']
        
        if not slope_hazards:
            return {'max_slope': 0.0, 'avg_slope': 0.0, 'slope_variance': 0.0}
        
        slopes = []
        for hazard in slope_hazards:
            if 'metadata' in hazard and 'max_slope_deg' in hazard['metadata']:
                slopes.append(hazard['metadata']['max_slope_deg'])
        
        if not slopes:
            return {'max_slope': 0.0, 'avg_slope': 0.0, 'slope_variance': 0.0}
        
        return {
            'max_slope': max(slopes),
            'avg_slope': sum(slopes) / len(slopes),
            'slope_variance': np.var(slopes) if len(slopes) > 1 else 0.0
        }
    
    def _estimate_surface_roughness(self, hazards: List[Dict]) -> float:
        """Estimate surface roughness from hazard data"""
        if not hazards:
            return 0.0
        
        roughness_factors = {
            'large_rock': 0.8,
            'rock_cluster': 0.6,
            'sand_trap': 0.3,
            'steep_slope': 0.7,
            'cliff_edge': 0.9,
            'uncertain_terrain': 0.5,
            'shadow_region': 0.2
        }
        
        total_roughness = 0.0
        for hazard in hazards:
            hazard_type = hazard.get('type', 'unknown')
            roughness = roughness_factors.get(hazard_type, 0.5)
            area = hazard.get('area', 0)
            total_roughness += roughness * (area / 10000.0)  # Normalize by area
        
        return min(1.0, total_roughness)
    
    def _generate_recommended_path(self, pose: Dict, hazards: List[Dict], 
                                  traversability: float) -> List[Tuple[float, float]]:
        """Generate recommended navigation path"""
        try:
            if not pose or 'position' not in pose:
                return []
            
            current_pos = pose['position'][:2]  # x, y
            
            # Simple path generation - avoid hazards
            path = [current_pos]
            
            # Add waypoints that avoid critical hazards
            for hazard in hazards:
                if hazard.get('severity') in ['HIGH', 'CRITICAL']:
                    # Calculate avoidance waypoint
                    hazard_pos = hazard.get('position', (0, 0))
                    if isinstance(hazard_pos, list) and len(hazard_pos) >= 2:
                        # Offset waypoint to avoid hazard
                        offset_distance = 2.0  # meters
                        angle = math.atan2(hazard_pos[1] - current_pos[1], 
                                         hazard_pos[0] - current_pos[0])
                        
                        avoid_x = current_pos[0] + offset_distance * math.cos(angle + math.pi/2)
                        avoid_y = current_pos[1] + offset_distance * math.sin(angle + math.pi/2)
                        
                        path.append((avoid_x, avoid_y))
            
            return path
            
        except Exception as e:
            logger.error(f'Error generating recommended path: {e}')
            return []
    
    def _generate_alternative_paths(self, pose: Dict, hazards: List[Dict]) -> List[List[Tuple[float, float]]]:
        """Generate alternative navigation paths"""
        # Simplified alternative path generation
        base_path = self._generate_recommended_path(pose, hazards, 0.0)
        
        if not base_path:
            return []
        
        # Generate variations of the base path
        alternatives = []
        
        # Left alternative
        left_path = [(p[0] - 1.0, p[1]) for p in base_path]
        alternatives.append(left_path)
        
        # Right alternative
        right_path = [(p[0] + 1.0, p[1]) for p in base_path]
        alternatives.append(right_path)
        
        return alternatives
    
    def _analyze_geological_features(self, hazards: List[Dict], pose: Dict) -> Optional[ScienceObservation]:
        """Analyze geological features for science observations"""
        try:
            rock_hazards = [h for h in hazards if 'rock' in h.get('type', '').lower()]
            
            if not rock_hazards:
                return None
            
            # Calculate rock density and distribution
            rock_count = len(rock_hazards)
            total_area = sum(h.get('area', 0) for h in rock_hazards)
            
            # Determine observation priority
            if rock_count > 5:
                priority = 3  # High priority for many rocks
            elif rock_count > 2:
                priority = 2  # Medium priority
            else:
                priority = 1  # Low priority
            
            position = pose.get('position', [0, 0, 0])[:3]
            
            observation = ScienceObservation(
                observation_id=f"geo_{self.science_observation_id}",
                timestamp=time.time(),
                location=tuple(position),
                observation_type='geology',
                confidence=min(0.9, 0.5 + (rock_count * 0.1)),
                description=f"Rock field observed: {rock_count} rocks in area",
                measurements={
                    'rock_count': rock_count,
                    'total_rock_area': total_area,
                    'rock_density': rock_count / max(1, total_area / 1000)
                },
                priority=priority
            )
            
            self.science_observation_id += 1
            return observation
            
        except Exception as e:
            logger.error(f'Error analyzing geological features: {e}')
            return None
    
    def _analyze_surface_composition(self, hazards: List[Dict], traversability: float) -> Optional[ScienceObservation]:
        """Analyze surface composition"""
        try:
            sand_hazards = [h for h in hazards if h.get('type') == 'sand_trap']
            
            if not sand_hazards and traversability > 70:
                return None
            
            position = [0, 0, 0]  # Default position
            priority = 2
            
            if sand_hazards:
                description = f"Sandy terrain detected: {len(sand_hazards)} sand traps"
                confidence = 0.8
                measurements = {
                    'sand_trap_count': len(sand_hazards),
                    'avg_sand_trap_area': sum(h.get('area', 0) for h in sand_hazards) / len(sand_hazards),
                    'traversability': traversability
                }
                priority = 3
            else:
                description = f"Firm terrain: traversability {traversability:.1f}%"
                confidence = 0.7
                measurements = {'traversability': traversability}
                priority = 1
            
            observation = ScienceObservation(
                observation_id=f"comp_{self.science_observation_id}",
                timestamp=time.time(),
                location=tuple(position),
                observation_type='geology',
                confidence=confidence,
                description=description,
                measurements=measurements,
                priority=priority
            )
            
            self.science_observation_id += 1
            return observation
            
        except Exception as e:
            logger.error(f'Error analyzing surface composition: {e}')
            return None
    
    def _analyze_terrain_stability(self, hazards: List[Dict], traversability: float) -> Optional[ScienceObservation]:
        """Analyze terrain stability"""
        try:
            slope_hazards = [h for h in hazards if h.get('type') == 'steep_slope']
            
            if not slope_hazards and traversability > 60:
                return None
            
            stability_score = max(0, 100 - traversability)
            
            if stability_score > 40:
                priority = 3
                description = f"Unstable terrain detected: stability {stability_score:.1f}%"
                confidence = 0.8
            elif stability_score > 20:
                priority = 2
                description = f"Moderate terrain stability: {stability_score:.1f}%"
                confidence = 0.6
            else:
                priority = 1
                description = f"Stable terrain: stability {100-stability_score:.1f}%"
                confidence = 0.7
            
            observation = ScienceObservation(
                observation_id=f"stab_{self.science_observation_id}",
                timestamp=time.time(),
                location=(0, 0, 0),  # Default position
                observation_type='physics',
                confidence=confidence,
                description=description,
                measurements={
                    'stability_score': stability_score,
                    'slope_count': len(slope_hazards),
                    'traversability': traversability
                },
                priority=priority
            )
            
            self.science_observation_id += 1
            return observation
            
        except Exception as e:
            logger.error(f'Error analyzing terrain stability: {e}')
            return None
    
    def _generate_navigation_recommendation(self, hazards: List[Dict], traversability: float) -> Optional[MissionRecommendation]:
        """Generate navigation recommendations"""
        try:
            critical_hazards = [h for h in hazards if h.get('severity') == 'CRITICAL']
            
            if critical_hazards:
                return MissionRecommendation(
                    recommendation_id=f"nav_{self.recommendation_id}",
                    timestamp=time.time(),
                    priority=4,  # High priority
                    category='safety',
                    action='Avoid critical hazards immediately',
                    reasoning=f"Found {len(critical_hazards)} critical hazards",
                    estimated_duration=30.0,
                    risk_level='high'
                )
            elif traversability < 30:
                return MissionRecommendation(
                    recommendation_id=f"nav_{self.recommendation_id}",
                    timestamp=time.time(),
                    priority=3,
                    category='navigation',
                    action='Seek alternative route',
                    reasoning=f"Low traversability: {traversability:.1f}%",
                    estimated_duration=60.0,
                    risk_level='medium'
                )
            
            self.recommendation_id += 1
            return None
            
        except Exception as e:
            logger.error(f'Error generating navigation recommendation: {e}')
            return None
    
    def _generate_science_recommendation(self, hazards: List[Dict], pose: Dict) -> Optional[MissionRecommendation]:
        """Generate science recommendations"""
        try:
            rock_hazards = [h for h in hazards if 'rock' in h.get('type', '').lower()]
            
            if rock_hazards and len(rock_hazards) >= 3:
                return MissionRecommendation(
                    recommendation_id=f"sci_{self.recommendation_id}",
                    timestamp=time.time(),
                    priority=2,
                    category='science',
                    action='Investigate rock field',
                    reasoning=f"Interesting geological formation: {len(rock_hazards)} rocks",
                    estimated_duration=300.0,
                    risk_level='low'
                )
            
            self.recommendation_id += 1
            return None
            
        except Exception as e:
            logger.error(f'Error generating science recommendation: {e}')
            return None
    
    def _generate_safety_recommendation(self, hazards: List[Dict], traversability: float) -> Optional[MissionRecommendation]:
        """Generate safety recommendations"""
        try:
            if traversability < 20:
                return MissionRecommendation(
                    recommendation_id=f"safety_{self.recommendation_id}",
                    timestamp=time.time(),
                    priority=5,  # Critical
                    category='safety',
                    action='Stop and reassess mission',
                    reasoning=f"Extremely low traversability: {traversability:.1f}%",
                    estimated_duration=600.0,
                    risk_level='critical'
                )
            
            self.recommendation_id += 1
            return None
            
        except Exception as e:
            logger.error(f'Error generating safety recommendation: {e}')
            return None
    
    def _generate_efficiency_recommendation(self, analysis: Dict[str, Any]) -> Optional[MissionRecommendation]:
        """Generate efficiency recommendations"""
        try:
            # Simple efficiency recommendation based on current state
            if analysis.get('traversability', 0) > 80:
                return MissionRecommendation(
                    recommendation_id=f"eff_{self.recommendation_id}",
                    timestamp=time.time(),
                    priority=1,
                    category='efficiency',
                    action='Increase exploration speed',
                    reasoning="Good terrain conditions for faster movement",
                    estimated_duration=0.0,
                    risk_level='low'
                )
            
            self.recommendation_id += 1
            return None
            
        except Exception as e:
            logger.error(f'Error generating efficiency recommendation: {e}')
            return None
    
    def _calculate_optimal_path(self, pose: Dict, hazards: List[Dict], traversability: float) -> Optional[List[Tuple[float, float]]]:
        """Calculate optimal navigation path"""
        try:
            return self._generate_recommended_path(pose, hazards, traversability)
        except Exception as e:
            logger.error(f'Error calculating optimal path: {e}')
            return None
    
    # Publishing methods
    def _publish_terrain_analysis(self, terrain_analysis: TerrainAnalysis):
        """Publish terrain analysis"""
        try:
            msg = String()
            msg.data = json.dumps(asdict(terrain_analysis))
            self.terrain_analysis_pub.publish(msg)
            
        except Exception as e:
            logger.error(f'Error publishing terrain analysis: {e}')
    
    def _publish_science_observation(self, observation: ScienceObservation):
        """Publish science observation"""
        try:
            msg = String()
            msg.data = json.dumps(asdict(observation))
            self.science_obs_pub.publish(msg)
            
        except Exception as e:
            logger.error(f'Error publishing science observation: {e}')
    
    def _publish_mission_recommendation(self, recommendation: MissionRecommendation):
        """Publish mission recommendation"""
        try:
            msg = String()
            msg.data = json.dumps(asdict(recommendation))
            self.mission_rec_pub.publish(msg)
            
        except Exception as e:
            logger.error(f'Error publishing mission recommendation: {e}')
    
    def _publish_navigation_path(self, path: List[Tuple[float, float]]):
        """Publish navigation path as ROS Path message"""
        try:
            if not path:
                return
            
            path_msg = Path()
            path_msg.header.stamp = self.get_clock().now().to_msg()
            path_msg.header.frame_id = 'map'
            
            for x, y in path:
                pose_stamped = PoseStamped()
                pose_stamped.header = path_msg.header
                pose_stamped.pose.position = Point(x=x, y=y, z=0.0)
                pose_stamped.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
                path_msg.poses.append(pose_stamped)
            
            self.nav_rec_pub.publish(path_msg)
            
        except Exception as e:
            logger.error(f'Error publishing navigation path: {e}')
    
    def _publish_mission_status(self, status: Dict[str, Any]):
        """Publish mission status"""
        try:
            msg = String()
            msg.data = json.dumps(status)
            self.mission_status_pub.publish(msg)
            
        except Exception as e:
            logger.error(f'Error publishing mission status: {e}')


def main(args=None):
    """Main function"""
    rclpy.init(args=args)
    
    try:
        node = MarsRoverAnalysisPublisher()
        
        logger.info('Starting Mars Rover Analysis Publisher...')
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
