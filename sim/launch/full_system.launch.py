# sim/launch/full_system.launch.py
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_dir = get_package_share_directory('rover_description')
    
    return LaunchDescription([
        # 1. Gazebo simulation with rover
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_dir, 'launch', 'gazebo.launch.py')
            )
        ),
        
        # 2. Nav2 navigation stack
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_dir, 'launch', 'nav2.launch.py')
            )
        ),
        
        # 3. AI Vision Node (Real-time terrain detection)
        Node(
            package='rover_integration',
            executable='ros_ai_bridge.py',
            name='rover_vision',
            output='screen',
            parameters=[{
                'inference_rate': 5.0,
                'confidence_threshold': 0.7,
                'publish_overlay': True
            }]
        ),
        
        # 4. Image viewer for debugging
        Node(
            package='rqt_image_view',
            executable='rqt_image_view',
            name='image_viewer',
            arguments=['/terrain/overlay']
        ),
        
        # 5. Dashboard data publisher
        Node(
            package='rover_integration',
            executable='dashboard_bridge.py',
            name='dashboard_bridge',
            output='screen'
        ),
    ])