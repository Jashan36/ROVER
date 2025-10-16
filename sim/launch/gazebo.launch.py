# sim/launch/gazebo.launch.py
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_dir = get_package_share_directory('rover_description')
    world_file = os.path.join(pkg_dir, 'worlds', 'mars_terrain.world')
    urdf_file = os.path.join(pkg_dir, 'urdf', 'rover.urdf.xacro')
    
    return LaunchDescription([
        # Gazebo server
        ExecuteProcess(
            cmd=['gazebo', '--verbose', '-s', 'libgazebo_ros_factory.so', world_file],
            output='screen'
        ),
        
        # Robot state publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': Command(['xacro ', urdf_file])}]
        ),
        
        # Spawn robot
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=['-entity', 'mars_rover', '-topic', 'robot_description', '-x', '0', '-y', '0', '-z', '2.0'],
            output='screen'
        ),
    ])