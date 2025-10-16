from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.actions import ExecuteProcess


def generate_launch_description():
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    params_file = LaunchConfiguration('params_file')
    slam = LaunchConfiguration('slam', default='false')
    
    # Get package share directories
    pkg_share = FindPackageShare('rover_description')
    nav2_bringup_share = FindPackageShare('nav2_bringup')
    
    # Default params file
    default_params_file = PathJoinSubstitution([
        pkg_share, 'config', 'nav2_params.yaml'
    ])
    
    # Nav2 bringup
    nav2_bringup_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                nav2_bringup_share,
                'launch',
                'navigation_launch.py'
            ])
        ]),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'params_file': params_file,
            'slam': slam,
        }.items()
    )
    
    # SLAM Toolbox (if slam is true)
    slam_toolbox_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('slam_toolbox'),
                'launch',
                'online_async_launch.py'
            ])
        ]),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'params_file': PathJoinSubstitution([
                pkg_share, 'config', 'slam_toolbox_params.yaml'
            ]),
        }.items(),
        condition=IfCondition(slam)
    )
    
    # Cartographer (alternative SLAM)
    cartographer_node = Node(
        package='cartographer_ros',
        executable='cartographer_node',
        name='cartographer_node',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        arguments=[
            '-configuration_directory', PathJoinSubstitution([pkg_share, 'config']),
            '-configuration_basename', 'cartographer.lua'
        ],
        condition=IfCondition(slam)
    )
    
    # EKF for sensor fusion
    ekf_filter_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[PathJoinSubstitution([pkg_share, 'config', 'ekf.yaml'])],
        remappings=[('odometry/filtered', 'odom')]
    )
    
    # RViz2
    rviz2_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', PathJoinSubstitution([pkg_share, 'rviz', 'nav2.rviz'])],
        parameters=[{'use_sim_time': use_sim_time}]
    )
    
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation (Gazebo) clock if true'
        ),
        
        DeclareLaunchArgument(
            'params_file',
            default_value=default_params_file,
            description='Full path to the Nav2 params file'
        ),
        
        DeclareLaunchArgument(
            'slam',
            default_value='false',
            description='Whether to run SLAM'
        ),
        
        ekf_filter_node,
        nav2_bringup_launch,
        slam_toolbox_launch,
        cartographer_node,
        rviz2_node,
    ])