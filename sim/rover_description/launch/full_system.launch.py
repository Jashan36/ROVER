from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.conditions import IfCondition


def generate_launch_description():
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    launch_gazebo = LaunchConfiguration('launch_gazebo', default='true')
    launch_nav2 = LaunchConfiguration('launch_nav2', default='true')
    launch_ai = LaunchConfiguration('launch_ai', default='true')
    slam = LaunchConfiguration('slam', default='true')
    
    # Get package share directories
    rover_desc_share = FindPackageShare('rover_description')
    integration_share = FindPackageShare('rover_integration')
    
    # Gazebo launch
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([rover_desc_share, 'launch', 'gazebo.launch.py'])
        ]),
        condition=IfCondition(launch_gazebo)
    )
    
    # Navigation launch
    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([rover_desc_share, 'launch', 'nav2.launch.py'])
        ]),
        launch_arguments={
            'slam': slam,
        }.items(),
        condition=IfCondition(launch_nav2)
    )
    
    # AI integration launch
    ai_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([integration_share, 'launch', 'ros_ai_bridge.launch.py'])
        ]),
        condition=IfCondition(launch_ai)
    )
    
    # Goal publisher (for automated navigation)
    goal_publisher = Node(
        package='rover_description',
        executable='goal_publisher.py',
        name='goal_publisher',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )
    
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation clock'
        ),
        
        DeclareLaunchArgument(
            'launch_gazebo',
            default_value='true',
            description='Launch Gazebo simulation'
        ),
        
        DeclareLaunchArgument(
            'launch_nav2',
            default_value='true', 
            description='Launch Nav2 navigation'
        ),
        
        DeclareLaunchArgument(
            'launch_ai',
            default_value='true',
            description='Launch AI integration'
        ),
        
        DeclareLaunchArgument(
            'slam',
            default_value='true',
            description='Run SLAM for mapping'
        ),
        
        LogInfo(msg="🚀 Starting Mars Rover AI Full System"),
        
        gazebo_launch,
        nav2_launch, 
        ai_launch,
        goal_publisher,
        
        LogInfo(msg="✅ Mars Rover AI Full System launched successfully"),
        LogInfo(msg="📊 Access points:"),
        LogInfo(msg="   - Gazebo: Simulation environment"),
        LogInfo(msg="   - RViz2: Navigation visualization"), 
        LogInfo(msg="   - Streamlit: http://localhost:8501"),
        LogInfo(msg="   - ROS2 Topics: ros2 topic list"),
    ])