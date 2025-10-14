Simulation folder notes

This folder will contain ROS2 packages, URDF for the rover, Gazebo/Ignition world files, and Nav2 launch files.

Windows notes:
- ROS2 Foxy/Galactic/Humble generally has better support on Ubuntu. For development, we recommend using Ubuntu (WSL2 on Windows) for full Gazebo/Ignition and ROS2 compatibility.
- If using Windows natively, install ROS2 desktop and Ignition as per ROS2 docs. Some Nav2 components may require WSL2 or Linux.

Planned files:
- urdf/rover.urdf.xacro
- launch/sim_launch.py (ROS2 launch)
- worlds/mars_dem.world
