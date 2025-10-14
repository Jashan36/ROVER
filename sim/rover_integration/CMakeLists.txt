cmake_minimum_required(VERSION 3.5)
project(rover_integration)

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)

ament_python_install_package(${PROJECT_NAME})

ament_package()
