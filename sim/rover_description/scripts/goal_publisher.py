#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
import math
import time


class GoalPublisher(Node):
    """
    Publishes navigation goals for the Mars Rover
    Can be used for automated testing and demonstrations
    """
    
    def __init__(self):
        super().__init__('goal_publisher')
        
        # Action client for navigation
        self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        
        # Wait for action server
        self.get_logger().info("⏳ Waiting for navigate_to_pose action server...")
        self.nav_to_pose_client.wait_for_server()
        self.get_logger().info("✅ Navigation action server available")
        
        # Pre-defined exploration goals (Mars-like coordinates)
        self.exploration_goals = [
            (5.0, 0.0, 0.0),      # Forward 5m
            (5.0, 3.0, 1.57),     # Right 3m, face 90°
            (2.0, 3.0, 3.14),     # Back 3m, face 180°  
            (2.0, 0.0, 0.0),      # Return to start
        ]
        
        self.current_goal_index = 0
        
        # Timer for publishing goals
        self.timer = self.create_timer(30.0, self.publish_next_goal)  # Every 30 seconds
        
        self.get_logger().info("🎯 Goal publisher initialized - starting automated exploration")
    
    def publish_next_goal(self):
        """Publish the next goal in the exploration sequence"""
        if self.current_goal_index >= len(self.exploration_goals):
            self.get_logger().info("✅ Exploration sequence completed")
            self.timer.cancel()
            return
        
        goal_x, goal_y, goal_yaw = self.exploration_goals[self.current_goal_index]
        
        # Create goal pose
        goal_pose = PoseStamped()
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.header.frame_id = 'map'
        
        goal_pose.pose.position.x = goal_x
        goal_pose.pose.position.y = goal_y
        goal_pose.pose.position.z = 0.0
        
        # Convert yaw to quaternion
        goal_pose.pose.orientation.x = 0.0
        goal_pose.pose.orientation.y = 0.0
        goal_pose.pose.orientation.z = math.sin(goal_yaw / 2)
        goal_pose.pose.orientation.w = math.cos(goal_yaw / 2)
        
        # Send goal
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = goal_pose
        
        self.get_logger().info(f"🎯 Sending goal {self.current_goal_index + 1}: ({goal_x:.1f}, {goal_y:.1f}, {math.degrees(goal_yaw):.1f}°)")
        
        # Send goal asynchronously
        self.nav_to_pose_client.send_goal_async(goal_msg)
        
        self.current_goal_index += 1
    
    def send_single_goal(self, x, y, yaw_degrees):
        """Send a single specific goal"""
        goal_pose = PoseStamped()
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.header.frame_id = 'map'
        
        goal_pose.pose.position.x = x
        goal_pose.pose.position.y = y
        goal_pose.pose.position.z = 0.0
        
        yaw_rad = math.radians(yaw_degrees)
        goal_pose.pose.orientation.z = math.sin(yaw_rad / 2)
        goal_pose.pose.orientation.w = math.cos(yaw_rad / 2)
        
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = goal_pose
        
        self.get_logger().info(f"🎯 Sending single goal: ({x:.1f}, {y:.1f}, {yaw_degrees:.1f}°)")
        self.nav_to_pose_client.send_goal_async(goal_msg)


def main(args=None):
    rclpy.init(args=args)
    
    try:
        goal_publisher = GoalPublisher()
        
        # Send initial goal immediately
        goal_publisher.publish_next_goal()
        
        rclpy.spin(goal_publisher)
        
    except KeyboardInterrupt:
        pass
    finally:
        goal_publisher.get_logger().info("🛑 Shutting down goal publisher")
        goal_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()