#!/usr/bin/env python3
# Simple ROS2 goal publisher stub (requires rclpy + action msgs)

import time

if __name__ == '__main__':
    print('This is a stub for publishing navigation goals to Nav2')
    for i in range(3):
        print(f'Would publish goal {i+1}')
        time.sleep(1)
