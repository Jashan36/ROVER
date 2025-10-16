#!/usr/bin/env python3

from setuptools import setup, find_packages
import os
from glob import glob

# Read requirements
with open('requirements.txt', 'r') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read README for long description
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="space_rover",
    version="1.0.0",
    description="Autonomous Mars Rover with Real-time AI Perception",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Mars Rover AI Team",
    author_email="vengalathurjashan@gmail.com",
    url="https://github.com/your-org/space_rover",
    
    # Package discovery
    packages=find_packages(include=['ai', 'integration', 'dashboard', 'scripts']),
    include_package_data=True,
    
    # Python version requirement
    python_requires='>=3.10',
    
    # Dependencies
    install_requires=requirements,
    
    # Entry points for command-line tools
    entry_points={
        'console_scripts': [
            # AI Pipeline
            'rover-ai-train=ai.training.train_segmentation:main',
            'rover-ai-infer=ai.inference.realtime_pipeline:main',
            
            # ROS2 Integration
            'rover-bridge=integration.ros_ai_bridge:main',
            'rover-dashboard-bridge=integration.dashboard_bridge:main',
            
            # Utility scripts
            'rover-download-data=scripts.download_data:main',
            'rover-system-check=scripts.system_check:main',
        ],
    },
    
    # Package data
    package_data={
        'ai': ['config/*.yaml', 'models/weights/*.pth'],
        'dashboard': ['assets/*', 'config/*.yaml'],
        'integration': ['config/*.yaml'],
    },
    
    # Data files
    data_files=[
        # ROS2 package files
        ('share/ament_index/resource_index/packages', ['resource/space_rover']),
        ('share/space_rover', ['package.xml']),
        
        # Launch files
        ('share/space_rover/launch', glob('sim/rover_description/launch/*.launch.py')),
        
        # Config files
        ('share/space_rover/config', glob('sim/rover_description/config/*.yaml') + 
                                   glob('sim/rover_description/config/*.lua')),
        
        # World files
        ('share/space_rover/worlds', glob('sim/rover_description/worlds/*')),
        
        # URDF files
        ('share/space_rover/urdf', glob('sim/rover_description/urdf/*.xacro')),
    ],
    
    # Classifiers
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Robotics',
    ],
    
    # Keywords
    keywords='mars rover ai robotics autonomous-navigation ros2 gazebo',
    
    # Project URLs
    project_urls={
        'Documentation': 'https://github.com/your-org/space_rover/docs',
        'Source': 'https://github.com/your-org/space_rover',
        'Tracker': 'https://github.com/your-org/space_rover/issues',
    },
    
    # License
    license='MIT',
    
    # Options
    zip_safe=False,
)

print("🚀 Space Rover AI Package Setup Complete!")
print("📦 Available commands:")
print("   - rover-ai-train: Train AI models")
print("   - rover-ai-infer: Run AI inference")
print("   - rover-bridge: Start ROS2-AI bridge")
print("   - rover-dashboard-bridge: Start dashboard bridge")
print("   - rover-download-data: Download datasets")
print("   - rover-system-check: System health check")