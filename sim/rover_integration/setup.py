from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'rover_integration'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Install launch files
        (os.path.join('share', package_name, 'launch'), 
         glob('launch/*.launch.py')),
        # Install config files
        (os.path.join('share', package_name, 'config'), 
         glob('config/*.yaml')),
        # Install resource files
        (os.path.join('share', package_name, 'resource'), 
         glob('resource/*')),
    ],
    install_requires=[
        'setuptools',
        'numpy>=1.24.0',
        'opencv-python>=4.7.0',
        'pillow>=9.5.0',
        'pyyaml>=6.0',
        'scipy>=1.10.0',
    ],
    zip_safe=False,
    maintainer='Mars Rover Team',
    maintainer_email='dev@marsrover.ai',
    description='ROS2-AI Integration for Mars Rover Real-time Perception and Navigation',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # Main bridge node
            'ros_ai_bridge = rover_integration.ros_ai_bridge:main',
            
            # Individual component nodes (for testing/debugging)
            'image_subscriber = rover_integration.image_subscriber:main',
            'analysis_publisher = rover_integration.analysis_publisher:main',
            'dashboard_bridge = rover_integration.dashboard_bridge:main',
            
            # Utility nodes
            'bridge_status = rover_integration.utils:main_status_monitor',
        ],
    },
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Robotics',
    ],
    python_requires='>=3.8',
)