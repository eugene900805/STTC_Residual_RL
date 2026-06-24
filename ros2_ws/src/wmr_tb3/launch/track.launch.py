"""Launch only the tracking controller node (Gazebo / robot started separately).

    ros2 launch wmr_tb3 track.launch.py
    ros2 launch wmr_tb3 track.launch.py trajectory:=line mode:=nocomp
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('wmr_tb3')
    cfg = os.path.join(pkg, 'config', 'tb3_track.yaml')

    args = [
        DeclareLaunchArgument('trajectory', default_value='circle'),
        DeclareLaunchArgument('mode', default_value='paper'),
        DeclareLaunchArgument('pose_source', default_value='odom'),
    ]
    node = Node(
        package='wmr_tb3',
        executable='tracking_node',
        name='wmr_tracking',
        output='screen',
        parameters=[cfg, {
            'trajectory': LaunchConfiguration('trajectory'),
            'mode': LaunchConfiguration('mode'),
            'pose_source': LaunchConfiguration('pose_source'),
        }],
    )
    return LaunchDescription(args + [node])
