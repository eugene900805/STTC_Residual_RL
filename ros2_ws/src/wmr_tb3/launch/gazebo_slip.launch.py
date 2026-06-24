"""Start Gazebo (empty world, headless gzserver) and spawn a low-friction
TurtleBot3 burger that exhibits real wheel slip, with a ground-truth pose
plugin (/odom_truth).  Run the controller separately with pose_source:=truth.

    ros2 launch wmr_tb3 gazebo_slip.launch.py
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    pkg_gazebo = get_package_share_directory('gazebo_ros')
    pkg_tb3g = get_package_share_directory('turtlebot3_gazebo')
    pkg = get_package_share_directory('wmr_tb3')

    world = os.path.join(pkg_tb3g, 'worlds', 'empty_world.world')
    model = os.path.join(pkg, 'models', 'tb3_burger_slip', 'model.sdf')

    gzserver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo, 'launch', 'gzserver.launch.py')),
        launch_arguments={'world': world}.items())

    spawn = Node(
        package='gazebo_ros', executable='spawn_entity.py',
        arguments=['-entity', 'tb3_slip', '-file', model,
                   '-x', '0.0', '-y', '0.0', '-z', '0.01'],
        output='screen')

    return LaunchDescription([gzserver, spawn])
