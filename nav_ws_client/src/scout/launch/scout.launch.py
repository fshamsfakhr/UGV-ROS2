from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    return LaunchDescription([
        Node(
            package='scout',
            executable='localisation.py',
            name='scout',
            output='screen',
        )
    ])

