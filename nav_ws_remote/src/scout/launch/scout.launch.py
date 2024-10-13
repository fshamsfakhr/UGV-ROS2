from launch import LaunchDescription
from launch_ros.actions import Node

import os
import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    
    pkg_scout = get_package_share_directory('scout')

    # Process the URDF xacro file
    xacro_file = os.path.join(pkg_scout, 'urdf', 'robot.urdf.xacro')
    urdf = xacro.process_file(xacro_file)
    
    # Include the robot_state_publisher to publish transforms
    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': urdf.toxml()}]
    )

    # Include the Gazebo launch file, provided by the gazebo_ros package
    world_file_path = os.path.join(pkg_scout, 'worlds', 'maze.world') 
    gazebo = IncludeLaunchDescription(
                PythonLaunchDescriptionSource([os.path.join(
                    get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')]),
                launch_arguments={'world': world_file_path}.items()
    )

    # Run the spawner node from the gazebo_ros package to spawn the robot in the simulation
    spawn_entity = Node(
                package='gazebo_ros', 
                executable='spawn_entity.py',
                output='screen',
                arguments=['-topic', 'robot_description',  # The robot description is published by the rsp node on the /robot_description topic
                           '-entity', 'scout'],           # The name of the entity to spawn
    )

    # Add RViz node
    rviz_config_file = os.path.join(pkg_scout, 'config', 'simple_config.rviz')
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file]  # Pass the RViz configuration file as an argument
    )
    return LaunchDescription([
        Node(
            package='scout',
            executable='main.py',
            name='localisation',
            output='screen',
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation/Gazebo clock'
        ),
        rsp,
        gazebo,
        spawn_entity,
        rviz  # Add RViz to the launch description
    ])

