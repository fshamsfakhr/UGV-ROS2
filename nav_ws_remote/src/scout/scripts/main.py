#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import matplotlib.pyplot as plt
import numpy as np
from utils_state_est import *
from ament_index_python.packages import get_package_share_directory
import os

class UWBPosPlotter(Node):
    def __init__(self):
        super().__init__('uwb_pos_plotter')
        self.ekf_subscription = self.create_subscription(
            Float64MultiArray,
            '/EKF_Pose',
            self.listener_callback,
            10  # QoS (Quality of Service) depth
        )
        self.position_data = []
        self.trajectory_points = np.empty((0, 3))
        self.trajectory_inputs = np.empty((0, 2))
        # Subscribers for trajectory points and inputs
        self.trajectory_points_subscriber = self.create_subscription(
            Float64MultiArray,
            '/trajectory_points',
            self.trajectory_points_callback,
            10
        )
        
        self.trajectory_inputs_subscriber = self.create_subscription(
            Float64MultiArray,
            '/trajectory_inputs',
            self.trajectory_inputs_callback,
            10
        )
        # Get the path to the current package
        package_path = get_package_share_directory('scout')

        # Dynamically construct the file paths
        wp_file = os.path.join(package_path, 'config', 'WP1.txt')
        #map_file = os.path.join(package_path, 'config', 'map.txt')
        
        ## Load map points
        #self.map_coords, self.x_min, self.x_max, self.y_min, self.y_max = read_points(map_file)
        #self.x_map, self.y_map = zip(*self.map_coords)
        
        # Load waypoint points
        self.wp_coords, self.x_min, self.x_max, self.y_min, self.y_max = read_points(wp_file)  # You can use different min/max if needed
        self.x_wp, self.y_wp = zip(*self.wp_coords)

    def listener_callback(self, msg):
        x = msg.data[0]
        y = msg.data[1]
        theta = msg.data[2]
        self.position_data.append((x, y))
        self.update_plot(theta)  # Pass the latest theta value

    def trajectory_points_callback(self, msg):
        # Reshape the data into (n)x3
        n = len(msg.data) // 3  # Calculate n
        self.trajectory_points = np.array(msg.data).reshape(n, 3)
        #self.get_logger().info(f'Received trajectory points: {trajectory_points}')

    def trajectory_inputs_callback(self, msg):
        # Reshape the data into (n)x3
        n = len(msg.data) // 2  # Calculate n
        self.trajectory_inputs = np.array(msg.data).reshape(n, 2)
        #self.get_logger().info(f'Received trajectory inputs: {trajectory_inputs}')

        
    def update_plot(self, heading_i):
        plt.clf()  # Clear the current figure
    
        ## Plot map coordinates
        #plt.scatter(self.x_map, self.y_map, c='blue', s=5, label='Map Points')
    
        # Plot waypoint coordinates
        plt.scatter(self.x_wp, self.y_wp, c='red', s=5, label='Waypoints')

        # Plot trajectory points
        if self.trajectory_points.size > 0:
            x_tp, y_tp = self.trajectory_points[:, 0], self.trajectory_points[:, 1]  # Extract first two columns
            plt.scatter(x_tp, y_tp, c='blue', s=5, label='Trajectory Points')
        
        # Plot EKF position data
        if self.position_data:
            x_data, y_data = zip(*self.position_data)
            plt.scatter(x_data, y_data, c='green', s=5, label='Filter Points')
        
            # Plot quiver for the latest heading
            plt.quiver(x_data[-1], y_data[-1], np.cos(heading_i), np.sin(heading_i),
                       angles='xy', scale_units='xy', scale=1, color='purple', label='Heading')

        plt.xlim(self.x_min-5, self.x_max+5)
        plt.ylim(self.y_min-5, self.y_max+5)
        plt.title("UWB Position Data")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
    
        plt.grid(True)
        plt.gca().set_aspect('equal', adjustable='box')
    
        plt.legend()
        plt.pause(0.01)



def main(args=None):
    rclpy.init(args=args)
    plotter = UWBPosPlotter()

    plt.ion()  # Turn on interactive mode
    plt.figure()

    try:
        rclpy.spin(plotter)
    except KeyboardInterrupt:
        pass
    finally:
        plotter.destroy_node()
        rclpy.shutdown()

    plt.ioff()  # Turn off interactive mode
    plt.show()  # Show the plot window at the end

if __name__ == '__main__':
    main()

