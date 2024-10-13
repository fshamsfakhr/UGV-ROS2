#!/usr/bin/env python3

import rclpy
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64MultiArray
import math
from statsmodels.nonparametric.smoothers_lowess import lowess
from utils_state_est import *
from ament_index_python.packages import get_package_share_directory
import os

class SensorDataSubscriber(Node):
    def __init__(self):
        super().__init__('sensor_data_subscriber')
        # Publisher for EKF_Pose
        self.ekf_pose_publisher = self.create_publisher(Float64MultiArray, '/EKF_Pose', 10)
        # Publisher for trajectory points
        self.trajectory_points_publisher = self.create_publisher(Float64MultiArray, '/trajectory_points', 10)
        # Publisher for trajectory inputs
        self.trajectory_inputs_publisher = self.create_publisher(Float64MultiArray, '/trajectory_inputs', 10)
        # Create a publisher for the /cmd_vel topic
        self.vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        # Subscriber for UWB Position
        self.uwb_subscriber = self.create_subscription(
            Pose,
            '/UWB/Pos',
            self.uwb_callback,
            10
        )

        # Subscriber for IMU Data
        self.imu_subscriber = self.create_subscription(
            Imu,
            '/imu_data',
            self.imu_callback,
            10
        )
        # Get the path to the current package
        package_path = get_package_share_directory('scout')

        # Dynamically construct the file paths
        wp_file = os.path.join(package_path, 'config', 'WP1.txt')
        #map_file = os.path.join(package_path, 'config', 'map.txt')
        # Set the desired processing frequency (e.g., 5 Hz)
        self.freq = 10.0
        self.timer = self.create_timer(1.0 / self.freq, self.process_data)
              
        # Initialize data containers
        self.uwb_data = None
        self.imu_data = None

        # Variables for operations
        self.uwb_coords = np.empty((0, 3))
        self.S_EKF = np.empty((0, 3))
        self.ss_k = np.zeros(3)
        self.ss_k[2] = -1.4
        self.P = np.eye(3)
        self.Q = np.diag([0.1**2, 0.02**2])
        self.R = np.diag([0.05**2, 0.05**2])
        self.i = 0
        self.n = 30
        self.pivot = 10
        self.cc = 0
        self.U = np.zeros((self.n, 2))   # Initialize U
        self.best_s = np.zeros((self.n, 2))  # Initialize best_s with the same shape as U or another suitable default

        # Load map points
        # self.map_coords, self.x_min, self.x_max, self.y_min, self.y_max = read_points(map_file)
        # self.x_map, self.y_map = zip(*self.map_coords)
        # self.map = np.array(self.map_coords)
        
        # Load waypoint points
        self.wp_coords, _, _, _, _ = read_points(wp_file) 
        self.x_wp, self.y_wp = zip(*self.wp_coords)
        self.path = np.array(self.wp_coords)
  

        x, y, z = np.meshgrid([0.3, 0.4], np.sort(np.concatenate([np.linspace(-0.3, 0.3, 10), [0]])), [0, 1], indexing='ij')

        self.prms = {
            'InputMat': np.column_stack((x.ravel(), y.ravel(), z.ravel())),  # Missing comma was here
            'n': 30,  # Number of steps
            'r': 3.0,  # Radius for subpath
            'dk': 0.2,  # Time step
            'step': 4  # Subsampling step
        }

    def uwb_callback(self, msg):
        self.uwb_data = {
            'position': (msg.position.x, msg.position.y, msg.position.z),
            'orientation': (msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w)
        }

    def imu_callback(self, msg):
        self.imu_data = {
            'orientation': (msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w),
            'angular_velocity': (msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z),
            'linear_acceleration': (msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z)
        }

    def process_data(self):
        if self.imu_data and self.uwb_data:
            if self.uwb_data is not None and 'position' in self.uwb_data:
                uwb_position_data = np.array(self.uwb_data['position'])
            else:
                uwb_position_data = [np.inf, np.inf, np.inf]

            if self.imu_data is not None and 'angular_velocity' in self.imu_data:
                angular_velocity = self.imu_data['angular_velocity']
                W_gyro = np.deg2rad(angular_velocity[2] * 0.03)
            else:
                W_gyro = np.inf
            self.uwb_coords = np.vstack([self.uwb_coords, uwb_position_data])
            z_k1 = uwb_position_data[0:2]
            if not np.isinf(sum(z_k1)):
                self.ss_k, self.P = ekf_fun(self.ss_k, z_k1, .4, W_gyro, self.P, self.Q, self.R, self.prms['dk'])
                self.S_EKF = np.vstack([self.S_EKF, self.ss_k])
                #self.get_logger().info(f'self.ss_k : {self.ss_k}')
                if self.i % 10 == 0:
                    self.cc = -1
                    self.best_s, self.U, cost = traj_optimization(self.path, self.ss_k, self.prms)
                    #if np.isinf(cost) or (np.linalg.norm(self.ss_k[:2] - self.path[-1, :]) < 0.001):
                        #break

		    # Prepare and publish trajectory points
                    trajectory_points_msg = Float64MultiArray()
                    trajectory_points_msg.data = self.best_s.flatten().tolist()  # Flatten best_s
                    self.trajectory_points_publisher.publish(trajectory_points_msg)

                    # Prepare and publish trajectory inputs
                    trajectory_inputs_msg = Float64MultiArray()
                    trajectory_inputs_msg.data = self.U.flatten().tolist()  # Flatten U
                    self.trajectory_inputs_publisher.publish(trajectory_inputs_msg)
                # publish velocities
                self.cc += 1 
                msg = Twist()
                msg.linear.x = self.U[self.cc, 0]  
                msg.angular.z = self.U[self.cc, 1] 
                self.vel_publisher.publish(msg)
                # Prepare and publish EKF pose data
                ekf_pose_msg = Float64MultiArray()
                ekf_pose_msg.data = self.S_EKF[self.i, 0:3].tolist()
                self.ekf_pose_publisher.publish(ekf_pose_msg)
                self.i += 1
                
            self.imu_data = None
            self.uwb_data = None
            self.gps_data = None


def main(args=None):
    rclpy.init(args=args)
    sensor_data_subscriber = SensorDataSubscriber()

    try:
        rclpy.spin(sensor_data_subscriber)
    except KeyboardInterrupt:
        pass

    sensor_data_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

