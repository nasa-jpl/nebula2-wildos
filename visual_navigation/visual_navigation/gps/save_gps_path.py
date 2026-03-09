"""
ROS2 Node for GPS Visualization
"""

import rclpy
from rclpy.node import Node

from gps_visualization.msg import NavSatFixArray

import os
import json

class SaveGPSPathNode(Node):
    def __init__(self):
        super().__init__('save_gps_path_node')
        self.declare_parameter('save_path', 'gps_path.json')
        self.gps_coords = []
        self.subscription = self.create_subscription(
            NavSatFixArray,
            '/spot1/gps_path',
            self.listener_callback,
            10
        )
        self.subscription  # prevent unused variable warning
        self.save_path = self.get_parameter('save_path').value

        if not os.path.exists(self.save_path):
            with open(self.save_path, 'w') as f:
                json.dump([], f)

    def listener_callback(self, msg):
        self.gps_coords = [(fix.latitude, fix.longitude) for fix in msg.gps_fixes]
        self.save_gps_path()

    def save_gps_path(self):
        with open(self.save_path, 'w') as f:
            json.dump(self.gps_coords, f)
        self.get_logger().info(f"Saved GPS path to {self.save_path}")


def main(args=None):
    rclpy.init(args=args)
    save_gps_path_node = SaveGPSPathNode()
    rclpy.spin(save_gps_path_node)

    # Destroy the node explicitly
    save_gps_path_node.destroy_node()
    rclpy.shutdown() 

if __name__ == '__main__':
    main()