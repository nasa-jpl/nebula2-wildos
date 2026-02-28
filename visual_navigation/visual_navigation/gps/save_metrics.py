"""
ROS2 Node for GPS Visualization
"""

import rclpy
from rclpy.node import Node

from gps_visualization.msg import NavSatFixArray
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import Odometry, Path as PathMsg
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image as ImageMsg
import os
import json
import numpy as np

class RunMetricsNode(Node):
    def __init__(self):
        super().__init__('run_metrics_node')
        self.declare_parameter('save_path', 'metrics_path.json')

        # self.odom_sub = Subscriber(self, Odometry, "/spot1/odom", qos_profile=10)
        # # self.model_viz_sub = Subscriber(self, ImageMsg, "/spot1/model_visualization", qos_profile=10)
        # # self.model_viz_sub = Subscriber(self, PathMsg, "/spot1/graphnav_planner/path", qos_profile=10)
        # self.model_viz_sub = Subscriber(self, PoseStamped, "/spot1/goal_pose", qos_profile=10)
        # ts_subs = [self.odom_sub, self.model_viz_sub]
        # self.ts = ApproximateTimeSynchronizer(
        #     ts_subs, queue_size=10, slop=0.1
        # )
        # self.ts.registerCallback(self.listener_callback)
        self.odom_sub = self.create_subscription(
            Odometry,
            '/spot1/odom',
            self.listener_callback,
            10
        )
        self.save_path = self.get_parameter('save_path').value

        self.metrics = {
            "time": [],
            "distance": [],
            "fps": []
        }
        self.start_time = None
        self.prev_position = None
        self.total_distance = 0.0
        self.frame_count = 0

        if not os.path.exists(self.save_path):
            with open(self.save_path, 'w') as f:
                json.dump(self.metrics, f)

    def listener_callback(self, msg_odom):
        stamp = msg_odom.header.stamp.sec + msg_odom.header.stamp.nanosec * 1e-9
        if self.start_time is None:
            self.start_time = stamp
        elapsed_time = stamp - self.start_time
        self.frame_count += 1
        current_position = np.array([
            msg_odom.pose.pose.position.x,
            msg_odom.pose.pose.position.y,
            msg_odom.pose.pose.position.z
        ])
        if self.prev_position is not None:
            self.total_distance += np.linalg.norm(current_position - self.prev_position)
        self.prev_position = current_position

        self.metrics["time"] = [elapsed_time]
        self.metrics["distance"] = [self.total_distance]
        self.metrics["fps"] = [self.frame_count / elapsed_time if elapsed_time > 0 else 0]

        self.get_logger().info(f"Time: {elapsed_time:.2f}s, Distance: {self.total_distance:.2f}m, FPS: {self.metrics['fps'][-1]:.2f}")
        self.save_metrics()

    def save_metrics(self):
        with open(self.save_path, 'w') as f:
            json.dump(self.metrics, f)
        self.get_logger().info(f"Saved metrics to {self.save_path}")


def main(args=None):
    rclpy.init(args=args)
    run_metrics_node = RunMetricsNode()
    rclpy.spin(run_metrics_node)

    # Destroy the node explicitly
    run_metrics_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()