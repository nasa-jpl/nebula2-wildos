import rclpy
from rclpy.node import Node
import struct

import numpy as np
from sensor_msgs.msg import PointCloud2
from triangulation3d.pcl_utils import create_colored_pointcloud2
from sensor_msgs_py import point_cloud2 as pc2


class ColorizePointCloud(Node):
    def __init__(self):
        super().__init__('colorize_pointcloud')

        # Set desired color here (0–255)
        self.r = 255
        self.g = 255    
        self.b = 255

        self.subscription = self.create_subscription(
            PointCloud2,
            '/spot1/object_hypotheses',
            self.callback,
            10)

        self.publisher = self.create_publisher(
            PointCloud2,
            '/spot1/object_hypotheses_colored',
            10)

        self.get_logger().info("PointCloud colorizer node started.")

    def callback(self, msg):
        # Convert incoming PointCloud2 to python list of points
        points = list(pc2.read_points(msg, field_names=("x", "y", "z", "rgb"), skip_nans=True))
        new_msg = create_colored_pointcloud2(
            np.array([[pt[0], pt[1], pt[2]] for pt in points]),
            frame_id=msg.header.frame_id,
            color=np.array([self.r, self.g, self.b], dtype=np.uint8)
        )

        # Publish
        self.publisher.publish(new_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ColorizePointCloud()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
