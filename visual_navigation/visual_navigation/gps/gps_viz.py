"""
ROS2 Node for GPS Visualization
"""

import rclpy
from rclpy.node import Node
from rclpy.time import Time

from gps_visualization.msg import NavSatFixArray
from sensor_msgs.msg import NavSatFix, Image as ImageMsg
from nav_msgs.msg import Path

import folium
import io
from PIL import Image
import cv2
import numpy as np

class GPSVizNode(Node):
    def __init__(self):
        super().__init__('gps_viz_node')

        self.gps_coords = []
        self.subscription = self.create_subscription(
            NavSatFixArray,
            '/spot1/gps_path',
            self.listener_callback,
            10
        )
        self.subscription  # prevent unused variable warning
        self.timer = self.create_timer(1.0, self.show_map)
        self.gps_bev_pub = self.create_publisher(
            ImageMsg,
            '/spot1/gps_bev_image',
            qos_profile=rclpy.qos.QoSProfile(
                depth=10,
                reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
                durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
                lifespan=rclpy.duration.Duration(seconds=7)
            )
        )

    def listener_callback(self, msg):
        self.gps_coords = [(fix.latitude, fix.longitude) for fix in msg.gps_fixes]

    def show_map(self):
        if not self.gps_coords:
            self._logger.warning("No GPS data received yet.")
            return
        map = folium.Map(
            location=[self.gps_coords[0][0], self.gps_coords[0][1]],
            zoom_start=28
            # zoom_start=17
        )
        folium.PolyLine(
            locations=self.gps_coords,
            color='blue',
            weight=5,
            opacity=0.7
        ).add_to(map)

        # Save the map to an image
        img_data = map._to_png(5)
        img = Image.open(io.BytesIO(img_data))
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGR)
        img_msg = ImageMsg()
        img_msg.height = img.shape[0]
        img_msg.width = img.shape[1]
        img_msg.encoding = 'bgr8'
        img_msg.is_bigendian = False
        img_msg.data = img.tobytes()
        self.gps_bev_pub.publish(img_msg)
        self._logger.info("Map updated with latest GPS data.")

def main(args=None):
    rclpy.init(args=args)
    gps_viz_node = GPSVizNode()
    rclpy.spin(gps_viz_node)

    # Destroy the node explicitly
    gps_viz_node.destroy_node()
    rclpy.shutdown() 

if __name__ == '__main__':
    main()