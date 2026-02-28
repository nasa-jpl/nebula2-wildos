import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge
import cv2
from sensor_msgs.msg import CompressedImage, Image
import numpy as np


class ImageSubscriber(Node):

    def __init__(self):
        super().__init__('realsense_subscriber')
        self.subscription = self.create_subscription(
            # CompressedImage,
            Image,
            # '/spot1/realsense/front/color/image_raw/compressed',
            '/spot1/realsense/front/color/image_raw',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

        # Used to convert between ROS and OpenCV images
        self.br = CvBridge()
        self.i = 0

    def listener_callback(self, msg):
        self.get_logger().info(f"Frame ID: {msg.header.frame_id}")

        # Convert ROS Image message to OpenCV image
        # current_frame = self.br.compressed_imgmsg_to_cv2(msg)
        current_frame = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        current_frame = np.rot90(current_frame, k=2)
        cv2.imwrite(f'/home/scarecrow/data/nebula/{self.i:04d}.png', current_frame)
        self.i += 1

        # Display image
        # cv2.imshow("camera", current_frame)
        
        # cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)

    img_sub = ImageSubscriber()
    rclpy.spin(img_sub)

    img_sub.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
