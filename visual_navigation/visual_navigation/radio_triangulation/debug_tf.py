import rclpy
from rclpy.node import Node

from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import Image as ImageMsg, CameraInfo
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.duration import Duration
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from omegaconf import OmegaConf
import numpy as np
from scipy.spatial.transform import Rotation as R
import mrcal

import cv2
from cv_bridge import CvBridge

class MessageBuffer:
    def __init__(self, max_size=10):
        self.max_size = max_size
        self.buffer = []

    def add_msg(self, msg: dict, timestamp):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)  # Remove the oldest image
        self.buffer.append((msg, timestamp))

    def get_oldest_msg(self) -> dict:
        if not self.buffer:
            return None
        return self.buffer[0][0]

    def get_closest_msg(self, timestamp):
        if not self.buffer:
            return None
        closest_msg = min(self.buffer, key=lambda x: abs(x[1] - timestamp))
        return closest_msg[0]
    
    def pop_oldest_msg(self):
        if self.buffer:
            self.buffer.pop(0)
    

class DebugTF(Node):
   
    def __init__(self, config: OmegaConf=OmegaConf.create()):
        super().__init__('radio_triangulator')

        # Used to convert between ROS and OpenCV images
        self.br = CvBridge()

        # Subscribers and Publishers
        self.img_sub = Subscriber(self, ImageMsg, '/spot1/realsense/front/color/image_raw')
        self.camera_info_sub = Subscriber(self, CameraInfo, '/spot1/realsense/front/color/camera_info')

        self.ts = ApproximateTimeSynchronizer([self.img_sub, self.camera_info_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.listener_callback)
        self.msg_buffer = MessageBuffer(max_size=5)

        self.tf_buffer = Buffer(cache_time=Duration(seconds=10))
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

        self.timer = self.create_timer(0.05, self.check_tf_exists)

        self.marker_pub = self.create_publisher(
            MarkerArray,
            'cam_points',
            10
        )
        self.cam_points = MarkerArray()

    def listener_callback(self, img_msg, cam_info_msg):
        self.msg_buffer.add_msg(
            {"img_msg": img_msg, "cam_info_msg": cam_info_msg}, 
            img_msg.header.stamp.sec + img_msg.header.stamp.nanosec * 1e-9
        )


    def check_tf_exists(self):
        if self.msg_buffer.buffer:
            oldest_msg = self.msg_buffer.get_oldest_msg()
            print(f"Img Header frame: {oldest_msg["img_msg"].header.frame_id}")
            try:
                tf_oldest_msg = self.tf_buffer.lookup_transform(
                    'spot1/map', 
                    oldest_msg["img_msg"].header.frame_id, 
                    oldest_msg["img_msg"].header.stamp, 
                    timeout=Duration(seconds=0)
                )
                self.get_logger().info(f"TF found for {oldest_msg['img_msg'].header.frame_id} at time {oldest_msg['img_msg'].header.stamp}")
                self.do_processing(oldest_msg, tf_oldest_msg)
            except Exception as e:
                self.get_logger().info(f"TF not found for {oldest_msg['img_msg'].header.frame_id} at time {oldest_msg['img_msg'].header.stamp}: {e}")
                return

    def do_processing(self, msg, tf_msg):

        img_msg = msg["img_msg"]
        cam_info_msg = msg["cam_info_msg"]

        # Convert ROS Image message to OpenCV image
        current_frame = self.br.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        
        # Sample points in the image        
        pts_x = np.arange(0, current_frame.shape[1]//2, 50)
        pts_y = np.ones_like(pts_x) * 50  # Fixed y-coordinate for the points
        pts_z = np.ones_like(pts_x)
        depth = 2 # Fixed depth for the points in meters
        pts_pixels = np.vstack((pts_x, pts_y, pts_z)).T

        for i in range(len(pts_pixels)):
            current_frame = cv2.circle(current_frame, (int(pts_pixels[i][0]), int(pts_pixels[i][1])), 5, (0, 255, 0), -1)
        
        cv2.imshow('Current Frame', current_frame)
        cv2.waitKey(1)

        # Convert pts_pixels to map frame and publish as MarkerArray
        R_wc_quat = np.array([
            tf_msg.transform.rotation.x,
            tf_msg.transform.rotation.y,
            tf_msg.transform.rotation.z,
            tf_msg.transform.rotation.w
        ])
        R_wc = R.from_quat(R_wc_quat).as_matrix()
        t_wc = np.array([
            tf_msg.transform.translation.x,
            tf_msg.transform.translation.y,
            tf_msg.transform.translation.z
        ])
        K = np.array(cam_info_msg.k).reshape(3, 3)

        pts_cam = np.dot(np.linalg.inv(K), depth * pts_pixels.T).T

        Rt = np.vstack((R_wc, t_wc.reshape(1,3)))
        pts_map = mrcal.transform_point_Rt(
            Rt, pts_cam
        )
        # pts_map = np.dot(R_wc, pts_cam.T).T + t_wc
        
        pts_map = pts_map.astype(np.float64)

        # Append to MarkerArray
        num_markers = len(self.cam_points.markers)
        for i in range(len(pts_map)):
            marker = Marker()
            marker.header.frame_id = 'spot1/map'
            marker.header.stamp = img_msg.header.stamp
            marker.id = i + num_markers
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = pts_map[i][0]
            marker.pose.position.y = pts_map[i][1]
            marker.pose.position.z = pts_map[i][2]
            marker.scale.x = 0.3
            marker.scale.y = 0.3
            marker.scale.z = 0.3
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0

            self.cam_points.markers.append(marker)
        self.marker_pub.publish(self.cam_points)
        

def main(args=None):
    rclpy.init(args=args)

    dino_sub = DebugTF()
    rclpy.spin(dino_sub)

    dino_sub.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
