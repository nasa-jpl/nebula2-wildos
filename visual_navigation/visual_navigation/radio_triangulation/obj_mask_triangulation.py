import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from img_vlms.utils.buffer import MessageBuffer
from img_vlms.utils.tf_lookup_sub import TFLookupSubscriber

from sensor_msgs_py import point_cloud2
from object_search_msgs.msg import ObjectMaskWithTf
from visualization_msgs.msg import Marker
from nav_msgs.msg import Odometry, Path as PathMsg
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image as ImageMsg, CameraInfo, PointCloud2
from message_filters import ApproximateTimeSynchronizer, Subscriber
from cv_bridge import CvBridge

from pathlib import Path
from omegaconf import OmegaConf
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R


from triangulation3d.camera_data import Camera
from triangulation3d.particle_generator import ParticleGenerator
from triangulation3d.bbox_generator import BoundingBoxGenerator
from triangulation3d.triangulator import Triangulator

CAMERA_MAPPING = {
    0: "front",
    1: "left",
    2: "right"
}    

class ObjectMaskTriangulator(Node):
    default_config = {
        # Robot Parameters
        "num_cameras": 3,  # Number of cameras to use
        "parent_frame": "spot1/odom",
        "cam_frame": "spot1/realsense/{}_color_optical_frame",
        "lidar_frame": "spot1/ouster/front/os_lidar",
        "object_mask_topic": "/spot1/object_mask",  # Topic for object mask input
        "lidar_topic": "/spot1/ouster/front/points_filtered",  # Topic for lidar input

        # Publisher topics
        "triangulated_object_topic": "/spot1/triangulated_object",
        "navigation_goal_topic": "/spot1/imgnav_waypoint",
        "particle_viz_topic": "/spot1/object_hypotheses",

        # ROS2 subscriber params
        "qos_history_depth": 10,
        "syncsub_queue_size": 10,
        "syncsub_slop": 0.2,

        # Triangulation config
        "max_views": 350,  # Maximum number of cameras to initialize
        "min_lidar_points": 150,  # Minimum number of lidar points in the object mask for triangulation
        "min_view_distance": 1.0,  # Minimum distance between camera views to consider them different
        "particle_generator_config": {
            "num_particles": 1000, # Number of particles to generate
            "depth_range": [1.0, 100.0],  # Depth range in meters
            "add_odom_drift": False, # Whether to add odometry drift to the camera pose
        },
        "use_mask_for_projection": True,  # Whether to use the object mask for projection


        # TFLookup Config
        "buffer_size": 10,       # number of messages
        "timer_duration": 0.2,   # seconds
    }

    def __init__(self, config: OmegaConf=OmegaConf.create()):
        config = OmegaConf.merge(OmegaConf.create(self.default_config), config)
        super().__init__('obj_mask_triangulator')

        np.random.seed(42)

        # Robot Parameters
        self.num_cameras = config.num_cameras
        assert self.num_cameras in [1, 3], "Only 1 or 3 cameras are supported."

        # Triangulation parameters and initializations
        self.views = []
        self.max_views = config.max_views
        self.min_lidar_points = config.min_lidar_points
        self.min_view_distance = config.min_view_distance
        self.use_mask_for_projection = config.use_mask_for_projection

        self.particle_generator = ParticleGenerator(config.particle_generator_config)
        self.triangulator = Triangulator()
        self.triangulated_position = None
        self.prev_view_pos = None
        self.found_lidar_in_mask = False

        # Subscribers and Publishers
        self.init_publishers(config)
        self.init_subscribers(config)

        # Frames and Topic Names
        self.global_frame = config.parent_frame
        self.cam_tf_frame = config.cam_frame
        self.lidar_tf_frame = config.lidar_frame

        # TFLookup
        self.tf_buffer = Buffer(cache_time=Duration(seconds=10))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.msg_buffer = MessageBuffer(
            max_size=config.buffer_size, wait_for_oldest=True
        )
        self.clbk_cntr = 0

        self.timer = self.create_timer(config.timer_duration, self.do_processing)

    def init_publishers(self, config: OmegaConf):
        self.triangulated_obj_publisher = self.create_publisher(
            Marker, config.triangulated_object_topic, 10
        )
        self.nav_goal_publisher = self.create_publisher(
            PoseStamped, config.navigation_goal_topic, 10
        )
        self.particle_viz_publisher = self.create_publisher(
            PointCloud2, config.particle_viz_topic, 10
        )

    def init_subscribers(self, config):
        self.object_mask_sub = Subscriber(
            self, ObjectMaskWithTf, config.object_mask_topic, qos_profile=config.qos_history_depth
        )
        self.lidar_sub = Subscriber(
            self, PointCloud2, config.lidar_topic, qos_profile=config.qos_history_depth
        )

        ts_subs = [self.object_mask_sub, self.lidar_sub]
        self.ts = ApproximateTimeSynchronizer(
            ts_subs, queue_size=config.syncsub_queue_size, slop=config.syncsub_slop
        )
        self.ts.registerCallback(self.listener_callback)

    def listener_callback(self, object_mask_msg, lidar_msg):
        self.clbk_cntr += 1

        self.get_logger().info(f"Received callback {self.clbk_cntr}")
        assert lidar_msg.header.frame_id == self.lidar_tf_frame, \
            f"LiDAR frame_id {lidar_msg.header.frame_id} does not match expected {self.lidar_tf_frame}"

        self.msg_buffer.add_msg(
            msg={
                "obj_mask": object_mask_msg,
                "lidar": lidar_msg
            },
            stamp=object_mask_msg.header.stamp,
        )

    def do_processing(self):
        if not self.msg_buffer.buffer:
            self.get_logger().warn("Message buffer is empty, waiting for messages...")
            return

        # Extract messages and data
        oldest_msg, _, _ = self.msg_buffer.pop_oldest_msg()

        obj_mask_msg = oldest_msg["obj_mask"]
        lidar_msg = oldest_msg["lidar"]
        all_cam_data = self.extract_data_from_obj_mask_msg(obj_mask_msg)
        cur_pos = np.array([
            obj_mask_msg.odom.pose.pose.position.x,
            obj_mask_msg.odom.pose.pose.position.y,
            obj_mask_msg.odom.pose.pose.position.z
        ])

        # Check if LiDAR points fall within the object mask
        self.check_lidar_in_mask(all_cam_data, lidar_msg)

        # Triangulate using multiple views if LiDAR triangulation was not successful
        if not self.found_lidar_in_mask:
            if self.prev_view_pos is not None:
                
                dist = np.linalg.norm(cur_pos - self.prev_view_pos)
                if dist < self.min_view_distance:
                    self.get_logger().info(f"Current view is too close to previous view (distance: {dist:.2f}m), skipping...")
                    return

            self.add_views(all_cam_data)
            self.prev_view_pos = cur_pos
            if len(self.views) >= 2:
                self.triangulated_position = self.triangulator.triangulate(self.views)
                self.get_logger().info(f"Triangulated position using multiple views: {self.triangulated_position}")
            else:
                self.get_logger().info("Not enough views for triangulation.")
        else:
            self.get_logger().info("Object LOCK using LiDAR!")
        
        # Publish triangulated position as a navigation goal
        self.publish_navigation_goal_and_marker()
        self.publish_goal_hypotheses()

    def extract_data_from_obj_mask_msg(self, obj_mask_msg):
        """
        Extract relevant data from ObjectMaskWithTf message.
        """
        obj_mask = self.get_objmask_from_multiarray(obj_mask_msg.object_mask)
        assert obj_mask.shape[0] == self.num_cameras, f"Expected {self.num_cameras} object masks, but got {obj_mask.shape[0]}"

        camera_infos = obj_mask_msg.cam_infos
        assert len(camera_infos) == self.num_cameras, f"Expected {self.num_cameras} camera infos, but got {len(camera_infos)}"

        camera_tfs = obj_mask_msg.cam_transforms.transforms
        assert len(camera_tfs) == self.num_cameras, f"Expected {self.num_cameras} camera transforms, but got {len(camera_tfs)}"

        all_cam_data = []
        for cam_id in range(self.num_cameras):
            cam_data = {}
            cam_frame = self.cam_tf_frame.format(CAMERA_MAPPING[cam_id])

            assert camera_infos[cam_id].header.frame_id[1:] == cam_frame, \
                f"CameraInfo frame_id {camera_infos[cam_id].header.frame_id} does not match expected {cam_frame}"
            
            assert camera_tfs[cam_id].child_frame_id == cam_frame, \
                f"Camera TF child_frame_id {camera_tfs[cam_id].child_frame_id} does not match expected {cam_frame}"

            cam_data["object_mask"] = obj_mask[cam_id, 0, :, :]  # HxW
            cam_data["K"] = np.array(camera_infos[cam_id].k).reshape(3, 3)
            cam_data["R_wc"] = R.from_quat([
                camera_tfs[cam_id].transform.rotation.x,
                camera_tfs[cam_id].transform.rotation.y,
                camera_tfs[cam_id].transform.rotation.z,
                camera_tfs[cam_id].transform.rotation.w
            ]).as_matrix()
            cam_data["t_wc"] = np.array([
                camera_tfs[cam_id].transform.translation.x,
                camera_tfs[cam_id].transform.translation.y,
                camera_tfs[cam_id].transform.translation.z
            ]).reshape(3, 1)
            cam_data["width"] = camera_infos[cam_id].width
            cam_data["height"] = camera_infos[cam_id].height
            cam_data["frame_id"] = cam_frame
            cam_data["camera_info"] = camera_infos[cam_id]
            cam_data["camera_tf"] = camera_tfs[cam_id]
        
            all_cam_data.append(cam_data)

        return all_cam_data

    def check_lidar_in_mask(self, all_cam_data, lidar_msg):
        """
        Check if LiDAR points fall within the object mask.
        """
        
        # Check if lidar points fall within the mask
        lidar_points_3d = point_cloud2.read_points(lidar_msg, field_names=("x", "y", "z"), skip_nans=True)
        lidar_points_3d = np.array([np.array(list(pt)) for pt in lidar_points_3d])  # Nx3

        for cam_id in range(self.num_cameras):
            
            cam_name = CAMERA_MAPPING[cam_id]
            cam_frame =  all_cam_data[cam_id]["frame_id"]
            cam_obj_mask = all_cam_data[cam_id]["object_mask"]  # HxW
            h = all_cam_data[cam_id]["height"]
            w = all_cam_data[cam_id]["width"]
            if cam_obj_mask.shape[0] != h or cam_obj_mask.shape[1] != w:
                raise ValueError(f"Object mask shape {cam_obj_mask.shape} does not match expected {(h, w)} for camera {cam_name}")

            if np.sum(cam_obj_mask) == 0:
                self.get_logger().warn(f"No object detected in camera {cam_name}, skipping...")
                continue

            cam_from_lidar = self.tf_buffer.lookup_transform(
                cam_frame,
                self.lidar_tf_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.0)
            )
            R_cl = R.from_quat([
                cam_from_lidar.transform.rotation.x,
                cam_from_lidar.transform.rotation.y,
                cam_from_lidar.transform.rotation.z,
                cam_from_lidar.transform.rotation.w
            ]).as_matrix()
            t_cl = np.array([
                cam_from_lidar.transform.translation.x,
                cam_from_lidar.transform.translation.y,
                cam_from_lidar.transform.translation.z
            ]).reshape(3, 1)
            K = all_cam_data[cam_id]["K"]

            lidar_points_cam = (R_cl @ lidar_points_3d.T).T + t_cl.T  # Nx3
            valid_points = lidar_points_cam[:, 2] > 0.1  # Keep points in front of the camera
            lidar_points_cam = lidar_points_cam[valid_points]
            if lidar_points_cam.shape[0] == 0:
                continue

            # Project to image plane
            lidar_pix = K @ lidar_points_cam.T  # 3xN
            lidar_pix = lidar_pix[:2, :] / lidar_pix[2, :]  # 2xN
            lidar_pix = lidar_pix.T.astype(np.int32)  # Nx2

            # consider points lying inside the image
            inside_image = np.logical_and.reduce((
                lidar_pix[:, 0] >= 0,
                lidar_pix[:, 0] < w,
                lidar_pix[:, 1] >= 0,
                lidar_pix[:, 1] < h
            ))
            lidar_pix = lidar_pix[inside_image]
            lidar_points_cam = lidar_points_cam[inside_image]

            # consider points lying inside the object mask
            inside_mask = cam_obj_mask[lidar_pix[:, 1], lidar_pix[:, 0]].astype(bool)
            lidar_points_cam = lidar_points_cam[inside_mask]

            if lidar_points_cam.shape[0] < self.min_lidar_points:
                self.get_logger().info(f"Not enough lidar points found in object mask for camera {cam_name}, skipping...")
                continue
            
            dists = np.linalg.norm(lidar_points_cam, axis=-1)
            median_idx = np.argsort(dists)[len(dists)//2]
            triangulated_position_cam = lidar_points_cam[median_idx]

            R_wc = all_cam_data[cam_id]["R_wc"]
            t_wc = all_cam_data[cam_id]["t_wc"]
            triangulated_position_world = R_wc @ triangulated_position_cam.reshape(3, 1) + t_wc  # 3x1
            self.triangulated_position = triangulated_position_world.flatten()

            self.get_logger().info(f"Triangulated position using LiDAR points in camera {cam_name}: {self.triangulated_position}")
            self.found_lidar_in_mask = True

    def add_views(self, all_cam_data):
        """
        Add camera views for triangulation.
        """
        if len(self.views) >= self.max_views:
            self.get_logger().info(f"Reached maximum number of views ({self.max_views}), not adding more.")
            return

        for cam_id in range(self.num_cameras):
            cam_name = CAMERA_MAPPING[cam_id]
            cam_data = all_cam_data[cam_id]

            if np.sum(cam_data["object_mask"]) == 0:
                self.get_logger().info(f"No object detected in camera {cam_name}, skipping...")
                continue
            
            bbox = cv2.boundingRect(cam_data["object_mask"].astype(np.uint8))
            x,y,w,h = bbox
            bbox = np.array([x,y,x+w,y+h])
            view = Camera(
                camera_info=cam_data["camera_info"],
                camera_tf=cam_data["camera_tf"],
                bounding_box=bbox,
                object_mask=cam_data["object_mask"],
                image=None
            )
            view = BoundingBoxGenerator.generate_ray_from_bbox(view)

            view = self.particle_generator.generate_particles(
                view,
                use_mask=self.use_mask_for_projection,
                pcl_frame_id=self.global_frame
            )

            self.views.append(view)

    def get_objmask_from_multiarray(self, multiarray_msg):
        """
        Convert MultiArray message to binary object mask.
        """
        offset = multiarray_msg.layout.data_offset
        obj_mask = np.array(multiarray_msg.data[offset:], dtype=np.uint8)
        dims = multiarray_msg.layout.dim
        assert len(dims) == 4, "Expected 4D MultiArray for object mask (Bx1xHxW)"
        obj_mask = obj_mask.reshape((multiarray_msg.layout.dim[0].size,
                                     multiarray_msg.layout.dim[1].size,
                                     multiarray_msg.layout.dim[2].size,
                                     multiarray_msg.layout.dim[3].size))
        return obj_mask
            
    def publish_navigation_goal_and_marker(self):
        if self.triangulated_position is None:
            self.get_logger().info("No triangulated position available to publish.")
            return

        # Publish as a navigation goal
        nav_goal = PoseStamped()
        nav_goal.header.frame_id = self.global_frame
        nav_goal.header.stamp = self.get_clock().now().to_msg()
        nav_goal.pose.position.x = float(self.triangulated_position[0])
        nav_goal.pose.position.y = float(self.triangulated_position[1])
        nav_goal.pose.position.z = float(self.triangulated_position[2])
        nav_goal.pose.orientation.w = 1.0  # Neutral orientation
        self.nav_goal_publisher.publish(nav_goal)
        self.get_logger().info(f"Published navigation goal at {self.triangulated_position}")

        # Publish as a visualization marker
        marker = Marker()
        marker.header.frame_id = self.global_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.ns = "triangulated_object"
        marker.id = 0
        marker.pose.position.x = float(self.triangulated_position[0])
        marker.pose.position.y = float(self.triangulated_position[1])
        marker.pose.position.z = float(self.triangulated_position[2])
        marker.pose.orientation.w = 1.0
        marker.scale.x = 1.2
        marker.scale.y = 1.2
        marker.scale.z = 1.2
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        self.triangulated_obj_publisher.publish(marker)

    def publish_goal_hypotheses(self):
        # Combine the points from all the cameras into a single PointCloud message
        combined_pcl_msg = self.triangulator.combine_points(self.views, pcl_frame_id=self.global_frame)
        self.particle_viz_publisher.publish(combined_pcl_msg)

def main(args=None):
    rclpy.init(args=args)
    from ament_index_python.packages import get_package_share_directory
    import os

    package_share_directory = Path(get_package_share_directory('img_vlms'))
    conf = package_share_directory / "configs" / "triangulation3d_objsearch_conf.yaml"

    mask_triang_node = ObjectMaskTriangulator(OmegaConf.load(conf))
    rclpy.spin(mask_triang_node)

    mask_triang_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
