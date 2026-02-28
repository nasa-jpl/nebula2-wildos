import rclpy
from rclpy.node import Node

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Odometry, Path as PathMsg
from sensor_msgs.msg import CompressedImage, Image as ImageMsg, CameraInfo
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.duration import Duration
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R
from skimage.graph import MCP_Geometric
import numpy as np
import torch
from torchvision import transforms

from img_vlms.third_party.nvidia_radio.radio_downstream import RADIODownstreamInference

from cv_bridge import CvBridge
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib
matplotlib.use("Agg")

from pathlib import Path

HOME_DIR = Path.home()
CAMERA_MAPPING = {
    0: "front",
    1: "left",
    2: "right"
}


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
    

class ImgFrontierNav(Node):
    default_config = {
        # Model Params
        "frontier_ckpt": "frontier_ckpt.ckpt",
        "traversability_ckpt": "traversability_ckpt.ckpt",
        "model_version": "c-radio_v3-b",
        "adaptor_version": None,
        "use_naclip": True,
        "use_summary_for_spatial": True,
        "radio_dim": 768,
        "static_scale_factor": 0.75,
        "model_precision": "FP16",

        # Nav Params
        "frontier_threshold": 0.8,
        "traversability_threshold": 0.8,
        "num_cameras": 3,
        "goal_reach_radius": 2.0,
        "goal_lock_range": 8.0,
        "way_points": None,

        # Pixel Scoring Params
        "frontier_w": 2.0,
        "goal_w": 3.0,
        "reachability_w": 2.0,

        # ROS2 params
        "parent_frame": "spot1/platform/odom",
        "cam_frame": "spot1/realsense/{}_color_optical_frame",
        "camera_img_topic": "/spot1/realsense/{}/color/image_raw/compressed",
        "camera_depth_topic": "/spot1/realsense/{}/aligned_depth_to_color/image_raw",
        "camera_info_topic": "/spot1/realsense/{}/color/camera_info",
        "odometry_topic": "/spot1/platform/odometry",
        "waypoint_topic": "imgnav_waypoint",
    }

    def __init__(self, config: OmegaConf=OmegaConf.create()):
        super().__init__('img_frontier_nav')

        # Used to convert between ROS and OpenCV images
        self.br = CvBridge()

        # ImgNav parameters and initializations
        config = OmegaConf.merge(OmegaConf.create(self.default_config), config)
        self.frontier_threshold = config.frontier_threshold
        self.traversability_threshold = config.traversability_threshold
        self.num_cameras = config.num_cameras
        assert self.num_cameras in [1, 3], "Only 1 or 3 cameras are supported."
        self.goal_reach_radius = config.goal_reach_radius
        self.goal_lock_range = config.goal_lock_range

        if config.way_points is None:
            self.way_points = None
            self.current_wp_idx = None
        else:
            self.way_points = np.array(config.way_points)
            self.current_wp_idx = 0

        np.random.seed(0)

        # vlm initializations
        self.device = "cuda"

        # radio model
        self.model = RADIODownstreamInference(
            frontier_ckpt= HOME_DIR / "ckpts" / config.frontier_ckpt,
            traversability_ckpt= HOME_DIR / "ckpts" / config.traversability_ckpt,
            model_version=config.model_version,
            adaptor_version=config.adaptor_version,
            use_naclip=config.use_naclip,
            use_summary_for_spatial=config.use_summary_for_spatial,
            radio_dim=config.radio_dim,
            static_scale_factor=config.static_scale_factor,
            model_precision=config.model_precision,
        )
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.get_logger().info('Finished initializing models!')

        # Initialize pixel scoring params
        self.pixel_scoring_params = {
            "frontier": config.frontier_w,
            "goal": config.goal_w,
            "reachability": config.reachability_w
        }
        self.img_start = lambda h, w: (h - 4, w//2) # (y,x): x is width, y is height

        # Projection Parameters
        self.path_max_depth = 5.0

        self.clbk_cntr = 0

        # Frames and Topic Names
        self.global_frame = config.parent_frame
        self.cam_tf_frame = config.cam_frame
        cameraimg_topic_str = config.camera_img_topic
        cameradepth_topic_str = config.camera_depth_topic
        camerainfo_topic_str = config.camera_info_topic

        self.using_compressed_imgs = "compressed" in cameraimg_topic_str
        img_msg_type = CompressedImage if self.using_compressed_imgs else ImageMsg

        # Subscribers and Publishers
        self.init_publishers()

        self.msg_buffer = MessageBuffer(max_size=1)

        self.tf_buffer = Buffer(cache_time=Duration(seconds=10))
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

        self.camera_subs = {}
        for i in range(config.num_cameras):
            self.camera_subs[i] = {
                "image": Subscriber(self, img_msg_type, cameraimg_topic_str.format(CAMERA_MAPPING[i])),
                "depth": Subscriber(self, ImageMsg, cameradepth_topic_str.format(CAMERA_MAPPING[i])),
                "info": Subscriber(self, CameraInfo, camerainfo_topic_str.format(CAMERA_MAPPING[i]))
            }
        self.odom_sub = Subscriber(self, Odometry, config.odometry_topic)
        self.waypoint_sub = self.create_subscription(
            PoseStamped,
            config.waypoint_topic,
            self.waypoint_callback,
            10
        )

        ts_subs = [self.odom_sub]
        for i in range(self.num_cameras):
            ts_subs.append(self.camera_subs[i]["image"])
            ts_subs.append(self.camera_subs[i]["depth"])
            ts_subs.append(self.camera_subs[i]["info"])

        self.ts = ApproximateTimeSynchronizer(
            ts_subs,
            # queue_size=20, slop=1.0)
            queue_size=1, slop=0.2)
        self.ts.registerCallback(self.listener_callback)

        self.oldest_time_processed = None
        self.timer = self.create_timer(0.5, self.check_tf_exists)

    def init_publishers(self):
        self.imgpath_publisher = self.create_publisher(
            PathMsg,
            'predicted_path',
            10
        )
        self.nav2_marker_publisher = self.create_publisher(
            Marker,
            'nav2_goal_marker',
            10
        )
        self.nav2_goal_publisher = self.create_publisher(
            PoseStamped,
            '/spot1/goal_pose',
            10
        )
        self.model_viz_pub = self.create_publisher(
            ImageMsg,
            'model_visualization',
            10
        )
        self.goal_direction_pub = self.create_publisher(
            Marker,
            'goal_direction',
            10
        )
        self.goal_waypoint_pub = self.create_publisher(
            MarkerArray,
            'goal_waypoints',
            10
        )
        self.publish_goal_waypoints()

    def fetch_cam_intrinsics_extrinsics(self, cam_info, tf_world_cam):
        """
        Fetch camera intrinsics and extrinsics using the CameraInfo message.
        """
        K = np.array(cam_info.k).reshape(3, 3)

        R_wc = R.from_quat([
            tf_world_cam.transform.rotation.x,
            tf_world_cam.transform.rotation.y,
            tf_world_cam.transform.rotation.z,
            tf_world_cam.transform.rotation.w
        ]).as_matrix()
        t_wc = np.array([
            tf_world_cam.transform.translation.x,
            tf_world_cam.transform.translation.y,
            tf_world_cam.transform.translation.z
        ]).reshape(3, 1)

        return {
            "K": K,
            "R_wc": R_wc,
            "t_wc": t_wc
        }
    
    def compute_goal_heading(self, odom_msg):
        """
        Compute goal heading and goal reach condition
        """
        current_pos = np.array([
            odom_msg.pose.pose.position.x,
            odom_msg.pose.pose.position.y,
            odom_msg.pose.pose.position.z
        ])
        current_waypoint = self.way_points[self.current_wp_idx]
        self.publish_goal_direction(current_pos, current_waypoint)

        goal_heading = current_waypoint - current_pos
        goal_heading[2] = 0  # Ignore vertical component
        goal_error = np.linalg.norm(goal_heading)
        goal_heading = goal_heading / (goal_error + 1e-6)
        # print(f"Goal Error: {goal_error}, Cur Waypoint Index: {self.current_wp_idx}")

        if goal_error < self.goal_reach_radius:
            self.current_wp_idx += 1
            if self.current_wp_idx >= len(self.way_points):
                self.get_logger().warn("Reached the last waypoint.")
                self.current_wp_idx = None
                self.way_points = None
            else:
                self.get_logger().info(f"Moving to waypoint {self.current_wp_idx}")

        return goal_error, goal_heading
    
    def waypoint_callback(self, msg):
        waypoint = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])
        if self.way_points is None:
            self.way_points = waypoint.reshape(1, 3)
            self.current_wp_idx = 0
        else:
            self.way_points = np.vstack((self.way_points, waypoint))
        self.get_logger().info(f"Received new waypoint: {waypoint}, total waypoints: {len(self.way_points)}")
        self.publish_goal_waypoints()

    def listener_callback(self, odom_msg, *msgs):
        self.clbk_cntr += 1

        self.get_logger().info(f"Received callback {self.clbk_cntr}")
        assert odom_msg.header.frame_id == self.global_frame

        self.msg_buffer.add_msg(
            msg={
                "odom": odom_msg,
                "cam_msgs": msgs
            },
            timestamp=odom_msg.header.stamp.sec + odom_msg.header.stamp.nanosec * 1e-9
        )

    def do_processing(self, msg, tf_data):
        if self.way_points is None or self.current_wp_idx is None:
            self.get_logger().warn("No waypoints provided, skipping processing.")
            return

        print(f"Started Heavy")
        odom_msg = msg["odom"]
        msgs = msg["cam_msgs"]
        goal_error, goal_heading = self.compute_goal_heading(odom_msg)

        if goal_error < self.goal_lock_range:
            self.directly_publish_goal(self.way_points[self.current_wp_idx], odom_msg)
            self.get_logger().info(f"GOAL LOCK!!!!")
            return

        # Extract camera images and info
        rgb_imgs, depth_imgs, cam_info_msgs = [], [], []
        for i in range(len(self.camera_subs)):
            if self.using_compressed_imgs:
                rgb_imgs.append(
                    np.rot90(self.br.compressed_imgmsg_to_cv2(msgs[i * 3], desired_encoding='rgb8'), k=2)
                )
            else:
                rgb_imgs.append(
                    np.rot90(self.br.imgmsg_to_cv2(msgs[i * 3], desired_encoding='rgb8'), k=2)
                )
            depth_imgs.append(
                self.br.imgmsg_to_cv2(msgs[i * 3 + 1], desired_encoding='16UC1')
            )
            cam_info_msgs.append(msgs[i * 3 + 2])

        rgb_tensors = [self.transforms(img.copy()) for img in rgb_imgs]
        batch_tensor = torch.stack(rgb_tensors)
        batch_img_traversability, batch_img_frontiers, _ = self.model.forward(batch_tensor)

        batch_img_frontiers = batch_img_frontiers.cpu().numpy()
        batch_img_traversability = batch_img_traversability.cpu().numpy()

        nav_data = []
        all_cam_data = []
        chosen_scores = []
        for i, cam_info_msg in enumerate(cam_info_msgs):
            cam_data = self.fetch_cam_intrinsics_extrinsics(cam_info_msg, tf_data[i])
            all_cam_data.append(cam_data)

            chosen_score, path, score_map = self.find_img_paths(
                batch_img_frontiers[i][0], batch_img_traversability[i][0], goal_heading, cam_data)
            nav_data.append({
                "image": rgb_imgs[i],
                "traversability": batch_img_traversability[i][0],
                "frontiers": batch_img_frontiers[i][0],
                "frame_id": cam_info_msg.header.frame_id,
                "score_map": score_map,
                "chosen_score": chosen_score,
                "path": np.array(path)
            })
            chosen_scores.append(chosen_score)

        chosen_cam = np.argmax(np.stack(chosen_scores))
        self.project_path_to_3d(chosen_cam, nav_data[chosen_cam]["path"], depth_imgs[chosen_cam], all_cam_data[chosen_cam], odom_msg)
        self.visualize_model_det(nav_data, chosen_cam)
        self.publish_goal_waypoints()
        print(f"Finished Heavy")

    def find_img_paths(self, frontiers, traversability, goal_heading, cam_data):
        """
        Find image paths based on frontiers, traversability, and goal heading.

        :param frontiers: The frontier map (H, W)
        :param traversability: The traversability map (H, W)
        :param goal_heading: The goal heading vector (1, 3)
        :param current_odom: The current odometry message.
        :param cam_data: The camera intrinsics and extrinsics.
        """

        h, w = frontiers.shape
        goal_conf = self.get_goal_conf(h, w, cam_data, goal_heading)
        frontier_conf = self.get_frontier_conf(frontiers, traversability)
        reachability_conf, mcp = self.get_reachability_conf(frontiers, traversability)
        
        score_map = (
            self.pixel_scoring_params["frontier"] * frontier_conf +
            self.pixel_scoring_params["goal"] * goal_conf +
            self.pixel_scoring_params["reachability"] * reachability_conf
        ) / sum(self.pixel_scoring_params.values())

        best_index = np.unravel_index(np.argmax(score_map), score_map.shape)
        chosen_score = score_map[best_index]
        path = mcp.traceback((best_index[0], best_index[1]))

        return chosen_score, path, score_map
       
    def get_reachability_conf(self, frontiers, traversability):
        """
        Compute reachability cost based on frontiers and traversability.
        """
        h, w = traversability.shape

        mcp = MCP_Geometric(1/(traversability + 1e-3), fully_connected=True)
        costs, trbk = mcp.find_costs([self.img_start(h, w)])
        costs /= (h+w)

        # normalize to [0,1]
        reachability_cost = 1 - np.tanh(costs)
        return reachability_cost, mcp

    def get_goal_conf(self, h, w, cam_data, goal_heading):
        """
        Compute goal confidence for each pixel according to alignment with the goal heading.
        """
        K = cam_data['K']
        R_wc = cam_data['R_wc']
        T_wc = cam_data['t_wc']

        pixel_coords = np.array(np.meshgrid(np.arange(w), np.arange(h))).reshape(2, -1).T
        pixel_coords = np.hstack((pixel_coords, np.ones((pixel_coords.shape[0], 1))))
        coords_cam = np.linalg.inv(K) @ pixel_coords.T
        
        pixel_heading = R_wc @ coords_cam
        pixel_heading[2] = 0  # Ignore vertical component
        pixel_heading = pixel_heading / (np.linalg.norm(pixel_heading, axis=0) + 1e-6)

        goal_conf = goal_heading @ pixel_heading
        goal_conf = goal_conf.reshape(h, w)

        # normalize to [0,1]
        goal_conf = (1 + goal_conf) / 2.0
        goal_conf = np.rot90(goal_conf, k=2)

        return goal_conf

    def get_frontier_conf(self, frontiers, traversability):
        frontier_conf = frontiers.copy()
        frontier_conf[frontier_conf < self.frontier_threshold] = 0.0
        frontier_conf[traversability < self.traversability_threshold] = 0.0

        return frontier_conf

    def visualize_model_det(self, nav_data, chosen_cam_idx):
        fig, axes = plt.subplots(4, self.num_cameras, figsize=(17, 8))
        
        if self.num_cameras == 1:
            cam_order = [0]  # Only one camera
        else:
            cam_order = [1, 0, 2]  # LEFT, FRONT, RIGHT

        for i in range(self.num_cameras):
            plt_idx = cam_order[i]
            rgb_img = nav_data[i]["image"]
            axes[0, plt_idx].imshow(rgb_img)
            axes[0, plt_idx].set_title(f"Image {CAMERA_MAPPING[i]}")
            axes[0, plt_idx].axis('off')
            
            # dummy colorbar for alignment
            dummy_sm = mpl.cm.ScalarMappable(cmap='jet', norm=mpl.colors.Normalize(vmin=0, vmax=1))
            cbar = plt.colorbar(dummy_sm, ax=axes[0, plt_idx], fraction=0.046, pad=0.04)
            cbar.ax.set_visible(False)

            # overlay frontiers on the image
            frontier_map = nav_data[i]["frontiers"]
            axes[1, plt_idx].imshow(rgb_img)
            hm = axes[1, plt_idx].imshow(frontier_map, alpha=0.5, cmap='jet', vmin=0, vmax=1)
            axes[1, plt_idx].set_title("Frontiers Overlay")
            axes[1, plt_idx].axis('off')
            plt.colorbar(hm, ax=axes[1, plt_idx], fraction=0.046, pad=0.04)

            # overlay traversability on the image
            traversability_map = nav_data[i]["traversability"]
            axes[2, plt_idx].imshow(rgb_img)
            hm = axes[2, plt_idx].imshow(traversability_map, alpha=0.5, cmap='jet', vmin=0, vmax=1)
            axes[2, plt_idx].set_title("Traversability Overlay")
            axes[2, plt_idx].axis('off')
            plt.colorbar(hm, ax=axes[2, plt_idx], fraction=0.046, pad=0.04)

            # """
            axes[3, plt_idx].imshow(rgb_img)
            y, x = self.img_start(rgb_img.shape[0], rgb_img.shape[1])
            score = nav_data[i]["chosen_score"]
            path = nav_data[i]["path"]
            score_map = nav_data[i]["score_map"]

            if i == chosen_cam_idx:
                color = plt.cm.jet(score)  # Use jet colormap for scores
            else:
                color = 'white'
            axes[3, plt_idx].scatter(x, y, color=color, s=100, label=f"Frontier {CAMERA_MAPPING[i]} ({score:.2f})")
            axes[3, plt_idx].text(x, y, f"{CAMERA_MAPPING[i]}", color='black', fontsize=12, ha='center', va='center')
            axes[3, plt_idx].plot(path[:, 1], path[:, 0], color=color, linewidth=2, alpha=0.7)
            axes[3, plt_idx].scatter(path[-1, 1], path[-1, 0], color='white', marker='x', s=100)
            hm = axes[3, plt_idx].imshow(score_map, cmap='jet', vmin=0, vmax=1, alpha=0.5)
            plt.colorbar(hm, ax=axes[3, plt_idx], fraction=0.046, pad=0.04)
            axes[3, plt_idx].set_title("Paths")
            axes[3, plt_idx].legend(loc='upper right', fontsize='small')
            axes[3, plt_idx].axis('off')
            # """
        plt.tight_layout()
        
        # Convert plot to image
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:,:,1:]
        plt.close(fig)

        # Publish the image
        self.model_viz_pub.publish(self.br.cv2_to_imgmsg(data, encoding="rgb8"))

    def project_path_to_3d(self, chosen_cam, path, depth_img, cam_data, odom_msg):
        """
        Projects a 2D path in the image space to 3D space using the camera intrinsics and depth image.
        """
        K = cam_data['K']
        h, w = depth_img.shape
        path = [(w-x, h-y) for y, x in path]    # account for image rotation
        path = np.array(path)

        # choose path points within the max depth
        x_idx, y_idx = path[:, 0], path[:, 1]
        depths = depth_img[y_idx, x_idx] / 1000.0  # Convert to meters
        valid_mask = (0 < depths) & (depths < self.path_max_depth)
        valid_points = path[valid_mask]
        path_depths = depths[valid_mask]

        depth_sort_idx = np.argsort(path_depths)
        valid_points = valid_points[depth_sort_idx]
        path_depths = path_depths[depth_sort_idx]
        if len(valid_points) == 0:
            self.get_logger().warn("No valid path points found within max depth.")
            return

        # project to cam frame
        cam_path = np.hstack((valid_points, np.ones((valid_points.shape[0], 1))))
        cam_path = np.linalg.inv(K) @ cam_path.T
        cam_path = cam_path / cam_path[2]
        cam_path = cam_path.T
        cam_path = cam_path * path_depths[:, np.newaxis]

        # convert to Path message
        path_msg = PathMsg()
        path_msg.header.frame_id = self.cam_tf_frame.format(CAMERA_MAPPING[chosen_cam])
        # path_msg.header.stamp = odom_msg.header.stamp
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.poses = []
        for p in cam_path:
            pose = PoseStamped()
            pose.pose.position.x = p[0]
            pose.pose.position.y = p[1]
            pose.pose.position.z = p[2]
            path_msg.poses.append(pose)
        self.imgpath_publisher.publish(path_msg)
        heading = cam_path[-1] - cam_path[0]
        if np.linalg.norm(heading) > 0:
            heading = heading / np.linalg.norm(heading)
            # convert to quaternion
            heading = R.from_euler('y', np.arctan2(-heading[2], heading[0])).as_quat()
        else:
            raise ValueError("No valid heading found.")

        # publish nav2_goal as a Marker
        nav2_goal_marker = Marker()
        nav2_goal_marker.header.frame_id = self.cam_tf_frame.format(CAMERA_MAPPING[chosen_cam])
        nav2_goal_marker.header.stamp = self.get_clock().now().to_msg()
        nav2_goal_marker.ns = "nav2_goal_marker"
        nav2_goal_marker.id = 0
        nav2_goal_marker.type = Marker.CUBE
        nav2_goal_marker.action = Marker.ADD
        nav2_goal_marker.scale.x = 1.0
        nav2_goal_marker.scale.y = 1.0
        nav2_goal_marker.scale.z = 1.0
        nav2_goal_marker.color.a = 1.0
        nav2_goal_marker.color.r = 1.0
        nav2_goal_marker.color.g = 0.0
        nav2_goal_marker.color.b = 0.0
        nav2_goal_marker.pose.position.x = cam_path[-1, 0]
        nav2_goal_marker.pose.position.y = cam_path[-1, 1]
        nav2_goal_marker.pose.position.z = cam_path[-1, 2]

        self.nav2_marker_publisher.publish(nav2_goal_marker)

        nav2_goal = PoseStamped()
        nav2_goal.header.frame_id = self.cam_tf_frame.format(CAMERA_MAPPING[chosen_cam])
        nav2_goal.header.stamp = self.get_clock().now().to_msg()
        nav2_goal.pose.position.x = cam_path[-1, 0]
        nav2_goal.pose.position.y = cam_path[-1, 1]
        nav2_goal.pose.position.z = cam_path[-1, 2]
        nav2_goal.pose.orientation.x = heading[0]
        nav2_goal.pose.orientation.y = heading[1]
        nav2_goal.pose.orientation.z = heading[2]
        nav2_goal.pose.orientation.w = heading[3]

        self.nav2_goal_publisher.publish(nav2_goal)
    
    def directly_publish_goal(self, goal, odom_msg):
        nav2_goal = PoseStamped()
        nav2_goal.header.frame_id = self.global_frame
        nav2_goal.header.stamp = self.get_clock().now().to_msg()
        nav2_goal.pose.position.x = goal[0]
        nav2_goal.pose.position.y = goal[1]
        nav2_goal.pose.position.z = goal[2]
        nav2_goal.pose.orientation = odom_msg.pose.orientation

        self.nav2_goal_publisher.publish(nav2_goal)

    def publish_goal_direction(self, current_pos, goal_pos):
        marker = Marker()
        marker.header.frame_id = self.global_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "goal_direction"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        start = Point()
        start.x, start.y, start.z = current_pos
        end = Point()
        end.x, end.y, end.z = goal_pos
        marker.points.append(start)
        marker.points.append(end)
        marker.scale.x = 1.0
        marker.scale.y = 1.5
        marker.scale.z = 1.5
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        self.goal_direction_pub.publish(marker)

    def publish_goal_waypoints(self):
        if self.way_points is None:
            return

        marker_array = MarkerArray()
        for i, waypoint in enumerate(self.way_points):
            marker = Marker()
            marker.header.frame_id = self.global_frame
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "goal_waypoints"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = waypoint[0]
            marker.pose.position.y = waypoint[1]
            marker.pose.position.z = waypoint[2]
            marker.scale.x = 5.0
            marker.scale.y = 5.0
            marker.scale.z = 5.0
            marker.color.a = 1.0
            marker.color.r = 0.0 if i == self.current_wp_idx else 1.0
            marker.color.g = 1.0
            marker.color.b = 1.0
            marker_array.markers.append(marker)
        self.goal_waypoint_pub.publish(marker_array)

    def check_tf_exists(self):
        if self.msg_buffer.buffer:
            found_one_valid_ts = False
            found_invalid_after_valid = False
            valid_tfs = None
            valid_msg = None
            valid_ts = None

            for old_msg, old_msg_ts in self.msg_buffer.buffer:
                tfs = []
                for i in range(self.num_cameras):
                    try:
                        tf_oldest_msg = self.tf_buffer.lookup_transform(
                            self.global_frame,
                            self.cam_tf_frame.format(CAMERA_MAPPING[i]), 
                            old_msg["odom"].header.stamp, 
                            timeout=Duration(seconds=0)
                        )
                        tfs.append(tf_oldest_msg)

                    except Exception as e:
                        self.get_logger().info(f"TF not found for {self.cam_tf_frame.format(CAMERA_MAPPING[i])} at time {old_msg_ts}: {e}")
                        if not found_one_valid_ts:
                            return
                        else:
                            found_invalid_after_valid = True
                            break
                if found_invalid_after_valid:
                    break
                found_one_valid_ts = True
                valid_tfs = tfs.copy()
                valid_msg = old_msg
                valid_ts = old_msg_ts
                break
            
            if self.oldest_time_processed is None or self.oldest_time_processed < valid_ts:
                self.oldest_time_processed = valid_ts
                self.get_logger().info(f"TF found for camera frames at time {valid_ts}")
                self.do_processing(valid_msg, valid_tfs)
            else:
                self.get_logger().warn(f"Already processed TF for time {valid_ts}, skipping processing.")
                self.msg_buffer.pop_oldest_msg()
        else:
            self.get_logger().warn("Message buffer is empty, waiting for messages...")


def main(args=None):
    rclpy.init(args=args)
    from ament_index_python.packages import get_package_share_directory
    import os

    package_share_directory = Path(get_package_share_directory('img_vlms'))
    conf = package_share_directory / "configs" / "nav_conf_marsyard.yaml"

    imgfrontiernav_node = ImgFrontierNav(OmegaConf.load(conf))
    rclpy.spin(imgfrontiernav_node)

    imgfrontiernav_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
