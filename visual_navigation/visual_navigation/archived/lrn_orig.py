import rclpy
from rclpy.node import Node

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PoseStamped, Pose
from nav_msgs.msg import Odometry, Path as PathMsg
from sensor_msgs.msg import CompressedImage, Image as ImageMsg, CameraInfo
from builtin_interfaces.msg import Duration as MarkerDuration
from std_msgs.msg import ColorRGBA
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.duration import Duration
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import tf_transformations as transformations

import PIL
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
    

class LRN(Node):
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

        # Robot Env Params
        "num_cameras": 3,
        "goal_reach_radius": 2.0,
        "way_points": None,
        
        # Nav Params
        "hotspot_threshold": 0.7,  # Heat values below this threshold are ignored
        "angle_discretization_deg": 5,  # (degrees) Bin hotspots at this resolution
        "fixed_hotspot_dist": 5.0,  # fixed distance to pretend hotspots are at for projection
        "confidence_threshold": 0.0,  # Threshold if all hotspots below this value then it will just send goal
        "goal_lock_range": 12.0,  # If within range of goal then just send goal
        "honing_range": 30,  # If within honing range linearly decrease goal_std based on dist2goal/honing_range
        "costmap_range": 8.0,  # Radius of local costmap
        "ema_alpha": 0.1,   # EMA alpha for smoothing predictions over time
        "beta_degradation": 1.0,  # Degrade LRN to uniform distribution (1.0 = full LRN)

        # Cost parms
        "goal_std": 90.0,  # Goal gaussian std. Larger makes it care less about goal and visa-versa
        "prev_std": 110.0,  # Previous prediction consistency gaussian std. Larger makes it care less about previous predicted heading and visa-versa


        # ROS2 params
        "parent_frame": "spot1/platform/odom",
        "body_frame": "spot1/base_link",
        "cam_frame": "spot1/realsense/{}_color_optical_frame",
        "camera_img_topic": "/spot1/realsense/{}/color/image_raw/compressed",
        "camera_depth_topic": "/spot1/realsense/{}/aligned_depth_to_color/image_raw",
        "camera_info_topic": "/spot1/realsense/{}/color/camera_info",
        "odometry_topic": "/spot1/platform/odometry",
        "waypoint_topic": "imgnav_waypoint",
    }

    def __init__(self, config: OmegaConf=OmegaConf.create()):
        super().__init__('long_range_navigator')

        # Used to convert between ROS and OpenCV images
        self.br = CvBridge()

        # ImgNav parameters and initializations
        config = OmegaConf.merge(OmegaConf.create(self.default_config), config)
        self.num_cameras = config.num_cameras
        assert self.num_cameras in [1, 3], "Only 1 or 3 cameras are supported."
        self.goal_reach_radius = config.goal_reach_radius

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

        # Initialize nav params
        self.hotspot_threshold = config.hotspot_threshold
        self.angle_discretization_deg = config.angle_discretization_deg
        self.fixed_hotspot_dist = config.fixed_hotspot_dist
        self.confidence_threshold = config.confidence_threshold
        self.goal_lock_range = config.goal_lock_range
        self.honing_range = config.honing_range
        self.costmap_range = config.costmap_range
        self.ema_alpha = config.ema_alpha
        self.beta_degradation = config.beta_degradation
        self.goal_std = config.goal_std
        self.prev_std = config.prev_std

        self.clbk_cntr = 0

        # Frames and Topic Names
        self.global_frame = config.parent_frame
        self.body_frame = config.body_frame
        self.cam_tf_frame = config.cam_frame
        cameraimg_topic_str = config.camera_img_topic
        cameradepth_topic_str = config.camera_depth_topic
        camerainfo_topic_str = config.camera_info_topic

        self.using_compressed_imgs = "compressed" in cameraimg_topic_str
        img_msg_type = CompressedImage if self.using_compressed_imgs else ImageMsg

        # Subscribers and Publishers
        self.init_publishers()
        self.prev_heading = None
        self.filtered_scores = None

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
        self.scores_debug_publisher = self.create_publisher(
            ImageMsg,
            'lrn_scores_debug',
            10
        )
        self.heatring_publisher = self.create_publisher(
            MarkerArray,
            'lrn_heatring',
            10
        )
        self.model_viz_pub = self.create_publisher(
            ImageMsg,
            'model_visualization',
            10
        )
        self.direction_pub = self.create_publisher(
            Marker,
            'heading_direction',
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
    
    def compute_goal_heading_and_range(self, odom_msg):
        """
        Compute goal heading, range and goal reach condition
        """
        current_pos = np.array([
            odom_msg.pose.pose.position.x,
            odom_msg.pose.pose.position.y,
            odom_msg.pose.pose.position.z
        ])
        current_waypoint = self.way_points[self.current_wp_idx]

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

        return goal_heading, goal_error

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

        odom_from_baselink = tf_data[-1]
        tf_data = tf_data[:-1]  # Last one is odom to body

        odom_msg = msg["odom"]
        msgs = msg["cam_msgs"]
        goal_heading, goal_range = self.compute_goal_heading_and_range(odom_msg)

        if goal_range < self.goal_lock_range:
            if self.way_points is None or self.current_wp_idx is None:
                self.get_logger().warn("No waypoints provided, skipping goal lock.")
                return
            self.publish_pred_goal(self.way_points[self.current_wp_idx], odom_msg.header.stamp)
            self.get_logger().info(f"GOAL LOCK. Publishing predicted goal at {odom_msg.header.stamp}")
            return

        print(f"Started Heavy")
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
        _, batch_img_frontiers, _ = self.model.forward(batch_tensor)

        batch_img_frontiers = batch_img_frontiers.cpu().numpy()

        nav_data = []
        all_cam_data = []
        bin_scores = []
        for i, cam_info_msg in enumerate(cam_info_msgs):
            cam_data = self.fetch_cam_intrinsics_extrinsics(cam_info_msg, tf_data[i])
            all_cam_data.append(cam_data)
            
            raw_hotspots = np.array(np.where(batch_img_frontiers[i][0] > self.hotspot_threshold)).T

            # No hotspots found
            if raw_hotspots.shape[1] == 0:
                self.get_logger().warn(f"No hotspots found for camera {CAMERA_MAPPING[i]}, skipping this camera for scoring.")
                continue
            heat_scores = batch_img_frontiers[i][0, raw_hotspots[:, 0], raw_hotspots[:, 1]]
            cam_scores = self.get_frontier_bin_scores(
                heat_scores, raw_hotspots, cam_data)
            
            nav_data.append({
                "image": rgb_imgs[i],
                "frontiers": batch_img_frontiers[i][0],
                "frame_id": cam_info_msg.header.frame_id,
            })
            bin_scores.append(cam_scores)

        bin_scores = np.array(bin_scores)
        print(bin_scores)
        bin_scores = np.max(bin_scores, axis=0)  # (B,) max over cameras

        all_scores = self.get_all_scores(bin_scores, goal_heading, goal_range)
        if all_scores[2, :].max() < self.confidence_threshold:
            self.publish_pred_goal(self.way_points[self.current_wp_idx], odom_msg.header.stamp)
            self.get_logger().info(f"Low confidence {all_scores[2, :].max()}. Publishing predicted goal at {odom_msg.header.stamp}")
            heading = np.arctan2(goal_heading[1], goal_heading[0])
        else:
            heading = np.deg2rad(
                np.argmax(all_scores[0, :]) * self.angle_discretization_deg
            )
            goal_pos = np.array([
                odom_msg.pose.pose.position.x + self.fixed_hotspot_dist * np.cos(heading),
                odom_msg.pose.pose.position.y + self.fixed_hotspot_dist * np.sin(heading),
                odom_msg.pose.pose.position.z
            ])
            self.publish_pred_goal(goal_pos, odom_msg.header.stamp)

        self.publish_arrow_marker(heading, odom_msg.pose.pose.position, ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0), 1, 0, odom_msg.header.stamp)
        self.publish_arrow_marker(
            np.arctan2(goal_heading[1], goal_heading[0]), odom_msg.pose.pose.position,
            ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0), 0, 0, odom_msg.header.stamp)
        self.publish_scores_debug(all_scores, odom_from_baselink)
        self.publish_ring(all_scores, odom_msg.pose.pose.position)
        self.visualize_model_det(nav_data)
        self.publish_goal_waypoints()
        print(f"Finished Heavy")

    def get_frontier_bin_scores(self, heat_scores: np.ndarray, raw_hotspots: np.ndarray, cam_data: dict):
        """
        Score each bin using the frontier heatmaps.
        """
        K = cam_data['K']

        R_wc = cam_data['R_wc']
        T_wc = cam_data['t_wc']

        hotspot_pix = np.flip(raw_hotspots, axis=-1)  # (N, 2) y,x -> x,y
        hotspot_pix = np.hstack((hotspot_pix, np.ones((hotspot_pix.shape[0], 1))))
        hotspot_cam = np.linalg.inv(K) @ hotspot_pix.T
        hotspot_cam = hotspot_cam / hotspot_cam[2]
        hotspot_cam = hotspot_cam.T * self.fixed_hotspot_dist  # (N, 3)
        hotspot_odom = R_wc @ hotspot_cam.T
        hotspot_angles = np.arctan2(hotspot_odom[1], hotspot_odom[0])

        bins = (hotspot_angles // np.deg2rad(self.angle_discretization_deg)).astype(int) 
        bins = bins % int(360 / self.angle_discretization_deg)

        cam_scores = np.zeros(360 // self.angle_discretization_deg)
        np.add.at(cam_scores, bins, heat_scores)

        return cam_scores

    def get_all_scores(self, bin_scores: np.ndarray, goal_heading: np.ndarray, goal_range: float):
        """
        LRN accumulation.
        Score each bin using:
            - Prev Heading
            - Goal Heading
            - Frontier Heading
        """
        bs_sum = np.sum(bin_scores)
        if bs_sum > 0.001:
            bin_scores = bin_scores / bs_sum

        beta = self.beta_degradation
        uniform = np.ones_like(bin_scores) / len(bin_scores)
        bin_scores = beta * bin_scores + (1.0 - beta) * uniform

        # Add goal scoring
        goal_std = self.goal_std
        gaussian_center = np.rad2deg(np.arctan2(goal_heading[1], goal_heading[0])) % 360
        if (
            self.honing_range > 0
            and goal_range < self.honing_range
        ):
            goal_std *= goal_range / self.honing_range

        goal_scores = self.get_gaussian_score(gaussian_center, goal_std)

        # Prev heading consistency score
        prev_heading_scores = np.ones_like(goal_scores)
        if self.prev_heading is not None:
            prev_heading_scores = self.get_gaussian_score(
                self.prev_heading, self.prev_std
            )

        # Combine
        if self.filtered_scores is not None:
            self.filtered_scores = (
                self.ema_alpha * bin_scores
                + (1 - self.ema_alpha) * self.filtered_scores
            )
        else:
            self.filtered_scores = bin_scores

        total_scores = goal_scores * self.filtered_scores * prev_heading_scores

        self.prev_heading = (
            np.argmax(total_scores)
            * self.angle_discretization_deg
        )

        # Prevent nans on 0 scores
        if total_scores.sum() > 0.001:
            total_scores /= total_scores.sum()

        return np.vstack(
            (
                total_scores,
                goal_scores,
                bin_scores,
                prev_heading_scores,
            )
        )
    
    def get_gaussian_score(self, center_deg, std):
        angle_bins = np.arange(0, 360, self.angle_discretization_deg)
        # Find minimum circular distance between gaussian center and each angle. Must take min of A - B and B - A bc of circle.
        dists = np.minimum(
            (angle_bins - center_deg) % 360,
            (center_deg - angle_bins) % 360
        )
        gaussian = np.exp(-0.5 * (dists / std) ** 2)
        gaussian /= gaussian.sum()
        return gaussian

    def publish_scores_debug(self, scores, odom_from_baselink):

        yaw = R.from_quat([
                odom_from_baselink.transform.rotation.x,
                odom_from_baselink.transform.rotation.y,
                odom_from_baselink.transform.rotation.z,
                odom_from_baselink.transform.rotation.w,
        ]).inv().as_euler("xyz", degrees=True)[2]

        yaw_roll = int(yaw // self.angle_discretization_deg)
        centering_roll = int(scores.shape[1] / 2)
        all_scores = np.flip(
            np.roll(scores, shift=centering_roll + yaw_roll, axis=1),
            axis=1,
        )
        total_scores_np = all_scores[0, :]
        goal_scores_np = all_scores[1, :]
        bin_scores_np = all_scores[2, :]
        goal_scores_np *= bin_scores_np.max()  # For viz

        img_array = self.draw_barchart(
            np.array([goal_scores_np, total_scores_np, bin_scores_np])
        )

        # Publish
        scores_msg = self.br.cv2_to_imgmsg(img_array, encoding="rgb8")
        self.scores_debug_publisher.publish(scores_msg)

    def draw_barchart(self, data, width=800, height=600):
        # Create a new image with white background
        image = PIL.Image.new("RGB", (width, height), "white")
        draw = PIL.ImageDraw.Draw(image)

        # Padding for labels
        padding_left = 100
        padding_bottom = 50
        padding_top = 20
        padding_right = 20

        # Chart area dimensions
        chart_width = width - padding_left - padding_right
        chart_height = height - padding_bottom - padding_top

        max_value = data.max()
        bar_width = int(chart_width / (360 / self.angle_discretization_deg))
        if max_value > 0:

            # Draw bars
            colors = ["yellow", "green", "red"]
            for didx, data_column in enumerate(data):
                for idx, value in enumerate(data_column):
                    bar_height = int((value / max_value) * chart_height)
                    x0 = padding_left + idx * bar_width
                    y0 = height - padding_bottom - bar_height
                    x1 = padding_left + (idx + 1) * bar_width
                    y1 = height - padding_bottom
                    draw.rectangle([x0, y0, x1, y1], fill=colors[didx])

            # Draw X-axis
            draw.line(
                [
                    (padding_left, height - padding_bottom),
                    (width - padding_right, height - padding_bottom),
                ],
                fill="black",
            )

            # Draw Y-axis
            draw.line(
                [(padding_left, padding_top), (padding_left, height - padding_bottom)],
                fill="black",
            )

            # Draw X-axis labels from -180 to 180
            font_ticks = PIL.ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 12
            )
            for i, angle in enumerate(range(-180, 180, self.angle_discretization_deg)):
                if i % 2 == 0:
                    x_pos = padding_left + i * bar_width + bar_width // 2
                    draw.text(
                        (x_pos, height - padding_bottom + 5),
                        str(angle),
                        fill="black",
                        anchor="mt",
                        font=font_ticks,
                    )

            # Draw Y-axis labels (Score)
            font_ticks = PIL.ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14
            )
            for i in range(0, 10):
                y_pos = height - padding_bottom - int((i / 10) * chart_height)
                draw.text(
                    (padding_left - 50, y_pos),
                    str(round(max_value * (i / 10), 2)),
                    fill="black",
                    # anchor="mr",
                    font=font_ticks,
                )

            font = PIL.ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 20
            )

            # Draw "Score" label vertically
            score_label = "Score"
            score_label_size = draw.textlength(score_label)
            draw.text(
                (
                    padding_left - score_label_size - 80,
                    (padding_top + chart_height) // 2
                    + 20 // 2,  # TODO hardcoded fontsize
                ),
                score_label,
                fill="black",
                anchor="mm",
                font=font,
            )

            # Draw "Angles" label
            angle_label = "Angles"
            draw.text(
                (width // 2, height - padding_bottom + 30),
                angle_label,
                fill="black",
                font=font,
                anchor="mt",
            )

        return np.array(image)
    
    def publish_ring(self, scores, robot_position):
        total_scores_np = scores[2, :]
        total_scores_np /= total_scores_np.sum()  # Keep distribution shape
        total_scores_np *= 1.0 / total_scores_np.max()  # Scale to [0, 1]

        # Ring properties
        ring_radius = self.costmap_range + 3.5  # meters
        num_segments = int(
            360 / self.angle_discretization_deg
        )  # Number of sections in the ring

        marker_array = MarkerArray()
        for i in range(num_segments):
            marker = Marker()
            marker.header.frame_id = self.global_frame
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.id = i

            # Set position -- needed to prevent rviz warning
            marker.pose.position.x = 0.0
            marker.pose.position.y = 0.0
            marker.pose.position.z = 0.0
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0

            # Calculate start and end angles for the segment
            angle_start = i * 2 * np.pi / num_segments
            angle_end = (i + 1) * 2 * np.pi / num_segments

            # Generate points for the segment
            num_points = 2
            for j in range(num_points + 1):
                angle = angle_start + (angle_end - angle_start) * (j / num_points)
                x = (ring_radius * np.cos(angle)) + robot_position.x
                y = (ring_radius * np.sin(angle)) + robot_position.y
                marker.points.append(Point(x=x, y=y, z=0.0))

            # Assign color
            marker.color = self.get_score_color(total_scores_np[i])

            # Marker properties
            marker.scale.x = 0.5  # Line thickness
            marker.lifetime = MarkerDuration(sec=0)  # Persistent
            marker_array.markers.append(marker)
        self.heatring_publisher.publish(marker_array)

    def get_score_color(self, score):
        cmap = matplotlib.colormaps["inferno"]
        mpl_color = cmap(score)
        return ColorRGBA(r=mpl_color[0], g=mpl_color[1], b=mpl_color[2], a=mpl_color[3])

    def visualize_model_det(self, nav_data):
        fig, axes = plt.subplots(2, self.num_cameras, figsize=(17, 8))
        
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

        plt.tight_layout()
        
        # Convert plot to image
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:,:,1:]
        plt.close(fig)

        # Publish the image
        self.model_viz_pub.publish(self.br.cv2_to_imgmsg(data, encoding="rgb8"))

    def publish_pred_goal(self, goal_pos: np.ndarray, stamp):
        # publish nav2_goal as a Marker
        nav2_goal_marker = Marker()
        nav2_goal_marker.header.frame_id = self.global_frame
        nav2_goal_marker.header.stamp = stamp
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
        nav2_goal_marker.pose.position.x = goal_pos[0]
        nav2_goal_marker.pose.position.y = goal_pos[1]
        nav2_goal_marker.pose.position.z = goal_pos[2]

        self.nav2_marker_publisher.publish(nav2_goal_marker)

        nav2_goal = PoseStamped()
        nav2_goal.header.frame_id = self.global_frame
        nav2_goal.header.stamp = self.get_clock().now().to_msg()
        nav2_goal.pose.position.x = goal_pos[0]
        nav2_goal.pose.position.y = goal_pos[1]
        nav2_goal.pose.position.z = goal_pos[2]

        self.nav2_goal_publisher.publish(nav2_goal)

    def publish_arrow_marker(self, heading, robot_position, color, id, action, stamp):
        marker = Marker()
        marker.header.frame_id = self.global_frame
        marker.header.stamp = stamp
        marker.type = 0
        marker.id = id
        marker.scale.x = 2.0
        marker.scale.y = 0.5
        marker.scale.z = 1.0
        marker.color = color
        marker.action = action
        dist = self.costmap_range + 4
        marker.pose = Pose()
        marker.pose.position.x = robot_position.x + dist * np.cos(heading)
        marker.pose.position.y = robot_position.y + dist * np.sin(heading)
        marker.pose.position.z = robot_position.z
        quat = transformations.quaternion_from_euler(0.0, 0.0, heading)
        marker.pose.orientation.x = quat[0]
        marker.pose.orientation.y = quat[1]
        marker.pose.orientation.z = quat[2]
        marker.pose.orientation.w = quat[3]
        self.direction_pub.publish(marker)

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
                for i in range(self.num_cameras+1):
                    src_frame = self.cam_tf_frame.format(CAMERA_MAPPING[i]) if i < self.num_cameras else self.body_frame
                    try:
                        tf_oldest_msg = self.tf_buffer.lookup_transform(
                            self.global_frame,
                            src_frame,
                            old_msg["odom"].header.stamp,
                            timeout=Duration(seconds=0)
                        )
                        tfs.append(tf_oldest_msg)

                    except Exception as e:
                        self.get_logger().info(f"TF not found for {src_frame} at time {old_msg_ts}: {e}")
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
    conf = package_share_directory / "configs" / "lrn_conf_marsyard.yaml"

    lrn_node = LRN(OmegaConf.load(conf))
    rclpy.spin(lrn_node)

    lrn_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
