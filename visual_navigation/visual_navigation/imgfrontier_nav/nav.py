import rclpy

from nav_msgs.msg import Odometry, Path as PathMsg
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CompressedImage, Image as ImageMsg, CameraInfo
from message_filters import ApproximateTimeSynchronizer, Subscriber
from object_search_msgs.msg import ObjectMaskWithTf
from cv_bridge import CvBridge

from pathlib import Path
from omegaconf import OmegaConf
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
from torchvision import transforms

from visual_navigation.utils.tf_lookup_sub import TFEdge
from visual_navigation.utils.goal_navigator import GoalNavigator
from visual_navigation.utils.scoring import ScoringGeometricFrontiers
from visual_navigation.geofrontier_nav.viz import VisualizeGeoFrontierScoring
from visual_navigation.utils.object_search_utils import localize_query, get_objectmask_msg
from visual_navigation.imgfrontier_nav.viz import get_path_msg
from visual_navigation.third_party.nvidia_radio.radio_downstream import RADIODownstreamInference

HOME_DIR = Path.home()
CAMERA_MAPPING = {
    0: "front",
    1: "left",
    2: "right"
}    

class ImgFrontierNav(GoalNavigator):
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
        "num_cameras": 3,
        "cams_inverted": True,

        # Pixel Scoring Params
        "frontier_threshold": 0.6,
        "traversability_threshold": 0.8,
        "frontier_opening_kernel_size": 0,
        "frontier_w": 2.0,
        "goal_w": 3.0,
        "reachability_w": 2.0,
        "scoring_method": "ADDITIVE",
        "reach_scale": 0.25,

        # ROS2 frames and topics
        "parent_frame": "spot1/odom",
        "cam_frame": "spot1/realsense/{}_color_optical_frame",
        "camera_img_topic": "/spot1/realsense/{}/color/image_raw/compressed",
        "camera_depth_topic": "/spot1/realsense/{}/aligned_depth_to_color/image_raw",
        "camera_info_topic": "/spot1/realsense/{}/color/camera_info",
        "odometry_topic": "/spot1/odom",

        # Publish topic names
        "model_viz_topic": "/spot1/model_visualization",
        "imgpath_topic": "/spot1/projected_img_path",
        "nav2_goal_topic": "/spot1/goal_pose",

        # ROS2 subscriber params
        "qos_history_depth": 1,
        "syncsub_queue_size": 2,
        "syncsub_slop": 0.2,

        # Object Search Params
        "object_search_config": {
            "text_queries": ["car"],
            "pixel_level_seg": True,
            "mask_threshold": 0.2,
            "obj_frontier_score": 1.0,
            "obj_trav_score": 0.8
        },

        # Goal Navigator Config
        "goal_navigator_config": {
            # Nav
            "goal_reach_radius": 2.0,
            "reach_in_2D": True,
            "waypoints": None,
            "overwrite_waypoints": True,

            # ROS2
            "waypoint_frame": "spot1/odom",
            "waypoint_topic": "spot1/imgnav_waypoint",
            "waypoint_viz_topic": "spot1/goal_waypoints",
            "goal_dir_viz_topic": "spot1/goal_direction"
        },
        "goal_lock_range": 6.0,  # meters
        "path_max_depth": 5.0,   # meters

        # TFLookup Config
        "tf_lookup_config": {
            "buffer_size": 1,       # number of messages
            "cache_time": 10,       # seconds
            "timer_duration": 0.5,  # seconds
            "lookup_timeout": 0,     # seconds
            "qos_history_depth": 1,  # depth for QoS profile
            "wait_for_oldest": True,  # whether to wait when buffer is full
            "clear_buffer_on_process": True,  # whether to clear buffer after processing
            "spin_thread": False,     # whether to spin tf listener in a separate thread
        }
    }

    def __init__(self, config: OmegaConf=OmegaConf.create(), do_object_search=False):
        config = OmegaConf.merge(OmegaConf.create(self.default_config), config)

        # init model before to prevent tf listener from spinning too early
        np.random.seed(42)
        self.init_model(config, do_object_search)

        super().__init__(
            node_name='img_frontier_nav',
            nav_config=config.goal_navigator_config,
            tf_lookup_config=config.tf_lookup_config
        )
        self.get_logger().info('Finished initializing models!')
        if self.object_search_mode:
            self.get_logger().info(f'Computed Text Feats for {self.text_queries}!')

        # Used to convert between ROS and OpenCV images
        self.br = CvBridge()

        # Nav parameters and initializations
        self.num_cameras = config.num_cameras
        self.cam_inverted = config.cams_inverted
        assert self.num_cameras in [1, 3], "Only 1 or 3 cameras are supported."
        self.goal_lock_range = config.goal_lock_range
        self.path_max_depth = config.path_max_depth

        # Visualization
        self.model_viz = VisualizeGeoFrontierScoring(
            camera_mapping=CAMERA_MAPPING,
            num_cameras=self.num_cameras
        )

        # Initialize pixel scoring params
        frontier_threshold = config.frontier_threshold
        traversability_threshold = config.traversability_threshold
        pixel_scoring_params = {
            "frontier": config.frontier_w,
            "goal": config.goal_w,
            "reachability": config.reachability_w,
            "method": config.scoring_method,
        }
        reach_scale = config.reach_scale
        self.scorer = ScoringGeometricFrontiers(
            pixel_scoring_params=pixel_scoring_params,
            frontier_threshold=frontier_threshold,
            frontier_opening_kernel_size=config.frontier_opening_kernel_size,
            traversability_threshold=traversability_threshold,
            reach_in_2D=self.reach_in_2D,
            cam_inverted=self.cam_inverted,
            reach_scale=reach_scale
        )
        self.img_start = lambda h, w: np.array([(h - 4, w//2)]) # (y,x): x is width, y is height

        self.clbk_cntr = 0

        # Frames and Topic Names
        self.global_frame = config.parent_frame
        self.cam_tf_frame = config.cam_frame
        self.using_compressed_imgs = "compressed" in config.camera_img_topic

        # TFLookup
        self.required_transforms = {
            f"world_from_cam{idx}": TFEdge(
                source_frame=self.cam_tf_frame.format(CAMERA_MAPPING[idx]),
                target_frame=self.global_frame
            )
            for idx in range(self.num_cameras)
        }

        # Subscribers and Publishers
        self.init_publishers(config)
        self.init_subscribers(config)
        self.start_timer()

    def init_model(self, config, do_object_search):
        # vlm initializations
        self.device = "cuda"

        if do_object_search and config.adaptor_version is None:
            config.adaptor_version = "siglip2"

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
        self.object_search_mode = False

        if do_object_search:
            self.object_search_mode = True
            self.text_queries = config.object_search_config.text_queries
            self.pixel_level_seg = config.object_search_config.pixel_level_seg
            self.mask_threshold = config.object_search_config.mask_threshold
            self.obj_frontier_score = config.object_search_config.obj_frontier_score
            self.obj_trav_score = config.object_search_config.obj_trav_score

            assert len(self.text_queries) == 1, "Only single object search is supported in this version."

            self.text_feats = self.model.forward_on_text(self.text_queries)

    def init_publishers(self, config: OmegaConf):
        self.model_viz_pub = self.create_publisher(
            ImageMsg,
            config.model_viz_topic,
            10
        )
        self.imgpath_publisher = self.create_publisher(
            PathMsg,
            config.imgpath_topic,
            10
        )
        self.nav2_goal_publisher = self.create_publisher(
            PoseStamped,
            config.nav2_goal_topic,
            10
        )
        if self.object_search_mode:
            self.object_mask_publisher = self.create_publisher(
                ObjectMaskWithTf,
                "/spot1/object_mask",
                10
            )

    def init_subscribers(self, config):

        cameraimg_topic_str = config.camera_img_topic
        cameradepth_topic_str = config.camera_depth_topic
        camerainfo_topic_str = config.camera_info_topic
        img_msg_type = CompressedImage if self.using_compressed_imgs else ImageMsg

        self.camera_subs = {}
        for i in range(self.num_cameras):
            self.camera_subs[i] = {
                "image": Subscriber(self, img_msg_type, cameraimg_topic_str.format(CAMERA_MAPPING[i]), qos_profile=config.qos_history_depth),
                "depth": Subscriber(self, ImageMsg, cameradepth_topic_str.format(CAMERA_MAPPING[i]), qos_profile=config.qos_history_depth),
                "info": Subscriber(self, CameraInfo, camerainfo_topic_str.format(CAMERA_MAPPING[i]), qos_profile=config.qos_history_depth)
            }
        self.odom_sub = Subscriber(self, Odometry, config.odometry_topic, qos_profile=config.qos_history_depth)

        ts_subs = [self.odom_sub]
        for i in range(self.num_cameras):
            ts_subs.append(self.camera_subs[i]["image"])
            ts_subs.append(self.camera_subs[i]["depth"])
            ts_subs.append(self.camera_subs[i]["info"])

        self.ts = ApproximateTimeSynchronizer(
            ts_subs, queue_size=config.syncsub_queue_size, slop=config.syncsub_slop
        )
        self.ts.registerCallback(self.listener_callback)

    def listener_callback(self, odom_msg, *msgs):
        self.clbk_cntr += 1

        self.get_logger().info(f"Received callback {self.clbk_cntr}")
        assert odom_msg.header.frame_id == self.global_frame, \
            f"Odom frame {odom_msg.header.frame_id} does not match global frame {self.global_frame}"

        self.msg_buffer.add_msg(
            msg={
                "odom": odom_msg,
                "cam_msgs": msgs
            },
            stamp=odom_msg.header.stamp,
        )

    def do_processing(self, msg, tf_data):
        if self.waypoints is None or self.current_wp_idx is None:
            self.get_logger().warn("No waypoints provided, skipping processing.")
            return

        print(f"Started Heavy")

        # Extract messages
        odom_msg = msg["odom"]
        msgs = msg["cam_msgs"]
        goal_error, goal_heading = self.compute_goal_range_heading(odom_msg)

        if goal_error < self.goal_lock_range:
            self.handle_goal_lock(odom_msg)
            return

        # Extract camera images and info
        rgb_imgs, depth_imgs, cam_info_msgs = [], [], []
        for i in range(self.num_cameras):
            if self.using_compressed_imgs:
                convert_func = self.br.compressed_imgmsg_to_cv2
            else:
                convert_func = self.br.imgmsg_to_cv2

            if self.cam_inverted:
                rgb_imgs.append(
                    np.rot90(convert_func(msgs[i * 3], desired_encoding='rgb8'), k=2)
                )
            else:
                rgb_imgs.append(
                    convert_func(msgs[i * 3], desired_encoding='rgb8')
                )
            depth_imgs.append(
                self.br.imgmsg_to_cv2(msgs[i * 3 + 1], desired_encoding='16UC1')
            )
            cam_info_msgs.append(msgs[i * 3 + 2])

        all_cam_data = []
        for i, cam_info_msg in enumerate(cam_info_msgs):
            cam_data = self.fetch_cam_intrinsics_extrinsics(
                cam_info_msg, tf_data[f"world_from_cam{i}"]
        )
            all_cam_data.append(cam_data)

        # Model Forward pass
        rgb_tensors = [self.transforms(img.copy()) for img in rgb_imgs]
        batch_tensor = torch.stack(rgb_tensors)
        batch_img_traversability, batch_img_frontiers, spatial_feats = self.model.forward(batch_tensor)
        
        if self.object_search_mode:
            if self.model.model_precision.is_fp16():
                spatial_feats = spatial_feats.half()

            text_sim_spatial, binary_mask = localize_query(
                text_feats=self.text_feats,
                spatial_feats=spatial_feats,
                orig_img_shape=rgb_imgs[0].shape[:2],
                pixel_level_seg=self.pixel_level_seg,
                mask_threshold=self.mask_threshold
            )
            if np.sum(binary_mask) > 0:
                tf_list = [tf_data[f"world_from_cam{i}"] for i in range(self.num_cameras)]
                self.object_mask_publisher.publish(
                    get_objectmask_msg(binary_mask, self.cam_inverted, odom_msg, tf_list, cam_info_msgs)
                )
            else:
                self.get_logger().warn("No object detected in the scene, skipping object mask publish.")

        batch_img_frontiers = batch_img_frontiers.cpu().numpy()
        batch_img_traversability = batch_img_traversability.cpu().numpy()
        h, w = batch_img_frontiers.shape[2:4]

        # Pixels with object presence are always frontiers and traversable
        if self.object_search_mode:
            batch_img_frontiers = np.maximum(batch_img_frontiers, self.obj_frontier_score*binary_mask)
            batch_img_traversability = np.maximum(batch_img_traversability, self.obj_trav_score*binary_mask)

        # Scoring Geometric Frontiers
        nav_data = []
        all_cam_scores = []
        for i in range(self.num_cameras):
            cam_data = all_cam_data[i]

            scores, paths, score_maps = self.scorer.score_geofrontiers(
                geometric_frontiers=self.img_start(h, w),
                img_frontiers=batch_img_frontiers[i][0],
                traversability=batch_img_traversability[i][0],
                goal_heading=goal_heading,
                cam_data=cam_data
            )
            nav_data.append({
                "image": rgb_imgs[i],
                "object_mask": binary_mask[i][0] if self.object_search_mode else None,
                "traversability": batch_img_traversability[i][0],
                "img_frontiers": batch_img_frontiers[i][0],
                "score_map": score_maps[0],
                "geo_frontiers": self.img_start(h, w),
                "scores": scores,
                "paths": paths,
            })
            all_cam_scores.append(scores[0])

        self.model_viz_pub.publish(
            self.br.cv2_to_imgmsg(
                self.model_viz.visualize_model_det(nav_data), encoding="rgb8"
            )
        )

        # Project path in img space to world and publish
        chosen_cam = np.argmax(all_cam_scores)
        self.compute_nav2_goal(
            self.cam_tf_frame.format(CAMERA_MAPPING[chosen_cam]),
            nav_data[chosen_cam]["paths"][0],
            all_cam_data[chosen_cam],
            depth_imgs[chosen_cam]
        )
        self.publish_goal_waypoints()
        print(f"Finished Heavy")

    def compute_nav2_goal(self, cam_frame_id, path, cam_data, depth_img):
        cam_path_3d = self.project_img_path(cam_frame_id, path, cam_data, depth_img)
        if len(cam_path_3d) == 0:
            self.get_logger().warn("No valid 3D path could be projected, skipping goal publish.")
            return
        
        # compute heading
        if len(cam_path_3d) == 1:
            heading = cam_path_3d[0] - np.array([0,0,0.1])
        else:
            heading = cam_path_3d[-1] - cam_path_3d[0]

        # convert to quaternion
        if np.linalg.norm(heading) > 0:
            heading = heading / np.linalg.norm(heading)
            heading = R.from_euler('y', np.arctan2(-heading[2], heading[0])).as_quat()
        else:
            raise ValueError(f"Something wrong with the projected 3D path: {cam_path_3d}")

        goal_msg = PoseStamped()
        goal_msg.header.frame_id = cam_frame_id
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.position.x = cam_path_3d[-1, 0]
        goal_msg.pose.position.y = cam_path_3d[-1, 1]
        goal_msg.pose.position.z = cam_path_3d[-1, 2]
        goal_msg.pose.orientation.x = heading[0]
        goal_msg.pose.orientation.y = heading[1]
        goal_msg.pose.orientation.z = heading[2]
        goal_msg.pose.orientation.w = heading[3]
        self.nav2_goal_publisher.publish(goal_msg)

    def project_img_path(self, frame_id, path, cam_data, depth_img):
        """
        Projects a 2D path in the image space to 3D space using the camera pose (and intrinsics) and depth image.
        """
        K = cam_data['K']
        h, w = depth_img.shape

        path = path[:, ::-1]  # (y,x) -> (x,y)
        if self.cam_inverted:
            path = np.array([w, h]) - path    # account for image rotation
        
        # choose path points within the max depth
        x_idx, y_idx = path[:, 0], path[:, 1]
        depths = depth_img[y_idx.astype(int), x_idx.astype(int)] / 1000.0  # Convert to meters
        valid_mask = (0 < depths) & (depths < self.path_max_depth)
        valid_points = path[valid_mask]
        path_depths = depths[valid_mask]

        depth_sort_idx = np.argsort(path_depths)
        valid_points = valid_points[depth_sort_idx]
        path_depths = path_depths[depth_sort_idx]
        if len(valid_points) == 0:
            self.get_logger().warn("No valid path points found within max depth.")
            return np.array([])

        # project to cam frame
        cam_path = np.hstack((valid_points, np.ones((valid_points.shape[0], 1))))
        cam_path = np.linalg.inv(K) @ cam_path.T
        cam_path = cam_path / cam_path[2]
        cam_path = cam_path.T
        cam_path = cam_path * path_depths[:, np.newaxis]
        self.imgpath_publisher.publish(
            get_path_msg(cam_path, frame_id, self.get_clock().now().to_msg())
        )
        
        return cam_path        

    def handle_goal_lock(self, odom_msg: Odometry):
        if self.waypoints is None:
            return
        goal_msg = PoseStamped()
        goal_msg.header.frame_id = self.global_frame
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.position.x = self.waypoints[self.current_wp_idx][0]
        goal_msg.pose.position.y = self.waypoints[self.current_wp_idx][1]
        goal_msg.pose.position.z = self.waypoints[self.current_wp_idx][2]
        goal_msg.pose.orientation = odom_msg.pose.pose.orientation
        self.nav2_goal_publisher.publish(goal_msg)
        self.get_logger().info(f"GOAL LOCK!!!!")


def main(args=None):
    rclpy.init(args=args)
    from ament_index_python.packages import get_package_share_directory
    import argparse
    
    # Separate ROS args from your custom args
    custom_args = rclpy.utilities.remove_ros_args(args)
    
    # Now parse the remaining (non-ROS) args with argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Configuration file name (YAML) for the node.")
    def str2bool(v):
        return v.lower() in ('true')
    parser.add_argument("--do_object_search", type=str2bool, default=False, help="Enable object search.")
    custom_args = parser.parse_args(custom_args[1:])

    conf_name = f"{custom_args.config}"
    if conf_name.endswith(".yaml") is False:
        conf_name += ".yaml"

    package_share_directory = Path(get_package_share_directory('visual_navigation'))
    conf = package_share_directory / "configs" / conf_name

    imgfrontiernav_node = ImgFrontierNav(OmegaConf.load(conf), do_object_search=custom_args.do_object_search)
    rclpy.spin(imgfrontiernav_node)

    imgfrontiernav_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
