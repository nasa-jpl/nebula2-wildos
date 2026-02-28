import rclpy

from nav_msgs.msg import Odometry
from visualization_msgs.msg import MarkerArray
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

from img_vlms.utils.tf_lookup_sub import TFEdge
from img_vlms.utils.goal_navigator import GoalNavigator
from img_vlms.lrn.scoring import LRNScoring
from img_vlms.lrn.viz import LRNVisualizer
from img_vlms.third_party.nvidia_radio.radio_downstream import RADIODownstreamInference
from img_vlms.utils.object_search_utils import localize_query, get_objectmask_msg

HOME_DIR = Path.home()
CAMERA_MAPPING = {
    0: "front",
    1: "left",
    2: "right"
}    

class LRN(GoalNavigator):
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

        # LRN Params
        "hotspot_threshold": 0.7,  # Heat values below this threshold are ignored
        "frontier_opening_kernel_size": 0,  # Size of the kernel for morphological opening on frontier map
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

        # ROS2 frames and topics
        "parent_frame": "spot1/odom",
        "body_frame": "spot1/base_link",
        "cam_frame": "spot1/realsense/{}_color_optical_frame",
        "camera_img_topic": "/spot1/realsense/{}/color/image_raw/compressed",
        "camera_info_topic": "/spot1/realsense/{}/color/camera_info",
        "odometry_topic": "/spot1/odom",

        # Publish topic names
        "model_viz_topic": "/spot1/model_visualization",
        "score_viz_topic": "/spot1/lrn_scores_debug",
        "heatring_viz_topic": "/spot1/lrn_heatring",
        "nav2_goal_topic": "/spot1/goal_pose",

        # ROS2 subscriber params
        "qos_history_depth": 1,
        "syncsub_queue_size": 1,
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

        # TFLookup Config
        "tf_lookup_config": {
            "buffer_size": 1,
            "cache_time": 10,
            "timer_duration": 0.5,
            "lookup_timeout": 0,
            "qos_history_depth": 1,
            "wait_for_oldest": True,
            "clear_buffer_on_process": True,
            "spin_thread": False,
        }
    }

    def __init__(self, config: OmegaConf=OmegaConf.create(), do_object_search=False):
        config = OmegaConf.merge(OmegaConf.create(self.default_config), config)

        # init model before to prevent tf listener from spinning too early
        np.random.seed(42)
        self.init_model(config, do_object_search)

        super().__init__(
            node_name='lrn',
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
        

        # Visualization
        self.model_viz = LRNVisualizer(
            camera_mapping=CAMERA_MAPPING,
            num_cameras=self.num_cameras,
            angle_discretization_deg=config.angle_discretization_deg,
        )

        # Initialize LRN specific parameters
        self.angle_discretization_deg = config.angle_discretization_deg
        self.fixed_hotspot_dist = config.fixed_hotspot_dist
        self.confidence_threshold = config.confidence_threshold
        self.goal_lock_range = config.goal_lock_range
        self.costmap_range = config.costmap_range

        self.prev_heading = None
        self.scorer = LRNScoring(
            hotspot_threshold=config.hotspot_threshold,
            frontier_opening_kernel_size=config.frontier_opening_kernel_size,
            angle_discretization_deg=config.angle_discretization_deg,
            fixed_hotspot_dist=config.fixed_hotspot_dist,
            honing_range=config.honing_range,
            ema_alpha=config.ema_alpha,
            beta_degradation=config.beta_degradation,
            goal_std=config.goal_std,
            prev_std=config.prev_std,
            cam_inverted=self.cam_inverted
        )

        self.clbk_cntr = 0

        # Frames and Topic Names
        self.global_frame = config.parent_frame
        self.body_frame = config.body_frame
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
        self.required_transforms["world_from_body"] = TFEdge(
            source_frame=self.body_frame,
            target_frame=self.global_frame
        )

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
        self.markers_pub = self.create_publisher(
            MarkerArray,
            config.heatring_viz_topic,
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
        camerainfo_topic_str = config.camera_info_topic
        img_msg_type = CompressedImage if self.using_compressed_imgs else ImageMsg

        self.camera_subs = {}
        for i in range(self.num_cameras):
            self.camera_subs[i] = {
                "image": Subscriber(self, img_msg_type, cameraimg_topic_str.format(CAMERA_MAPPING[i]), qos_profile=config.qos_history_depth),
                "info": Subscriber(self, CameraInfo, camerainfo_topic_str.format(CAMERA_MAPPING[i]), qos_profile=config.qos_history_depth)
            }
        self.odom_sub = Subscriber(self, Odometry, config.odometry_topic, qos_profile=config.qos_history_depth)

        ts_subs = [self.odom_sub]
        for i in range(self.num_cameras):
            ts_subs.append(self.camera_subs[i]["image"])
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
        odom_from_baselink = tf_data["world_from_body"]

        odom_msg = msg["odom"]
        msgs = msg["cam_msgs"]
        goal_error, goal_heading = self.compute_goal_range_heading(odom_msg)

        # convert goal heading from vector to angle
        goal_heading = np.arctan2(goal_heading[1], goal_heading[0])

        if goal_error < self.goal_lock_range:
            self.handle_goal_lock(odom_msg)
            self.markers_pub.publish(self.model_viz.delete_markers())
            return

        # Extract camera images and info
        rgb_imgs, cam_info_msgs = [], []
        for i in range(self.num_cameras):
            if self.using_compressed_imgs:
                convert_func = self.br.compressed_imgmsg_to_cv2
            else:
                convert_func = self.br.imgmsg_to_cv2

            if self.cam_inverted:
                rgb_imgs.append(
                    np.rot90(convert_func(msgs[i * 2], desired_encoding='rgb8'), k=2)
                )
            else:
                rgb_imgs.append(
                    convert_func(msgs[i * 2], desired_encoding='rgb8')
                )
            cam_info_msgs.append(msgs[i * 2 + 1])

        all_cam_data = []
        for i, cam_info_msg in enumerate(cam_info_msgs):
            cam_data = self.fetch_cam_intrinsics_extrinsics(
                cam_info_msg, tf_data[f"world_from_cam{i}"]
            )
            all_cam_data.append(cam_data)

        # Model Forward pass
        rgb_tensors = [self.transforms(img.copy()) for img in rgb_imgs]
        batch_tensor = torch.stack(rgb_tensors)
        _, batch_img_frontiers, spatial_feats = self.model.forward(batch_tensor)
        
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

        # Pixels with object presence are always frontiers and traversable
        if self.object_search_mode:
            batch_img_frontiers = np.maximum(batch_img_frontiers, self.obj_frontier_score*binary_mask)

        # Scoring Bins
        model_pred_data = []
        bin_scores = []
        for i in range(self.num_cameras):
            img_frontiers = batch_img_frontiers[i, 0]  # (H, W)
            cam_data = all_cam_data[i]
            cam_bin_scores = self.scorer.score_bins_from_frontiers(img_frontiers, cam_data)
            model_pred_data.append({
                "image": rgb_imgs[i],
                "img_frontiers": img_frontiers,
                "object_mask": binary_mask[i][0] if self.object_search_mode else None,
            })
            bin_scores.append(cam_bin_scores)

        bin_scores = np.stack(bin_scores)
        combined_bin_scores = np.max(bin_scores, axis=0)    # (num_bins,)

        # Post process scores
        all_scores = self.scorer.get_final_scores(
            combined_bin_scores, self.prev_heading, goal_heading, goal_error
        )

        # Compute final heading and publish goal
        final_scores = all_scores['combined_scores']
        frontier_scores = all_scores['frontier_scores']
        if np.max(frontier_scores) < self.confidence_threshold:
            self.publish_nav2_goal(goal_heading, odom_msg)
            self.prev_heading = goal_heading
            self.get_logger().warn(f"Low confidence {np.max(frontier_scores):.3f}, sending goal direction")
        else:
            best_bin = np.argmax(final_scores)
            heading_angle = np.deg2rad(best_bin * self.angle_discretization_deg)
            self.publish_nav2_goal(heading_angle, odom_msg)
            self.prev_heading = heading_angle

        # Visualizations
        self.model_viz_pub.publish(
            self.br.cv2_to_imgmsg(
                self.model_viz.visualize_model_det(
                    model_pred_data, all_scores.copy(), odom_from_baselink), encoding="rgb8"
            )
        )
        self.markers_pub.publish(
            self.model_viz.visualize_heatring_and_headings(
                goal_heading,
                self.prev_heading,
                odom_msg.pose.pose.position,
                all_scores['frontier_scores'].copy(),
                self.costmap_range,
                frame_id=self.global_frame
            )
        )
        self.publish_goal_waypoints()

    def publish_nav2_goal(self, heading: float, odom_msg: Odometry):
        goal_msg = PoseStamped()
        goal_msg.header.frame_id = self.global_frame
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.position.x = odom_msg.pose.pose.position.x + self.fixed_hotspot_dist * np.cos(heading)
        goal_msg.pose.position.y = odom_msg.pose.pose.position.y + self.fixed_hotspot_dist * np.sin(heading)
        goal_msg.pose.position.z = 0.0

        orientation = R.from_euler('z', heading).as_quat()
        goal_msg.pose.orientation.x = orientation[0]
        goal_msg.pose.orientation.y = orientation[1]
        goal_msg.pose.orientation.z = orientation[2]
        goal_msg.pose.orientation.w = orientation[3]
        self.nav2_goal_publisher.publish(goal_msg)

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

    package_share_directory = Path(get_package_share_directory('img_vlms'))
    conf = package_share_directory / "configs" / conf_name

    lrn_node = LRN(OmegaConf.load(conf), do_object_search=custom_args.do_object_search)
    rclpy.spin(lrn_node)

    lrn_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
