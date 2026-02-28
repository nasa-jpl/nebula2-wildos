import rclpy

from nav_msgs.msg import Odometry
from sensor_msgs.msg import CompressedImage, Image as ImageMsg, CameraInfo
from message_filters import ApproximateTimeSynchronizer, Subscriber
from graphnav_msgs.msg import NavigationGraph
from visualization_msgs.msg import MarkerArray
from cv_bridge import CvBridge

from pathlib import Path
from omegaconf import OmegaConf
import numpy as np
import torch
from torchvision import transforms

from img_vlms.utils.tf_lookup_sub import TFEdge
from img_vlms.utils.goal_navigator import GoalNavigator
from img_vlms.utils.scoring import ScoringGeometricFrontiers
from img_vlms.geofrontier_nav.geofrontier_to_image import GeoFrontierToImage
from img_vlms.geofrontier_nav.viz import VisualizeGeoFrontierScoring
from img_vlms.third_party.nvidia_radio.radio_downstream import RADIODownstreamInference

HOME_DIR = Path.home()
CAMERA_MAPPING = {
    0: "front",
    1: "left",
    2: "right"
}    

class GeoFrontierNav(GoalNavigator):
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

        # Nav graph params
        "frontiers_range": 12.0,
        "traversability_class": "default",

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
        "camera_info_topic": "/spot1/realsense/{}/color/camera_info",
        "odometry_topic": "/spot1/odom",
        "navigation_graph_topic": "/spot1/nav_graph",

        # ROS2 Publisher topics
        "model_viz_topic": "/spot1/model_visualization",
        "valid_geofrontiers_topic": "/spot1/within_range_geofrontiers",

        # ROS2 subscriber params
        "qos_history_depth": 1,
        "syncsub_queue_size": 2,
        "syncsub_slop": 0.2,

        # Goal Navigator Config
        "goal_navigator_config": {
            # Nav
            "goal_reach_radius": 2.0,
            "reach_in_2D": True,
            "waypoints": None,
            "overwrite_waypoints": True,

            # ROS2
            "waypoint_frame": "spot1/odom",
            "waypoint_topic": "/spot1/imgnav_waypoint",
            "waypoint_viz_topic": "/spot1/goal_waypoints",
            "goal_dir_viz_topic": "/spot1/goal_direction"
        },

        # TFLookup Config
        "tf_lookup_config": {
            "buffer_size": 1,       # number of messages
            "cache_time": 10,       # seconds
            "timer_duration": 0.1,  # seconds
            "lookup_timeout": 0,    # seconds
            "qos_history_depth": 1,  # depth for QoS profile
            "wait_for_oldest": True,  # whether to wait when buffer is full
            "clear_buffer_on_process": True,  # whether to clear buffer after processing
            "spin_thread": False,     # whether to spin tf listener in a separate thread
        }
    }

    def __init__(self, config: OmegaConf=OmegaConf.create()):
        config = OmegaConf.merge(OmegaConf.create(self.default_config), config)

        # init model before to prevent tf listener from spinning too early
        np.random.seed(42)
        self.init_model(config)

        super().__init__(
            node_name='geo_frontier_nav',
            nav_config=config.goal_navigator_config,
            tf_lookup_config=config.tf_lookup_config
        )
        self.get_logger().info('Finished initializing models!')

        # Used to convert between ROS and OpenCV images
        self.br = CvBridge()

        # Nav parameters and initializations
        self.num_cameras = config.num_cameras
        self.cam_inverted = config.cams_inverted
        assert self.num_cameras in [1, 3], "Only 1 or 3 cameras are supported."

        # Navgraph frontiers to img
        self.geofrontier_to_image = GeoFrontierToImage(
            camera_mapping=CAMERA_MAPPING,
            frontiers_range=config.frontiers_range,
            traversability_class=config.traversability_class,
            cams_inverted=self.cam_inverted,
            reach_in_2D=False
        )
        self.geofrontier_viz_colors = np.array([
            [0.528, 0.471, 0.701],
            [0.772, 0.432, 0.102],
            [0.572, 0.586, 0.0],
        ])
        self.viz = VisualizeGeoFrontierScoring(
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

    def init_model(self, config):
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

    def init_publishers(self, config: OmegaConf):
        self.model_viz_pub = self.create_publisher(
            ImageMsg,
            config.model_viz_topic,
            10
        )
        self.withinrange_geofront_pub = self.create_publisher(
            MarkerArray,
            config.valid_geofrontiers_topic,
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
        self.navgraph_sub = Subscriber(self, NavigationGraph, config.navigation_graph_topic, qos_profile=config.qos_history_depth)

        ts_subs = [self.odom_sub, self.navgraph_sub]
        for i in range(self.num_cameras):
            ts_subs.append(self.camera_subs[i]["image"])
            ts_subs.append(self.camera_subs[i]["info"])

        self.ts = ApproximateTimeSynchronizer(
            ts_subs, queue_size=config.syncsub_queue_size, slop=config.syncsub_slop
        )
        self.ts.registerCallback(self.listener_callback)

    def listener_callback(self, odom_msg, navgraph_msg, *msgs):
        self.clbk_cntr += 1

        self.get_logger().info(f"Received callback {self.clbk_cntr}")
        assert odom_msg.header.frame_id == self.global_frame, \
            f"Odom frame {odom_msg.header.frame_id} does not match global frame {self.global_frame}"
        assert navgraph_msg.header.frame_id == self.global_frame, \
            f"Navgraph frame {navgraph_msg.header.frame_id} does not match global frame {self.global_frame}"

        self.msg_buffer.add_msg(
            msg={
                "odom": odom_msg,
                "navgraph": navgraph_msg,
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
        navgraph_msg = msg["navgraph"]
        msgs = msg["cam_msgs"]

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

        # Get geofrontiers from navgraph_msg
        all_cam_data = []
        for i, cam_info_msg in enumerate(cam_info_msgs):
            cam_data = self.fetch_cam_intrinsics_extrinsics(
                cam_info_msg, tf_data[f"world_from_cam{i}"])
            all_cam_data.append(cam_data)

        try:
            geofrontiers = self.geofrontier_to_image.extract_geofrontiers(
                current_odom_msg=odom_msg,
                navgraph=navgraph_msg,
                all_cam_data=all_cam_data
            )
        except Exception as e:
            self.get_logger().error(f"Error extracting geofrontiers: {e}")
            return

        self.viz.delete_markers(self.withinrange_geofront_pub)
        self.withinrange_geofront_pub.publish(
            self.viz.viz_valid_geofrontiers(geofrontiers, all_cam_data, odom_msg.header, self.geofrontier_viz_colors))
        # DEBUG
        # self.model_viz_pub.publish(
        #     self.br.cv2_to_imgmsg(self.viz.visualize_geofrontiers_in_image(geofrontiers, rgb_imgs), encoding="rgb8")
        # )

        # Model Forward pass
        rgb_tensors = [self.transforms(img.copy()) for img in rgb_imgs]
        batch_tensor = torch.stack(rgb_tensors)
        batch_img_traversability, batch_img_frontiers, _ = self.model.forward(batch_tensor)

        batch_img_frontiers = batch_img_frontiers.cpu().numpy()
        batch_img_traversability = batch_img_traversability.cpu().numpy()

        # Scoring Geometric Frontiers
        nav_data = []
        _, goal_heading = self.compute_goal_range_heading(odom_msg)
        for i in range(self.num_cameras):
            cam_data = all_cam_data[i]
            if not geofrontiers[i]:
                # no geometric frontiers for this camera
                nav_data.append({
                    "image": rgb_imgs[i],
                    "traversability": batch_img_traversability[i][0],
                    "img_frontiers": batch_img_frontiers[i][0],
                })
                continue

            scores, paths, score_maps = self.scorer.score_geofrontiers(
                geometric_frontiers=geofrontiers[i]["frontier_pixel_coords"],
                img_frontiers=batch_img_frontiers[i][0],
                traversability=batch_img_traversability[i][0],
                goal_heading=goal_heading,
                cam_data=cam_data
            )
            chosen_frontier = np.argmax(scores)
            nav_data.append({
                "image": rgb_imgs[i],
                "traversability": batch_img_traversability[i][0],
                "img_frontiers": batch_img_frontiers[i][0],
                "score_map": score_maps[chosen_frontier],
                "geo_frontiers": geofrontiers[i]["frontier_pixel_coords"],
                "scores": scores,
                "paths": paths,
            })

        self.model_viz_pub.publish(
            self.br.cv2_to_imgmsg(self.viz.visualize_model_det(nav_data), encoding="rgb8")
        )
        self.publish_goal_waypoints()
        print(f"Finished Heavy")

def main(args=None):
    rclpy.init(args=args)
    from ament_index_python.packages import get_package_share_directory
    import os

    package_share_directory = Path(get_package_share_directory('img_vlms'))
    conf = package_share_directory / "configs" / "geofrontier_nav_conf.yaml"

    # imgfrontiernav_node = GeoFrontierNav(OmegaConf.load(conf))
    imgfrontiernav_node = GeoFrontierNav()
    rclpy.spin(imgfrontiernav_node)

    imgfrontiernav_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
