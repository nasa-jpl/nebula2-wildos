import os
import rclpy
from ament_index_python.packages import get_package_share_directory

from enum import Enum

# from image_transport_py import ImageTransport
from nav_msgs.msg import Odometry
from visualization_msgs.msg import MarkerArray
from sensor_msgs.msg import CompressedImage, Image as ImageMsg, CameraInfo, PointCloud2
from message_filters import ApproximateTimeSynchronizer, Subscriber
from cv_bridge import CvBridge
from sensor_msgs_py import point_cloud2

import cv2
from pathlib import Path
from omegaconf import OmegaConf
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
from torchvision import transforms

from visual_navigation.utils.tf_lookup_sub import TFEdge, TFLookupSubscriber
from visual_navigation.utils.object_search_utils import localize_query
from visual_navigation.third_party.nvidia_radio.radio_downstream import RADIODownstreamInference
from visual_navigation.radio_triangulation.triangulator_viz import TriangulationViz

from triangulation3d.camera_data import Camera
from triangulation3d.particle_generator import ParticleGenerator
from triangulation3d.bbox_generator import BoundingBoxGenerator
from triangulation3d.triangulator import Triangulator

HOME_DIR = Path.home()

CAMERA_MAPPING = {
    0: "front",
    1: "left",
    2: "right"
}

class RobotName(Enum):
    SPOT1 = "spot1"
    GHOST1 = "ghost1"
    HUSKY1 = "husky1"


RobotCameraInfo = {
    RobotName.SPOT1: {
        "num_cameras": 3,
        "camera_mapping": CAMERA_MAPPING,
        "inverted_cameras": [0, 1, 2]  # all cameras are inverted
    },
}


class RadioTriangulator(TFLookupSubscriber):
    default_config = {
        # Model Params
        "model_version": "c-radio_v3-b",
        "adaptor_version": "siglip2",
        "use_naclip": True,
        "use_summary_for_spatial": True,
        "radio_dim": 768,
        "static_scale_factor": 0.75,
        "model_precision": "FP16",

        # Nav Params
        "num_cameras": 3,

        # ROS2 frames and topics
        "parent_frame": "{robot_name}/odom",
        "cam_frame": "{robot_name}/realsense/{cam_name}_color_optical_frame",
        "lidar_frame": "{robot_name}/ouster/front/os_lidar",
        "camera_img_topic": "realsense/{cam_name}/color/image_raw/compressed",
        "camera_info_topic": "realsense/{cam_name}/color/camera_info",
        "lidar_topic": "ouster/front/points_filtered",
        "odometry_topic": "odom",

        # Publish topic names
        "model_viz_topic": "~/query_visualization",
        "triangulated_object_topic": "~/triangulated_objects",
        "particle_viz_topic": "~/query_hypotheses",

        # ROS2 subscriber params
        "qos_history_depth": 1,
        "syncsub_queue_size": 2,
        "syncsub_slop": 0.2,

        # Object Search Params
        "object_search_config": {
            "text_queries": ["car", "tree"],
            "pixel_level_seg": False,
            "mask_threshold": 0.095,
        },

        # Triangulation config
        "max_views": 350,
        "min_lidar_points": 150,
        "min_view_distance": 1.0,
        "particle_generator_config":{
            "num_particles": 1000,
            "depth_range": [1.0, 100.0],
            "add_odom_drift": False,
        },
        "use_mask_for_projection": True,

        # TFLookup Config
        "tf_lookup_config": {
            "buffer_size": 1,
            "cache_time": 10,
            "timer_duration": 0.1,
            "lookup_timeout": 0,
            "qos_history_depth": 1,
            "wait_for_oldest": True,
            "clear_buffer_on_process": True,
            "spin_thread": False,
        }
    }

    def __init__(self, config: OmegaConf=OmegaConf.create()):

        config = OmegaConf.merge(OmegaConf.create(self.default_config), config)

        # Model path initialization
        self.model_path = HOME_DIR / "ckpts" / config.model_version
        self.adaptor_ckpt_path = HOME_DIR / "ckpts" / config.adaptor_version

        # init model before to prevent tf listener from spinning too early
        np.random.seed(42)
        self.init_model(config)

        super().__init__(
            node_name='radio_triangulator',
            config=config.tf_lookup_config
        )
        self.get_logger().info(f"Model filepath: {self.model_path}")
        self.get_logger().info(f"Adaptor path location: {self.adaptor_ckpt_path}")

        self.get_logger().info('Finished initializing models!')

        # set ros2 parameter for robot name
        self.declare_parameter("robot_name", RobotName.SPOT1.value)
        self.robot_name = RobotName(self.get_parameter("robot_name").value)

        # Used to convert between ROS and OpenCV images
        self.br = CvBridge()

        # Robot parameters and initializations
        self.num_cameras = RobotCameraInfo[self.robot_name]["num_cameras"]
        assert self.num_cameras in [1, 3], "Only 1 or 3 cameras are supported."

        self.cam_inverted = RobotCameraInfo[self.robot_name]["inverted_cameras"]

        # Triangulation parameters and initializations
        self.view_data = {
            query: {
                "views": [],
                "last_added_pos": None,
                "triangulated_position": None,
                "found_lidar_in_mask": False,
            }
            for query in self.text_queries
        }
        self.max_views = config.max_views
        self.min_lidar_points = config.min_lidar_points
        self.min_view_distance = config.min_view_distance
        self.use_mask_for_projection = config.use_mask_for_projection

        self.particle_generator = ParticleGenerator(config.particle_generator_config)
        self.triangulator = Triangulator()

        # Visualization
        self.viz = TriangulationViz(
            camera_mapping=CAMERA_MAPPING,
            num_cameras=self.num_cameras
        )
        
        self.clbk_cntr = 0

        # Frames and Topic Names
        self.global_frame = config.parent_frame.format(robot_name=self.robot_name.value)
        self.cam_tf_frame = config.cam_frame.format(robot_name=self.robot_name.value, cam_name="{cam_name}")
        self.using_compressed_imgs = "compressed" in config.camera_img_topic

        # TFLookup
        self.required_transforms = {}
        for idx in range(self.num_cameras):
            self.required_transforms[f"world_from_cam{idx}"] = TFEdge(
                source_frame=self.cam_tf_frame.format(cam_name=CAMERA_MAPPING[idx]),
                target_frame=self.global_frame
            )
            self.required_transforms[f"cam{idx}_from_lidar"] = TFEdge(
                source_frame=config.lidar_frame.format(robot_name=self.robot_name.value),
                target_frame=self.cam_tf_frame.format(cam_name=CAMERA_MAPPING[idx])
            )

        # Subscribers and Publishers
        self.init_publishers(config)
        self.init_subscribers(config)
        self.start_timer()

    def init_model(self, config):
        # vlm initializations
        self.device = "cuda"

        # radio model
        self.model = RADIODownstreamInference(
            frontier_ckpt=None,
            traversability_ckpt= None,
            model_version=self.model_path,
            adaptor_version=config.adaptor_version,
            adaptor_ckpt_path=self.adaptor_ckpt_path,
            use_naclip=config.use_naclip,
            use_summary_for_spatial=config.use_summary_for_spatial,
            radio_dim=config.radio_dim,
            static_scale_factor=config.static_scale_factor,
            model_precision=config.model_precision,
            device=self.device,
        )
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.text_queries = config.object_search_config.text_queries
        self.pixel_level_seg = config.object_search_config.pixel_level_seg
        self.mask_threshold = config.object_search_config.mask_threshold

        self.text_feats = self.model.forward_on_text(self.text_queries)

    def init_publishers(self, config: OmegaConf):
        self.model_viz_pub = self.create_publisher(
            ImageMsg, config.model_viz_topic, 10
        )
        self.triangulated_obj_publisher = self.create_publisher(
            MarkerArray, config.triangulated_object_topic, 10
        )
        self.particle_viz_publisher = self.create_publisher(
            PointCloud2, config.particle_viz_topic, 10
        )

    def init_subscribers(self, config):

        cameraimg_topic_str = config.camera_img_topic
        camerainfo_topic_str = config.camera_info_topic
        img_msg_type = CompressedImage if self.using_compressed_imgs else ImageMsg

        self.camera_subs = {}
        for i in range(self.num_cameras):
            self.camera_subs[i] = {
                "image": Subscriber(
                    self, img_msg_type, cameraimg_topic_str.format(cam_name=CAMERA_MAPPING[i]), qos_profile=config.qos_history_depth
                ),
                "info": Subscriber(
                    self, CameraInfo, camerainfo_topic_str.format(cam_name=CAMERA_MAPPING[i]), qos_profile=config.qos_history_depth
                )
            }
        self.odom_sub = Subscriber(
            self, Odometry, config.odometry_topic, qos_profile=config.qos_history_depth
        )
        self.lidar_sub = Subscriber(
            self, PointCloud2, config.lidar_topic, qos_profile=config.qos_history_depth
        )
        ts_subs = [self.odom_sub, self.lidar_sub]

        for i in range(self.num_cameras):
            ts_subs.append(self.camera_subs[i]["image"])
            ts_subs.append(self.camera_subs[i]["info"])

        self.ts = ApproximateTimeSynchronizer(
            ts_subs, queue_size=config.syncsub_queue_size, slop=config.syncsub_slop
        )
        self.ts.registerCallback(self.listener_callback)

    def listener_callback(self, odom_msg, lidar_msg, *msgs):
        self.clbk_cntr += 1

        self.get_logger().info(f"Received callback {self.clbk_cntr}")

        self.msg_buffer.add_msg(
            msg={
                "odom": odom_msg,
                "lidar": lidar_msg,
                "cam_msgs": msgs
            },
            stamp=odom_msg.header.stamp,
        )

    def do_processing(self, msg, tf_data):

        print(f"Started Heavy")

        # Extract messages
        odom_msg = msg["odom"]
        lidar_msg = msg["lidar"]
        msgs = msg["cam_msgs"]
        cur_pos = np.array([
            odom_msg.pose.pose.position.x,
            odom_msg.pose.pose.position.y,
            odom_msg.pose.pose.position.z
        ])

        # Extract camera images and info
        rgb_imgs, cam_info_msgs = [], []
        for cam_idx in range(self.num_cameras):
            if self.using_compressed_imgs:
                convert_func = self.br.compressed_imgmsg_to_cv2
            else:
                convert_func = self.br.imgmsg_to_cv2

            if cam_idx in self.cam_inverted:
                rgb_imgs.append(
                    np.rot90(convert_func(msgs[cam_idx * 2], desired_encoding='rgb8'), k=2)
                )
            else:
                rgb_imgs.append(
                    convert_func(msgs[cam_idx * 2], desired_encoding='rgb8')
                )
            cam_info_msgs.append(msgs[cam_idx * 2 + 1])


        # Model Forward pass
        binary_mask = self.get_object_masks(rgb_imgs) # (B, Q, H, W)
        self.model_viz_pub.publish(
            self.br.cv2_to_imgmsg(
                self.viz.visualize_model_det({
                    i: {
                        "image": rgb_imgs[i],
                        "masks": {
                            self.text_queries[q_idx]: binary_mask[i, q_idx]
                            for q_idx in range(len(self.text_queries))
                        }
                    }
                    for i in range(self.num_cameras)
                }), encoding="rgb8"
            )
        )
        if np.sum(binary_mask) == 0:
            self.get_logger().info("No object detected in any camera, skipping...")
            return
        
        processed_mask = []
        for cam_idx in range(self.num_cameras):
            if cam_idx in self.cam_inverted:
                processed_mask.append(np.rot90(binary_mask[cam_idx], k=2, axes=(1, 2)))
            else:
                processed_mask.append(binary_mask[cam_idx])
        binary_mask = np.array(processed_mask) # (B, Q, H, W)

        # Fetch camera intrinsics and extrinsics
        all_cam_data = []
        for i, cam_info_msg in enumerate(cam_info_msgs):
            cam_data = self.fetch_cam_intrinsics_extrinsics(
                cam_info_msg, tf_data[f"world_from_cam{i}"]
            )
            cam_data["camera_info"] = cam_info_msg
            cam_data["camera_tf"] = tf_data[f"world_from_cam{i}"]
            all_cam_data.append(cam_data)

        # Check if lidar points fall within the masks
        self.check_lidar_in_mask(all_cam_data, lidar_msg, binary_mask, tf_data)

        # Add views for triangulation
        binary_mask = binary_mask.transpose(1, 0, 2, 3)  # (Q, B, H, W)
        for q_idx, query in enumerate(self.text_queries):
            q_mask = binary_mask[q_idx]  # (B, H, W)
            
            if self.view_data[query]["found_lidar_in_mask"]:
                continue

            if np.sum(q_mask) == 0:
                self.get_logger().info(f"No object detected for query '{query}', skipping...")
                continue

            if self.view_data[query]["last_added_pos"] is not None:
                dist_moved = np.linalg.norm(cur_pos - self.view_data[query]["last_added_pos"])
                if dist_moved < self.min_view_distance:
                    self.get_logger().info(
                        f"Not enough movement since last view for query '{query}' ({dist_moved:.2f}m), skipping..."
                    )
                    continue

            self.add_views(all_cam_data, query, q_mask, q_idx)
            self.view_data[query]["last_added_pos"] = cur_pos
            if len(self.view_data[query]["views"]) >= 2:
                self.view_data[query]["triangulated_position"] = self.triangulator.triangulate(
                    self.view_data[query]["views"],
                )
                self.get_logger().info(f"Triangulated position for query '{query}': {self.view_data[query]['triangulated_position']}")

        # Publish triangulated positions
        self.triangulated_obj_publisher.publish(
            self.viz.get_triangulated_markers(
                self.view_data, self.global_frame, self.get_clock().now().to_msg()
            )
        )

        # Publish goal hypotheses
        goal_hyp = self.viz.get_goal_hypotheses(
            self.view_data, self.triangulator, frame_id=self.global_frame, stamp=self.get_clock().now().to_msg()
        )
        if goal_hyp is not None:
            self.particle_viz_publisher.publish(goal_hyp)

        print(f"Finished Heavy")

    def get_object_masks(self, rgb_imgs):
        """
        Get object masks from the model: per camera, per query.
        """
        rgb_tensors = [self.transforms(img.copy()) for img in rgb_imgs]
        batch_tensor = torch.stack(rgb_tensors)
        _, _, spatial_feats = self.model.forward(batch_tensor)

        if self.model.model_precision.is_fp16():
            spatial_feats = spatial_feats.half()

        _, binary_mask = localize_query(
            text_feats=self.text_feats,
            spatial_feats=spatial_feats,
            orig_img_shape=rgb_imgs[0].shape[:2],
            pixel_level_seg=self.pixel_level_seg,
            mask_threshold=self.mask_threshold
        )

        return binary_mask

    def check_lidar_in_mask(self, all_cam_data, lidar_msg, binary_mask, tf_data):
        """
        Check if LiDAR points fall within the object masks.
        """
        
        # Check if lidar points fall within the mask
        lidar_points_3d = point_cloud2.read_points(lidar_msg, field_names=("x", "y", "z"), skip_nans=True)
        lidar_points_3d = np.array([np.array(list(pt)) for pt in lidar_points_3d])  # Nx3

        for cam_id in range(self.num_cameras):
            
            cam_name = CAMERA_MAPPING[cam_id]
            h = all_cam_data[cam_id]["height"]
            w = all_cam_data[cam_id]["width"]
            if binary_mask.shape[2] != h or binary_mask.shape[3] != w:
                raise ValueError(f"Object mask shape {binary_mask.shape} does not match expected {(h, w)} for camera {cam_name}")

            if np.sum(binary_mask) == 0:
                self.get_logger().info(f"No object detected in camera {cam_name}, skipping...")
                continue

            cam_from_lidar = tf_data[f"cam{cam_id}_from_lidar"]
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

            R_wc = all_cam_data[cam_id]["R_wc"]
            t_wc = all_cam_data[cam_id]["t_wc"]

            # consider points lying inside the object mask for each query
            for q_idx, query in enumerate(self.text_queries):
                cam_obj_mask = binary_mask[cam_id, q_idx]  # HxW
                if np.sum(cam_obj_mask) == 0:
                    self.get_logger().info(f"{query} not detected in camera {cam_name}, skipping...")
                    continue

                inside_mask = cam_obj_mask[lidar_pix[:, 1], lidar_pix[:, 0]].astype(bool)
                lidar_points_q = lidar_points_cam[inside_mask]

                if lidar_points_q.shape[0] < self.min_lidar_points:
                    self.get_logger().info(f"Not enough lidar points found in {query} mask for camera {cam_name}, skipping...")
                    continue
                
                dists = np.linalg.norm(lidar_points_q, axis=-1)
                median_idx = np.argsort(dists)[len(dists)//2]
                triangulated_position_cam = lidar_points_q[median_idx]

                triangulated_position_world = R_wc @ triangulated_position_cam.reshape(3, 1) + t_wc  # 3x1
                self.view_data[query]["triangulated_position"] = triangulated_position_world.flatten()

                self.get_logger().info(
                    f"Triangulated {query} using LiDAR points in camera {cam_name}: {self.view_data[query]['triangulated_position']}"
                )
                self.view_data[query]["found_lidar_in_mask"] = True

    def add_views(self, all_cam_data, query, query_mask, query_idx):
        """
        Add camera views for triangulation.
        """
        if len(self.view_data[query]["views"]) >= self.max_views:
            self.get_logger().warn(f"Reached maximum number of views ({self.max_views}) for {query}, not adding more.")
            return

        for cam_id in range(self.num_cameras):
            cam_name = CAMERA_MAPPING[cam_id]
            cam_data = all_cam_data[cam_id]
            cam_q_mask = query_mask[cam_id]

            if np.sum(cam_q_mask) == 0:
                self.get_logger().info(f"No object detected in camera {cam_name}, skipping...")
                continue
            
            bbox = cv2.boundingRect(cam_q_mask.astype(np.uint8))
            x,y,w,h = bbox
            bbox = np.array([x,y,x+w,y+h])
            view = Camera(
                camera_info=cam_data["camera_info"],
                camera_tf=cam_data["camera_tf"],
                bounding_box=bbox,
                object_mask=cam_q_mask,
                image=None
            )
            view = BoundingBoxGenerator.generate_ray_from_bbox(view)

            view = self.particle_generator.generate_particles(
                view,
                use_mask=self.use_mask_for_projection,
                pcl_frame_id=self.global_frame,
                color=(self.viz.viz_colors[query_idx % len(self.viz.viz_colors)]*255).astype(np.uint8)
            )

            self.view_data[query]["views"].append(view)


def main(args=None):
    rclpy.init(args=args)
    import argparse
    
    # Separate ROS args from your custom args
    custom_args = rclpy.utilities.remove_ros_args(args)
    
    # Now parse the remaining (non-ROS) args with argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Configuration file name (YAML) for the node.")
    custom_args = parser.parse_args(custom_args[1:])

    conf_name = f"{custom_args.config}"
    if conf_name.endswith(".yaml") is False:
        conf_name += ".yaml"

    package_share_directory = Path(get_package_share_directory('visual_navigation'))
    conf = package_share_directory / "configs" / conf_name

    radio_triangulator_node = RadioTriangulator(OmegaConf.load(conf))
    rclpy.spin(radio_triangulator_node)

    radio_triangulator_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
