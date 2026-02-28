
from omegaconf import OmegaConf

import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import matplotlib

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Pose
from builtin_interfaces.msg import Duration as MarkerDuration
from std_msgs.msg import ColorRGBA

from img_vlms.utils.viz import (
    make_subplot_grid, overlay_heatmap, make_histogram, make_colorbar, pad_image
)

class LRNVisualizer:
    default_config = {
        "fig_resize_factor": 0.2,   # factor to resize imgs during plotting
    }

    def __init__(self, camera_mapping, num_cameras: int, angle_discretization_deg: int, config: OmegaConf=OmegaConf.create()):
        self.config = OmegaConf.merge(OmegaConf.create(self.default_config), config)
        self.camera_mapping = camera_mapping
        self.num_cameras = num_cameras
        self.fig_resize_factor = self.config.fig_resize_factor
        self.angle_discretization_deg = angle_discretization_deg

        if self.num_cameras == 1:
            self.cam_order = [0]  # Only one camera
        else:
            self.cam_order = [1, 0, 2]  # LEFT, FRONT, RIGHT

    def visualize_model_det(self, model_pred_data, all_scores, odom_from_baselink):
        """
        Visualize model predictions including frontiers, scores, and headings.

        :param model_pred_data: List of dictionaries containing model predictions for each camera.
        :param all_scores: Dictionary containing scoring information.
        :param odom_from_baselink: TransformStamped representing tf odom from base_link.
        :return: Combined visualization image as a numpy array.
        """
        img_grid = {}
        num_rows = 2
        num_cols = self.num_cameras
        histogram_img_shape = None

        for i in range(self.num_cameras):
            plt_idx = self.cam_order[i]

            rgb_img = model_pred_data[i]["image"]
            rgb_img = cv2.resize(rgb_img, (0,0), fx=self.fig_resize_factor, fy=self.fig_resize_factor)
            img_grid[(0, plt_idx)] = (rgb_img, f"Image {self.camera_mapping[i]}")
            histogram_img_shape = rgb_img.shape[:2]

            if "object_mask" in model_pred_data[i] and model_pred_data[i]["object_mask"] is not None:
                obj_mask = model_pred_data[i]["object_mask"].astype(np.float32)
                obj_mask = cv2.resize(obj_mask, (0,0), fx=self.fig_resize_factor, fy=self.fig_resize_factor)
                mask_overlay = overlay_heatmap(rgb_img, obj_mask, alpha=0.5)
                img_grid[(0, plt_idx)] = (mask_overlay, f"Image {self.camera_mapping[i]} + Obj Mask")

            # overlay frontiers on the image
            frontier_map = model_pred_data[i]["img_frontiers"].astype(np.float32)
            frontier_map = cv2.resize(frontier_map, (0,0), fx=self.fig_resize_factor, fy=self.fig_resize_factor)
            frontier_overlay = overlay_heatmap(rgb_img, frontier_map)
            img_grid[(1, plt_idx)] = (frontier_overlay, "Frontiers Overlay")

        grid = make_subplot_grid(img_grid, (num_rows, num_cols), pad=15)
        cbar = make_colorbar(
            height=grid.shape[0] - 100,
            width=20,
            vmin=0,
            vmax=1,
            cmap=cv2.COLORMAP_JET,
            num_ticks=10,
            font_scale=0.5,
            pad=50
        )
        grid = cv2.hconcat([grid, cbar])

        # Get hisograms of all scores
        score_img = self.get_histogram_image(all_scores, histogram_img_shape, odom_from_baselink)

        # Pad imgs to same width
        width = max(grid.shape[1], score_img.shape[1])
        if grid.shape[1] < width:
            diff = width - grid.shape[1]
            grid = pad_image(grid, left=diff//2, right=diff - diff//2)
        if score_img.shape[1] < width:
            diff = width - score_img.shape[1]
            score_img = pad_image(score_img, left=diff//2, right=diff - diff//2)
        combined = cv2.vconcat([grid, score_img])
        return combined

    def get_histogram_image(self, all_scores, img_shape, odom_from_baselink):
        
        if odom_from_baselink is not None:
            # convert scores from global to base_link frame
            yaw = R.from_quat([
                odom_from_baselink.transform.rotation.x,
                odom_from_baselink.transform.rotation.y,
                odom_from_baselink.transform.rotation.z,
                odom_from_baselink.transform.rotation.w,
            ]).inv().as_euler("xyz", degrees=True)[2]

            # yaw roll to align with base_link frame
            yaw_roll = int(yaw // self.angle_discretization_deg)

            # centering roll to put forward in the middle i.e. from [0, 360] to [-180, 180]
            centering_roll = int(len(all_scores['combined_scores'])/ 2)

            for key in all_scores:
                # flip for [-180, 180] and roll to align with base_link
                all_scores[key] = np.flip(
                    np.roll(all_scores[key], yaw_roll + centering_roll)
                )

        # Create histogram image
        img_grid = {}
        num_rows = 1
        num_cols = len(all_scores)
        for i, (key, scores) in enumerate(all_scores.items()):
            img_grid[(0, i)] = (
                make_histogram(
                scores,
                np.arange(-180, 180 + self.angle_discretization_deg, self.angle_discretization_deg),
                img_shape,
                pad=20
            ),
                f"{key} (max: {np.max(scores):.2f})"
            )

        grid = make_subplot_grid(img_grid, (num_rows, num_cols), pad=15, title_font_scale=0.3)
        return grid
    
    def visualize_heatring_and_headings(
        self,
        goal_heading: float,
        predicted_heading: float,
        robot_position: Point,
        scores: np.ndarray,
        costmap_range: float,
        frame_id: str
    ):
        scores /= scores.sum()  # Keep distribution shape
        scores *= 1.0 / scores.max()  # Scale to [0, 1]

        # Ring properties
        ring_radius = costmap_range + 3.5  # meters
        num_segments = int(
            360 / self.angle_discretization_deg
        )  # Number of sections in the ring

        # colors
        cmap = matplotlib.colormaps["inferno"]
        
        marker_array = MarkerArray()
        for i in range(num_segments):
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.id = i
            marker.ns = "heatring"

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
            mpl_color = cmap(scores[i])
            marker.color = ColorRGBA(
                r=mpl_color[0], g=mpl_color[1], b=mpl_color[2], a=mpl_color[3]
            )

            # Marker properties
            marker.scale.x = 0.5  # Line thickness
            marker.lifetime = MarkerDuration(sec=0)  # Persistent
            marker_array.markers.append(marker)
        
        # Add arrow for goal heading and predicted heading
        dist = costmap_range + 4.0

        goal_marker = Marker()
        goal_marker.header.frame_id = frame_id
        goal_marker.type = Marker.ARROW
        goal_marker.id = 0
        goal_marker.ns = "goal_heading"
        goal_marker.scale.x = 2.0
        goal_marker.scale.y = 0.5
        goal_marker.scale.z = 1.0
        goal_marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
        goal_marker.pose = Pose()
        goal_marker.pose.position.x = robot_position.x + dist * np.cos(goal_heading)
        goal_marker.pose.position.y = robot_position.y + dist * np.sin(goal_heading)
        goal_marker.pose.position.z = robot_position.z
        quat = R.from_euler("z", goal_heading).as_quat()
        goal_marker.pose.orientation.x = quat[0]
        goal_marker.pose.orientation.y = quat[1]
        goal_marker.pose.orientation.z = quat[2]
        goal_marker.pose.orientation.w = quat[3]

        predicted_marker = Marker()
        predicted_marker.header.frame_id = frame_id
        predicted_marker.type = Marker.ARROW
        predicted_marker.id = 0
        predicted_marker.ns = "predicted_heading"
        predicted_marker.scale.x = 2.0
        predicted_marker.scale.y = 0.5
        predicted_marker.scale.z = 1.0
        predicted_marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
        predicted_marker.pose = Pose()
        predicted_marker.pose.position.x = robot_position.x + dist * np.cos(predicted_heading)
        predicted_marker.pose.position.y = robot_position.y + dist * np.sin(predicted_heading)
        predicted_marker.pose.position.z = robot_position.z
        quat = R.from_euler("z", predicted_heading).as_quat()
        predicted_marker.pose.orientation.x = quat[0]
        predicted_marker.pose.orientation.y = quat[1]
        predicted_marker.pose.orientation.z = quat[2]
        predicted_marker.pose.orientation.w = quat[3]

        marker_array.markers.append(goal_marker)
        marker_array.markers.append(predicted_marker)

        return marker_array
    
    def delete_markers(self):
        marker_array = MarkerArray()
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)
        return marker_array