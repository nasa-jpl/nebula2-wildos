from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point

from omegaconf import OmegaConf
import numpy as np
import cv2
import matplotlib.pyplot as plt

from visual_navigation.utils.viz import (
    make_subplot_grid, overlay_heatmap, draw_point, draw_text, draw_path, make_colorbar, pad_image
)

colors_rgb_u8 = {
    "orange": (251,151,39),
    "mangenta": (150,36,145),
    "blue": (67,110,176),
    "red": (210, 43,38),
    "cyan": (66, 173, 187),
    "green": (167,204,110),
    "red_light": (230,121,117),
    "orange_light": (252,188,115),
    "mangenta_light": (223,124,218),
    "blue_light": (137, 166,210),
    "cyan_light": (164, 216, 223), 
    "green_light": (192,218,152),
}
colors_rgb_f = { k: (float(v[0])/255.0,float(v[1])/255.0,float(v[2])/255.0)  for k,v in colors_rgb_u8.items()}

class TriangulationViz:
    default_config = {
        "fig_resize_factor": 0.2,   # factor to resize imgs during plotting
    }

    def __init__(self, camera_mapping, num_cameras: int, config: OmegaConf=OmegaConf.create()):
        config = OmegaConf.merge(OmegaConf.create(self.default_config), config)
        self.camera_mapping = camera_mapping
        self.num_cameras = num_cameras
        self.fig_resize_factor = config.fig_resize_factor

        if self.num_cameras == 1:
            self.cam_order = [0]  # Only one camera
        else:
            self.cam_order = [1, 0, 2]  # LEFT, FRONT, RIGHT

        self.viz_colors = np.array(list(colors_rgb_f.values()))

    def visualize_model_det(self, mask_data):
        """
        Visualize img_frontiers, traversability, and geo_frontier scores.
        """
        img_grid = {}
        num_rows = 4
        num_cols = self.num_cameras

        for i in range(self.num_cameras):
            plt_idx = self.cam_order[i]

            for q_idx, (query, mask) in enumerate(mask_data[i]["masks"].items()):
                rgb_img = mask_data[i]["image"]
                rgb_img = cv2.resize(rgb_img, (0,0), fx=self.fig_resize_factor, fy=self.fig_resize_factor)
                obj_mask = mask.astype(np.float32)
                obj_mask = cv2.resize(obj_mask, (0,0), fx=self.fig_resize_factor, fy=self.fig_resize_factor)
                mask_overlay = overlay_heatmap(rgb_img, obj_mask, alpha=0.5)
                img_grid[(q_idx, plt_idx)] = (mask_overlay, f"Image {self.camera_mapping[i]} + {query}")

        grid = make_subplot_grid(img_grid, (num_rows, num_cols), pad=15)

        return grid
                
    def get_triangulated_markers(self, view_data, global_frame, stamp):
        print("Generating triangulated markers...")
        markerarr = MarkerArray()
        for q_idx, query in enumerate(view_data.keys()):
            triangulated_position = view_data[query]["triangulated_position"]
            if triangulated_position is None:
                continue
            
            marker = Marker()
            marker.header.frame_id = global_frame
            marker.header.stamp = stamp
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.ns = f"triangulated_goal_{query}"
            marker.id = 0
            marker.pose.position.x = float(triangulated_position[0])
            marker.pose.position.y = float(triangulated_position[1])
            marker.pose.position.z = float(triangulated_position[2])
            marker.pose.orientation.w = 1.0
            marker.scale.x = 1.2
            marker.scale.y = 1.2
            marker.scale.z = 1.2
            marker.color.a = 1.0
            marker.color.r = self.viz_colors[q_idx % len(self.viz_colors)][0]
            marker.color.g = self.viz_colors[q_idx % len(self.viz_colors)][1]
            marker.color.b = self.viz_colors[q_idx % len(self.viz_colors)][2]

            markerarr.markers.append(marker)

        return markerarr

    def get_goal_hypotheses(self, view_data, triangulator, frame_id, stamp):
        # Combine the points from all the cameras for each query into a single PointCloud message
        views = []
        for query in view_data.keys():
            views.extend(view_data[query]["views"])
        if len(views) == 0:
            return None
        combined_pcl_msg = triangulator.combine_points(views, pcl_frame_id=frame_id)
        combined_pcl_msg.header.stamp = stamp
        return combined_pcl_msg

    def delete_markers(self, publisher):
        markerarr = MarkerArray()
        marker = Marker()
        marker.action = 3
        markerarr.markers.append(marker)
        publisher.publish(markerarr)