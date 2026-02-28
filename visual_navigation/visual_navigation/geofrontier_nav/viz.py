from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point

from omegaconf import OmegaConf
import numpy as np
import cv2
import matplotlib.pyplot as plt

from img_vlms.utils.viz import (
    make_subplot_grid, overlay_heatmap, draw_point, draw_text, draw_path, make_colorbar, pad_image
)

class VisualizeGeoFrontierScoring:
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


    def visualize_model_det(self, nav_data):
        """
        Visualize img_frontiers, traversability, and geo_frontier scores.
        """
        img_grid = {}
        num_rows = 4
        num_cols = self.num_cameras

        for i in range(self.num_cameras):
            plt_idx = self.cam_order[i]

            rgb_img = nav_data[i]["image"]
            rgb_img = cv2.resize(rgb_img, (0,0), fx=self.fig_resize_factor, fy=self.fig_resize_factor)
            img_grid[(0, plt_idx)] = (rgb_img, f"Image {self.camera_mapping[i]}")

            if "object_mask" in nav_data[i] and nav_data[i]["object_mask"] is not None:
                obj_mask = nav_data[i]["object_mask"].astype(np.float32)
                obj_mask = cv2.resize(obj_mask, (0,0), fx=self.fig_resize_factor, fy=self.fig_resize_factor)
                mask_overlay = overlay_heatmap(rgb_img, obj_mask, alpha=0.5)
                img_grid[(0, plt_idx)] = (mask_overlay, f"Image {self.camera_mapping[i]} + Obj Mask")

            # overlay frontiers on the image
            frontier_map = nav_data[i]["img_frontiers"].astype(np.float32)
            frontier_map = cv2.resize(frontier_map, (0,0), fx=self.fig_resize_factor, fy=self.fig_resize_factor)
            frontier_overlay = overlay_heatmap(rgb_img, frontier_map)
            img_grid[(1, plt_idx)] = (frontier_overlay, "Frontiers Overlay")

            # overlay traversability on the image
            traversability_map = nav_data[i]["traversability"].astype(np.float32)
            traversability_map = cv2.resize(traversability_map, (0,0), fx=self.fig_resize_factor, fy=self.fig_resize_factor)
            trav_overlay = overlay_heatmap(rgb_img, traversability_map)
            img_grid[(2, plt_idx)] = (trav_overlay, "Traversability Overlay")

            # Show projected geo_frontiers and paths to straight-line goal from camera
            path_overlay = rgb_img.copy()
            if "geo_frontiers" not in nav_data[i]:
                img_grid[(3, plt_idx)] = (None, "No Geometric Frontiers")
                continue

            geo_frontiers = nav_data[i]["geo_frontiers"] * self.fig_resize_factor
            scores = nav_data[i]["scores"]
            paths = nav_data[i]["paths"]
            score_map = nav_data[i]["score_map"].astype(np.float32)
            score_map = cv2.resize(score_map, (0,0), fx=self.fig_resize_factor, fy=self.fig_resize_factor)
            path_overlay = overlay_heatmap(path_overlay, score_map, alpha=0.5)
            
            for ((y,x), score, path) in zip(geo_frontiers, scores, paths):
                path = np.array(path) * self.fig_resize_factor
                color = plt.cm.jet(score)
                color = tuple(int(c * 255) for c in color[:3])

                draw_point(path_overlay, (y,x), color, radius=6)
                draw_point(path_overlay, path[-1], (255,255,255), radius=2)  # goal point
                draw_text(path_overlay, (y,x), f"{score:.2f}", color=(255,255,255))
                draw_path(path_overlay, path, color)

            img_grid[(3, plt_idx)] = (path_overlay, "Paths")


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

        return grid


    def viz_valid_geofrontiers(self, geofrontiers, all_cam_data, header, colors):
        """
        Visualize valid geometric frontier directions.
        """
        num_cams = len(geofrontiers)
        marker_array = MarkerArray()

        for cam_idx in range(num_cams):
            R_wc = all_cam_data[cam_idx]['R_wc']
            t_wc = all_cam_data[cam_idx]['t_wc'].T[0]
            cam_heading = R_wc @ np.array([0, 0, 1])
            cam_heading = cam_heading / np.linalg.norm(cam_heading)

            cam_marker = Marker()
            cam_marker.header = header
            cam_marker.ns = f"camera_heading_{self.camera_mapping[cam_idx]}"
            cam_marker.id = len(marker_array.markers)
            cam_marker.type = Marker.ARROW
            cam_marker.action = Marker.ADD
            start_pt = t_wc
            end_pt = t_wc + 1.0 * cam_heading
            cam_marker.points.append(Point(x=start_pt[0], y=start_pt[1], z=start_pt[2]))
            cam_marker.points.append(Point(x=end_pt[0], y=end_pt[1], z=end_pt[2]))
            cam_marker.scale.x = 0.1
            cam_marker.scale.y = 0.2
            cam_marker.scale.z = 0.2
            cam_marker.color.r = 1.0
            cam_marker.color.g = 1.0
            cam_marker.color.b = 0.0
            cam_marker.color.a = 1.0
            marker_array.markers.append(cam_marker)

            if not geofrontiers[cam_idx]:
                continue
            frontier_pos = geofrontiers[cam_idx]["frontier_positions"]
            frontier_heading = geofrontiers[cam_idx]["frontier_headings"]
            
            for pos, heading in zip(frontier_pos, frontier_heading):
                marker = Marker()
                marker.header = header
                marker.ns = f"geofrontier_{self.camera_mapping[cam_idx]}"
                marker.id = len(marker_array.markers)
                marker.type = Marker.ARROW
                marker.action = Marker.ADD

                start_pt = pos
                end_pt = pos + 3.0 * heading
                marker.points.append(Point(x=start_pt[0], y=start_pt[1], z=start_pt[2]))
                marker.points.append(Point(x=end_pt[0], y=end_pt[1], z=end_pt[2]))
                marker.scale.x = 0.1
                marker.scale.y = 0.2
                marker.scale.z = 0.2
                marker.color.r = colors[cam_idx, 0]
                marker.color.g = colors[cam_idx, 1]
                marker.color.b = colors[cam_idx, 2]
                marker.color.a = 1.0
                marker_array.markers.append(marker)

        return marker_array

    def delete_markers(self, publisher):
        markerarr = MarkerArray()
        marker = Marker()
        marker.action = 3
        markerarr.markers.append(marker)
        publisher.publish(markerarr)