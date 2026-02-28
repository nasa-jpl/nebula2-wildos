from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point

from omegaconf import OmegaConf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib
from matplotlib.gridspec import GridSpec


class VisualizeGeoFrontierScoring:
    default_config = {
        "fig_size": (10, 8),        # size of matplotlib figure
        "fig_resize_factor": 0.2,   # factor to resize imgs during plotting
    }

    def __init__(self, camera_mapping, num_cameras: int, config: OmegaConf=OmegaConf.create()):
        config = OmegaConf.merge(OmegaConf.create(self.default_config), config)
        self.camera_mapping = camera_mapping
        self.num_cameras = num_cameras
        self.fig_size = config.fig_size
        self.fig_resize_factor = config.fig_resize_factor

        if self.num_cameras == 1:
            self.cam_order = [0]  # Only one camera
        else:
            self.cam_order = [1, 0, 2]  # LEFT, FRONT, RIGHT

        self.main_fig = plt.figure(figsize=self.fig_size)
        gs = GridSpec(4, num_cameras + 1, figure=self.main_fig, width_ratios=[1]*num_cameras + [0.05])

        # Create axes
        self.main_axes = np.empty((4, num_cameras), dtype=object)
        for row in range(4):
            for col in range(num_cameras):
                self.main_axes[row, col] = self.main_fig.add_subplot(gs[row, col])
                self.main_axes[row, col].axis('off')

        # Add one shared colorbar axis
        cax = self.main_fig.add_subplot(gs[:, -1])
        cb = mpl.cm.ScalarMappable(cmap='jet', norm=mpl.colors.Normalize(vmin=0, vmax=1))
        self.main_fig.colorbar(cb, cax=cax, orientation='vertical')

        self.main_fig.tight_layout()

    def visualize_geofrontiers_in_image(self, geofrontiers, rgb_imgs):
        """
        DEBUG: Visualize geofrontiers in the given RGB images.
        """
        fig, axes = plt.subplots(1, self.num_cameras, figsize=self.fig_size)

        for i in range(self.num_cameras):
            rgb_img = rgb_imgs[i]
            axes[i].imshow(rgb_img)
            axes[i].set_title(f"Image {i}")

            if geofrontiers[i]:
                for (y, x) in geofrontiers[i]["frontier_pixel_coords"]:
                    axes[i].scatter(x, y, color='red', s=100)

        plt.tight_layout()
        
        # Convert plot to image
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:,:,1:]
        plt.close(fig)

        # Return the image
        return data

    def visualize_model_det(self, nav_data):
        """
        Visualize img_frontiers, traversability, and geo_frontier scores.
        """

        for i in range(self.num_cameras):
            plt_idx = self.cam_order[i]

            rgb_img = nav_data[i]["image"]
            rgb_img = cv2.resize(rgb_img, (0,0), fx=self.fig_resize_factor, fy=self.fig_resize_factor)
            self.main_axes[0, plt_idx].imshow(rgb_img)

            if "object_mask" in nav_data[i] and nav_data[i]["object_mask"] is not None:
                obj_mask = nav_data[i]["object_mask"].astype(np.float32)
                obj_mask = cv2.resize(obj_mask, (0,0), fx=self.fig_resize_factor, fy=self.fig_resize_factor)
                self.main_axes[0, plt_idx].imshow(obj_mask, alpha=0.5, cmap='gray')

            self.main_axes[0, plt_idx].set_title(f"Image {self.camera_mapping[i]}")

            # overlay frontiers on the image
            frontier_map = nav_data[i]["img_frontiers"].astype(np.float32)
            frontier_map = cv2.resize(frontier_map, (0,0), fx=self.fig_resize_factor, fy=self.fig_resize_factor)
            self.main_axes[1, plt_idx].imshow(rgb_img)
            hm = self.main_axes[1, plt_idx].imshow(frontier_map, alpha=0.5, cmap='jet', vmin=0, vmax=1)
            self.main_axes[1, plt_idx].set_title("Frontiers Overlay")

            # overlay traversability on the image
            traversability_map = nav_data[i]["traversability"].astype(np.float32)
            traversability_map = cv2.resize(traversability_map, (0,0), fx=self.fig_resize_factor, fy=self.fig_resize_factor)
            self.main_axes[2, plt_idx].imshow(rgb_img)
            hm = self.main_axes[2, plt_idx].imshow(traversability_map, alpha=0.5, cmap='jet', vmin=0, vmax=1)
            self.main_axes[2, plt_idx].set_title("Traversability Overlay")

            self.main_axes[3, plt_idx].imshow(rgb_img)
            if "geo_frontiers" not in nav_data[i]:
                continue

            geo_frontiers = nav_data[i]["geo_frontiers"] * self.fig_resize_factor
            scores = nav_data[i]["scores"]
            paths = nav_data[i]["paths"]
            score_map = nav_data[i]["score_map"].astype(np.float32)
            score_map = cv2.resize(score_map, (0,0), fx=self.fig_resize_factor, fy=self.fig_resize_factor)

            for ((y,x), score, path) in zip(geo_frontiers, scores, paths):
                path = np.array(path) * self.fig_resize_factor
                color = plt.cm.jet(score)  # Use jet colormap for scores

                self.main_axes[3, plt_idx].scatter(x, y, color=color, s=100, label=f"Frontier {self.camera_mapping[i]} ({score:.2f})")
                self.main_axes[3, plt_idx].text(x, y, f"{score:.2f}", color='black', fontsize=12, ha='center', va='center')
                self.main_axes[3, plt_idx].plot(path[:, 1], path[:, 0], color=color, linewidth=2, alpha=0.7)
                self.main_axes[3, plt_idx].scatter(path[-1, 1], path[-1, 0], color='white', marker='x', s=100)
            
            
            hm = self.main_axes[3, plt_idx].imshow(score_map, cmap='jet', vmin=0, vmax=1, alpha=0.5)
            self.main_axes[3, plt_idx].set_title("Paths")
            # self.main_axes[3, plt_idx].legend(loc='upper right', fontsize='small')
    
        # Convert plot to image
        self.main_fig.canvas.draw()
        data = np.frombuffer(self.main_fig.canvas.tostring_argb(), dtype=np.uint8)
        data = data.reshape(self.main_fig.canvas.get_width_height()[::-1] + (4,))[:,:,1:]
        
        # clear axes for next use
        for i in range(4):
            for j in range(self.num_cameras):
                self.main_axes[i, j].cla()
                self.main_axes[i, j].axis('off')

        # Return the image
        return data


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