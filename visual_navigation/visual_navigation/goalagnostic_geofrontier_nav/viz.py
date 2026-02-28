from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA

import numpy as np
import cv2
import matplotlib.pyplot as plt

from visual_navigation.geofrontier_nav.viz import VisualizeGeoFrontierScoring
from visual_navigation.utils.viz import (
    make_subplot_grid, overlay_heatmap, draw_point, draw_text, draw_path, make_colorbar, pad_image, show_mask
)


class VisualizeGoalAgnosticGeoFrontierScoring(VisualizeGeoFrontierScoring):
    def __init__(self, angular_bins: np.ndarray, **kwargs):
        super().__init__(**kwargs)
        self.discretization_angle = 2 * np.pi / len(angular_bins)
        self.bin_starts = angular_bins
        self.num_bins = len(angular_bins)

        # for visualizing the score map, for each camera show the score map for this heading
        self.goal_cam_relative_headings = 0.0
        self.ring_radius = 1.0  # radius of the score ring in meters

        # marker ids
        self.marker_id = 0
        self.uuid_to_marker_id = {}

    def visualize_navgraph(self, navgraph_msg, frame_id, stamp):
        """
        Visualize the navigation graph.
        """
        marker_array = MarkerArray()

        # Node marker (non-frontier nodes)
        node_marker = Marker()
        node_marker.header.frame_id = str(frame_id)
        node_marker.header.stamp = stamp
        node_marker.frame_locked = True
        node_marker.ns = "nodes"
        node_marker.id = 0
        node_marker.type = Marker.SPHERE_LIST
        node_marker.action = Marker.ADD
        node_marker.scale.x = 0.5
        node_marker.scale.y = 0.5
        node_marker.scale.z = 0.5
        node_marker.color = ColorRGBA()
        node_marker.color.a = 1.0
        node_marker.color.r = 0.0
        node_marker.color.g = 1.0
        node_marker.color.b = 0.0

        # Frontier node markers per traversability class
        frontier_node_markers = {}

        # Frontier points and lines from frontier point to its node
        frontier_point_marker = Marker()
        frontier_point_marker.header.frame_id = str(frame_id)
        frontier_point_marker.header.stamp = stamp
        frontier_point_marker.frame_locked = True
        frontier_point_marker.ns = "frontier_points"
        frontier_point_marker.id = 0
        frontier_point_marker.type = Marker.CUBE_LIST
        frontier_point_marker.action = Marker.ADD
        frontier_point_marker.scale.x = 0.3
        frontier_point_marker.scale.y = 0.3
        frontier_point_marker.scale.z = 0.3
        frontier_point_marker.color = ColorRGBA()
        frontier_point_marker.color.a = 1.0
        frontier_point_marker.color.r = 1.0
        frontier_point_marker.color.g = 0.0
        frontier_point_marker.color.b = 1.0

        frontier_point_to_node_marker = Marker()
        frontier_point_to_node_marker.header.frame_id = str(frame_id)
        frontier_point_to_node_marker.header.stamp = stamp
        frontier_point_to_node_marker.frame_locked = True
        frontier_point_to_node_marker.ns = "frontier_point_to_node"
        frontier_point_to_node_marker.id = 0
        frontier_point_to_node_marker.type = Marker.LINE_LIST
        frontier_point_to_node_marker.action = Marker.ADD
        frontier_point_to_node_marker.scale.x = 0.06
        frontier_point_to_node_marker.color = ColorRGBA()
        frontier_point_to_node_marker.color.a = 1.0
        frontier_point_to_node_marker.color.b = 1.0

        # Free radius counters and explored radius
        free_radius_counters = {}
        explored_radius_counter = 0

        # explored_radius reset marker (DELETEALL)
        explored_radius_reset = Marker()
        explored_radius_reset.header.frame_id = str(frame_id)
        explored_radius_reset.header.stamp = stamp
        explored_radius_reset.ns = "explored_radius"
        explored_radius_reset.id = explored_radius_counter
        explored_radius_reset.action = Marker.DELETEALL
        marker_array.markers.append(explored_radius_reset)

        # Edge and relative position markers
        edge_marker = Marker()
        edge_marker.header.frame_id = str(frame_id)
        edge_marker.header.stamp = stamp
        edge_marker.frame_locked = True
        edge_marker.ns = "edges"
        edge_marker.id = 1
        edge_marker.type = Marker.LINE_LIST
        edge_marker.action = Marker.ADD
        edge_marker.scale.x = 0.01
        edge_marker.color = ColorRGBA()
        edge_marker.color.a = 1.0
        edge_marker.color.r = 1.0
        edge_marker.color.g = 0.0
        edge_marker.color.b = 0.0

        rel_pos_marker = Marker()
        rel_pos_marker.header.frame_id = str(frame_id)
        rel_pos_marker.header.stamp = stamp
        rel_pos_marker.frame_locked = True
        rel_pos_marker.ns = "relative_positions"
        rel_pos_marker.id = 2
        rel_pos_marker.type = Marker.LINE_LIST
        rel_pos_marker.action = Marker.ADD
        rel_pos_marker.scale.x = 0.05
        rel_pos_marker.color = ColorRGBA()
        rel_pos_marker.color.a = 1.0
        rel_pos_marker.color.r = 0.0
        rel_pos_marker.color.g = 0.0
        rel_pos_marker.color.b = 1.0

        # Precompute node positions list for edges references
        node_positions = []
        for node in navgraph_msg.nodes:
            p = node.pose.position
            node_positions.append((p.x, p.y, p.z))

        # Iterate nodes and create markers
        for idx, node in enumerate(navgraph_msg.nodes):
            p = node.pose.position
            node_position = Point(x=p.x, y=p.y, z=p.z)

            for trav_idx, trav_prop in enumerate(node.trav_properties):
                try:
                    trav_class = str(navgraph_msg.trav_classes[trav_idx])
                except Exception:
                    trav_class = str(f"class_{trav_idx}")

                # If not frontier, add to node sphere list
                if not trav_prop.is_frontier:
                    node_marker.points.append(node_position)

                # Add explored radius sphere (use value from trav_prop if available)
                explored_radius = Marker()
                explored_radius.header.frame_id = str(frame_id)
                explored_radius.header.stamp = stamp
                explored_radius.ns = "explored_radius"
                explored_radius.id = explored_radius_counter + 1
                explored_radius_counter += 1
                explored_radius.action = Marker.ADD
                explored_radius.type = Marker.SPHERE
                explored_radius.scale.x = getattr(trav_prop, 'explored_radius', 0.0) * 2.0
                explored_radius.scale.y = getattr(trav_prop, 'explored_radius', 0.0) * 2.0
                explored_radius.scale.z = 0.01
                explored_radius.color = ColorRGBA()
                explored_radius.color.a = 0.2
                explored_radius.color.r = 0.0
                explored_radius.color.g = 0.0
                explored_radius.color.b = 1.0
                # position
                explored_radius.pose.position = node_position
                marker_array.markers.append(explored_radius)

                # Free radius per traversability class
                if trav_class not in free_radius_counters:
                    # add DELETEALL for this free radius namespace
                    free_radius_reset = Marker()
                    free_radius_reset.header.frame_id = str(frame_id)
                    free_radius_reset.header.stamp = stamp
                    free_radius_reset.ns = f"free_radius_{trav_class}"
                    free_radius_reset.id = 0
                    free_radius_reset.action = Marker.DELETEALL
                    marker_array.markers.append(free_radius_reset)
                    free_radius_counters[trav_class] = 1

                free_radius = Marker()
                free_radius.header.frame_id = str(frame_id)
                free_radius.header.stamp = stamp
                free_radius.ns = f"free_radius_{trav_class}"
                free_radius.id = free_radius_counters[trav_class]
                free_radius_counters[trav_class] += 1
                free_radius.action = Marker.ADD
                free_radius.type = Marker.SPHERE
                free_radius.scale.x = getattr(trav_prop, 'free_radius', 0.0) * 2.0
                free_radius.scale.y = getattr(trav_prop, 'free_radius', 0.0) * 2.0
                free_radius.scale.z = 0.01
                free_radius.color = ColorRGBA()
                free_radius.color.a = 0.2
                free_radius.color.r = 1.0
                free_radius.color.g = 0.0
                free_radius.color.b = 0.0
                free_radius.pose.position = node_position
                marker_array.markers.append(free_radius)

                # Frontier nodes grouped by class
                if trav_prop.is_frontier:
                    if trav_class not in frontier_node_markers:
                        m = Marker()
                        m.header.frame_id = str(frame_id)
                        if stamp is not None:
                            m.header.stamp = stamp
                        m.frame_locked = True
                        m.ns = f"frontier_nodes_{trav_class}"
                        m.id = 0
                        m.type = Marker.SPHERE_LIST
                        m.action = Marker.ADD
                        m.scale.x = 0.6
                        m.scale.y = 0.6
                        m.scale.z = 0.6
                        m.color = ColorRGBA()
                        m.color.a = 1.0
                        m.color.b = 1.0
                        frontier_node_markers[trav_class] = m
                    frontier_node_markers[trav_class].points.append(node_position)
                    # push a color for this point
                    c = ColorRGBA()
                    c.a = 1.0
                    c.b = 1.0
                    frontier_node_markers[trav_class].colors.append(c)

                # Add frontier points (positions) and line to node
                for fp in trav_prop.frontier_points:
                    fp_point = Point(x=fp.x, y=fp.y, z=fp.z)
                    frontier_point_marker.points.append(fp_point)
                    frontier_point_to_node_marker.points.append(node_position)
                    frontier_point_to_node_marker.points.append(fp_point)

        # Edges
        for edge in navgraph_msg.edges:
            try:
                from_p = node_positions[edge.from_idx]
                to_p = node_positions[edge.to_idx]
            except Exception:
                continue
            from_point = Point(x=from_p[0], y=from_p[1], z=from_p[2])
            to_point = Point(x=to_p[0], y=to_p[1], z=to_p[2])
            edge_marker.points.append(from_point)
            edge_marker.points.append(to_point)

        # Relative position lines
        for node in navgraph_msg.nodes:
            try:
                ref_p = node.pose.position
                node_p = node.pose.position
                ref_pt = Point(x=ref_p.x, y=ref_p.y, z=ref_p.z)
                node_pt = Point(x=node_p.x, y=node_p.y, z=node_p.z)
                rel_pos_marker.points.append(ref_pt)
                rel_pos_marker.points.append(node_pt)
            except Exception:
                pass

        # Optional text ids
        if hasattr(navgraph_msg, 'nodes'):
            text_delete_all = Marker()
            text_delete_all.header.frame_id = frame_id
            if stamp is not None:
                text_delete_all.header.stamp = stamp
            text_delete_all.frame_locked = True
            text_delete_all.ns = "ids"
            text_delete_all.id = 0
            text_delete_all.action = Marker.DELETEALL
            marker_array.markers.append(text_delete_all)
            id_counter = 1
            for node in navgraph_msg.nodes:
                try:
                    p = node.pose.position
                    pos_pt = Point(x=p.x, y=p.y, z=p.z)
                    text_marker = Marker()
                    text_marker.header.frame_id = frame_id
                    if stamp is not None:
                        text_marker.header.stamp = stamp
                    text_marker.frame_locked = True
                    text_marker.ns = "ids"
                    text_marker.id = id_counter
                    id_counter += 1
                    text_marker.type = Marker.TEXT_VIEW_FACING
                    text_marker.action = Marker.ADD
                    text_marker.scale.z = 0.05

                    try:
                        uuid_val = node.uuid
                        text_marker.text = "".join([f"{x:03}" for x in uuid_val.id])
                    except Exception:
                        text_marker.text = ""
                    text_marker.color = ColorRGBA()
                    text_marker.color.a = 1.0
                    text_marker.color.r = 1.0
                    text_marker.color.g = 1.0
                    text_marker.color.b = 1.0
                    text_marker.pose.position.x = pos_pt.x
                    text_marker.pose.position.y = pos_pt.y
                    text_marker.pose.position.z = pos_pt.z + 0.1
                    marker_array.markers.append(text_marker)
                except Exception:
                    continue

        # Append aggregated markers
        marker_array.markers.append(node_marker)
        for _, m in frontier_node_markers.items():
            marker_array.markers.append(m)
        marker_array.markers.append(frontier_point_marker)
        marker_array.markers.append(frontier_point_to_node_marker)
        marker_array.markers.append(edge_marker)
        marker_array.markers.append(rel_pos_marker)

        return marker_array
        

    def visualize_all_heading_scores(self, frontier_uuid_to_scores, removed_uuids, updated_uuids, frame_id, stamp):
        """
        Visualize the heading scores for all frontier nodes in RViz as a ring.

        :param frontier_uuid_to_scores: Dictionary mapping frontier UUIDs to their heading scores.
        :param removed_uuids: List of UUIDs that were removed in the latest update.
        :param updated_uuids: Set of UUIDs that were updated in the latest update.
        :param frame_id: The frame ID for the markers.
        :param stamp: The timestamp for the markers.
        :return: MarkerArray for visualization in RViz.
        """
        marker_array = MarkerArray()

        # Remove markers for removed frontiers
        for uuid in removed_uuids:
            for i in range(self.num_bins):
                marker = Marker()
                marker.header.frame_id = frame_id
                marker.header.stamp = stamp
                marker.ns = "geofrontier_score_ring"
                marker.id = self.uuid_to_marker_id[uuid][i]
                marker.action = Marker.DELETE
                marker_array.markers.append(marker)

        # Add/modify markers for updated frontiers
        for uuid in updated_uuids:
            scores, node = frontier_uuid_to_scores[uuid]

            node_pos = np.array([
                node.pose.position.x,
                node.pose.position.y,
                node.pose.position.z
            ])

            marker_action = Marker.MODIFY
            if uuid not in self.uuid_to_marker_id:
                self.uuid_to_marker_id[uuid] = np.arange(self.marker_id, self.marker_id + self.num_bins).tolist()
                self.marker_id += self.num_bins
                marker_action = Marker.ADD

            for bin_idx, score in enumerate(scores):
                color = plt.cm.jet(score)  # Use jet colormap for scores
                angle_st = self.bin_starts[bin_idx]
                angle_end = angle_st + self.discretization_angle

                marker = Marker()
                marker.header.frame_id = frame_id
                marker.header.stamp = stamp
                marker.ns = "geofrontier_score_ring"
                marker.action = marker_action
                marker.id = self.uuid_to_marker_id[uuid][bin_idx]
                marker.type = Marker.LINE_STRIP

                start_pt = node_pos + self.ring_radius * np.array([np.cos(angle_st), np.sin(angle_st), 0])
                end_pt = node_pos + self.ring_radius * np.array([np.cos(angle_end), np.sin(angle_end), 0])
                start_pt = start_pt.astype(np.float64)
                end_pt = end_pt.astype(np.float64)
                marker.points.append(Point(x=start_pt[0], y=start_pt[1], z=start_pt[2]))
                marker.points.append(Point(x=end_pt[0], y=end_pt[1], z=end_pt[2]))

                marker.scale.x = 0.5  # Line width
                marker.color.r = color[0]
                marker.color.g = color[1]
                marker.color.b = color[2]
                marker.color.a = 1.0
                marker_array.markers.append(marker)

        return marker_array
    
    def visualize_model_det_front(self, nav_data, all_cam_data):
        img_grid = {}
        num_rows = 1
        num_cols = 2
        fig_resize_factor = 0.75


        rgb_img = nav_data[0]["image"]
        # rgb_img = cv2.resize(rgb_img, (0,0), fx=self.fig_resize_factor, fy=self.fig_resize_factor)
        frontier_overlay = rgb_img.copy()
        trav_overlay = rgb_img.copy()

        frontier_map = nav_data[0]["img_frontiers"].astype(np.float32)
        traversability_map = nav_data[0]["traversability"].astype(np.float32)


        frontier_map[frontier_map < 0.6] = 0.0
        frontier_map[traversability_map < 0.9] = 0.0

        # Morphological opening to remove small noisy frontier regions
        kernel = np.ones((20, 20), np.uint8)
        valid = cv2.morphologyEx((frontier_map>0).astype(np.uint8), cv2.MORPH_OPEN, kernel)
        frontier_map[valid==0] = 0.0

        frontier_hm = overlay_heatmap(rgb_img, frontier_map, alpha=1.0)
        valid_frontier = (frontier_map > 0)
        valid_mask_3d = np.stack([valid_frontier] * 3, axis=-1)
        frontier_overlay[valid_mask_3d] = frontier_hm[valid_mask_3d]
        frontier_overlay = cv2.resize(frontier_overlay, (0,0), fx=fig_resize_factor, fy=fig_resize_factor)
        img_grid[(0, 0)] = (frontier_overlay, "Frontiers Overlay")
        # return frontier_overlay

        # overlay traversability on the image
        traversability_map[traversability_map < 0.9] = 0.0

        trav_hm = overlay_heatmap(rgb_img, 1-traversability_map, alpha=0.5)
        valid_traversability = (traversability_map > 0)
        valid_mask_3d = np.stack([valid_traversability] * 3, axis=-1)
        trav_overlay[valid_mask_3d] = trav_hm[valid_mask_3d]
        trav_overlay = cv2.resize(trav_overlay, (0,0), fx=fig_resize_factor, fy=fig_resize_factor)
        img_grid[(0, 1)] = (trav_overlay, "Traversability Overlay")
        # return trav_overlay

        fin_img_chosen = None
        if "object_mask" in nav_data[0] and nav_data[0]["object_mask"] is not None:
            chosen_cam_idx = -1
            max_obj_pix = -1
            for i in range(self.num_cameras):
                img = nav_data[i]["image"]
                H, W, C = img.shape
                
                obj_mask_2d = nav_data[i]["object_mask"].squeeze() 
                valid_mask_2d = obj_mask_2d > 0
                num_obj_pixels = np.sum(valid_mask_2d)

                if num_obj_pixels <= max_obj_pix:
                    continue

                fin_img = show_mask(img, obj_mask_2d)

                DARK_GRAY = (50, 50, 50) 
                WHITE = (255, 255, 255)
                FONT = cv2.FONT_HERSHEY_DUPLEX 
                FONT_SCALE = 0.7
                FONT_THICKNESS = 1
                PADDING = 10 # Padding around the text
                text_to_display = f"Current View: {self.camera_mapping[i].title()} Camera"
                (text_w, text_h), baseline = cv2.getTextSize(
                    text_to_display, FONT, FONT_SCALE, FONT_THICKNESS
                )

                text_x = PADDING
                text_y = PADDING + text_h

                p1 = (PADDING, PADDING) # Top-left corner of the background
                p2 = (PADDING + text_w + PADDING, PADDING + text_h + baseline + PADDING) # Bottom-right corner

                cv2.rectangle(fin_img, p1, p2, DARK_GRAY, -1) # -1 fills the rectangle

                cv2.putText(
                    fin_img, 
                    text_to_display, 
                    (text_x, text_y), 
                    FONT, 
                    FONT_SCALE, 
                    WHITE, 
                    FONT_THICKNESS, 
                    cv2.LINE_AA
                )

                max_obj_pix = num_obj_pixels
                chosen_cam_idx = i
                fin_img_chosen = fin_img

            if fin_img_chosen is not None:
                fin_img_chosen = cv2.resize(fin_img_chosen, (0,0), fx=fig_resize_factor, fy=fig_resize_factor)
                img_grid[(0, 2)] = (fin_img_chosen, "Object Detection Overlay")
                num_cols = 3

        grid = make_subplot_grid(img_grid, (num_rows, num_cols), pad=15)

        return grid

    def visualize_model_det(self, nav_data, all_cam_data):
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
                mask_overlay = show_mask(rgb_img, obj_mask)
                img_grid[(0, plt_idx)] = (mask_overlay, f"Image {self.camera_mapping[i]} + Obj Mask")

            # overlay frontiers on the image
            frontier_map = nav_data[i]["img_frontiers"].astype(np.float32)
            frontier_map = cv2.resize(frontier_map, (0,0), fx=self.fig_resize_factor, fy=self.fig_resize_factor)
            frontier_overlay = overlay_heatmap(rgb_img, frontier_map)
            img_grid[(1, plt_idx)] = (frontier_overlay, "Frontier Conf.")

            # overlay traversability on the image
            traversability_map = nav_data[i]["traversability"].astype(np.float32)
            traversability_map = cv2.resize(traversability_map, (0,0), fx=self.fig_resize_factor, fy=self.fig_resize_factor)
            trav_overlay = overlay_heatmap(rgb_img, traversability_map)
            img_grid[(2, plt_idx)] = (trav_overlay, "Traversability Conf.")

            # Show projected geo_frontiers and paths to straight-line goal from camera
            path_overlay = rgb_img.copy()
            if "geo_frontiers" not in nav_data[i]:
                dummy_img = path_overlay
                cv2.putText(
                    dummy_img,
                    "No Valid Geometric Frontiers",
                    (20, dummy_img.shape[0]//2),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.45,
                    (0,0,0),
                    0
                )
                img_grid[(3, plt_idx)] = (dummy_img, "Frontier Nodes")
                continue

            geo_frontiers = nav_data[i]["geo_frontiers"] * self.fig_resize_factor
            cam_heading = all_cam_data[i]["R_wc"].astype(np.float32) @ np.array([0, 0, 1], dtype=np.float32)
            cam_heading = cam_heading[:2]
            cam_heading = cam_heading / np.linalg.norm(cam_heading)
            cam_angle = np.arctan2(cam_heading[1], cam_heading[0])
            goal_heading = cam_angle + self.goal_cam_relative_headings
            heading_bin = int(goal_heading / self.discretization_angle) % len(self.bin_starts)

            scores = nav_data[i]["scores"]
            paths = nav_data[i]["paths"]
            score_map = nav_data[i]["score_map"][0][:,:,heading_bin].astype(np.float32)
            score_map = cv2.resize(score_map, (0,0), fx=self.fig_resize_factor, fy=self.fig_resize_factor)
            path_overlay_hm = overlay_heatmap(path_overlay, score_map, alpha=0.5)
            valid_map = (score_map > 0)
            valid_mask_3d = np.stack([valid_map] * 3, axis=-1)
            path_overlay[valid_mask_3d] = path_overlay_hm[valid_mask_3d]

            for ((y,x), score, path) in zip(geo_frontiers, scores, paths):
                path = np.array(path[heading_bin]) * self.fig_resize_factor
                color = plt.cm.jet(score[heading_bin])
                color = tuple(int(c * 255) for c in color[:3])

                # draw_point(path_overlay, (y,x), color, radius=6)
                # draw_point(path_overlay, path[-1], (255,255,255), radius=2)  # goal point
                # draw_text(path_overlay, (y,x), f"{score[heading_bin]:.2f}", color=(255,255,255))
                draw_path(path_overlay, path, color)
                draw_point(path_overlay, (y,x), (0,0,255), radius=8)
            img_grid[(3, plt_idx)] = (path_overlay, "Frontier Nodes")


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
