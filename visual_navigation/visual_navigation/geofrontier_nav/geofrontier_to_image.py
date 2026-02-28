from typing import Dict, List
import numpy as np

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose
from graphnav_msgs.msg import NavigationGraph, Node

class GeoFrontierToImage:
    def __init__(
        self,
        camera_mapping: Dict[int, str],
        frontiers_range: float,
        traversability_class: str,
        cams_inverted: bool,
        heading_sim_thresh: float=0.0,
        reach_in_2D: bool=False
    ):
        self.camera_mapping = camera_mapping
        self.frontiers_range = frontiers_range
        self.traversability_class = traversability_class
        self.cams_inverted = cams_inverted
        self.heading_sim_thresh = heading_sim_thresh
        self.reach_in_2D = reach_in_2D

    def extract_geofrontiers(
        self, current_odom_msg: Odometry, navgraph: NavigationGraph, all_cam_data: List
    ) -> Dict:
        geofronts = {}
        num_cams = len(all_cam_data)

        # extract frontier nodes from nav graph within frontiers_range
        frontier_nodes = []
        frontier_headings = []
        frontier_pos = []

        trav_class_idx = navgraph.trav_classes.index(self.traversability_class)
        for node in navgraph.nodes:
            if node.trav_properties[trav_class_idx].is_frontier:
                # Check if the node is within the frontiers_range of the current_odom_msg
                if self.is_within_range(current_odom_msg.pose.pose, node.pose):
                    frontier_nodes.append(node)

        if len(frontier_nodes) == 0:
            return {
                cam_idx: {} for cam_idx in range(num_cams)
            }

        # get the direction of each frontier node
        for node in frontier_nodes:
            front_heading = self.get_frontier_heading(node, trav_class_idx)
            frontier_headings.append(front_heading)
            frontier_pos.append(np.array([
                node.pose.position.x,
                node.pose.position.y,
                node.pose.position.z
            ]))
        frontier_headings = np.array(frontier_headings)
        frontier_pos = np.array(frontier_pos)

        # choose frontiers visible in each camera
        for cam_idx in range(num_cams):
            cam_info = all_cam_data[cam_idx]
            valid_frontiers, valid_frontier_pix = self.get_visible_frontiers(cam_info, frontier_pos, frontier_headings)
            if len(valid_frontiers) == 0:
                geofronts[cam_idx] = {}
            else:
                geofronts[cam_idx] = {
                    "frontier_nodes": [frontier_nodes[i] for i in valid_frontiers],
                    "frontier_positions": frontier_pos[valid_frontiers],
                    "frontier_headings": frontier_headings[valid_frontiers],
                    "frontier_pixel_coords": valid_frontier_pix
                }
        return geofronts

    def is_within_range(self, cur_pose: Pose, node_pose: Pose):
        cur_pos = np.array([
            cur_pose.position.x,
            cur_pose.position.y,
            cur_pose.position.z
        ])
        node_pos = np.array([
            node_pose.position.x,
            node_pose.position.y,
            node_pose.position.z
        ])
        if self.reach_in_2D:
            cur_pos[2] = 0.0
            node_pos[2] = 0.0

        return np.linalg.norm(cur_pos - node_pos) <= self.frontiers_range
    
    def get_frontier_heading(self, node: Node, trav_class_idx: str) -> np.ndarray:
        # Get the direction vector from the frontier node to frontier points
        node_pos = np.array([
            node.pose.position.x,
            node.pose.position.y,
            node.pose.position.z
        ])

        frontier_points = [
            np.array([pt.x, pt.y, pt.z])
            for pt in node.trav_properties[trav_class_idx].frontier_points
        ]
        frontier_points = np.array(frontier_points)

        direction = np.mean(frontier_points, axis=0) - node_pos
        return direction / np.linalg.norm(direction)
    
    def get_visible_frontiers(
        self, cam_info: Dict, frontier_pos: np.ndarray, frontier_headings: np.ndarray
    ) -> Dict:
        
        K = cam_info['K']

        # world from cam
        R_wc = cam_info['R_wc']
        t_wc = cam_info['t_wc']

        # cam from world
        R_cw = R_wc.T
        t_cw = -R_cw @ t_wc

        frontier_pos_cam = (R_cw @ frontier_pos.T).T + t_cw.T
        frontier_pix = (K @ frontier_pos_cam.T).T
        frontier_pix = frontier_pix[:, :2] / frontier_pix[:, 2:]

        cam_heading = R_wc @ np.array([0, 0, 1])
        cam_heading = cam_heading / np.linalg.norm(cam_heading)

        front_cam_cos = frontier_headings @ cam_heading

        valid_frontiers = np.logical_and.reduce((
            frontier_pix[:, 0] >= 0,
            frontier_pix[:, 0] < cam_info['width'],
            frontier_pix[:, 1] >= 0,
            frontier_pix[:, 1] < cam_info['height'],
            front_cam_cos > self.heading_sim_thresh,
            frontier_pos_cam[:, 2] > 0
        ))

        valid_frontiers = np.where(valid_frontiers)[0]

        # convert frontier_pix (x,y) to (y,x)
        frontier_pix = frontier_pix[:, [1, 0]]
        if self.cams_inverted:
            frontier_pix[:, 0] = cam_info['height'] - frontier_pix[:, 0]
            frontier_pix[:, 1] = cam_info['width'] - frontier_pix[:, 1]

        valid_frontier_pix = frontier_pix[valid_frontiers].astype(int)

        return valid_frontiers, valid_frontier_pix