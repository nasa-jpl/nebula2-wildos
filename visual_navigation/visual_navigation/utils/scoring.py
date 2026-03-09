from typing import Dict, List, Tuple
from enum import Enum, auto

import cv2
import numpy as np
from skimage.graph import MCP_Geometric

class ScoringMethod(Enum):
    ADDITIVE = auto()
    MULTIPLICATIVE = auto()
    COMBINATION = auto()


class ScoringGeometricFrontiers:
    def __init__(
        self,
        pixel_scoring_params: Dict,
        frontier_threshold: float,
        frontier_opening_kernel_size: int,
        traversability_threshold: float,
        reach_in_2D: bool = True,
        cam_inverted: bool = False,
        reach_scale: float = 0.25,
    ):
        self.pixel_scoring_params = pixel_scoring_params
        self.scoring_method = ScoringMethod[pixel_scoring_params["method"]]
        self.frontier_threshold = frontier_threshold
        self.frontier_opening_kernel_size = frontier_opening_kernel_size
        self.traversability_threshold = traversability_threshold
        self.reach_in_2D = reach_in_2D
        self.cam_inverted = cam_inverted
        self.reach_scale = reach_scale

        self.cam_intrinsics_and_coords = {}

    def init_camera_coordinates(self, h: int, w: int, K: np.ndarray, frame_id: str, force_reinit: bool = False):
        """
        Initialize camera coordinates in the camera frame.
        """
        if (not force_reinit) and (frame_id in self.cam_intrinsics_and_coords):
            if self.cam_intrinsics_and_coords[frame_id]["height"] == h and \
               self.cam_intrinsics_and_coords[frame_id]["width"] == w and \
               np.array_equal(self.cam_intrinsics_and_coords[frame_id]["K"], K):
                return

        pixel_coords = np.array(np.meshgrid(np.arange(w), np.arange(h))).reshape(2, -1).T
        pixel_coords = np.hstack((pixel_coords, np.ones((pixel_coords.shape[0], 1))))
        coords_cam = np.linalg.inv(K) @ pixel_coords.T
        coords_cam = coords_cam.astype(np.float32)

        self.cam_intrinsics_and_coords[frame_id] = {
            "height": h,
            "width": w,
            "K": K,
            "coords_cam": coords_cam
        }

    def score_geofrontiers(
        self,
        geometric_frontiers: np.ndarray,
        img_frontiers: np.ndarray,
        traversability: np.ndarray,
        goal_heading: np.ndarray,
        cam_data: Dict
    ) -> Dict:
        """
        Find image paths based on frontiers, traversability, and goal heading.

        :param geometric_frontiers: List of geometric frontier pixel coordinates, numpy array of shape (N, 2).
        :param img_frontiers: The frontier map (H, W)
        :param traversability: The traversability map (H, W)
        :param goal_heading: The goal heading vector (1, 3)
        :param current_odom: The current odometry message.
        :param cam_data: The camera intrinsics and extrinsics.
        """

        h, w = img_frontiers.shape
        goal_conf = self.get_goal_conf(h, w, cam_data, goal_heading)
        frontier_conf = self.get_frontier_conf(img_frontiers, traversability)

        trav = cv2.resize(traversability.astype(np.float32), (0,0), fx=self.reach_scale, fy=self.reach_scale)
        trav_costs = 1 / (trav + 1e-3)
        trav_costs[trav < self.traversability_threshold] = -1
        mcp = MCP_Geometric(trav_costs, fully_connected=True)

        scores = []
        paths = []
        score_maps = []

        for geofrontier in geometric_frontiers:
            reachability_conf, mcp = self.get_reachability_conf(geofrontier * self.reach_scale, (h,w), mcp)

            if self.scoring_method == ScoringMethod.ADDITIVE:
                score_map = (
                    self.pixel_scoring_params["frontier"] * frontier_conf +
                    self.pixel_scoring_params["goal"] * goal_conf +
                    self.pixel_scoring_params["reachability"] * reachability_conf
                ) / (
                        self.pixel_scoring_params["frontier"] +\
                        self.pixel_scoring_params["goal"] +\
                        self.pixel_scoring_params["reachability"]
                    )
            
            elif self.scoring_method == ScoringMethod.COMBINATION:
                score_map = (
                    (
                        self.pixel_scoring_params["frontier"] * frontier_conf +
                        self.pixel_scoring_params["goal"] * goal_conf
                    ) / (self.pixel_scoring_params["frontier"] + self.pixel_scoring_params["goal"])
                ) * reachability_conf
            
            elif self.scoring_method == ScoringMethod.MULTIPLICATIVE:
                score_map = reachability_conf * frontier_conf * goal_conf

            best_index = np.unravel_index(np.argmax(score_map), score_map.shape)
            chosen_score = score_map[best_index]
            try:
                best_index = (int(best_index[0]*self.reach_scale), int(best_index[1]*self.reach_scale))
                path = np.array(mcp.traceback((best_index[0], best_index[1]))) * (1 / self.reach_scale)
            except Exception as e:
                path = np.array([geofrontier])

            scores.append(chosen_score)
            paths.append(path)
            score_maps.append(score_map)

        return scores, paths, score_maps

    def get_reachability_conf(
        self,
        img_src: Tuple[int, int],
        img_shape: Tuple[int, int],
        mcp: MCP_Geometric,
    ):
        """
        Compute reachability cost based on img_frontiers and traversability.
        """
        h, w = img_shape
        costs, trbk = mcp.find_costs([img_src])
        costs /= (h+w) * self.reach_scale

        # normalize to [0,1]
        reachability_cost = 1 - np.tanh(costs)
        reachability_cost = cv2.resize(reachability_cost.astype(np.float32), (w,h))
        return reachability_cost, mcp

    def get_goal_conf(self, h: int, w: int, cam_data: Dict, goal_heading: np.ndarray):
        """
        Compute goal confidence for each pixel according to alignment with the goal heading.
        """
        K = cam_data['K']
        R_wc = cam_data['R_wc']

        self.init_camera_coordinates(h, w, K, cam_data['frame_id'])
        coords_cam = self.cam_intrinsics_and_coords[cam_data['frame_id']]['coords_cam']

        pixel_heading = R_wc @ coords_cam
        if self.reach_in_2D:
            pixel_heading[2, :] = 0.0
        pixel_heading = pixel_heading / (np.linalg.norm(pixel_heading, axis=0) + 1e-6)

        goal_conf = goal_heading @ pixel_heading
        goal_conf = goal_conf.reshape(h, w)

        if self.scoring_method == ScoringMethod.MULTIPLICATIVE:
            goal_conf = (1 + goal_conf) / 4.0 + 0.5 # normalize to [0.5, 1.0]
        else:
            goal_conf = (goal_conf + 1) / 2.0  # normalize to [0, 1]
        
        if self.cam_inverted:
            goal_conf = np.rot90(goal_conf, k=2)

        return goal_conf

    def get_frontier_conf(self, img_frontiers: np.ndarray, traversability: np.ndarray):
        """
        Compute long range frontier confidence for each pixel based on image frontiers and traversability.
        """
        img_frontier_conf = img_frontiers.copy()
        img_frontier_conf[img_frontier_conf < self.frontier_threshold] = 0.0
        img_frontier_conf[traversability < self.traversability_threshold] = 0.0

        # Morphological opening to remove small noisy frontier regions
        if self.frontier_opening_kernel_size > 0:
            kernel = np.ones((self.frontier_opening_kernel_size, self.frontier_opening_kernel_size), np.uint8)
            valid = cv2.morphologyEx((img_frontier_conf>0).astype(np.uint8), cv2.MORPH_OPEN, kernel)
            img_frontier_conf[valid==0] = 0.0

        return img_frontier_conf