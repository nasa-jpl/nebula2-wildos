from typing import Dict
import cv2
import numpy as np
from skimage.graph import MCP_Geometric

from visual_navigation.utils.scoring import ScoringGeometricFrontiers, ScoringMethod

class GoalAgnosticScoring(ScoringGeometricFrontiers):
    """
    Score image pixels based on frontiers, traversability. Compute scores for all possible goal headings by dividing 
    the heading space into discrete angular bins.
    """
    def __init__(
        self,
        num_angular_bins: int,
        **kwargs
    ):
        """
        :param num_angular_bins: Number of discrete angular bins to divide the heading space into.
        :param kwargs: Other parameters for ScoringGeometricFrontiers.
        """
        super().__init__(**kwargs)

        self.num_angular_bins = num_angular_bins
        self.all_goal_headings = self.compute_all_goal_headings(num_angular_bins)

    def compute_all_goal_headings(self, num_bins: int) -> np.ndarray:
        """
        Compute all possible goal headings by dividing the heading space into discrete angular bins.

        :param num_bins: Number of discrete angular bins.
        :return: Array of shape (num_bins, 3) representing the goal heading vectors.
        """
        self.angles = np.linspace(0, 2 * np.pi, num_bins, endpoint=False)
        self.angles_deg = np.degrees(self.angles)
        goal_headings = np.array([[np.cos(angle), np.sin(angle), 0] for angle in self.angles], dtype=np.float32)
        return goal_headings

    def score_geofrontiers(
        self,
        geometric_frontiers: np.ndarray,
        img_frontiers: np.ndarray,
        traversability: np.ndarray,
        cam_data: Dict,
        compute_paths: bool = True,
    ) -> Dict:
        """
        Find image paths based on frontiers, traversability, and goal heading.

        :param geometric_frontiers: List of geometric frontier pixel coordinates, numpy array of shape (N, 2).
        :param img_frontiers: The frontier map (H, W)
        :param traversability: The traversability map (H, W)
        :param current_odom: The current odometry message.
        :param cam_data: The camera intrinsics and extrinsics.
        """

        h, w = img_frontiers.shape
        goal_conf = self.get_goal_conf(h, w, cam_data)  # (H, W, num_angular_bins)
        frontier_conf = self.get_frontier_conf(img_frontiers, traversability)[..., np.newaxis] # (H, W, 1)

        trav = cv2.resize(traversability.astype(np.float32), (0,0), fx=self.reach_scale, fy=self.reach_scale)
        trav_costs = 1 / (trav + 1e-3)
        trav_costs[trav < self.traversability_threshold] = -1
        mcp = MCP_Geometric(trav_costs, fully_connected=True)

        scores = []
        paths = []
        score_maps = []

        for geofrontier in geometric_frontiers:
            reachability_conf, mcp = self.get_reachability_conf(geofrontier * self.reach_scale, (h,w), mcp)
            reachability_conf = reachability_conf[..., np.newaxis] # (H, W, 1)

            score_map = np.zeros_like(goal_conf, dtype=np.float32)

            if self.scoring_method == ScoringMethod.ADDITIVE:
                score_map += self.pixel_scoring_params["goal"] * goal_conf
                score_map += self.pixel_scoring_params["frontier"] * frontier_conf
                score_map += self.pixel_scoring_params["reachability"] * reachability_conf
                score_map /= (
                    self.pixel_scoring_params["goal"] + 
                    self.pixel_scoring_params["frontier"] + 
                    self.pixel_scoring_params["reachability"]
                )

            elif self.scoring_method == ScoringMethod.COMBINATION:
                score_map += self.pixel_scoring_params["goal"] * goal_conf
                score_map += self.pixel_scoring_params["frontier"] * frontier_conf
                score_map /= (self.pixel_scoring_params["goal"] + self.pixel_scoring_params["frontier"])
                score_map *= reachability_conf

            elif self.scoring_method == ScoringMethod.MULTIPLICATIVE:
                score_map += goal_conf
                score_map *= frontier_conf
                score_map *= reachability_conf

            # best score per angular bin
            flattened_scores = score_map.reshape(-1, self.num_angular_bins)  # (H*W, num_angular_bins)

            all_goal_paths = []
            if not compute_paths:
                all_goal_paths = [np.array([geofrontier]) for _ in range(self.num_angular_bins)]
                best_scores = np.max(flattened_scores, axis=0)

            else:
                best_score_idx = np.argmax(flattened_scores, axis=0)  # (num_angular_bins,)
                best_scores = flattened_scores[best_score_idx, np.arange(self.num_angular_bins)]

                for i in range(self.num_angular_bins):
                    best_index = np.unravel_index(best_score_idx[i], (h, w))
                    try:
                        best_index = (int(best_index[0]*self.reach_scale), int(best_index[1]*self.reach_scale))
                        path = np.array(mcp.traceback((best_index[0], best_index[1]))) * (1 / self.reach_scale)
                    except Exception as e:
                        path = np.array([geofrontier])
                    all_goal_paths.append(path)

            scores.append(best_scores)
            paths.append(all_goal_paths)
            score_maps.append(score_map)

        return scores, paths, score_maps

    def get_goal_conf(self, h: int, w: int, cam_data: Dict) -> np.ndarray:
        """
        Compute goal confidence for each pixel according to alignment with all possible goal headings.

        :param h: Image height.
        :param w: Image width.
        :param cam_data: The camera intrinsics and extrinsics.
        :return: Goal confidence map of shape (H, W, num_angular_bins). Each channel corresponds to a discrete goal heading.
        """
        K = cam_data['K'].astype(np.float32, copy=False)
        R_wc = cam_data['R_wc'].astype(np.float32, copy=False)
        
        self.init_camera_coordinates(h, w, K, cam_data['frame_id'])
        coords_cam = self.cam_intrinsics_and_coords[cam_data['frame_id']]['coords_cam']

        pixel_heading = R_wc @ coords_cam
        if self.reach_in_2D:
            pixel_heading[2, :] = 0.0
        pixel_heading = pixel_heading / (np.linalg.norm(pixel_heading, axis=0) + 1e-6)

        goal_conf = self.all_goal_headings @ pixel_heading  # (num_bins, 3) @ (3, H*W) -> (num_bins, H*W)
        goal_conf = goal_conf.reshape(self.num_angular_bins, h, w).transpose(1, 2, 0)

        if self.scoring_method == ScoringMethod.MULTIPLICATIVE:
            goal_conf = (1.0 + goal_conf) * 0.25 + 0.5 # normalize to [0.5, 1.0]
        else:
            goal_conf = (1.0 + goal_conf) * 0.5  # normalize to [0, 1]

        if self.cam_inverted:
            goal_conf = np.rot90(goal_conf, k=2, axes=(0,1))

        return goal_conf

    def get_default_scores(self, node, trav_class_idx, std, def_max_score) -> np.ndarray:
        """
        Get default scores for a frontier node that has not been scored yet.
        Compute the frontier heading and assign gaussian scores centered around that heading.

        :param node: The frontier node.
        :param trav_class_idx: The index of the traversability class.
        :param std: Standard deviation for the gaussian scoring (in degrees).
        :param def_max_score: The maximum score to assign.

        :return: Default scores as a numpy array of shape (num_angular_bins,).
        """
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
        heading = direction / np.linalg.norm(direction)

        heading_angle = np.rad2deg(np.arctan2(heading[1], heading[0]))
        gauss_scores = self.get_gauss_scores(heading_angle, std, def_max_score)

        return gauss_scores


    def get_gauss_scores(self, heading_angle: float, std: float, max_score: float) -> np.ndarray:
        """
        Get gaussian scores centered around a given heading angle.

        :param heading_angle: The heading angle (in degrees).
        :param std: Standard deviation for the gaussian scoring (in degrees).
        :param max_score: The maximum score to assign.

        :return: Gaussian scores as a numpy array of shape (num_angular_bins,).
        """
        dists = np.minimum(
            (self.angles_deg - heading_angle) % 360,
            (heading_angle - self.angles_deg) % 360
        )
        gauss_scores = np.exp(-0.5 * (dists / std) ** 2)
        gauss_scores = (gauss_scores / (np.max(gauss_scores) + 1e-6)) * max_score

        return gauss_scores