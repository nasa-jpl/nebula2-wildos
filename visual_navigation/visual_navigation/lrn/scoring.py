from typing import Dict

import numpy as np
import cv2

class LRNScoring:
    def __init__(
        self,
        hotspot_threshold: float,
        frontier_opening_kernel_size: int,
        angle_discretization_deg: float,
        fixed_hotspot_dist: float,
        honing_range: float,
        ema_alpha: float,
        beta_degradation: float,
        goal_std: float,
        prev_std: float,
        cam_inverted: bool = False,
    ):
        self.hotspot_threshold = hotspot_threshold
        self.frontier_opening_kernel_size = frontier_opening_kernel_size
        self.angle_discretization_deg = angle_discretization_deg
        self.fixed_hotspot_dist = fixed_hotspot_dist
        self.honing_range = honing_range
        self.ema_alpha = ema_alpha
        self.beta_degradation = beta_degradation
        self.goal_std = goal_std
        self.prev_std = prev_std
        self.cam_inverted = cam_inverted

        self.num_bins = int(360 // self.angle_discretization_deg)
        self.filtered_scores = None

    def score_bins_from_frontiers(
        self,
        img_frontiers: np.ndarray,
        cam_data: Dict,
    ):
        """
        Score the bins based on the image frontiers.
        
        :param img_frontiers: np.ndarray of shape (H, W) representing the image frontiers.
        :param cam_data: Dictionary containing camera intrinsic and extrinsic parameters.
        """
        if self.cam_inverted:
            img_frontiers = np.rot90(img_frontiers, k=2)

        # Morphological opening to remove small noisy frontier regions
        if self.frontier_opening_kernel_size > 0:
            kernel = np.ones((self.frontier_opening_kernel_size, self.frontier_opening_kernel_size), np.uint8)
            valid = cv2.morphologyEx((img_frontiers>0).astype(np.uint8), cv2.MORPH_OPEN, kernel)
            img_frontiers[valid==0] = 0.0

        cam_scores = np.zeros(self.num_bins)
        raw_hotspots = np.array(np.where(img_frontiers > self.hotspot_threshold)).T

        # No hotspots found
        if raw_hotspots.shape[1] == 0:
            return cam_scores

        heat_scores = img_frontiers[raw_hotspots[:, 0], raw_hotspots[:, 1]]
        
        K = cam_data['K']
        R_wc = cam_data['R_wc']
        
        hotspot_pix = np.flip(raw_hotspots, axis=-1)  # (N, 2) y,x -> x,y
        hotspot_pix = np.hstack((hotspot_pix, np.ones((hotspot_pix.shape[0], 1))))
        hotspot_cam = np.linalg.inv(K) @ hotspot_pix.T
        hotspot_cam = hotspot_cam / hotspot_cam[2]
        hotspot_cam = hotspot_cam.T * self.fixed_hotspot_dist  # (N, 3)
        hotspot_odom = R_wc @ hotspot_cam.T
        hotspot_angles = np.arctan2(hotspot_odom[1], hotspot_odom[0])

        bins = (hotspot_angles // np.deg2rad(self.angle_discretization_deg)).astype(int) 
        bins = bins % self.num_bins

        np.add.at(cam_scores, bins, heat_scores)

        return cam_scores
    
    def get_final_scores(
        self,
        frontier_scores: np.ndarray,
        prev_heading: float,
        goal_heading: float,
        goal_range: float
    ):
        """
        Combine frontier scores with heading information to get final scores.
        
        :param frontier_scores: np.ndarray of shape (num_bins,) representing the frontier scores.
        :param prev_heading: Previous heading in radians.
        :param goal_heading: Goal heading in radians.
        :param goal_range: Distance to the goal.
        :return: np.ndarray of shape (num_bins,) representing the final scores.
        """

        # Process frontier scores with EMA
        frontier_score_sum = np.sum(frontier_scores)
        if frontier_score_sum > 0.001:
            frontier_scores = frontier_scores / frontier_score_sum

        uniform_scores = np.ones_like(frontier_scores) / self.num_bins
        frontier_scores_degraded = (
            (1 - self.beta_degradation) * uniform_scores
            + self.beta_degradation * frontier_scores
        )

        if self.filtered_scores is None:
            self.filtered_scores = frontier_scores_degraded
        else:
            self.filtered_scores = (
                self.ema_alpha * frontier_scores_degraded
                + (1 - self.ema_alpha) * self.filtered_scores
            )

        # Process goal heading scores
        goal_gausian_center_deg = np.rad2deg(goal_heading) % 360
        goal_std = self.goal_std
        if (
            self.honing_range > 0 
            and goal_range < self.honing_range
        ):
            goal_std *= goal_range / self.honing_range
        goal_scores = self.get_gaussian_scores(
            mean=goal_gausian_center_deg,
            std=goal_std,
        )

        # Process previous heading scores
        prev_heading_scores = np.ones_like(frontier_scores)
        if prev_heading is not None:
            prev_heading_scores = self.get_gaussian_scores(
                mean=np.rad2deg(prev_heading) % 360,
                std=self.prev_std,
            )

        combined_scores = (
            self.filtered_scores 
            * goal_scores 
            * prev_heading_scores
        )

        combined_score_sum = np.sum(combined_scores)
        if combined_score_sum > 0.001:
            combined_scores = combined_scores / combined_score_sum

        return {
            'combined_scores': combined_scores,
            'frontier_scores': self.filtered_scores,
            'goal_scores': goal_scores,
            'prev_heading_scores': prev_heading_scores,
            # 'raw_frontier_scores': frontier_scores,
        }
    
    def get_gaussian_scores(self, mean: float, std: float, normalize: bool = False):
        """
        Generate Gaussian scores for each bin.
        
        :param mean: Mean angle in degrees.
        :param std: Standard deviation in degrees.
        :param normalize: Whether to normalize the scores to sum to 1.

        :return: np.ndarray of shape (num_bins,) representing the Gaussian scores.
        """
        angle_bins = np.arange(0, 360, self.angle_discretization_deg)
        
        assert len(angle_bins) == self.num_bins, \
            f"Angle bins length mismatch: {len(angle_bins)} vs {self.num_bins}"

        # Find minimum circular distance between gaussian center and each angle.
        # Must take min of A - B and B - A bc of circle.
        dists = np.minimum(
            (angle_bins - mean) % 360,
            (mean - angle_bins) % 360
        )
        gauss_scores = np.exp(-0.5 * (dists / std) ** 2)
        if normalize:
            gauss_scores_sum = np.sum(gauss_scores)
            gauss_scores = gauss_scores / gauss_scores_sum

        return gauss_scores