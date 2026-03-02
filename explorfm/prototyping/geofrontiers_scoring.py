from typing import Any, Dict, Optional, Tuple, List
import os
from tqdm import tqdm
import cv2
import networkx as nx
from skimage.graph import MCP_Geometric
import matplotlib.pyplot as plt
import numpy as np

from datasets.rugd import RUGDTraversabilityDataset
from datasets.nebula import NebulaDataset
from ..explorfm_model import ExploRFMInference

class ExploRFMScoringTest:
    def __init__(self,
        dataset: RUGDTraversabilityDataset,
        model: ExploRFMInference,
        num_frontiers: int = 10,
        num_radial_bins: int = 8,
        top_k_frontiers: int = 5,
        discretize_frontiers: bool = True,
        traversability_threshold: float = 0.1,
        frontier_threshold: float = 0.1,
    ):
        self.dataset = dataset
        self.model = model
        self.num_frontiers = num_frontiers
        self.num_radial_bins = num_radial_bins
        self.top_k_frontiers = top_k_frontiers
        self.discretize_frontiers = discretize_frontiers
        self.traversability_threshold = traversability_threshold
        self.frontier_threshold = frontier_threshold

    def sample_goal_heading(self) -> float:
        """Sample a random goal heading in radians."""
        angle_y = np.random.uniform(0, 2 * np.pi)
        return np.array([np.sin(angle_y), np.cos(angle_y), 0])
    
    def sample_frontiers(self, image_res: Tuple[int, int], robot_pose: bool = True, random_sample: bool=False) -> List[Tuple[int, int]]:
        """Sample random frontiers within the lower half of the image.
        Frontiers should be sampled uniformly if random_sample is True,
        otherwise they are sampled in a single horizontal line equally spaced.

        If robot_pose is True, the a single frontier is sampled at the center of the bottom row.
        """
        h, w = image_res
        num_frontiers = self.num_frontiers
        frontiers = []

        if random_sample:
            for _ in range(num_frontiers):
                x = np.random.randint(0, w)
                y = np.random.randint(h // 2, h)
                frontiers.append((x, y))
        elif not robot_pose:
            step = w // num_frontiers
            y = h - np.random.randint(0, h // 2)
            for i in range(num_frontiers):
                x = i * step + step // 2
                if x >= w:
                    x = w - 1
                frontiers.append((x, y))
        else:
            frontiers = [(w//2, h - 4)]
        return frontiers
    
    def build_weighted_graph(self, img: np.ndarray, node_threshold: int) -> nx.Graph:
        """Build a weighted graph from the image where each pixel is a node.
        
        :param img: The input image as a numpy array.
        :return: A NetworkX graph with pixel nodes and weighted edges.
        """
        h, w = img.shape
        G = nx.grid_2d_graph(h, w)
        G.add_edges_from([
            ((y,x), (y+1, x+1))
            for y in range(h-1) for x in range(w-1)
        ] + [
            ((y+1, x), (y, x+1))
            for y in range(h-1) for x in range(w-1)
        ])
        
        for y in range(h):
            for x in range(w):
                node = (y, x)
                if img[y, x] < node_threshold:
                    G.remove_node(node)
                    continue

        # Assign weight based on pixel value
        for node in G.nodes:
            y, x = node
            for neighbor in G.neighbors(node):
                weight = (img[y, x] + img[neighbor[0], neighbor[1]]) / 2
                G[node][neighbor]['weight'] = 1/(weight + 1e-6)
        return G
    
    def discretize_img_frontiers(
        self, img_frontiers: np.ndarray, traversability: np.ndarray, visualize: bool = False
    ) -> Tuple[Optional[np.ndarray], Dict[Any, Any]]:
        """Discretize the frontier image into a specified number of bins.

        :param img_frontiers: The frontier map.
        :param traversability: The traversability map.
        :param visualize: Whether to visualize the discretized frontiers.
        :return: Visualized discretized frontiers and a dictionary of bin centers.
        """
        h, w = img_frontiers.shape
        num_bins = self.num_radial_bins
        bin_size = w // num_bins
        masked_frontiers = img_frontiers > self.frontier_threshold
        
        frontier_info = {}
        viz_frontiers = None
        if visualize:
            viz_frontiers = np.zeros_like(img_frontiers, dtype=np.float32)
        idx_array = np.arange(traversability.size).reshape(traversability.shape)

        for i in range(num_bins):
            start_x = i * bin_size
            end_x = (i + 1) * bin_size if (i + 1) * bin_size < w else w
            cur_bin = masked_frontiers[:, start_x:end_x]
            idx, frontiers = cv2.connectedComponents(
                cur_bin.astype(np.uint8), connectivity=8, ltype=cv2.CV_32S
            )

            if visualize:
                viz_frontiers[:, start_x:start_x+1] = 1  # Mark the start of the bin

            for j in range(1, idx):
                mask = (frontiers == j)
                if np.sum(mask) > 0:
                    # center = np.argwhere(mask)
                    # center_y = np.mean(center[:, 0]).astype(int)
                    # center_x = np.mean(center[:, 1]).astype(int) + start_x
                    traversability_bin = traversability[:, start_x:end_x][mask]
                    idx_bin = idx_array[:, start_x:end_x][mask]
                    max_idx = idx_bin[np.argmax(traversability_bin)]
                    center_y, center_x = np.unravel_index(max_idx, traversability.shape)

                    mean_conf = np.mean(img_frontiers[:, start_x:end_x][mask])
                    if visualize:
                        viz_frontiers[:, start_x:end_x][mask] = mean_conf
                    frontier_info[(center_y, center_x)] = {
                        'mean_conf': mean_conf,
                        'bin': i,
                    }
        print(f"Found {len(frontier_info)} discrete frontiers.")
        # choose top K frontiers based on confidence
        if self.top_k_frontiers > 0:
            sorted_frontiers = sorted(
                frontier_info.items(), key=lambda x: x[1]['mean_conf'], reverse=True
            )[:self.top_k_frontiers]
            frontier_info = {k: v for k, v in sorted_frontiers}

        return viz_frontiers, frontier_info

    def score_frontiers(
        self,
        geometric_frontiers: List[Tuple[int, int]],
        traversability: np.ndarray,
        img_frontier_info: Dict[Any, Any],
        goal_heading: np.ndarray,
        weighted_graph: bool = False,
        score_scaling: float = 100.0
    ) -> Tuple[List[float], List[List[Tuple[int, int]]]]:
        """Score the sampled frontiers based on traversability and frontier maps.
        
        :param geometric_frontiers: List of geometric frontiers as (x, y) tuples.
        :param traversability: The traversability map.
        :param img_frontier_info: The frontier information (discretized centers and confidences).
        :param goal_heading: The goal heading vector.
        :return: A tuple containing scores and paths for each frontier.
        """
        scores = []
        paths = []

        h, w = traversability.shape

        if weighted_graph:
            img_graph = self.build_weighted_graph(traversability, self.traversability_threshold)
        else:
            img_graph = nx.grid_2d_graph(h, w)
            img_graph.add_edges_from([
                ((y,x), (y+1, x+1))
                for y in range(h-1) for x in range(w-1)
            ] + [
                ((y+1, x), (y, x+1))
                for y in range(h-1) for x in range(w-1)
            ])
            for y in range(h):
                for x in range(w):
                    if traversability[y, x] < self.traversability_threshold:
                        img_graph.remove_node((y, x))
        
        distance_maps = {}
        for target in img_frontier_info.keys():
            target_y, target_x = target
            if (target_y, target_x) in img_graph:
                if weighted_graph:
                    distance_maps[target] = nx.single_source_dijkstra_path_length(
                        img_graph, source=(target_y, target_x), weight='weight'
                    )
                else:
                    distance_maps[target] = nx.single_source_shortest_path_length(
                        img_graph, source=(target_y, target_x)
                    )

        for geo_frontier in geometric_frontiers:
            x, y = geo_frontier
            cur_scores = []
            cur_paths = []
            if (y, x) not in img_graph:
                # Find nearest node in the graph
                nearest_node = min(img_graph.nodes, key=lambda node: (node[1] - x) ** 2 + (node[0] - y) ** 2)
                y, x = nearest_node

            for target in img_frontier_info.keys():
                print(f"Scoring frontier at {geo_frontier} with target {target}")
                target_y, target_x = target
                frontier_conf = img_frontier_info[target]['mean_conf']

                frontier_heading = ((target_x - w//2) / (w//2))*(self.dataset.fov/2.0)
                frontier_heading = np.deg2rad(frontier_heading)
                frontier_vec = np.array([
                    np.sin(frontier_heading),
                    np.cos(frontier_heading),
                    0
                ])
                goal_conf = 1+np.dot(goal_heading, frontier_vec)

                if (target_y, target_x) not in img_graph:
                    continue

                # Find shortest path using Dijkstra's algorithm
                try:
                    path_cost = distance_maps[(target_y, target_x)][(y, x)]
                    path = []
                    """
                    if weighted_graph:
                        path = nx.shortest_path(img_graph, source=(y, x), target=(target_y, target_x), weight='weight')
                        path_cost = sum(img_graph[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))
                    else:
                        path = nx.shortest_path(img_graph, source=(y, x), target=(target_y, target_x))
                        path_cost = len(path)
                    """
                    # path_length = len(path)
                    score = (
                        (frontier_conf * goal_conf) / ((path_cost) + 1e-6)
                    ) * score_scaling
                except nx.NetworkXNoPath:
                    score = 0.0
                    path = []

                cur_scores.append(score)
                cur_paths.append(path)
            
            # append the best score and path for this frontier
            if cur_scores:
                best_index = np.argmax(cur_scores)
                scores.append(cur_scores[best_index])
                paths.append(cur_paths[best_index])
            else:
                scores.append(0.0)
                paths.append([])
        return scores, paths
    
    def score_frontiers_mcp(
        self,
        geometric_frontiers: List[Tuple[int, int]],
        traversability: np.ndarray,
        img_frontiers: np.ndarray,
        img_frontier_info: Dict[Any, Any],
        goal_heading: np.ndarray,
        weighted_graph: bool = False,
        score_scaling: float = 100.0
    ) -> Tuple[List[float], List[List[Tuple[int, int]]]]:
        """Score the sampled frontiers based on traversability and frontier maps.
        
        :param geometric_frontiers: List of geometric frontiers as (x, y) tuples.
        :param traversability: The traversability map.
        :param img_frontier_info: The frontier information (discretized centers and confidences).
        :param goal_heading: The goal heading vector.
        :return: A tuple containing scores and paths for each frontier.
        """
        scores = []
        paths = []

        h, w = traversability.shape

        mcp = MCP_Geometric(1/(traversability + 1e-6), fully_connected=True)
        # mcp = MCP_Geometric(1-traversability, fully_connected=True)

        for geo_frontier in geometric_frontiers:
            x, y = geo_frontier
            cur_scores = []
            cur_paths = []
            costs, trbk = mcp.find_costs([(y,x)])
            # costs[(y, x)] = 0.0  # Set the cost of the starting point to 0
            # costs = ((h+w) - costs) / (h+w) # Invert costs to make higher costs better

            if img_frontier_info is not None:
                for target in img_frontier_info.keys():
                    print(f"Scoring frontier at {geo_frontier} with target {target}")
                    target_y, target_x = target
                    frontier_conf = img_frontier_info[target]['mean_conf']

                    frontier_heading = ((target_x - w//2) / (w//2))*(self.dataset.fov/2.0)
                    frontier_heading = np.deg2rad(frontier_heading)
                    frontier_vec = np.array([
                        np.sin(frontier_heading),
                        np.cos(frontier_heading),
                        0
                    ])
                    goal_conf = 1+np.dot(goal_heading, frontier_vec)
                    # goal_conf = (1+np.dot(goal_heading, frontier_vec)) / 2.0

                    path_cost = costs[(target_y, target_x)]
                    path = mcp.traceback((target_y, target_x))
                    score = (
                        (frontier_conf * goal_conf) / ((path_cost) + 1e-6)
                        # (frontier_conf * goal_conf * path_cost)
                    ) * score_scaling

                    cur_scores.append(score)
                    cur_paths.append(path)
            
            else:
                # use all pixels as targets
                pixel_indx = np.indices(img_frontiers.shape)
                frontier_heading = ((pixel_indx[1] - w//2) / (w//2))*(self.dataset.fov/2.0)
                frontier_heading = np.deg2rad(frontier_heading)
                frontier_vec = np.stack([
                    np.sin(frontier_heading),
                    np.cos(frontier_heading),
                    np.zeros_like(frontier_heading)
                ], axis=-1)
                goal_conf = (1 + np.dot(frontier_vec, goal_heading)) / 2.0
                frontier_conf = img_frontiers.copy()
                frontier_conf[frontier_conf < self.frontier_threshold] = 0.0
                frontier_conf[traversability < self.traversability_threshold] = 0.0

                score = (
                    # (frontier_conf * goal_conf) / ((costs + 1e-6) / ((h+w)))
                    (frontier_conf * goal_conf) / ((costs + 1e-6))
                    # (frontier_conf * goal_conf * costs)
                ) * score_scaling

                best_index = np.unravel_index(np.argmax(score), score.shape)
                cur_scores = score.flatten().tolist()
                cur_paths = [[]] * len(cur_scores)
                cur_paths[best_index[0] * w + best_index[1]] = mcp.traceback((best_index[0], best_index[1]))

            # append the best score and path for this frontier
            if cur_scores:
                best_index = np.argmax(cur_scores)
                scores.append(cur_scores[best_index])
                paths.append(cur_paths[best_index])
            else:
                scores.append(0.0)
                paths.append([])
        print(scores)
        return scores, paths
    
    def score_frontiers_mcp_add(
        self,
        geometric_frontiers: List[Tuple[int, int]],
        traversability: np.ndarray,
        img_frontiers: np.ndarray,
        img_frontier_info: Dict[Any, Any],
        goal_heading: np.ndarray,
        weighted_graph: bool = False,
        score_scaling: float = 100.0
    ) -> Tuple[List[float], List[List[Tuple[int, int]]]]:
        """Score the sampled frontiers based on traversability and frontier maps.
        
        :param geometric_frontiers: List of geometric frontiers as (x, y) tuples.
        :param traversability: The traversability map.
        :param img_frontier_info: The frontier information (discretized centers and confidences).
        :param goal_heading: The goal heading vector.
        :return: A tuple containing scores and paths for each frontier.
        """
        scores = []
        paths = []

        h, w = traversability.shape

        mcp = MCP_Geometric(1/(traversability + 1e-3), fully_connected=True)
        # mcp = MCP_Geometric(1-traversability, fully_connected=True)
        alpha, beta, gamma = 2.0, 3.0, 2.0
        for geo_frontier in geometric_frontiers:
            x, y = geo_frontier
            cur_scores = []
            cur_paths = []
            costs, trbk = mcp.find_costs([(y,x)])
            costs /= (h+w)


            if img_frontier_info is not None:
                for target in img_frontier_info.keys():
                    print(f"Scoring frontier at {geo_frontier} with target {target}")
                    target_y, target_x = target
                    frontier_conf = img_frontier_info[target]['mean_conf']

                    frontier_heading = ((target_x - w//2) / (w//2))*(self.dataset.fov/2.0)
                    frontier_heading = np.deg2rad(frontier_heading)
                    frontier_vec = np.array([
                        np.sin(frontier_heading),
                        np.cos(frontier_heading),
                        0
                    ])
                    goal_conf = (1+np.dot(goal_heading, frontier_vec)) / 2.0

                    path_cost = 1 - np.tanh(costs[(target_y, target_x)])
                    path = mcp.traceback((target_y, target_x))
                    score = (
                        alpha*frontier_conf + beta*goal_conf + gamma*path_cost
                    ) / (alpha + beta + gamma)
                    cur_scores.append(score)
                    cur_paths.append(path)
            
            else:
                # use all pixels as targets
                pixel_indx = np.indices(img_frontiers.shape)
                frontier_heading = ((pixel_indx[1] - w//2) / (w//2))*(self.dataset.fov/2.0)
                frontier_heading = np.deg2rad(frontier_heading)
                frontier_vec = np.stack([
                    np.sin(frontier_heading),
                    np.cos(frontier_heading),
                    np.zeros_like(frontier_heading)
                ], axis=-1)
                goal_conf = (1 + np.dot(frontier_vec, goal_heading)) / 2.0
                frontier_conf = img_frontiers.copy()
                frontier_conf[frontier_conf < self.frontier_threshold] = 0.0
                frontier_conf[traversability < self.traversability_threshold] = 0.0

                score = (
                    alpha*frontier_conf + beta*goal_conf + gamma*(1 - np.tanh(costs))
                ) / (alpha + beta + gamma)

                best_index = np.unravel_index(np.argmax(score), score.shape)
                cur_scores = score.flatten().tolist()
                cur_paths = [[]] * len(cur_scores)
                cur_paths[best_index[0] * w + best_index[1]] = mcp.traceback((best_index[0], best_index[1]))

            # append the best score and path for this frontier
            if cur_scores:
                best_index = np.argmax(cur_scores)
                scores.append(cur_scores[best_index])
                paths.append(cur_paths[best_index])
            else:
                scores.append(0.0)
                paths.append([])
        print(scores)
        return scores, paths, score

    def process_index(self, index: int):
        image, annotation = self.dataset.get_image_and_annotation(index)
        # gt_traversability_map = self.dataset.get_traversability(annotation)

        # Preprocess image for RADIO model
        traversability_map, img_frontiers, _ = self.model.forward_on_numpy(image.copy())
        traversability_map = traversability_map.squeeze().cpu().numpy()
        img_frontiers = img_frontiers.squeeze().cpu().numpy()
        viz_discrete_frontiers, img_frontier_info = self.discretize_img_frontiers(img_frontiers, traversability_map, visualize=True)

        goal_heading = self.sample_goal_heading()
        geometric_frontiers = self.sample_frontiers(image.shape[:2])

        if self.discretize_frontiers:
            scores, paths = self.score_frontiers_mcp(
                geometric_frontiers, traversability_map, img_frontiers, img_frontier_info, goal_heading
            )
        else:
            scores, paths, score_hm = self.score_frontiers_mcp_add(
            # scores, paths = self.score_frontiers_mcp(
                geometric_frontiers, traversability_map, img_frontiers, None, goal_heading
            )
        if len(geometric_frontiers) > 1:
            score_hm = None
        return self.visualize_results(
            index, image, traversability_map, img_frontiers, geometric_frontiers, goal_heading, scores, paths, viz_discrete_frontiers, img_frontier_info, score_hm
        )


    def visualize_results(
        self,
        index: int,
        image: np.ndarray,
        traversability: np.ndarray,
        img_frontiers: np.ndarray,
        geometric_frontiers: List[Tuple[int, int]],
        goal_heading: np.ndarray,
        scores: List[float],
        paths: List[List[Tuple[int, int]]],
        viz_discrete_frontiers: np.ndarray,
        img_frontier_info: Dict[Any, Any],
        score_hm: Optional[np.ndarray] = None
    ):
        """Visualize the results of the model.

        :param index: The index of the image in the dataset.
        :param image: The original image.
        :param traversability: The predicted traversability map.
        :param img_frontiers: The predicted frontier map.
        :param geometric_frontiers: The sampled geometric frontiers.
        :param goal_heading: The goal heading vector.
        :param scores: The scores for each frontier.
        :param paths: The best in-image paths for each frontier.
        :param viz_discrete_frontiers: The visualized discretized frontier confidences.
        :param img_frontier_info: The frontier information (discretized centers and confidences).
        """
        img_name = self.dataset.index_to_path[index]
        fig, axes = plt.subplots(2, 3, figsize=(17, 10))
        axes = axes.flatten()

        axes[0].imshow(image)
        axes[0].set_title(f"Image: {img_name}")
        axes[0].axis('off')

        # overlay frontiers on the image
        axes[1].imshow(image)
        hm = axes[1].imshow(img_frontiers, alpha=0.5, cmap='jet', vmin=0, vmax=1)
        axes[1].set_title("Frontiers Overlay")
        axes[1].axis('off')
        plt.colorbar(hm, ax=axes[1], fraction=0.046, pad=0.04)

        # overlay traversability on the image
        axes[2].imshow(image)
        hm = axes[2].imshow(traversability, alpha=0.5, cmap='jet', vmin=0, vmax=1)
        axes[2].set_title("Traversability Overlay")
        axes[2].axis('off')
        plt.colorbar(hm, ax=axes[2], fraction=0.046, pad=0.04)

        # plot geometric frontiers and their paths 
        # color the frontiers based on their scores
        axes[3].imshow(image)
        for i, (geo_frontier, score) in enumerate(zip(geometric_frontiers, scores)):
            x, y = geo_frontier
            color = plt.cm.jet(score)  # Use jet colormap for scores
            axes[3].scatter(x, y, color=color, s=100, label=f"Frontier {i+1} ({score:.2f})")
            axes[3].text(x, y, f"{i+1}", color='white', fontsize=12, ha='center', va='center')
            if paths[i]:
                path = np.array(paths[i])
                axes[3].plot(path[:, 1], path[:, 0], color=color, linewidth=2, alpha=0.7)
                axes[3].scatter(path[-1, 1], path[-1, 0], color='white', marker='x', s=100)
        plt.colorbar(hm, ax=axes[3], fraction=0.046, pad=0.04)
        if score_hm is not None:
            axes[3].imshow(score_hm, cmap='jet', vmin=0, vmax=1, alpha=0.5)
        axes[3].set_title("Geometric Frontiers and Paths")
        axes[3].legend(loc='upper right', fontsize='small')
        axes[3].axis('off')

        # show the goal heading and current heading
        # axes[4].imshow(image)
        h, w = image.shape[:2]
        axes[4].quiver(w // 2, h // 2, goal_heading[0], goal_heading[1], color='r', scale=5, label='Goal Heading')
        axes[4].quiver(w // 2, h // 2, 0, 1, color='b', scale=5, label='Current Heading')  # Current heading (up)
        axes[4].set_title(f"Goal Heading: {goal_heading}")
        axes[4].legend(loc='upper right', fontsize='small')
        axes[4].axis('off')

        # show discretized frontiers
        axes[5].imshow(image)
        hm = axes[5].imshow(viz_discrete_frontiers, alpha=0.5, cmap='jet', vmin=0, vmax=1)
        for (center, info) in img_frontier_info.items():
            center_y, center_x = center
            mean_conf = info['mean_conf']
            axes[5].scatter(center_x, center_y, color='white', marker='x', s=50, label=f"Center ({center_x}, {center_y}) Conf: {mean_conf:.2f}")
        axes[5].legend(loc='upper right', fontsize='small')
        axes[5].set_title("Discretized Frontiers")
        plt.colorbar(hm, ax=axes[5], fraction=0.046, pad=0.04)
        axes[5].axis('off')
        plt.tight_layout()

        # show the score heatmap if available
        # if score_hm is not None:
        #     axes[5].imshow(score_hm, cmap='jet', vmin=0, vmax=1)
        #     axes[5].set_title("Score Heatmap")
        #     axes[5].axis('off')
        #     plt.colorbar(axes[5].images[0], ax=axes[5], fraction=0.046, pad=0.04)
        # axes[5].axis('off')
        # axes[7].axis('off')
        
        # Convert plot to image
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:,:,1:]
        data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
        plt.close(fig)

        cv2.imshow("Results", data)
        key = cv2.waitKey(0) & 0xFF
        if key == 27:  # ESC key to exit
            return False
        
        return True


    def run(self):
        """Run the model on the dataset and visualize results."""
        for index in tqdm(range(len(self.dataset))):
            if not self.process_index(index):
                print(f"Exiting at index {index} due to user input.")
                break

if __name__ == "__main__":
    frontier_ckpt = "ckpts/frontier_head.ckpt"
    traversability_ckpt = "ckpts/trav_head.ckpt"

    rugd_dataset = RUGDTraversabilityDataset("/home/$USER/data/RUGD")
    nebula_dataset = NebulaDataset("/home/$USER/data/nebula")
    radio_dn_model = ExploRFMInference(
        frontier_ckpt=frontier_ckpt,
        traversability_ckpt=traversability_ckpt,
        model_version="c-radio_v3-b",
        adaptor_version="siglip2",
        use_naclip=True,
        use_summary_for_spatial=True,
        radio_dim=768,
        static_scale_factor=0.5,
        model_precision="FP16",
    )

    model = ExploRFMScoringTest(
        dataset=nebula_dataset,
        # dataset=rugd_dataset,
        model=radio_dn_model,
        num_frontiers=4,
        # num_radial_bins=1,
        top_k_frontiers=5,
        discretize_frontiers=False,
        traversability_threshold=0.6,
        frontier_threshold=0.5,
    )
    
    model.run()
    cv2.destroyAllWindows()