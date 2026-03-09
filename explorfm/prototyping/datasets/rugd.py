import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class RUGDTraversabilityDataset:
    def __init__(
        self, data_dir: str, fov: float = 90.0,
    ):
        self.data_dir = data_dir
        self.fov = fov
        self.raw_frames_path = os.path.join(self.data_dir, "RUGD_frames")
        self.annotations_path = os.path.join(self.data_dir, "RUGD_annotations")
        self.colormap_path = os.path.join(self.annotations_path, "RUGD_annotation-colormap.txt")
        self.seg_colormap = self.load_annotations()

        self.scenes = os.listdir(self.raw_frames_path)
        self.scenes = [scene for scene in self.scenes if os.path.isdir(os.path.join(self.annotations_path, scene))]

        self.scene_data = {}
        self.index_to_path = {}
        self.dataset_size = 0
        for scene in self.scenes:
            scene_frames = sorted(os.listdir(os.path.join(self.raw_frames_path, scene)))
            for frame in scene_frames:
                frame_path = os.path.join(self.raw_frames_path, scene, frame)
                annotation_path = os.path.join(self.annotations_path, scene, frame)
                if os.path.exists(annotation_path):
                    self.scene_data[frame] = {
                        'image': frame_path,
                        'annotation': annotation_path
                    }
                    self.index_to_path[self.dataset_size] = f"{scene}/{frame}"
                    self.dataset_size += 1
        self.cmap = plt.get_cmap('jet')

        self.safe_labels = [
            "dirt", "sand", "grass", "asphalt", "gravel", "mulch", "rock-bed", "concrete"
        ]

    def __len__(self):
        return self.dataset_size


    def load_annotations(self):
        colormap = {}
        with open(self.colormap_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 5:
                    _, label, r, g, b = parts
                    colormap[label] = (int(r), int(g), int(b))
                else:
                    raise ValueError(f"Invalid line in annotations file: {line.strip()}")
        return colormap
    

    def get_image_and_annotation(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        if index < 0 or index >= self.dataset_size:
            raise IndexError("Index out of bounds for dataset size.")
        
        frame_name = list(self.scene_data.keys())[index]
        frame_info = self.scene_data[frame_name]

        image = np.array(Image.open(frame_info['image']).convert('RGB'))
        annotation = np.array(Image.open(frame_info['annotation']).convert('RGB'))

        return image, annotation
    
    def get_traversability(self, gt_img: np.ndarray) -> np.ndarray:
        """Convert ground truth image to traversability map."""
        safe_mask = np.zeros(gt_img.shape[:2], dtype=np.uint8)

        for label in self.safe_labels:
            if label not in self.seg_colormap:
                raise ValueError(f"Label '{label}' not found in segmentation colormap.")
            color = self.seg_colormap[label]
            mask = np.all(gt_img == np.array(color), axis=-1)
            safe_mask[mask] = 1

        return safe_mask
    