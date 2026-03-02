import os
from typing import Tuple

import numpy as np
from PIL import Image

class NebulaDataset:
    def __init__(self, data_dir: str, fov: float = 120.0, frame_skip: int = 10):
        self.data_dir = data_dir
        self.fov = fov
        self.images_path = self.data_dir

        all_images = sorted(os.listdir(self.images_path))
        self.images = []
        self.index_to_path = {}
        for i, img in enumerate(all_images):
            if i % frame_skip == 0:
                self.index_to_path[len(self.images)] = img
                self.images.append(img)
        self.dataset_size = len(self.images)
    
    def __len__(self):
        return self.dataset_size

    def get_image_and_annotation(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        if index < 0 or index >= self.dataset_size:
            raise IndexError("Index out of bounds for dataset size.")

        image_path = os.path.join(self.images_path, self.index_to_path[index])
        image = np.array(Image.open(image_path).convert('RGB'))
        annotation = None

        return image, annotation
    
    def get_traversability(self, gt_img: np.ndarray) -> np.ndarray:
        return None