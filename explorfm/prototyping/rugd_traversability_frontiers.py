from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

import numpy as np
import torch

from datasets.rugd import RUGDTraversabilityDataset
from datasets.nebula import NebulaDataset
from ..explorfm_model import ExploRFMInference

class ExploRFMFrontierTraversability:
    def __init__(self,
        dataset: RUGDTraversabilityDataset,
        model: ExploRFMInference
    ):
        self.dataset = dataset
        self.model = model

    def process_index(self, index: int):
        image, annotation = self.dataset.get_image_and_annotation(index)
        traversability_map = self.dataset.get_traversability(annotation)

        # Preprocess image for RADIO model
        traversability, frontiers, _ = self.model.forward_on_numpy(image.copy())
        return self.visualize_results(index, image, traversability, frontiers)


    def visualize_results(self, index: int, image: np.ndarray, traversability: torch.Tensor, frontiers: torch.Tensor):
        """Visualize the results of the model.

        :param index: The index of the image in the dataset.
        :param image: The original image.
        :param frontiers: The predicted frontier map.
        :param traversability: The predicted traversability map.
        """
        img_name = self.dataset.index_to_path[index]
        fig, axes = plt.subplots(1, 3, figsize=(15, 8))
        axes = axes.flatten()

        axes[0].imshow(image)
        axes[0].set_title(f"Image: {img_name}")
        axes[0].axis('off')

        # overlay frontiers on the image
        frontier_map = frontiers.squeeze().cpu().numpy()
        axes[1].imshow(image)
        hm = axes[1].imshow(frontier_map, alpha=0.5, cmap='jet', vmin=0, vmax=1)
        axes[1].set_title("Frontiers Overlay")
        axes[1].axis('off')
        plt.colorbar(hm, ax=axes[1], fraction=0.046, pad=0.04)

        # overlay traversability on the image
        traversability_map = traversability.squeeze().cpu().numpy()
        axes[2].imshow(image)
        hm = axes[2].imshow(traversability_map, alpha=0.5, cmap='jet', vmin=0, vmax=1)
        axes[2].set_title("Traversability Overlay")
        axes[2].axis('off')
        plt.colorbar(hm, ax=axes[2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        
        # Convert plot to image
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:,:,1:]
        data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
        plt.close(fig)

        cv2.imshow("Results", data)
        key = cv2.waitKey(1) & 0xFF
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

    model = ExploRFMFrontierTraversability(
        # dataset=rugd_dataset,
        dataset=nebula_dataset,
        model=radio_dn_model
    )
    
    model.run()
    cv2.destroyAllWindows()