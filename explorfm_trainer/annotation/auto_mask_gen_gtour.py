import os
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import cv2

import torch
import torchvision

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

class GrandTour_Loader:
    """ Loader for selected images from the GrandTour Dataset
    """
    def __init__(self, data_path):
        self.data_path = data_path
        self.images_path = os.path.join(data_path, "RGB_frames")
        self.boundaries_path = os.path.join(data_path, "SAM_boundaries")
        self.rect_path = os.path.join(data_path, "RGB_rectified")
        os.makedirs(self.boundaries_path, exist_ok=True)
        os.makedirs(self.rect_path, exist_ok=True)

        self.img_list = list(Path(self.images_path).rglob("*.png"))
        self.img_list = [str(img).split("/")[-1] for img in self.img_list]

        # ---- Calibration Parameters ----
        self.DISTORTION_MODEL = "equidistant"
        self.D = np.array(
                [-0.06226555154591874, 0.006984942920386333, -0.005291335660179726, 0.001455018149071658]
            )
        self.K = np.array(
                [[984.8643239835407, 0.0, 946.7086445278437],
                [0.0, 984.5008837495056, 636.9616008492758],
                [0.0, 0.0, 1.0]]
            )

    def rectify_fisheye(self, image):
        h, w = image.shape[:2]
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            self.K, self.D, (w, h), np.eye(3), balance=0
        )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            self.K, self.D, np.eye(3), new_K, (w, h), cv2.CV_16SC2
        )
        undistorted = cv2.remap(
            image, map1, map2,
            interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )
        return undistorted


    def __len__(self):
        return len(self.img_list)
    
    def load_index(self, index):
        """ Load an image by index.
        """
        if index < 0 or index >= len(self.img_list):
            raise IndexError("Index out of range")
        
        img_name = self.img_list[index]
        img_path = os.path.join(self.images_path, img_name)
        return img_path, img_name
    
class AutoMaskGenerator:
    """ Class to generate boundaries using SAM2
    """

    def __init__(self, data_path, device, thickness, viz=False):
        self.loader = GrandTour_Loader(data_path)

        sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
        model_cfg = "//cluster/home/$USER/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"

        sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

        self.mask_generator = SAM2AutomaticMaskGenerator(
            model=sam2,
            points_per_side=64,
            points_per_batch=128,
            pred_iou_thresh=0.7,
            stability_score_thresh=0.92,
            stability_score_offset=0.7,
            crop_n_layers=0,
            box_nms_thresh=0.7,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=45.0,
            use_m2m=False,
        )
        self.thickness = thickness
        self.viz = viz

    def compute_boundaries(self, anns):
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

        img = np.zeros((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        for ann in sorted_anns:
            m = ann['segmentation']
            
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (1, 1, 1, 1), thickness=self.thickness)
        
        return img

    def visualize_masks(self, index, rgb_image, mask_image):
        plt.figure(figsize=(20, 20))
        plt.imshow(rgb_image)
        ax = plt.gca()
        ax.set_autoscale_on(False)
        ax.imshow(mask_image)

        plt.axis('off')
        plt.savefig(f"tmp/mask_{index}.png", bbox_inches='tight', pad_inches=0)
        plt.close()

    def generate_mask(self, index):
        """ Generate mask for a given index.
        """
        img_path, img_name = self.loader.load_index(index)
        image = np.array(Image.open(img_path).convert("RGB"))
        if self.loader.DISTORTION_MODEL == "equidistant":
            image = self.loader.rectify_fisheye(image)

        masks = self.mask_generator.generate(image)
        mask_image = self.compute_boundaries(masks)

        del masks

        if self.viz:
            self.visualize_masks(index, image, mask_image)
        
        # Save the mask image (convert to single channel)
        mask_image = (mask_image[:, :, 0] * 255).astype(np.uint8)
        mask_save_path = os.path.join(self.loader.boundaries_path, img_name.replace('rgb', 'bound'))
        cv2.imwrite(mask_save_path, mask_image)

        # Save the rectified image
        if self.loader.DISTORTION_MODEL == "equidistant":
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            rect_save_path = os.path.join(self.loader.rect_path, img_name.replace('rgb', 'rect'))
            cv2.imwrite(rect_save_path, image)     

    def generate_masks(self):
        """ Generate masks for all images in the dataset.
        """
        for i in tqdm(range(len(self.loader))):
            self.generate_mask(i)
        print("All masks generated.")

if __name__ == "__main__":
    data_path = "/cluster/home/$USER/scratch/data/grand_tour_selected"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mask_generator = AutoMaskGenerator(data_path, device, thickness=6, viz=True)
    mask_generator.generate_masks()