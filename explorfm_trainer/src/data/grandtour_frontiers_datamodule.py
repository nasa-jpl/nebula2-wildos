from typing import Optional, List, Tuple, Dict, Any
import os
from enum import Enum, auto

import json
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from functools import partial

from lightning import LightningDataModule
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

class FrontierAnnotationType(Enum):
    """
    Enum for different types of annotations in the GrandTour dataset.
    """
    SAM_BOUNDARY = auto()
    BBOX = auto()


class GrandTourFrontiersDataset(Dataset):
    """
    Pytorch Dataset for Image Frontiers in the GrandTour dataset.
    """
    def __init__(
        self,
        data_dir: str,
        gt_type: str,
        boundary_dilation: int = 0,
        labels: Optional[List[str]] = None,
        dataset_size: Optional[int] = None,
        boundary_frontier_thickness: int = 3,
        resize_factor: Optional[float] = None,
        augmentations_prob: float = 0.0,
        fixed_bbox_ht: Optional[int] = None,
    ):
        self.data_dir = data_dir
        self.img_dir = os.path.join(data_dir, "RGB_rectified")
        self.boundary_dir = os.path.join(data_dir, "SAM_boundaries")
        self.annotations_dir = os.path.join(data_dir, "annotations")

        all_img_names = sorted([
            f for f in os.listdir(self.img_dir) if f.endswith('.png')
        ])
        self.img_names = []
        for img_name in all_img_names:
            if not os.path.exists(os.path.join(self.boundary_dir, img_name.replace('rect', 'bound'))):
                continue
            if not os.path.exists(os.path.join(self.annotations_dir, img_name.replace('rect', 'annotation').replace('.png', '.json'))):
                continue
            self.img_names.append(img_name)
        
        if dataset_size is not None:
            self.img_names = self.img_names[:dataset_size]

        self.gt_type = FrontierAnnotationType[gt_type]
        self.labels = labels
        self.boundary_frontier_thickness = boundary_frontier_thickness

        self.boundary_dilation = boundary_dilation
        if self.boundary_dilation > 0:
            self.dilation_kernel = np.ones((self.boundary_dilation, self.boundary_dilation), np.uint8)
        else:
            self.dilation_kernel = None
        self.resize_factor = resize_factor if resize_factor is not None else 1.0
        self.fixed_bbox_ht = int(fixed_bbox_ht * self.resize_factor)
        
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
            ], p=augmentations_prob),
        ])


    def __len__(self):
        """ Returns the length of the dataset. """
        return len(self.img_names)

    def sort_bbox(self, bbox: List[int], bounds: Tuple[int, int]) -> List[int]:
        """ Sorts the bounding box coordinates in the order [x1, y1, x2, y2]. """
        # ensure coordinates are top-left and bottom-right
        new_box = [min(bbox[0], bbox[2]), min(bbox[1], bbox[3]), max(bbox[0], bbox[2]), max(bbox[1], bbox[3])]
        x1, y1, x2, y2 = new_box

        if self.fixed_bbox_ht is not None:
            if y2 - y1 < self.fixed_bbox_ht:
                y2 = y1 + self.fixed_bbox_ht

        # ensure coordinates are within image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(bounds[1], x2)
        y2 = min(bounds[0], y2)
        return [x1, y1, x2, y2]

    def load_annotation(
            self,
            annotation_path: str,
            image_shape: Tuple[int, int]
        ) -> Tuple[np.ndarray, List[Dict], np.ndarray]:
        """
        Load the bbox annotations from a JSON file.
        Return the bounding boxes as semantic segmentation masks and a list of annotations.
        """
        if not os.path.exists(annotation_path):
            raise FileNotFoundError(f"Annotation file {annotation_path} does not exist.")
        
        with open(annotation_path, 'r') as f:
            annotations = json.load(f)

        masks = np.zeros(image_shape, dtype=np.uint8)
        viz_ann = np.zeros(image_shape + (3,), dtype=np.uint8)  # For visualization
        for annotation in annotations:
            box = annotation['start'] + annotation['end']
            box = [int(coord * self.resize_factor) for coord in box]
            box = self.sort_bbox(box, (image_shape[0]-1, image_shape[1]-1))

            label = annotation['label']

            x1, y1, x2, y2 = box
            if self.labels is None or (label in self.labels):
                if label == "image_boundary_frontier":
                    assert x1==0 or x2==image_shape[1]-1, print(f"Image shape: {image_shape}, x1: {x1}, x2: {x2}")
                    if x1 == 0:
                        x2 = x1 + self.boundary_frontier_thickness
                    elif x2 == image_shape[1] - 1:
                        x1 = x2 - self.boundary_frontier_thickness
                    y2 = y1 + self.boundary_frontier_thickness
                    masks[y1:y2, x1:x2] = 2
                else:
                    masks[y1:y2, x1:x2] = 1
                cv2.rectangle(viz_ann, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(viz_ann, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return masks, annotations, viz_ann
    
    def process_annotation(
        self,
        mask_img: np.ndarray,
        boundary_img: np.ndarray,
    ) -> np.ndarray:
        """
        Convert the bbox masks to SAM boundaries that lie inside the bounding boxes.
        Dilation is applied to the boundaries if specified.
        """
        if self.dilation_kernel is not None:
            boundary_img = cv2.dilate(boundary_img, self.dilation_kernel, iterations=1)
        
        processed_mask = np.logical_or(mask_img==2, np.logical_and(mask_img==1, boundary_img)).astype(np.uint8)
        return processed_mask

    def __getitem__(self, idx: int) -> dict:
        """ Returns a dictionary containing the image and its corresponding annotation. """
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        boundary_path = os.path.join(self.boundary_dir, img_name.replace('rect', 'bound'))
        annotation_path = os.path.join(self.annotations_dir, img_name.replace('rect', 'annotation').replace('.png', '.json'))

        raw_img = np.array(Image.open(img_path).convert("RGB"))
        boundary_img = cv2.imread(boundary_path, cv2.IMREAD_UNCHANGED) // 255

        if self.resize_factor != 1.0:
            raw_img = cv2.resize(raw_img, (0, 0), fx=self.resize_factor, fy=self.resize_factor)
            boundary_img = cv2.resize(boundary_img, (raw_img.shape[1], raw_img.shape[0]))

        mask_img, _, viz_ann = self.load_annotation(annotation_path, raw_img.shape[:2])
        
        if self.gt_type == FrontierAnnotationType.SAM_BOUNDARY:
            mask_img = self.process_annotation(mask_img, boundary_img)
        mask_img = (mask_img > 0).astype(np.uint8)  # Ensure mask is binary
        
        raw_img = self.transforms(raw_img)
        mask_img = torch.tensor(mask_img).unsqueeze(0)  # Add channel dimension
        boundary_img = torch.tensor(boundary_img).unsqueeze(0)
        viz_ann = torch.tensor(viz_ann).permute(2, 0, 1)  # Convert to CxHxW format

        return {
            'image': raw_img,
            'mask': mask_img,
            'boundary': boundary_img,
            'bbox_annotations': viz_ann,
            'image_name': img_name
        }
    
class GrandTourFrontiersDataModule(LightningDataModule):
    """
    Pytorch Lightning DataModule for the GrandTour Frontiers dataset.
    """
    def __init__(
            self,
            data_dir: str,
            gt_type: str,
            boundary_dilation: int = 0,
            labels: Optional[List[str]] = None,
            dataset_size: Optional[int] = None,
            boundary_frontier_thickness: int = 3,
            train_val_split: List[int] = [0.8, 0.2],
            augmentations_prob: float = 0.0,
            resize_factor: Optional[float] = None,
            fixed_bbox_ht: Optional[int] = None,
            batch_size: int = 64,
            num_workers: int = 0,
            pin_memory: bool = False,
    ) -> None:
        """Initialize the GrandTourFrontiersDataModule.
        
        :param data_dir: The directory containing the GrandTour dataset.
        :param gt_type: The type of ground truth annotations to use (e.g., 'SAM_BOUNDARY').
        :param boundary_dilation: The amount of dilation to apply to the SAM boundaries.
        :param labels: The list of labels to consider for bounding boxes.
        :param dataset_size: The size of the dataset to use. If None, use the full dataset.
        :param boundary_frontier_thickness: The thickness of the boundary frontier.
        :param batch_size: The batch size for data loading.
        :param num_workers: The number of workers for data loading.
        :param pin_memory: Whether to pin memory for data loading.
        """

        super().__init__()

        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None

        self.batch_size_per_device = batch_size
        
    def setup(self, stage: Optional[str] = None) -> None:
        """Setup the dataset for training and validation."""
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val:
            dataset = partial(
                GrandTourFrontiersDataset,
                data_dir=self.hparams.data_dir,
                gt_type=self.hparams.gt_type,
                boundary_dilation=self.hparams.boundary_dilation,
                labels=self.hparams.labels,
                dataset_size=self.hparams.dataset_size,
                boundary_frontier_thickness=self.hparams.boundary_frontier_thickness,
                resize_factor=self.hparams.resize_factor,
                fixed_bbox_ht=self.hparams.fixed_bbox_ht,
            )

            if len(self.hparams.train_val_split) == 1:
                self.data_train = dataset(
                    augmentations_prob=self.hparams.augmentations_prob
                )

                num_val = self.hparams.train_val_split[0]
                self.data_val, _ = random_split(
                    dataset=dataset(augmentations_prob=0.0),
                    lengths=[num_val, 1 - num_val],
                    generator=torch.Generator().manual_seed(42),
                )
            else:
                self.data_train, self.data_val = random_split(
                    dataset=dataset(
                        augmentations_prob=self.hparams.augmentations_prob
                    ),
                    lengths=self.hparams.train_val_split,
                    generator=torch.Generator().manual_seed(42),
                )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

if __name__ == "__main__":
    module = GrandTourFrontiersDataModule(
        data_dir="/cluster/home/$USER/scratch/data/grand_tour_selected",
        gt_type="BBOX",     # or "BBOX"
        boundary_dilation=10,
        labels=['frontier'],
        dataset_size=1000,
        boundary_frontier_thickness=50,
        train_val_split=[0.2],
        resize_factor=0.25,
        augmentations_prob=1.0,
        fixed_bbox_ht=50,
        batch_size=4,
        num_workers=4,
        pin_memory=True
    )
    module.prepare_data()
    module.setup()

    train_loader = module.train_dataloader()
    val_loader = module.val_dataloader()

    print(f"Number of training batches: {len(train_loader)}")
    print(f"Batch size per device: {module.batch_size_per_device}")

    num_neg = 0
    num_pos = 0

    for batch in val_loader:

        raw_img = batch["image"]
        mask_img = batch["mask"]
        boundary_img = batch["boundary"]
        img_name = batch["image_name"]
        bbox_ann = batch["bbox_annotations"]

        B, C, H, W = raw_img.shape
        print(f"Processing batch with {B} samples")
        print(f"Image batch shape: {raw_img.shape}")
        print(f"Raw image max: {raw_img.max()}, min: {raw_img.min()}")
        assert raw_img.max() <= 1.0 and raw_img.min() >= 0.0, "Raw image tensor should be normalized to [0, 1] range."
        assert torch.all((mask_img == 0) | (mask_img == 1)), "Ground truth mask should be binary (0 or 1)."
        assert torch.all((boundary_img == 0) | (boundary_img == 1)), "Ground truth boundary should be binary (0 or 1)."

        num_neg += (mask_img == 0).sum().item()
        num_pos += (mask_img == 1).sum().item()

        for idx in range(B):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            img = raw_img[idx].permute(1, 2, 0).cpu().numpy() * 255
            img = img.astype(np.uint8)

            # Visualize the raw image with BBox labels
            img1 = img.copy()
            viz_ann = bbox_ann[idx].permute(1, 2, 0).cpu().numpy()
            img1[np.any(viz_ann, axis=-1)] = viz_ann[np.any(viz_ann, axis=-1)]  # Apply bbox annotations
            axes[0].imshow(img1)
            axes[0].set_title(f"Image with BBox Labels: {img_name[idx]}")
            axes[0].axis('off')

            # Visualize the ground truth boundary overlaid on the image
            boundary = boundary_img[idx].squeeze().cpu().numpy()
            boundary_overlay = img.copy()
            boundary_overlay[boundary > 0] = [255, 0, 0]  # Red overlay for boundary
            axes[1].imshow(boundary_overlay)
            axes[1].set_title(f"Ground Truth Boundary")
            axes[1].axis('off')

            # Visualize the ground truth mask overlaid on the image
            mask = mask_img[idx].squeeze().cpu().numpy()
            mask_overlay = img.copy()
            mask_overlay[mask > 0] = [0, 255, 0]  # Green overlay for mask
            axes[2].imshow(mask_overlay)
            axes[2].set_title(f"Ground Truth Mask")
            axes[2].axis('off')

            plt.tight_layout()
            plt.show()
    
    print(f"Total negative pixels: {num_neg}, Total positive pixels: {num_pos}")
    print(f"Weight for BCE: {num_neg / num_pos}")