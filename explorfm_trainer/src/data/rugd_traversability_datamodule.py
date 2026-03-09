from typing import Any, Dict, Optional, Tuple, List
from enum import Enum
import os
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import numpy as np
import torch
from lightning import LightningDataModule
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torchvision.transforms.functional import pil_to_tensor

np.random.seed(42)

class RUGDTraversabilityDataset(Dataset):
    """
    Pytorch Dataset for the RUGD dataset for traversability estimation.
    """

    def __init__(
        self,
        data_dir: str = "data/",
        resize_resolution: Optional[Tuple[int, int]] = None,
        phase: str = "train",
        dataset_size: Optional[int] = None,
    ) -> None:
        """Initialize a `RUGDTraversabilityDataset`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param transforms: List of transformations to apply. Defaults to `None`.
        """
        self.data_dir = data_dir
        self.resize_resolution = resize_resolution
        self.phase = phase
        
        self.safe_labels = [
            "dirt", "sand", "grass", "asphalt", "gravel", "mulch", "rock-bed", "concrete"
        ]
        self.train_scenes = [
            "park-1", "park-2", "trail", "trail-3", "trail-4",
            "trail-5", "trail-6", "trail-7", "trail-9", "trail-10", "trail-11",
            "trail-12", "trail-13", "trail-14"
        ]
        self.val_scenes = ["park-8", "trail-15"]
        self.split_scenes = ["creek", "village"]
        self.split_percentages = [0.8, 0.2]  # 80% for training, 20% for validation

        self.raw_frames_path = os.path.join(self.data_dir, "RUGD_frames")
        self.annotations_path = os.path.join(self.data_dir, "RUGD_annotations")
        self.colormap_path = os.path.join(self.annotations_path, "RUGD_annotation-colormap.txt")
        self.seg_colormap = self.load_annotations()

        self.train_paths, self.val_paths = self.load_datapaths()

        # transforms
        self.train_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.val_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])

        if self.phase == "train":
            self.data_paths = self.train_paths
            self.transforms = self.train_transforms
        elif self.phase == "val":
            self.data_paths = self.val_paths
            self.transforms = self.val_transforms
        else:
            raise ValueError(f"Invalid phase: {self.phase}. Use 'train' or 'val'.")
        if dataset_size is not None:
            self.data_paths = self.data_paths[:dataset_size]

        
    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.data_paths)

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
    
    def load_datapaths(self) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """Load paths for training and validation datasets."""
        train_paths = []
        val_paths = []

        for scene in tqdm(self.train_scenes, desc="Loading training paths"):
            imgs = os.listdir(os.path.join(self.raw_frames_path, scene))

            for img in imgs:
                raw_img_path = os.path.join(self.raw_frames_path, scene, img)
                annotation_path = os.path.join(self.annotations_path, scene, img)
                if os.path.exists(annotation_path):
                    train_paths.append((raw_img_path, annotation_path))

        for scene in tqdm(self.val_scenes, desc="Loading validation paths"):
            imgs = os.listdir(os.path.join(self.raw_frames_path, scene))

            for img in imgs:
                raw_img_path = os.path.join(self.raw_frames_path, scene, img)
                annotation_path = os.path.join(self.annotations_path, scene, img)
                if os.path.exists(annotation_path):
                    val_paths.append((raw_img_path, annotation_path))

        # Split the split scenes into train and val sets
        for scene in tqdm(self.split_scenes, desc="Splitting scenes"):
            imgs = os.listdir(os.path.join(self.raw_frames_path, scene))
            split_index = int(len(imgs) * self.split_percentages[0])

            imgs = np.random.permutation(imgs)  # Shuffle the images
            for img in imgs[:split_index]:
                raw_img_path = os.path.join(self.raw_frames_path, scene, img)
                annotation_path = os.path.join(self.annotations_path, scene, img)
                if os.path.exists(annotation_path):
                    train_paths.append((raw_img_path, annotation_path))

            for img in imgs[split_index:]:
                raw_img_path = os.path.join(self.raw_frames_path, scene, img)
                annotation_path = os.path.join(self.annotations_path, scene, img)
                if os.path.exists(annotation_path):
                    val_paths.append((raw_img_path, annotation_path))

        return train_paths, val_paths
    
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
    
    def __getitem__(self, idx: int) -> Dict:
        """Load an item from the dataset."""
        raw_img_path, annotation_path = self.data_paths[idx]
        raw_img = np.array(Image.open(raw_img_path).convert("RGB"))
        gt_img = np.array(Image.open(annotation_path).convert("RGB"))

        gt_traversability = self.get_traversability(gt_img)

        # Convert to tensor
        raw_img = self.transforms(raw_img)
        gt_traversability = torch.tensor(gt_traversability)
        gt_traversability = gt_traversability.unsqueeze(0)  # Add channel dimension
        gt_img = torch.tensor(gt_img)

        original_size = raw_img.shape[-2:]
        if self.resize_resolution:
            raw_img = F.interpolate(
                raw_img.unsqueeze(0),
                size=self.resize_resolution,
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

        return {
            "raw_img": raw_img,
            "gt_traversability": gt_traversability,
            "gt_img": gt_img,
            "original_size": original_size,
            "img_path": "/".join(raw_img_path.split("/")[-2:]),
        }


class RUGDTraversabilityDataModule(LightningDataModule):
    """`LightningDataModule` for the RUGD dataset for traversability estimation.
    """

    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        train_size: Optional[int] = None,
        val_size: Optional[int] = None,
    ) -> None:
        """Initialize a `RUGDTraversabilityDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None

        self.batch_size_per_device = batch_size
        self.resize_resolution = None

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val:
            self.data_train = RUGDTraversabilityDataset(
                data_dir=self.hparams.data_dir,
                resize_resolution=self.resize_resolution,
                phase="train",
                dataset_size=self.hparams.train_size,
            )
            self.data_val = RUGDTraversabilityDataset(
                data_dir=self.hparams.data_dir,
                resize_resolution= self.resize_resolution,
                phase="val",
                dataset_size=self.hparams.val_size,
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

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    module = RUGDTraversabilityDataModule(
        data_dir="/home/$USER/data/RUGD",
        batch_size=4,
        num_workers=4,
        pin_memory=True,
    )
    module.prepare_data()
    module.setup()

    train_loader = module.train_dataloader()

    print(f"Number of training batches: {len(train_loader)}")
    print(f"Batch size per device: {module.batch_size_per_device}")

    for batch in train_loader:
        raw_img = batch["raw_img"]
        gt_traversability = batch["gt_traversability"]
        gt_img = batch["gt_img"]

        B, C, H, W = raw_img.shape
        print(f"Image batch shape: {raw_img.shape}")
        print(f"Raw image max: {raw_img.max()}, min: {raw_img.min()}")
        assert raw_img.max() <= 1.0 and raw_img.min() >= 0.0, "Raw image tensor should be normalized to [0, 1] range."
        assert torch.all((gt_traversability == 0) | (gt_traversability == 1)), "Ground truth traversability should be binary (0 or 1)."

        for idx in range(B):
            fig, axes = plt.subplots(1, 3, figsize=(15, 8))
        
            # 1. Original RGB Image
            axes[0].imshow(raw_img[idx].permute(1, 2, 0).numpy())
            axes[0].set_title('Original RGB')
            axes[0].axis('off')
        
            # 2. Ground Truth Segmentation with Color Mapping
            axes[1].imshow(gt_img[idx])
            axes[1].set_title('Ground Truth Segmentation')
            axes[1].axis('off')

            # 3. Ground Truth Traversability Map
            axes[2].imshow(gt_traversability[idx][0], cmap='gray')
            axes[2].set_title('Ground Truth Safe Mask')
            axes[2].axis('off')

            # Add segmentation legend below all subplots
            handles = [
                mpatches.Patch(color=np.array(color)/255.0, label=label)
                for label, color in module.data_train.seg_colormap.items()
            ]
            fig.legend(handles=handles, loc='lower center', ncol=4, fontsize='small', frameon=False)

            plt.tight_layout(rect=[0, 0.1, 1, 1])  # Leave space for the legend
            plt.show()