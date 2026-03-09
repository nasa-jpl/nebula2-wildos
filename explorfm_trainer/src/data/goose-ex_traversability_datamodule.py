from typing import Any, Dict, Optional, Tuple, List
from enum import Enum
import os
import glob
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import cv2
import numpy as np
import csv
import torch
from lightning import LightningDataModule
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torchvision.transforms.functional import pil_to_tensor

np.random.seed(42)

def __check_labels(img_path: str, lbl_path: str) -> bool:
    """
    Check if pair of labels and images exist. Filter non-existing pairs.
    """
    name = os.path.basename(img_path)
    name, ext = name.split('.')
    name = name.split('_')[:-1]
    name = '_'.join(name)

    names = []
    for l in ['color', 'instanceids', 'labelids']:
        # Check if label exists
        lbl_name = name + '_' + l + '.' + ext
        if not os.path.exists(os.path.join(lbl_path, lbl_name)):
            return False, None
        names.append(lbl_name)

    return True, names

def __goose_datadict_folder(img_path: str, lbl_path: str):
    """
    Create a data Dictionary with image paths
    """
    subfolders = glob.glob(os.path.join(img_path, '*/'), recursive = False)
    subfolders = [f.split('/')[-2] for f in subfolders]

    valid_imgs = []
    valid_lbls = []
    valid_insta= []
    valid_color= []

    datadict = []

    for s in tqdm(subfolders):
        imgs_p = os.path.join(img_path, s)
        lbls_p = os.path.join(lbl_path, s)
        imgs = glob.glob(os.path.join(imgs_p, '*.png'))
        for i in imgs:
            valid, lbl_names = __check_labels(i, lbls_p)
            if not valid:
                continue

            valid_imgs.append(i)
            valid_color.append(os.path.join(lbls_p, lbl_names[0]))
            valid_insta.append(os.path.join(lbls_p, lbl_names[1]))
            valid_lbls.append(os.path.join(lbls_p,  lbl_names[2]))

    for i,m,p,c in zip(valid_imgs, valid_lbls, valid_insta, valid_color):
        datadict.append({
                'img_path': i,
                'semantic_path': m,
                'instance_path':p,
                'color_path': c,
            })   
    return datadict

def goose_create_dataDict(src_path: str, mapping_csv_name: str = 'goose_label_mapping.csv') -> Dict:
    """
    :param src_path:   path to dataset
    :param mapping_csv_name:  name of the csv file containing the mapping
                                between the original labels and the new labels
    :return:  a dictionary containing the data paths for train, val and test and the mapping
    """
    if mapping_csv_name is not None:
        mapping_path = os.path.join(src_path, mapping_csv_name)
        mapping = []
        with open(mapping_path, newline='') as f:
            reader = csv.DictReader(f)
            for r in reader:
                mapping.append(r)
    else:
        mapping = None

    img_path = os.path.join(src_path, 'images')
    lbl_path = os.path.join(src_path, 'labels')

    datadicts = []
    for c in ['test', 'train', 'val']:
        print("### " + c.capitalize() + " Data ###")
        datadicts.append(
            __goose_datadict_folder(
                os.path.join(img_path, c),
                os.path.join(lbl_path, c)
                )
            )

    test, train, val = datadicts

    return test, train, val, mapping

class GooseExTraversabilityDataset(Dataset):
    """
    Pytorch Dataset for the GooseEx dataset for traversability estimation.
    """

    def __init__(
        self,
        data_dict: List[Dict],
        mapping: List[Dict],
        resize_factor: Optional[float] = None,
        resize_size: Optional[Tuple[int, int]] = None,
        crop: bool = True,
        phase: str = "train",
        dataset_size: Optional[int] = None,
    ) -> None:
        """Initialize a `GooseExTraversabilityDataset`.

        :param data_dict: The data dictionary containing image and label paths.
        :param mapping: The mapping between text labels and their corresponding RGB values and IDs.
        :param resize_factor: The factor to resize the images by. If `None`, no resizing is applied. Defaults to `None`.
        :param dataset_size: The size of the dataset. If `None`, the entire dataset is used. Defaults to `None`.
        """
        self.dataset_dict = []
        self.mapping = mapping
        self.resize_factor = resize_factor
        self.phase = phase  # "train" or "val" or "test"

        self.resize_size = resize_size
        self.crop = crop
        if self.resize_factor is not None and self.resize_size is not None:
            raise ValueError("Only one of resize_factor or resize_size should be set.")
        
        if dataset_size is not None:
            idxs = np.random.choice(len(data_dict), dataset_size, replace=False)
            self.dataset_dict = data_dict[idxs]
        else:
            self.dataset_dict = data_dict

        self.safe_labels = [
            "cobble", "snow", "leaves", "bikeway", "pedestrian_crossing", "road_marking", "sidewalk", "curb",
            "asphalt", "gravel", "soil", "low_grass"
        ]
        self.safe_ids = []
        for m in self.mapping:
            if m['class_name'] in self.safe_labels:
                self.safe_ids.append(int(m['label_key']))
        print(f"Safe labels: {self.safe_labels}")
        print(f"Safe IDs: {self.safe_ids}")

        self.train_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.val_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])

        if self.phase == "train":
            self.transforms = self.train_transforms
        else:
            self.transforms = self.val_transforms

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.dataset_dict)
    
    def get_traversability(self, label_img: np.ndarray) -> np.ndarray:
        """Convert ground truth label to a binary traversability map."""
        safe_mask = np.zeros_like(label_img, dtype=np.uint8)

        for label in self.safe_ids:
            mask = label_img == label
            safe_mask[mask] = 1

        return safe_mask
    
    def preprocess(self, image):
        if image is None:
            return None

        if self.crop:
            # Square-Crop in the center
            s = min([image.width , image.height])
            image = transforms.CenterCrop((s,s)).forward(image)

        if self.resize_size is not None:
            # Resize to given size
            image = image.resize(self.resize_size, resample=Image.NEAREST)

        return image
    
    def __getitem__(self, idx: int) -> Dict:
        """Load an item from the dataset."""
        image = Image.open(self.dataset_dict[idx]['img_path']).convert('RGB')
        label = Image.open(self.dataset_dict[idx]['semantic_path']).convert('L')
        gt_img = Image.open(self.dataset_dict[idx]['color_path']).convert('RGB')

        image = self.preprocess(image)
        label = self.preprocess(label)
        gt_img = self.preprocess(gt_img)

        image = np.array(image)
        label = np.array(label)
        gt_img = np.array(gt_img)

        if self.resize_factor is not None:
            image = cv2.resize(image, (0,0), fx=self.resize_factor, fy=self.resize_factor)
            label = cv2.resize(label, (0,0), fx=self.resize_factor, fy=self.resize_factor)
            gt_img = cv2.resize(gt_img, (0,0), fx=self.resize_factor, fy=self.resize_factor)

        assert image.shape[0:2] == label.shape[0:2], "Image and label shape mismatch"
        assert image.shape[0:2] == gt_img.shape[0:2], "Image and gt_img shape mismatch"

        gt_traversability = self.get_traversability(np.array(label))

        # Convert to tensor
        raw_img = self.transforms(image)
        gt_traversability = torch.tensor(gt_traversability)
        gt_traversability = gt_traversability.unsqueeze(0)  # Add channel dimension
        gt_img = torch.tensor(gt_img)

        return {
            "raw_img": raw_img,
            "gt_traversability": gt_traversability,
            "gt_img": gt_img,
            "img_path": self.dataset_dict[idx]['img_path'],
        }


class GooseExTraversabilityDataModule(LightningDataModule):
    """`LightningDataModule` for the GooseEx dataset for traversability estimation.
    """

    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 64,
        resize_factor: Optional[float] = None,
        resize_size: Optional[Tuple[int, int]] = None,
        crop: bool = True,
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
        self.data_test: Optional[Dataset] = None

        self.data_test, self.train_dict, self.val_dict, self.mapping = goose_create_dataDict(data_dir)
        print(f"Number of training samples: {len(self.train_dict)}")
        print(f"Number of validation samples: {len(self.val_dict)}")
        print(f"Number of test samples: {len(self.data_test)}")

        self.batch_size_per_device = batch_size

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
            self.data_train = GooseExTraversabilityDataset(
                data_dict=self.train_dict,
                mapping=self.mapping,
                resize_factor=self.hparams.resize_factor,
                resize_size=self.hparams.resize_size,
                crop=self.hparams.crop,
                phase="train",
                dataset_size=self.hparams.train_size,
            )
            self.data_val = GooseExTraversabilityDataset(
                data_dict=self.val_dict,
                mapping=self.mapping,
                resize_factor=self.hparams.resize_factor,
                resize_size=self.hparams.resize_size,
                crop=self.hparams.crop,
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
    module = GooseExTraversabilityDataModule(
        data_dir="/home/$USER/data/goose-ex",
        resize_factor=None,
        resize_size=(768, 768),
        crop=True,
        batch_size=4,
        num_workers=1,
        pin_memory=False,
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

            plt.tight_layout(rect=[0, 0.1, 1, 1])  # Leave space for the legend
            plt.show()