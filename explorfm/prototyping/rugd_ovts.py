"""
Use:  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
when running on large images
"""

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

from PIL import Image
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import pil_to_tensor


from nvidia_radio.radio.pamr import PAMR
from nvidia_radio.hubconf import radio_model

class RADIO_Segmentation:
    """
    Class for performing segmentation using RADIO models.
    """
    def __init__(self):
        # model_version="radio_v2.5-g" # for RADIOv2.5-g model (ViT-H/14)
        # model_version="radio_v2.5-h" # for RADIOv2.5-H model (ViT-H/16)
        # model_version="radio_v2.5-l" # for RADIOv2.5-L model (ViT-L/16)
        # model_version="radio_v2.5-b" # for RADIOv2.5-B model (ViT-B/16)
        self.model_version="c-radio_v3-b" # for C_RADIOv3-B model (ViT-B/16)
        # model_version="e-radio_v2" # for E-RADIO
        self.adaptor_version="siglip2" # ["clip", siglip", "siglip2"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model, chk = radio_model(
            version=self.model_version,
            progress=True,
            skip_validation=True,
            adaptor_names=self.adaptor_version,
            return_checkpoint=True, 
            use_naclip=True, 
            naclip_strategy="kkonly", #"kkonly",
            naclip_gaussian_std=5.0,
            fixed_patch_dim=(40,40), #(45,80),
            gaussian_device='cuda',
            use_summary_for_spatial=True
        )
        self.model.to(self.device).eval()
        self.model.requires_grad_(False)  # Disable gradients for inference
        # self.model.make_preprocessor_external()
        print(f"Loaded model: {self.model_version} with adaptor: {self.adaptor_version}")

        size_model = 0
        for param in self.model.parameters():
            if param.data.is_floating_point():
                size_model += param.numel() * torch.finfo(param.data.dtype).bits
            else:
                size_model += param.numel() * torch.iinfo(param.data.dtype).bits
        print(f"model size: {size_model} / bit | {size_model / 8e6:.2f} / MB")

        # data
        self.rugd_data_path = "/home/$USER/data/RUGD_sample-data/"
        self.rugd_loader = RUGD_Loader(self.rugd_data_path)
        self.seg_colormap = self.rugd_loader.load_annotations()

        self.safe_text_queries = [
            "dirt", "sand", "grass", "asphalt", "gravel", "mulch", "concrete"
        ]
        self.text_feats = self.get_text_embeddings(self.safe_text_queries)

        # visualization parameters
        self.cmap = 'inferno'  # ['viridis', 'plasma', 'inferno', 'magma', 'seismic']
        self.viz_threshold = 0.09

        self.pixel_level_seg = False
        # if GPU size allows, use: 
            # num_iter = 50
            # dilations = [1, 2, 4, 8, 12, 24]
        self.pamr = PAMR(
            num_iter=50,
            dilations=[1, 2, 4, 8, 12, 24],
        ).to(self.device)
        self.i = 0


    def get_text_embeddings(self, text_queries):
        """ Get text embeddings for the given text queries.
        """
        adaptor = self.model.adaptors[self.adaptor_version]
        tokens = adaptor.tokenizer(text_queries).to(self.device)
        text_feats = adaptor.encode_text(tokens, normalize=True)
        print(f"Computed text features shape: {text_feats.shape}")

        return text_feats

    def get_nearest_supported_resolution(self, h, w):
        """ Get the nearest supported resolution for the model.
        """
        nearest_res = self.model.get_nearest_supported_resolution(
            h, w
        )
        print(f"Nearest supported resolution for input ({h}, {w}): {nearest_res}")
        return nearest_res
    
    def localize_query(self, spatial_features, img_rgb):
        """
        Localizes the text query in the spatial features.
        
        Args:
            spatial_features: Tensor (1, C, H, W), spatial features from the model
        """
        num_queries = len(self.text_feats)

        # Normalize features
        spatial_feats = spatial_features[0]  # (C, H, W)
        spatial_feats = spatial_feats / spatial_feats.norm(dim=0, keepdim=True)
        c, h, w = spatial_feats.shape
        spatial_feats = spatial_feats.view(c, h * w)

        # Compute similarity maps
        text_sim_spatial = self.text_feats @ spatial_feats  # (num_texts, H*W)
        text_sim_spatial = text_sim_spatial.view(num_queries, h, w)

        # Resize similarity maps to match the original image size
        if self.pixel_level_seg:
            text_sim_spatial = F.interpolate(
                text_sim_spatial.unsqueeze(0), size=(img_rgb.shape[0], img_rgb.shape[1]), mode='bilinear', align_corners=False
            )  # Shape: (1, num_queries, H, W)
            # Apply PAMR to the text similarity map (mask refinement: Patch to Pixels)
            x = pil_to_tensor(Image.fromarray(img_rgb)).to(dtype=torch.float32, device=self.device).unsqueeze(0)  # Convert to tensor and add batch dimension
            text_sim_spatial = self.pamr(x, text_sim_spatial).cpu().numpy()  # (num_texts, H, W)
            text_sim_spatial = text_sim_spatial.squeeze(0).transpose(1, 2, 0)  # Shape: (H, W, num_queries)

        else:
            text_sim_spatial = text_sim_spatial.cpu().numpy().transpose(1, 2, 0)  # Shape: (H, W, num_queries)
            text_sim_spatial = cv2.resize(
                text_sim_spatial, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_NEAREST
            )

        # Binary mask for visualization
        binary_mask = (text_sim_spatial > self.viz_threshold).astype(np.uint8)

        return text_sim_spatial, binary_mask
    
    def get_gt_safe_mask(self, gt_img):
        """ Get a binary mask for the ground truth image based on safe text queries.
        """
        gt_img = np.array(gt_img)
        safe_mask = np.zeros(gt_img.shape[:2], dtype=np.uint8)

        for label in self.safe_text_queries:
            if label not in self.seg_colormap:
                raise ValueError(f"Label '{label}' not found in segmentation colormap.")
            color = self.seg_colormap[label]
            if label in self.safe_text_queries:
                mask = np.all(gt_img == np.array(color), axis=-1)
                safe_mask[mask] = 1

        return safe_mask
    
    def get_ovts(self, img_path, gt_path):
        """ Get the OVTS (Open Vocabulary Traversability Segmentation) for the given image.
        """
        img = Image.open(img_path).convert('RGB')
        img_rgb = np.array(img)
        x = pil_to_tensor(img.copy()).to(dtype=torch.float32, device=self.device)
        x.div_(255.0)  # RADIO expects the input values to be between 0 and 1
        x = x.unsqueeze(0) # Add a batch dimension

        nearest_res = self.get_nearest_supported_resolution(*x.shape[-2:])
        x = F.interpolate(x, nearest_res, mode='bilinear', align_corners=False)

        # forward pass
        summary, spatial_features = self.model(x, feature_fmt='NCHW')[self.adaptor_version]
        text_sim_spatial, binary_mask = self.localize_query(spatial_features, img_rgb)

        gt_img = np.array(Image.open(gt_path).convert('RGB'))
        viz_img = self.visualize_predictions(
            img_rgb, gt_img, text_sim_spatial, binary_mask
        )

    def visualize_predictions(self, img_rgb, gt_img, text_sim_spatial, binary_mask):
        """
        Visualize the predictions. 
        Displays the original image, ground truth, the heatmap of text similarity, and the binary mask.
        Also display the legend for the segmentation categories and the heatmap.
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes[0, 2].axis('off')
        
        # 1. Original RGB Image
        axes[0, 0].imshow(img_rgb)
        axes[0, 0].set_title('Original RGB')
        axes[0, 0].axis('off')
        
        # 2. Ground Truth Segmentation with Color Mapping
        axes[0, 1].imshow(gt_img)
        axes[0, 1].set_title('Ground Truth Segmentation')
        axes[0, 1].axis('off')

        # 3. Heatmap (text similarity)
        text_sim_spatial = text_sim_spatial.max(axis=-1)
        heatmap = axes[1, 0].imshow(text_sim_spatial, cmap=self.cmap, vmin=0, vmax=0.2)
        axes[1, 0].set_title('Text Similarity Heatmap')
        axes[1, 0].axis('off')
        plt.colorbar(heatmap, ax=axes[1,0], fraction=0.046, pad=0.04)

        # 4. Binary mask (thresholded)
        binary_mask = (text_sim_spatial > self.viz_threshold).astype(np.uint8)
        axes[1, 1].imshow(binary_mask, cmap='gray')
        axes[1, 1].set_title(f'Binary Mask (>{self.viz_threshold})')
        axes[1, 1].axis('off')

        # 5. Ground Truth Safe Mask
        safe_mask = self.get_gt_safe_mask(gt_img)
        axes[1, 2].imshow(safe_mask, cmap='gray')
        axes[1, 2].set_title('Ground Truth Safe Mask')
        axes[1, 2].axis('off')

        # Add segmentation legend below all subplots
        handles = [
            mpatches.Patch(color=np.array(color)/255.0, label=label)
            for label, color in self.seg_colormap.items()
        ]
        fig.legend(handles=handles, loc='upper right', ncol=4, fontsize='small', frameon=False)

        plt.tight_layout(rect=[0, 0.1, 1, 1])  # Leave space for the legend
        # plt.show()
        plt.savefig(f"tmp/seg_{self.i:04d}.png", dpi=300, bbox_inches='tight')
        self.i+=1
        plt.close()

    def run(self):
        """ Run the segmentation on a sample image from RUGD dataset.
        """
        for idx in range(len(self.rugd_loader)):
            img_path, gt_path = self.rugd_loader.load_index(idx)
            print(f"Processing image: {img_path}")
            self.get_ovts(img_path, gt_path)


class RUGD_Loader:
    """ Loader for RUGD dataset images.
    """
    def __init__(self, data_path):
        self.data_path = data_path
        self.colormap_path = os.path.join(data_path, "RUGD_annotation-colormap.txt")
        self.images_path = os.path.join(data_path, "images")
        self.annotations_path = os.path.join(data_path, "annotations")

        self.img_list = os.listdir(self.images_path)

    def __len__(self):
        return len(self.img_list)
        
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
    
    def load_index(self, index):
        """ Load an image by index.
        """
        if index < 0 or index >= len(self.img_list):
            raise IndexError("Index out of range")
        
        img_name = self.img_list[index]
        img_path = os.path.join(self.images_path, img_name)
        gt_path = os.path.join(self.annotations_path, img_name)
        return img_path, gt_path
    

if __name__ == "__main__":
    rugd_segmentation = RADIO_Segmentation()
    rugd_segmentation.run()