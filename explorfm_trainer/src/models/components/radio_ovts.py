import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms.functional import pil_to_tensor

from nvidia_radio.radio.pamr import PAMR
from nvidia_radio.hubconf import radio_model
from src.models.components.radio_utils import RADIO_MODEL_VERSIONS, RADIO_ADAPTOR_VERSIONS

class RADIO_Segmentation(nn.Module):
    """
    Class for performing segmentation using RADIO models.
    """
    def __init__(
        self,
        model_version: str = "c-radio_v3-b",
        adaptor_version: str = "siglip2",
        pixel_level_seg: bool = False,
    ) -> None:
        super().__init__()
        
        self.model_version = model_version
        if model_version not in RADIO_MODEL_VERSIONS:
            raise ValueError(f"Invalid model version: {model_version}. Available versions: {RADIO_MODEL_VERSIONS}")
        self.adaptor_version = adaptor_version
        if adaptor_version not in RADIO_ADAPTOR_VERSIONS:
            raise ValueError(f"Invalid adaptor version: {adaptor_version}. Available versions: {RADIO_ADAPTOR_VERSIONS.keys()}")
        self.pixel_level_seg = pixel_level_seg

        self.radio_model, chk = radio_model(
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
        self.radio_model.eval()
        self.radio_model.requires_grad_(False)  # Disable gradients for inference
        print(f"Loaded model: {self.model_version} with adaptor: {self.adaptor_version}")

        size_model = 0
        for param in self.radio_model.parameters():
            if param.data.is_floating_point():
                size_model += param.numel() * torch.finfo(param.data.dtype).bits
            else:
                size_model += param.numel() * torch.iinfo(param.data.dtype).bits
        print(f"model size: {size_model} / bit | {size_model / 8e6:.2f} / MB")

        self.safe_text_queries = [
            "dirt", "sand", "grass", "asphalt", "gravel", "mulch", "rock-bed", "concrete"
        ]
        self.text_feats = None

        self.pamr = PAMR(
            num_iter=50,
            dilations=[1, 2, 4, 8, 12, 24],
        )

    def get_text_embeddings(self, text_queries, device):
        """ Get text embeddings for the given text queries.
        """
        adaptor = self.radio_model.adaptors[self.adaptor_version]
        tokens = adaptor.tokenizer(text_queries).to(device)
        text_feats = adaptor.encode_text(tokens, normalize=True)
        print(f"Computed text features shape: {text_feats.shape}")

        return text_feats

    def get_nearest_supported_resolution(self, h, w):
        """ Get the nearest supported resolution for the model.
        """
        nearest_res = self.radio_model.get_nearest_supported_resolution(
            h, w
        )
        return nearest_res
    
    def localize_query(self, spatial_features, img_rgb):
        """
        Localizes the text query in the spatial features.
        
        Args:
            spatial_features: Tensor (1, C, H, W), spatial features from the model
        """
        num_queries = len(self.text_feats)

        # Normalize features
        spatial_feats = spatial_features  # (B, C, H, W)
        spatial_feats = spatial_feats / spatial_feats.norm(dim=1, keepdim=True)
        b, c, h, w = spatial_feats.shape
        spatial_feats = spatial_feats.view(b, c, h * w)

        # Compute similarity maps
        text_sim_spatial = torch.einsum(
            "bcd,qc->bqd", spatial_feats, self.text_feats
        )  # (B, num_queries, H, W)
        text_sim_spatial = text_sim_spatial.view(b, num_queries, h, w)

        # Resize similarity maps to match the original image size
        if self.pixel_level_seg:
            text_sim_spatial = F.interpolate(
                text_sim_spatial, size=(img_rgb.shape[-2], img_rgb.shape[-1]), mode='bilinear', align_corners=False
            )  # Shape: (B, num_queries, H, W)

            # Apply PAMR to the text similarity map (mask refinement: Patch to Pixels)
            text_sim_spatial = self.pamr(img_rgb*255, text_sim_spatial)  # (B, num_texts, H, W)

        else:
            text_sim_spatial = F.interpolate(
                text_sim_spatial, size=(img_rgb.shape[-2], img_rgb.shape[-1]), mode='nearest'
            )

        text_sim_spatial, _ = text_sim_spatial.max(axis=1, keepdim=True)

        return text_sim_spatial
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Get the OVTS (Open Vocabulary Traversability Segmentation) for the given image.
        """
        if self.text_feats is None:
            self.text_feats = self.get_text_embeddings(self.safe_text_queries, x.device)

        nearest_res = self.get_nearest_supported_resolution(*x.shape[-2:])
        x_resized = F.interpolate(x, nearest_res, mode='bilinear', align_corners=False)

        # forward pass
        summary, spatial_features = self.radio_model(x_resized, feature_fmt='NCHW')[self.adaptor_version]
        text_sim_spatial = self.localize_query(spatial_features, x)

        return text_sim_spatial


if __name__ == "__main__":
    model = RADIO_Segmentation(
        model_version="c-radio_v3-b",
        adaptor_version="siglip2",
        pixel_level_seg=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    img_path = "assets/demo.png"
    img = Image.open(img_path).convert("RGB")
    img_tensor = pil_to_tensor(img).unsqueeze(0).float() / 255
    img_tensor = img_tensor.to(device)
    print(f"Input image shape: {img_tensor.shape}")

    with torch.no_grad():
        output = model(img_tensor)
        print(f"Output shape: {output.shape}")

    # Visualize the output
    cmap = 'inferno'  # ['viridis', 'plasma', 'inferno', 'magma', 'seismic']
    output_np = output.squeeze().cpu().numpy()

    assert output_np.shape == np.array(img).shape[:2], \
        f"Output shape {output_np.shape} does not match input image shape {np.array(img).shape[:2]}"

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(np.array(img))
    ax[0].axis('off')
    ax[0].set_title("Input Image")

    heatmap = ax[1].imshow(output_np, cmap=cmap, vmin=0, vmax=0.2)
    ax[1].axis('off')
    ax[1].set_title("OVTS Output")
    plt.colorbar(heatmap, ax=ax[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    
    # plt.show()
    plt.savefig("assets/ovts_output0.png", bbox_inches='tight', dpi=300)