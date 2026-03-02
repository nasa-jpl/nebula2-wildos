from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from src.models.components.radio_utils import RADIO_MODEL_VERSIONS, RADIO_ADAPTOR_VERSIONS
from nvidia_radio.radio.pamr import PAMR
from nvidia_radio.hubconf import radio_model

class RADIO_CNN(nn.Module):
    """A CNN decoder head on top of a RADIO model."""

    def __init__(
        self,
        model_version: str = "c-radio_v3-b",
        adaptor_version: Optional[str] = None,
        use_naclip: bool = False,
        use_summary_for_spatial: bool = False,
        sigmoid_out: bool = True,
    ) -> None:
        """Initialize a `RADIO_CNN` module.

        :param model_version: The version of the RADIO model to use.
        :param adaptor_version: The version of the adaptor to use.
        :param use_naclip: Whether to use the NA-CLIP changes.
        :param use_summary_for_spatial: Whether to use the summary adaptor for spatial features.
        """
        super().__init__()

        if model_version not in RADIO_MODEL_VERSIONS:
            raise ValueError(f"Invalid model version: {model_version}. Available versions: {RADIO_MODEL_VERSIONS}")
        self.model_version = model_version
        if adaptor_version is not None and adaptor_version not in RADIO_ADAPTOR_VERSIONS:
            raise ValueError(f"Invalid adaptor version: {adaptor_version}. Available versions: {RADIO_ADAPTOR_VERSIONS.keys()}")
        self.adaptor_version = adaptor_version
        
        if self.adaptor_version is None:
            self.dim = RADIO_ADAPTOR_VERSIONS["none"]
        else:
            self.dim = RADIO_ADAPTOR_VERSIONS[self.adaptor_version]
        self.use_naclip = use_naclip
        self.use_summary_for_spatial = use_summary_for_spatial        

        self.radio_model, chk = radio_model(
            version=self.model_version,
            progress=True,
            skip_validation=True,
            adaptor_names=self.adaptor_version,
            return_checkpoint=True, 
            use_naclip=self.use_naclip,
            naclip_strategy="kkonly", #"kkonly",
            naclip_gaussian_std=5.0,
            fixed_patch_dim=(40,40), #(45,80),
            gaussian_device='cuda',
            use_summary_for_spatial=self.use_summary_for_spatial,
        )
        self.radio_model.eval()
        self.radio_model.requires_grad_(False)  # Disable gradients
        
        print(f"Loaded model: {self.model_version} with adaptor: {self.adaptor_version}")

        self.head = nn.Sequential(
            nn.ConvTranspose2d(self.dim, self.dim//2, 2, stride=2),
            nn.Conv2d(self.dim//2, self.dim//2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.dim//2, self.dim//4, 2, stride=2),
            nn.Conv2d(self.dim//4, self.dim//4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.dim//4, self.dim//8, 2, stride=2),
            nn.Conv2d(self.dim//8, self.dim//8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.dim//8, 1, 2, stride=2),
        )
        self.sigmoid_out = sigmoid_out
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        nearest_res = self.radio_model.get_nearest_supported_resolution(*x.shape[-2:])
        x_resized = F.interpolate(x, nearest_res, mode='bilinear', align_corners=False)

        # forward pass
        if self.adaptor_version is not None:
            summary, spatial_features = self.radio_model(x_resized, feature_fmt='NCHW')[self.adaptor_version]
        else:
            summary, spatial_features = self.radio_model(x_resized, feature_fmt='NCHW')
        out = self.head(spatial_features)

        out = F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=False)

        if self.sigmoid_out:
            out = F.sigmoid(out)
        
        return out


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RADIO_CNN(
        model_version="c-radio_v3-b",
    ).to(device)

    from PIL import Image
    from torchvision.transforms.functional import pil_to_tensor
    import matplotlib.pyplot as plt
    import numpy as np

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
    
    plt.show()