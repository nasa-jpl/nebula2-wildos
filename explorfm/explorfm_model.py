from typing import Optional, Tuple, List
import os
from enum import Enum, auto

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms

from nvidia_radio.hubconf import radio_model

class ModelPrecision(Enum):
    FP32 = auto()
    FP16 = auto()

    def __str__(self) -> str:
        return self.name

    def is_fp16(self) -> bool:
        return self == ModelPrecision.FP16
    
    def is_fp32(self) -> bool:
        return self == ModelPrecision.FP32

class ExploRFM(nn.Module):
    def __init__(
        self,
        frontier_ckpt: Optional[str]=None,
        traversability_ckpt: Optional[str]=None,
        model_version: str = "c-radio_v3-b",
        adaptor_version: Optional[str] = "siglip2",
        adaptor_ckpt_path: Optional[str] = None,
        use_naclip: bool = True,
        use_summary_for_spatial: bool = True,
        radio_dim: int = 768,
        static_scale_factor: float = 1.0,
    ):
        super().__init__()

        self.model_version = model_version
        self.adaptor_version = adaptor_version
        self.adaptor_ckpt_path = adaptor_ckpt_path
        self.use_naclip = use_naclip
        self.use_summary_for_spatial = use_summary_for_spatial
        self.dim = radio_dim
        self.static_scale_factor = static_scale_factor

        self.init_radio_backbone()
        # self.radio_model.make_preprocessor_external() # Disabled since it was disabled during training
        
        self.traversability_head = None
        if traversability_ckpt:
            self.init_traversability_head(traversability_ckpt)

        self.frontier_head = None
        if frontier_ckpt:
            self.init_frontier_head(frontier_ckpt)

    def init_radio_backbone(self) -> None:
        """Initialize the RADIO backbone model."""
        self.radio_model, chk = radio_model(
            version=self.model_version,
            progress=True,
            skip_validation=True,
            adaptor_names=self.adaptor_version,
            adaptor_ckpt_path=self.adaptor_ckpt_path,
            return_checkpoint=True, 
            use_naclip=self.use_naclip,
            naclip_strategy="kkonly", #"kkonly",
            naclip_gaussian_std=5.0,
            fixed_patch_dim=(40,40), #(45,80),
            gaussian_device='cuda',
            use_summary_for_spatial=self.use_summary_for_spatial,
        )
    
    def init_traversability_head(self, traversability_ckpt: str) -> None:
        """Initialize the traversability detection head."""
        self.traversability_head = nn.Sequential(
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
        # load traversability checkpoint
        if os.path.exists(traversability_ckpt):
            orig_state_dict = torch.load(traversability_ckpt, map_location='cpu', weights_only=False)['state_dict']
            state_dict = {}
            for k, v in orig_state_dict.items():
                state_dict[k.replace('net.head.', '')] = v
            self.traversability_head.load_state_dict(state_dict)
            print(f"Loaded traversability head from {traversability_ckpt}")
        else:
            raise FileNotFoundError(f"Traversability checkpoint not found: {traversability_ckpt}")

    def init_frontier_head(self, frontier_ckpt: str) -> None:
        """Initialize the frontier detection head."""
        self.frontier_head = nn.Sequential(
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
        # load frontier checkpoint
        if os.path.exists(frontier_ckpt):
            orig_state_dict = torch.load(frontier_ckpt, map_location='cpu', weights_only=False)['state_dict']
            state_dict = {}
            for k, v in orig_state_dict.items():
                if 'criterion' in k:
                    continue
                state_dict[k.replace('net.head.', '')] = v
            self.frontier_head.load_state_dict(state_dict)
            print(f"Loaded frontier head from {frontier_ckpt}")
        else:
            raise FileNotFoundError(f"Frontier checkpoint not found: {frontier_ckpt}")
        
    def get_input_shape(self, input_tensor: torch.Tensor) -> Tuple[int, int]:
        """Calculate the input shape for resizing based on the input tensor."""
        height, width = torch.tensor(input_tensor.shape[-2:])
        height, width = (height * self.static_scale_factor).int(), (width * self.static_scale_factor).int()

        min_resolution_step = self.radio_model.min_resolution_step
        height = int(torch.round(height / min_resolution_step) * min_resolution_step)
        width = int(torch.round(width / min_resolution_step) * min_resolution_step)

        height = max(height, min_resolution_step)
        width = max(width, min_resolution_step)

        return height, width

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        model_input_res = self.get_input_shape(x)
        x_resized = F.interpolate(x, model_input_res, mode='bilinear', align_corners=False)
        
        # forward pass
        if self.adaptor_version is not None:
            model_output = self.radio_model(x_resized, feature_fmt='NCHW')
            _, ad_spatial_features = model_output[self.adaptor_version]
            _, spatial_features = model_output["backbone"]
        else:
            _, spatial_features = self.radio_model(x_resized, feature_fmt='NCHW')
            ad_spatial_features = None

        traversability = None
        if self.traversability_head is not None:
            traversability = self.traversability_head(spatial_features)
            traversability = F.interpolate(traversability, size=x.shape[-2:], mode='bilinear', align_corners=False)
            traversability = F.sigmoid(traversability)

        frontiers = None
        if self.frontier_head is not None:
            frontiers = self.frontier_head(spatial_features)
            frontiers = F.interpolate(frontiers, size=x.shape[-2:], mode='bilinear', align_corners=False)
            frontiers = F.sigmoid(frontiers)

        return traversability, frontiers, ad_spatial_features
    

class ExploRFMInference:
    """
    Perform inference using the ExploRFM model with different precision.
    Also supports exporting the model to ONNX format.
    """
    def __init__(
        self,
        frontier_ckpt: str,
        traversability_ckpt: str,
        model_version: str = "c-radio_v3-b",
        adaptor_version: Optional[str] = "siglip2",
        adaptor_ckpt_path: Optional[str] = None,
        use_naclip: bool = True,
        use_summary_for_spatial: bool = True,
        radio_dim: int = 768,
        static_scale_factor: float = 1.0,
        model_precision: str = "FP32",
        device: Optional[str] = None,
    ):
        self.model = ExploRFM(
            frontier_ckpt=frontier_ckpt,
            traversability_ckpt=traversability_ckpt,
            model_version=model_version,
            adaptor_version=adaptor_version,
            adaptor_ckpt_path=adaptor_ckpt_path,
            use_naclip=use_naclip,
            use_summary_for_spatial=use_summary_for_spatial,
            radio_dim=radio_dim,
            static_scale_factor=static_scale_factor,
        )
        self.model_precision = ModelPrecision[model_precision.upper()]
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.eval().to(self.device)
        self.model.requires_grad_(False)

        if self.model_precision.is_fp16():
            self.model.half()
            self.model.radio_model.input_conditioner.dtype = torch.float16
        elif self.model_precision.is_fp32():
            self.model.float()

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
        ])

        # Initialize Text Model
        if adaptor_version is not None:
            self.text_model = self.model.radio_model.adaptors[adaptor_version]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Perform a forward pass through the model.

        :param x: The input tensor.
        :return: A tuple containing the traversability, frontiers, and optional adaptor features.
        """
        x = x.to(self.device)
        if self.model_precision.is_fp16():
            x = x.half()
            """
            with torch.autocast("cuda", dtype=torch.float16):
                return self.model(x)
            """
        return self.model(x)
    
    def forward_on_numpy(
        self, input_img: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Perform a forward pass on a numpy array input.

        :param input_img: The input image as a numpy array (HWC).
        :return: A tuple containing the traversability, frontiers, and optional adaptor features.
        """
        input_tensor = self.transforms(input_img).unsqueeze(0)        
        return self.forward(input_tensor)
    
    def forward_on_text(
        self, text_queries: List[str]
    ) -> torch.Tensor:
        """Perform a forward pass on a list of text queries.

        :param text_queries: The input text queries.
        :return: The output text embeddings. (ND)
        """

        tokens = self.text_model.tokenizer(text_queries).to(self.device)
        return self.text_model.encode_text(tokens, normalize=True)

    def export_to_onnx(
        self,
        output_path_prefix: str = "ckpts/radio_downstream",
        input_shape: Tuple[int, int, int] = (3, 720, 1280),
        opset_version: int = 17,
    ) -> None:
        """Export the model to ONNX format.

        :param output_path_prefix: The prefix for the output ONNX file.
        :param input_shape: The shape of the dummy input tensor.
        :param opset_version: The ONNX opset version to use.
        """
        dummy_input = torch.randn(input_shape).to(self.device)

        true_input_shape = (input_shape[0],) + self.model.get_input_shape(dummy_input)
        inp_shape_str = "_".join(map(str, input_shape))
        tr_inp_shape_str = "_".join(map(str, true_input_shape))

        output_path = f"{output_path_prefix}_B_{inp_shape_str}_resize_{tr_inp_shape_str}_{self.model_precision}.onnx"

        dummy_input = dummy_input.unsqueeze(0)  # Add batch dimension
        if self.model_precision.is_fp16():
            dummy_input = dummy_input.half()
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["traversability", "frontiers", "text_feats"],
            dynamic_axes={
                "input": {0: "batch"},
                "traversability": {0: "batch"},
                "frontiers": {0: "batch"},
                "text_feats": {0: "batch"}
            }
        )
        print(f"ONNX model exported successfully to {output_path}.")

if __name__ == "__main__":
    # Example usage
    frontier_ckpt = "ckpts/frontier_head.ckpt"
    traversability_ckpt = "ckpts/trav_head.ckpt"

    model = ExploRFMInference(
        frontier_ckpt=frontier_ckpt,
        traversability_ckpt=traversability_ckpt,
        model_version="ckpts/c-radio_v3-b_half.pth.tar",
        adaptor_version="siglip2",
        adaptor_ckpt_path="ckpts/siglip2",
        use_naclip=True,
        use_summary_for_spatial=True,
        radio_dim=768,
        static_scale_factor=0.5,
        model_precision="FP32",
    )

    dummy_input = torch.randn(1, 3, 720, 1280)  # Example input tensor
    traversability, frontiers, text_feats = model.forward(dummy_input)
    print("Traversability shape:", traversability.shape)
    print("Frontiers shape:", frontiers.shape)
    if text_feats is not None:
        print("Adaptor features shape:", text_feats.shape)
    
    model.export_to_onnx(output_path_prefix="ckpts/explorfm_onnx_test", input_shape=(3, 720, 1280), opset_version=17)
    
    print("Inference completed successfully.")