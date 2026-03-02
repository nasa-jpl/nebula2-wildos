from typing import Optional
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from torchvision.transforms import transforms

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import onnxruntime as ort

from protoyping.datasets.rugd import RUGDTraversabilityDataset
from protoyping.datasets.nebula import NebulaDataset
from explorfm_model import ModelPrecision


class ExploRFMONNXInference:
    def __init__(self,
        onnx_ckpt: str,
        model_precision: str = "FP32",
        device: Optional[str] = None,
        use_cpu_fallback: bool = False,
        enable_profiling: bool = False
    ):
        self.model_precision = ModelPrecision[model_precision.upper()]
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Running on: {self.device}")
        self.use_cpu_fallback = use_cpu_fallback

        opts = ort.SessionOptions()
        opts.enable_profiling = enable_profiling

        self.session = ort.InferenceSession(
            onnx_ckpt,
            sess_options=opts,
            # providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )

        self.binding = self.session.io_binding()

        self.input_names = [i.name for i in self.session.get_inputs()]
        assert len(self.input_names) == 1, f"Expected 1 input, got {self.input_names}"
        self.input_name = self.input_names[0]

        self.output_names = [o.name for o in self.session.get_outputs()]
        expected = {'traversability', 'frontiers', 'text_feats'}
        missing = expected - set(self.output_names)
        if missing:
            raise RuntimeError(f"Model missing expected outputs: {missing}. Found: {self.output_names}")

        self.transforms = transforms.Compose([transforms.ToTensor()])

    def _ensure_tensor(self, x: torch.Tensor) -> torch.Tensor:
        # Expect NCHW float32, batch size 1, on self.device
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if self.device == "cuda":
            x = x.to(device="cuda", non_blocking=True)
        else:
            x = x.to(device="cpu")
        if self.model_precision.is_fp16():
            x = x.half()
        else:
            x = x.float()
        return x.contiguous()

    def _run_with_iobinding_cuda(self, x: torch.Tensor):
        # Clear any previous bindings
        self.binding.clear_binding_inputs()
        self.binding.clear_binding_outputs()

        self.binding.bind_input(
            name=self.input_name,
            device_type='cuda',
            device_id=0,
            element_type=np.float32 if self.model_precision.is_fp32() else np.float16,
            shape=tuple(x.shape),
            buffer_ptr=x.data_ptr()
        )

        for name in self.output_names:
            self.binding.bind_output(name=name, device_type='cuda', device_id=0)

        self.session.run_with_iobinding(self.binding)

        # Copy results to CPU
        outs = self.binding.copy_outputs_to_cpu()
        return {name: outs[i] for i, name in enumerate(self.output_names)}

    def model_forward(self, x: torch.Tensor):
        """
        Returns:
            traversability: torch.Tensor [1, 1, H?, W?]
            frontiers:      torch.Tensor [1, 1, H?, W?]
            text_feats:     torch.Tensor [1, 768, H?/16, W?/16] (shape depends on model)
        """
        x = self._ensure_tensor(x)
        if self.device == "cuda":
            try:
                name_to_np = self._run_with_iobinding_cuda(x)
            except RuntimeError as e:
                if not self.use_cpu_fallback:
                    raise
                print(f"[WARN] CUDA path failed, falling back to CPU run(): {e}")
                name_to_np = self._cpu_run(x)
        else:
            name_to_np = self._cpu_run(x)

        trav_np = name_to_np['traversability'].squeeze()
        front_np = name_to_np['frontiers'].squeeze()
        text_np = name_to_np['text_feats'].squeeze()

        return trav_np, front_np, text_np

    def _cpu_run(self, x: torch.Tensor):
        x_cpu = x.detach().to('cpu').numpy()
        outputs = self.session.run(None, {self.input_name: x_cpu})
        return {name: outputs[i] for i, name in enumerate(self.output_names)}
    
    def model_forward_on_numpy(self, image: np.ndarray):
        """Run the model on a numpy image array.
        """
        x = self.transforms(image.copy())
        return self.model_forward(x)


def process_index(onnx_model: ExploRFMONNXInference, index: int, dataset):
    image, annotation = dataset.get_image_and_annotation(index)
    _ = dataset.get_traversability(annotation)  # if you still need it downstream
    traversability, frontiers, _ = onnx_model.model_forward_on_numpy(image)

    img_name = dataset.index_to_path[index]
    return visualize_results(img_name, index, image, traversability, frontiers)


def visualize_results(img_name: str, index: int, image: np.ndarray, traversability: torch.Tensor, frontiers: torch.Tensor):
    """Visualize the results of the model.

    :param index: The index of the image in the dataset.
    :param image: The original image.
    :param frontiers: The predicted frontier map.
    :param traversability: The predicted traversability map.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 8))
    axes = axes.flatten()

    axes[0].imshow(image)
    axes[0].set_title(f"Image: {img_name}")
    axes[0].axis('off')

    # overlay frontiers on the image
    axes[1].imshow(image)
    hm = axes[1].imshow(frontiers, alpha=0.5, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title("Frontiers Overlay")
    axes[1].axis('off')
    plt.colorbar(hm, ax=axes[1], fraction=0.046, pad=0.04)

    # overlay traversability on the image
    axes[2].imshow(image)
    hm = axes[2].imshow(traversability, alpha=0.5, cmap='jet', vmin=0, vmax=1)
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

    cv2.imwrite(f"tmp/{index:04d}_results.png", data)
    return True


def run(onnx_model: ExploRFMONNXInference, dataset):
    """Run the model on the dataset and visualize results."""
    for index in tqdm(range(len(dataset))):
        if not process_index(onnx_model, index, dataset):
            print(f"Exiting at index {index} due to user input.")
            break

if __name__ == "__main__":
    onnx_ckpt = "ckpts/explorfm_onnx_test_B_3_720_1280_FP16.onnx"

    # rugd_dataset = RUGDTraversabilityDataset("/home/$USER/data/RUGD")
    nebula_dataset = NebulaDataset("/home/$USER/data/nebula")

    model = ExploRFMONNXInference(
        onnx_ckpt=onnx_ckpt,
        model_precision="FP16",
        device='cuda'
    )

    run(model, nebula_dataset)
    cv2.destroyAllWindows()