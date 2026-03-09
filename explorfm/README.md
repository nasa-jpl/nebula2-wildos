# ExploRFM — Exploration Foundation Model

ExploRFM is the vision module for WildOS, built on top of the [modified RADIO backbone](../nvidia_radio/). It simultaneously predicts:

- **Visual Traversability** — safe/unsafe terrain classification
- **Visual Frontiers** — candidate exploration directions visible beyond the depth horizon  
- **Open-Vocabulary Object Similarity** — text-conditioned similarity scoring for target object search

## Architecture

ExploRFM uses the language-aligned RADIO backbone (with NACLIP + SigLIP2 adaptor) to extract dense spatial features from RGB images. Lightweight CNN deconvolution heads decode these features into traversability and frontier predictions. For object similarity, the language-aligned patch features are directly compared against text embeddings.

## Usage

The [`ExploRFMInference`](explorfm_model.py) class provides the main inference interface:

```python
from explorfm.explorfm_model import ExploRFMInference

model = ExploRFMInference(
    frontier_ckpt="ckpts/frontier_head.ckpt",
    traversability_ckpt="ckpts/trav_head.ckpt",
    model_version="c-radio_v3-b",          # or local path: "ckpts/c-radio_v3-b_half.pth.tar"
    adaptor_version="siglip2",
    adaptor_ckpt_path="ckpts/siglip2",
    use_naclip=True,
    use_summary_for_spatial=True,
    radio_dim=768,
    static_scale_factor=0.5,
    model_precision="FP32",                 # "FP32" or "FP16"
)

# Forward pass on a tensor (NCHW)
traversability, frontiers, spatial_features = model.forward(input_tensor)

# Forward pass on a numpy image (HWC, uint8)
traversability, frontiers, spatial_features = model.forward_on_numpy(image_np)

# Encode text queries for open-vocabulary similarity
text_embeddings = model.forward_on_text(["house", "car", "tree"])
```

An ONNX export and inference wrapper is also available in [`explorfm_model_onnx.py`](explorfm_model_onnx.py).

## Prototyping Scripts

The `prototyping/` directory contains standalone scripts for testing and visualization:

| Script | Description |
|---|---|
| `geofrontiers_scoring.py` | Experiments with scoring strategies for geometric frontiers using traversability and frontier predictions. Randomly samples geometric frontiers in image space. |
| `rugd_traversability_frontiers.py` | Visualizes ExploRFM traversability and frontier outputs on the RUGD dataset. |
| `rugd_ovts.py` | Open-vocabulary traversability segmentation on RUGD using RADIO+SigLIP-summary with PAMR for patch-to-pixel segmentation. Manually defines "safe" and "risky" classes. |
| `test_naclip_ovss.py` | Tests whether the NACLIP modifications on RADIO affect open-vocabulary semantic segmentation performance. |

> See the [main README](../README.md) for installation and checkpoints, and [`explorfm_trainer/`](../explorfm_trainer/README.md) for training.
