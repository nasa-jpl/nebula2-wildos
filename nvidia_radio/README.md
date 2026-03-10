# NVIDIA RADIO — Modified for WildOS

A fork of [NVIDIA RADIO](https://github.com/NVlabs/RADIO) with modifications to produce language-aligned spatial features, following the approach proposed in [RayFronts](https://arxiv.org/abs/2504.06994).

## Modifications from Upstream RADIO

### 1. NACLIP Integration ([`radio/naclip.py`](radio/naclip.py))

The final self-attention layer is replaced with the [NACLIP](https://arxiv.org/abs/2404.08181) variant using KK<sup>T</sup> attention to enhance local spatial consistency. 

### 2. SigLIP Summary Adaptor for Spatial Features ([`radio/siglip2_adaptor.py`](radio/siglip2_adaptor.py))

RADIO's SigLIP summary adaptor is used to project patch embeddings into a language-aligned space, enabling text-conditioned similarity computation at the patch level.

### 3. PAMR ([`radio/pamr.py`](radio/pamr.py))

Pixel Adaptive Mask Refinement converts patch-level segmentation outputs to pixel-level resolution. Used in the prototyping scripts for open-vocabulary traversability segmentation.

### Key Arguments

The [`radio_model()`](hubconf.py) function in `hubconf.py` is the main entry point for loading the model. The following are the key arguments:

| Argument | Type | Default | Description |
|---|---|---|---|
| `version` | `str` | `""` | Model identifier (e.g., `"c-radio_v3-b"`) or a **local file path** to a checkpoint (e.g., `"ckpts/c-radio_v3-b_half.pth.tar"`) to skip downloading |
| `adaptor_names` | `str \| List[str]` | `None` | Adaptor(s) to load (e.g., `"siglip2"` for language-aligned features) |
| `adaptor_ckpt_path` | `str` | `"/home/$USER/cache/huggingface/hub"` | Path to the directory containing SigLIP2 adaptor weights |
| `use_naclip` | `bool` | `False` | Enable NACLIP attention (KK<sup>T</sup>) in the final self-attention layer |
| `use_summary_for_spatial` | `bool` | `False` | Project spatial patch features through the SigLIP summary adaptor into language-aligned space |
| `naclip_strategy` | `str` | `"naclip"` | NACLIP variant to use (e.g., `"kkonly"`) |

### Example: Loading the Model

```python
from nvidia_radio.hubconf import radio_model

model = radio_model(
    version="ckpts/c-radio_v3-b_half.pth.tar",  # local path or model id
    adaptor_names="siglip2",
    adaptor_ckpt_path="ckpts/siglip2",
    return_checkpoint=True,
    use_naclip=True,
    naclip_strategy="kkonly",
    fixed_patch_dim=(40, 40),
    use_summary_for_spatial=True,
)
```

For a full usage example with ExploRFM heads, see [`explorfm/explorfm_model.py`](../explorfm/explorfm_model.py).

### Main Modified Files

- [`radio/naclip.py`](radio/naclip.py) — NACLIP attention implementation
- [`radio/adaptor_generic.py`](radio/adaptor_generic.py) — Adaptor modifications for `use_summary_for_spatial`
- [`radio/radio_model.py`](radio/radio_model.py) — Model-level support for NACLIP and SigLIP summary adaptor
- [`radio/pamr.py`](radio/pamr.py) — PAMR for patch-to-pixel segmentation

> See the [main README](../README.md) for installation instructions and checkpoint downloads.
