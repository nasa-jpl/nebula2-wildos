"""
Use:  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
when running on large images (debug_img_nebula.png)
"""

from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

import torch
from torch.nn import functional as F
from torchvision.transforms.functional import pil_to_tensor

from nvidia_radio.hubconf import radio_model
from nvidia_radio.radio.pamr import PAMR


#model_version="radio_v2.5-g" # for RADIOv2.5-g model (ViT-H/14)
# model_version="radio_v2.5-h" # for RADIOv2.5-H model (ViT-H/16)
# model_version="radio_v2.5-l" # for RADIOv2.5-L model (ViT-L/16)
# model_version="radio_v2.5-b" # for RADIOv2.5-B model (ViT-B/16)
model_version="c-radio_v3-b" # for C_RADIOv3-B model (ViT-B/16)
# model_version="e-radio_v2" # for E-RADIO
adaptor_version="siglip2" # ["clip", siglip", "siglip2"]

model, chk = radio_model(
    version=model_version,
    progress=True,
    skip_validation=True,
    adaptor_names=adaptor_version,
    return_checkpoint=True, 
    use_naclip=True, 
    naclip_strategy="kkonly", #"kkonly",
    naclip_gaussian_std=5.0,
    fixed_patch_dim=(40,40), #(45,80),
    gaussian_device='cuda',
    use_summary_for_spatial=True
)

size_model = 0
for param in model.parameters():
    if param.data.is_floating_point():
        size_model += param.numel() * torch.finfo(param.data.dtype).bits
    else:
        size_model += param.numel() * torch.iinfo(param.data.dtype).bits
print(f"model size: {size_model} / bit | {size_model / 8e6:.2f} / MB")
model.cuda().eval()

# x = Image.open('assets/debug_img_nebula.png').convert('RGB')
x = Image.open('assets/demo.png').convert('RGB')
img = x.copy()
x = pil_to_tensor(x).to(dtype=torch.float32, device='cuda')
x.div_(255.0)  # RADIO expects the input values to be between 0 and 1
x = x.unsqueeze(0) # Add a batch dimension

nearest_res = model.get_nearest_supported_resolution(*x.shape[-2:])
x = F.interpolate(x, nearest_res, mode='bilinear', align_corners=False)

print(f"Original input shape: {x.shape}")
print(f"Nearest supported resolution: {nearest_res}")
print(f"Model Input shape: {x.shape}")

if "e-radio" in model_version:
    model.model.set_optimal_window_size(x.shape[2:]) #where it expects a tuple of (height, width) of the input image.

text_queries = [
    "car", "building", "streetlamp", "sky", "tree", "road", "flag", "other"
    # "bush", "tree trunk", "leaves", "trail", "sky", "other",
]
adaptor = model.adaptors[adaptor_version]
tokens = adaptor.tokenizer(text_queries).to("cuda")
text_feats = adaptor.encode_text(tokens, normalize=True)

summary, spatial_features = model(x, feature_fmt='NCHW')[adaptor_version]
assert spatial_features.ndim == 4

print(f"Spatial features shape: {spatial_features.shape}")
print(f"Summary shape: {summary.shape}")

# visualize cosine similarity between text embeddings and spatial features
spatial_feats = spatial_features[0] # remove batch dimension
spatial_feats = spatial_feats / spatial_feats.norm(dim=0, keepdim=True)
c, h, w = spatial_feats.shape
spatial_feats = spatial_feats.view(c, h * w)
text_sim_spatial = text_feats @ spatial_feats  # (num_texts, H*W)
text_sim_spatial = text_sim_spatial.view(len(text_queries), h, w)

pixel_level_seg = False
if pixel_level_seg:
    text_sim_spatial = F.interpolate(
        text_sim_spatial.unsqueeze(0), size=(x.shape[-2], x.shape[-1]), mode='bilinear', align_corners=False
    ).squeeze(0)  # (num_texts, H, W)

    # Apply PAMR to the text similarity map (mask refinement: Patch to Pixels)
    # if GPU size allows, use: 
    # num_iter = 50
    # dilations = [1, 2, 4, 8, 12, 24]
    pamr = PAMR(10, dilations=[1, 2]).to('cuda')
    text_sim_spatial = pamr(x*255, text_sim_spatial.unsqueeze(0)).squeeze(0)  # (num_texts, H, W)

pred_labels = text_sim_spatial.argmax(dim=0).cpu().numpy()  # (H, W)
pred_labels_resized = pred_labels

# To get patch level categories, resize the labels
if not pixel_level_seg:
    pred_labels_resized = cv2.resize(
        pred_labels, (x.shape[-1], x.shape[-2]), interpolation=cv2.INTER_NEAREST
    )

# visualize the predictions
image_res = (x.shape[-2], x.shape[-1])  # (H, W)
num_queries = len(text_queries)
cmap = plt.get_cmap('tab10', num_queries)
color_map = np.array([cmap(i)[:3] for i in range(num_queries)])  # [Q, 3], float [0, 1]

# Create segmap
segmap = np.zeros((image_res[0], image_res[1], 3), dtype=np.float32)
for q in range(num_queries):
    for c in range(3):
        segmap[..., c] += (pred_labels_resized == q) * color_map[q, c]

img_rgb = np.array(img.convert('RGB')) / 255.0  # Convert to float [0, 1]
segmap = np.clip(segmap, 0, 1)

viz_img = np.concatenate([
    img_rgb, segmap
], axis=1)  # Concatenate original image and segmap side by side
viz_img = (viz_img * 255).astype(np.uint8)


fig = plt.figure(figsize=(18, 10))
gs = gridspec.GridSpec(2, 1, height_ratios=[20, 1], hspace=0.1, wspace=0.9)

ax1 = fig.add_subplot(gs[0, 0])
ax1.set_title("Patch-Level Text Alignment")
ax1.imshow(viz_img)
ax1.axis('off')

# Legend below 2nd plot
ax_legend = fig.add_subplot(gs[1, 0])
ax_legend.axis('off')
handles = [plt.Line2D([0], [0], color=color_map[i], lw=6) for i in range(len(text_queries))]
labels = [text_queries[i] for i in range(len(text_queries))]
ax_legend.legend(handles, labels, loc='center', ncol=num_queries, fontsize='small', frameon=False)

plt.show()