import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge
import cv2
from sensor_msgs.msg import CompressedImage

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image

# radio
# import sys
# from pathlib import Path
# sys.path.append(str(Path(__file__).resolve().parent / "third_party" / "nvidia_radio"))

from img_vlms.third_party.nvidia_radio.hubconf import radio_model
from img_vlms.third_party.nvidia_radio.radio.pamr import PAMR

import cv2
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
matplotlib.use("Agg")

from sklearn.decomposition import PCA
from skimage import color

class RADIOSubscriber(Node):

    def __init__(self):
        super().__init__('realsense_subscriber')
        self.subscription = self.create_subscription(
            CompressedImage,
            '/spot1/realsense/front/color/image_raw/compressed',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

        # Used to convert between ROS and OpenCV images
        self.br = CvBridge()

        # vlm initializations
        self.device = "cuda"

        # radio model
        model_version="c-radio_v3-b" # for RADIOv2.5-B model (ViT-B/16) ["c-radio_v3-b", "radio_v2.5-b"]
        self.adaptor_version = "siglip2" #["siglip", "siglip2", "clip", "dino", "sam"]

        model, chk = radio_model(
            version=model_version,
            progress=True,
            skip_validation=True,
            adaptor_names=self.adaptor_version,
            return_checkpoint=True, 
            use_naclip=True, 
            naclip_strategy="kkonly",
            naclip_gaussian_std=5.0,
            fixed_patch_dim=(45,80),
            gaussian_device='cuda',
            use_summary_for_spatial=True
        )
        self.model = model.to(self.device).eval()
        self.model.requires_grad_(False)  # Disable gradients for inference

        # Text Queries
        # self.text_queries = ["a person", "trees", "dirt", "house", "trail", "shed", "sky", "other"]
        self.text_queries = ["trail", "tree", "house", "fence", "person", "grass"]
        adaptor = model.adaptors[self.adaptor_version]
        tokens = adaptor.tokenizer(self.text_queries).to(self.device)
        self.text_feats = adaptor.encode_text(tokens, normalize=True)

        # Visualization
        self.heatmap_ui = True
        self.grid_size = (3, 2)  # Grid size for heatmap visualization - max queries is 4
        self.max_queries = self.grid_size[0]*self.grid_size[1]
        self.cmap = 'inferno'  # ['viridis', 'plasma', 'inferno', 'magma', 'seismic']
        self.viz_threshold = 0.1

        if self.heatmap_ui:
            self.text_feats = self.text_feats[:self.max_queries]
            self.text_queries = self.text_queries[:self.max_queries]

        self.pixel_level_seg = False
        self.pamr = PAMR(
            num_iter=10,
            dilations=[1, 2],
        )

        self.get_logger().info('Finished initializing models!')

        self.clbk_cntr = 0
    
    def create_ui(self, viz_img, color_map):
        fig = plt.figure(figsize=(18, 10))
        gs = gridspec.GridSpec(2, 1, height_ratios=[20, 1], hspace=0.1, wspace=0.9)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_title("Patch-Level Text Alignment")
        ax1.imshow(viz_img)
        ax1.axis('off')

        # Legend below 2nd plot
        ax_legend = fig.add_subplot(gs[1, 0])
        ax_legend.axis('off')
        handles = [plt.Line2D([0], [0], color=color_map[i], lw=6) for i in range(len(self.text_queries))]
        labels = [self.text_queries[i] for i in range(len(self.text_queries))]
        ax_legend.legend(handles, labels, loc='center', ncol=len(self.text_queries), fontsize='small', frameon=False)


        # convert plot to image
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)

        cv2.imshow("window", data)
        cv2.waitKey(1)

        plt.close()

    def create_heatmap_ui(self, spatial_features, img_rgb):
        """
        - Left: Original image
        - Right: Grid of heatmaps showing cosine similarity scores

        Args:
            spatial_features: Tensor (1, C, H, W), spatial features from the model
            img_rgb: RGB image as a numpy array (H, W, 3)
        """
        num_queries = len(self.text_queries)
        grid_rows, grid_cols = self.grid_size
        assert num_queries <= grid_rows * grid_cols, "Grid size too small for number of queries."

        fig = plt.figure(figsize=(18, 10))
        
        # Left: Original Image
        ax_img = plt.subplot2grid((grid_rows, grid_cols*2), (0, 0), rowspan=grid_rows, colspan=grid_cols)
        ax_img.set_title("Original Image", fontsize=14)
        ax_img.imshow(img_rgb)
        ax_img.axis('off')

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
            ).squeeze(0).cpu().numpy()  # Shape: (num_queries, H, W)
        else:
            text_sim_spatial = text_sim_spatial.cpu().numpy().transpose(1, 2, 0)  # Shape: (H, W, num_queries)
            text_sim_spatial = cv2.resize(
                text_sim_spatial, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_NEAREST
            )
            text_sim_spatial = text_sim_spatial.transpose(2, 0, 1)  # Shape: (num_queries, H, W)

        # Colormap configuration
        cmap = self.cmap
        vmin, vmax = 0.0, 0.2

        # Right: Heatmaps
        for i in range(num_queries):
            row = i // grid_cols
            col = i % grid_cols
            ax = plt.subplot2grid((grid_rows, grid_cols*2), (row, col + grid_cols))  # shift right
            im = ax.imshow(text_sim_spatial[i], cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(self.text_queries[i], fontsize=12)
            ax.axis('off')

        # Optional: add a single colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])  # [left, bottom, width, height]
        fig.colorbar(im, cax=cbar_ax)

        # Convert plot to image
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)

        cv2.imshow("Similarity Visualization", data)
        cv2.waitKey(1)

        plt.close()

    def compute_segmentation(self, spatial_features, img_rgb) -> tuple:
        """ Compute segmentation based on spatial features and text queries.

        Args:
            spatial_features: Tensor of shape (C, H, W) containing spatial features.
            img_rgb: RGB image as a numpy array of shape (H, W, 3).
        Returns:
            viz_img: Numpy array of the visualization image containing the original image and segmentation map side by side.
            color_map: Color map used for visualization.
        """
        
        # convert image to tensor
        x = pil_to_tensor(Image.fromarray(img_rgb)).to(dtype=torch.float32, device=self.device)

        # visualize cosine similarity between text embeddings and spatial features
        spatial_feats = spatial_features[0] # remove batch dimension
        spatial_feats = spatial_feats / spatial_feats.norm(dim=0, keepdim=True)
        c, h, w = spatial_feats.shape
        spatial_feats = spatial_feats.view(c, h * w)
        text_sim_spatial = self.text_feats @ spatial_feats  # (num_texts, H*W)
        text_sim_spatial = text_sim_spatial.view(len(self.text_queries), h, w)

        if self.pixel_level_seg:
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
        if not self.pixel_level_seg:
            pred_labels_resized = cv2.resize(
                pred_labels, (x.shape[-1], x.shape[-2]), interpolation=cv2.INTER_NEAREST
            )

        # visualize the predictions
        image_res = (x.shape[-2], x.shape[-1])  # (H, W)
        num_queries = len(self.text_queries)
        cmap = plt.get_cmap('tab10', num_queries)
        color_map = np.array([cmap(i)[:3] for i in range(num_queries)])  # [Q, 3], float [0, 1]

        # Create segmap
        segmap = np.zeros((image_res[0], image_res[1], 3), dtype=np.float32)
        for q in range(num_queries):
            for c in range(3):
                segmap[..., c] += (pred_labels_resized == q) * color_map[q, c]

        img_rgb = img_rgb / 255.0  # Convert to float [0, 1]
        segmap = np.clip(segmap, 0, 1)

        viz_img = np.concatenate([
            img_rgb, segmap
        ], axis=1)  # Concatenate original image and segmap side by side
        viz_img = (viz_img * 255).astype(np.uint8)

        return viz_img, color_map
        

    def listener_callback(self, msg):
        self.clbk_cntr += 1

        if self.clbk_cntr % 10 != 0:
            return
        
        # Convert ROS Image message to OpenCV image
        current_frame = self.br.compressed_imgmsg_to_cv2(msg)
        current_frame = cv2.flip(current_frame, 0)
        
        # Convert OpenCV BGR image to RGB
        img_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb).convert('RGB')
        
        # Preprocess the image
        x = pil_to_tensor(img_pil).to(dtype=torch.float32, device=self.device)
        x.div_(255.0)  # RADIO expects the input values to be between 0 and 1
        x = x.unsqueeze(0) # Add a batch dimension

        nearest_res = self.model.get_nearest_supported_resolution(*x.shape[-2:])
        x = F.interpolate(x, nearest_res, mode='bilinear', align_corners=False)
        print(f"Input image shape: {img_rgb.shape}, Nearest supported resolution: {nearest_res}")

        summary, spatial_features = self.model(x, feature_fmt='NCHW')[self.adaptor_version]

        if self.heatmap_ui:
            # Create and display the heatmap UI
            self.create_heatmap_ui(spatial_features, img_rgb)
        else:
            viz_img, color_map = self.compute_segmentation(spatial_features, img_rgb)

            # Create and display the UI
            self.create_ui(viz_img, color_map)

def main(args=None):
    rclpy.init(args=args)

    dino_sub = RADIOSubscriber()
    rclpy.spin(dino_sub)

    dino_sub.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
