import rclpy
from rclpy.node import Node

from sensor_msgs.msg import CompressedImage, Image as ImageMsg
from cv_bridge import CvBridge

from pathlib import Path
from omegaconf import OmegaConf
import numpy as np
import torch
from torchvision import transforms

import os
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec

from explorfm import ExploRFMInference
from visual_navigation.utils.viz import overlay_heatmap

HOME_DIR = Path.home()
CAMERA_MAPPING = {
    0: "front",
    1: "left",
    2: "right"
}    

class VizModelPred(Node):
    default_config = {
        # Model Params
        "frontier_ckpt": "frontier_ckpt_new.ckpt",
        "traversability_ckpt": "traversability_ckpt.ckpt",
        "model_version": "c-radio_v3-b",
        "adaptor_version": None,
        "use_naclip": True,
        "use_summary_for_spatial": True,
        "radio_dim": 768,
        "static_scale_factor": 0.75,
        "model_precision": "FP16",

        "cams_inverted": True,

        # Heatmap Viz Params
        "frontier_threshold": 0.6,
        "traversability_threshold": 0.9,
        "frontier_opening_kernel_size": 20,

        # ROS2 frames and topics
        "camera_img_topic": "/spot1/realsense/{}/color/image_raw",
    }

    def __init__(self, config: OmegaConf=OmegaConf.create()):
        config = OmegaConf.merge(OmegaConf.create(self.default_config), config)

        # init model before to prevent tf listener from spinning too early
        np.random.seed(42)
        self.init_model(config)

        super().__init__('viz_model_pred')
        self.get_logger().info('Finished initializing models!')
        
        # Used to convert between ROS and OpenCV images
        self.br = CvBridge()

        # Nav parameters and initializations
        self.cam_inverted = config.cams_inverted

        # Initialize pixel scoring params
        self.frontier_threshold = config.frontier_threshold
        self.traversability_threshold = config.traversability_threshold
        self.frontier_opening_kernel_size = config.frontier_opening_kernel_size

        self.clbk_cntr = 0

        # Frames and Topic Names
        self.using_compressed_imgs = "compressed" in config.camera_img_topic

        # Viz
        self.fig_size = (10, 8)        # size of matplotlib figure
        self.main_fig = plt.figure(figsize=self.fig_size)
        gs = GridSpec(1, 4, figure=self.main_fig, width_ratios=[1]*3 + [0.05])

        # Create axes
        self.main_axes = np.empty((1, 3), dtype=object)
        row = 0
        for col in range(3):
            self.main_axes[row, col] = self.main_fig.add_subplot(gs[row, col])
            self.main_axes[row, col].axis('off')

        # Add one shared colorbar axis
        cax = self.main_fig.add_subplot(gs[:, -1])
        cb = mpl.cm.ScalarMappable(cmap='jet', norm=mpl.colors.Normalize(vmin=0, vmax=1))
        self.main_fig.colorbar(cb, cax=cax, orientation='vertical')

        self.main_fig.tight_layout()

        # Subscribers and Publishers
        self.init_subscribers(config)

        os.makedirs("viz_outputs", exist_ok=True)
        os.makedirs("viz_outputs/imgs", exist_ok=True)
        os.makedirs("viz_outputs/frontiers", exist_ok=True)
        os.makedirs("viz_outputs/traversability", exist_ok=True)
        os.makedirs("viz_outputs/combined", exist_ok=True)

    def init_model(self, config):
        # vlm initializations
        self.device = "cuda"

        # radio model
        self.model = ExploRFMInference(
            frontier_ckpt= HOME_DIR / "ckpts" / config.frontier_ckpt,
            traversability_ckpt= HOME_DIR / "ckpts" / config.traversability_ckpt,
            model_version=config.model_version,
            adaptor_version=config.adaptor_version,
            use_naclip=config.use_naclip,
            use_summary_for_spatial=config.use_summary_for_spatial,
            radio_dim=config.radio_dim,
            static_scale_factor=config.static_scale_factor,
            model_precision=config.model_precision,
        )
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
        ])

    def init_subscribers(self, config):

        self.subscription = self.create_subscription(
            CompressedImage if self.using_compressed_imgs else ImageMsg,
            config.camera_img_topic.format("front"),
            self.listener_callback,
            2
        )

    def listener_callback(self, img_msg):
        self.clbk_cntr += 1

        self.get_logger().info(f"Received callback {self.clbk_cntr}")
    
        print(f"Started Heavy")

        # Extract messages
        if self.using_compressed_imgs:
            convert_func = self.br.compressed_imgmsg_to_cv2
        else:
            convert_func = self.br.imgmsg_to_cv2

        if self.cam_inverted:
            rgb_img = (
                np.rot90(convert_func(img_msg, desired_encoding='rgb8'), k=2)
            )
        else:
            rgb_img = (
                convert_func(img_msg, desired_encoding='rgb8')
            )

        # Model Forward pass
        rgb_imgs = [rgb_img]
        rgb_tensors = [self.transforms(img.copy()) for img in rgb_imgs]
        batch_tensor = torch.stack(rgb_tensors)
        batch_img_traversability, batch_img_frontiers, spatial_feats = self.model.forward(batch_tensor)
        
        batch_img_frontiers = batch_img_frontiers.cpu().numpy()
        batch_img_traversability = batch_img_traversability.cpu().numpy()
        h, w = batch_img_frontiers.shape[2:4]

        fin_viz_img = None

        img_frontiers = batch_img_frontiers[0, 0, :, :]
        traversability = batch_img_traversability[0, 0, :, :]

        frontier_overlay = rgb_img.copy()
        traversability_overlay = rgb_img.copy()

        img_frontier_conf = img_frontiers.copy()
        img_frontier_conf[img_frontier_conf < self.frontier_threshold] = 0.0
        img_frontier_conf[traversability < self.traversability_threshold] = 0.0

        # Morphological opening to remove small noisy frontier regions
        if self.frontier_opening_kernel_size > 0:
            kernel = np.ones((self.frontier_opening_kernel_size, self.frontier_opening_kernel_size), np.uint8)
            valid = cv2.morphologyEx((img_frontier_conf>0).astype(np.uint8), cv2.MORPH_OPEN, kernel)
            img_frontier_conf[valid==0] = 0.0

        # frontier_hm = overlay_heatmap(rgb_img, img_frontier_conf, alpha=1.0)
        frontier_hm = overlay_heatmap(rgb_img, img_frontier_conf, alpha=0.5)
        valid_frontier = (img_frontier_conf > 0)
        valid_mask_3d = np.stack([valid_frontier] * 3, axis=-1)
        frontier_overlay[valid_mask_3d] = frontier_hm[valid_mask_3d]
        cv2.imwrite(f"viz_outputs/frontiers/frontier_{self.clbk_cntr:03d}.png", frontier_overlay[:,:,::-1])

        img_trav_conf = traversability.copy()
        img_trav_conf[img_trav_conf < self.traversability_threshold] = 0.0

        trav_hm = overlay_heatmap(rgb_img, 1-img_trav_conf, alpha=0.5)
        valid_traversability = (img_trav_conf > 0)
        valid_mask_3d = np.stack([valid_traversability] * 3, axis=-1)
        traversability_overlay[valid_mask_3d] = trav_hm[valid_mask_3d]
        cv2.imwrite(f"viz_outputs/traversability/traversability_{self.clbk_cntr:03d}.png", traversability_overlay[:,:,::-1])

        self.main_axes[0,0].imshow(rgb_img)
        self.main_axes[0,0].set_title("RGB Image")

        self.main_axes[0,1].imshow(frontier_overlay)
        self.main_axes[0,1].set_title("Frontiers Overlay")

        self.main_axes[0,2].imshow(traversability_overlay)
        self.main_axes[0,2].set_title("Traversability Overlay")

        # Convert plot to image
        self.main_fig.canvas.draw()
        data = np.frombuffer(self.main_fig.canvas.tostring_argb(), dtype=np.uint8)
        data = data.reshape(self.main_fig.canvas.get_width_height()[::-1] + (4,))[:,:,1:]
        
        # clear axes for next use
        for i in range(3):
            self.main_axes[0, i].cla()
            self.main_axes[0, i].axis('off')
        fin_viz_img = data

        cv2.imshow("viz", fin_viz_img[:,:,::-1])
        cv2.waitKey(1)
        cv2.imwrite(f"viz_outputs/imgs/rgb_{self.clbk_cntr:03d}.png", rgb_img[:,:,::-1])

        print(f"Finished Heavy")



def main(args=None):
    rclpy.init(args=args)

    viz_node = VizModelPred()
    rclpy.spin(viz_node)

    viz_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
