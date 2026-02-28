
from pathlib import Path
from omegaconf import OmegaConf
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
from torchvision import transforms

import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib
from matplotlib.gridspec import GridSpec

from img_vlms.third_party.nvidia_radio.radio_downstream import RADIODownstreamInference
from img_vlms.utils.viz import (
    make_subplot_grid, overlay_heatmap, draw_point, draw_text, draw_path, make_colorbar, pad_image
)
HOME_DIR = Path.home()
CAMERA_MAPPING = {
    0: "front",
    1: "left",
    2: "right"
}    

class VizModelPred:
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

        # Nav Params
        "num_cameras": 1,
        "cams_inverted": True,

        # Pixel Scoring Params
        "frontier_threshold": 0.6,
        "traversability_threshold": 0.9,
        "frontier_opening_kernel_size": 20,

        # ROS2 frames and topics
        "parent_frame": "spot1/odom",
        "cam_frame": "spot1/realsense/{}_color_optical_frame",
        "camera_img_topic": "/spot1/realsense/{}/color/image_raw",
        "camera_depth_topic": "/spot1/realsense/{}/aligned_depth_to_color/image_raw",
        "camera_info_topic": "/spot1/realsense/{}/color/camera_info",

        # Publish topic names
        "model_viz_topic": "/spot1/model_visualization",

        # ROS2 subscriber params
        "qos_history_depth": 1,
        "syncsub_queue_size": 2,
        "syncsub_slop": 0.2,
    }

    def __init__(self, config: OmegaConf=OmegaConf.create()):
        config = OmegaConf.merge(OmegaConf.create(self.default_config), config)

        # init model before to prevent tf listener from spinning too early
        np.random.seed(42)
        self.init_model(config)


        # Nav parameters and initializations
        self.num_cameras = config.num_cameras
        self.cam_inverted = config.cams_inverted
        assert self.num_cameras in [1, 3], "Only 1 or 3 cameras are supported."

        # Initialize pixel scoring params
        self.frontier_threshold = config.frontier_threshold
        self.traversability_threshold = config.traversability_threshold
        self.frontier_opening_kernel_size = config.frontier_opening_kernel_size

        self.clbk_cntr = 0

        # Frames and Topic Names
        self.global_frame = config.parent_frame
        self.cam_tf_frame = config.cam_frame
        self.using_compressed_imgs = "compressed" in config.camera_img_topic

        #Viz
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


    def init_model(self, config):
        # vlm initializations
        self.device = "cuda"

        # radio model
        self.model = RADIODownstreamInference(
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


    def listener_callback(self, rgb_img):
        self.clbk_cntr += 1

    
        print(f"Started Heavy")

        # Model Forward pass
        rgb_imgs = [rgb_img]
        rgb_tensors = [self.transforms(img.copy()) for img in rgb_imgs]
        batch_tensor = torch.stack(rgb_tensors)
        batch_img_traversability, batch_img_frontiers, spatial_feats = self.model.forward(batch_tensor)
        
        batch_img_frontiers = batch_img_frontiers.cpu().numpy()
        batch_img_traversability = batch_img_traversability.cpu().numpy()
        h, w = batch_img_frontiers.shape[2:4]

        fin_viz_img = None
        fig, axes = plt.subplots(1, self.num_cameras, figsize=(10,2))

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

        frontier_hm = overlay_heatmap(rgb_img, img_frontier_conf, alpha=1.0)
        valid_frontier = (img_frontier_conf > 0)
        valid_mask_3d = np.stack([valid_frontier] * 3, axis=-1)
        frontier_overlay[valid_mask_3d] = frontier_hm[valid_mask_3d]
        # cv2.imwrite(str(Path.home() / "data" / "rugd" / f"frontiers_{self.clbk_cntr:03d}.png"), frontier_overlay[:,:,::-1])
        cv2.imwrite(str(Path.home() / "data" / f"frontiers_{self.clbk_cntr:03d}.png"), frontier_overlay[:,:,::-1])


        img_trav_conf = traversability.copy()
        img_trav_conf[img_trav_conf < self.traversability_threshold] = 0.0

        trav_hm = overlay_heatmap(rgb_img, 1-img_trav_conf, alpha=0.5)
        valid_traversability = (img_trav_conf > 0)
        valid_mask_3d = np.stack([valid_traversability] * 3, axis=-1)
        traversability_overlay[valid_mask_3d] = trav_hm[valid_mask_3d]
        # cv2.imwrite(str(Path.home() / "data" / "rugd" / f"trav_{self.clbk_cntr:03d}.png"), traversability_overlay[:,:,::-1])
        # cv2.imwrite(str(Path.home() / "data" / "rugd" / f"img_{self.clbk_cntr:03d}.png"), rgb_img[:,:,::-1])

        cv2.imwrite(str(Path.home() / "data" / f"trav_{self.clbk_cntr:03d}.png"), traversability_overlay[:,:,::-1])
        cv2.imwrite(str(Path.home() / "data" / f"img_{self.clbk_cntr:03d}.png"), rgb_img[:,:,::-1])

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
        cv2.waitKey(0)
        # cv2.imwrite(str(Path.home() / "data" / "rugd" / f"viz_img_{self.clbk_cntr:03d}.png"), fin_viz_img[:,:,::-1])
        print(f"Finished Heavy")



def main(args=None):
    from pathlib import Path
    import os
    from tqdm import tqdm
    home_dir = Path.home()
    viz_node = VizModelPred()
    # img_dir = home_dir / "data" / "rugd" / "creek" 
    # file_ls = os.listdir(img_dir)
    # file_ls.sort()
    # for files in tqdm(os.listdir(img_dir)):
    #     img_path = img_dir / files
    #     rgb_img = cv2.imread(str(img_path))
    #     rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    #     viz_node.listener_callback(rgb_img)

    
    img_path = Path.home() / "data" / "watertank.jpg"
    rgb_img = cv2.imread(str(img_path))
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    viz_node.listener_callback(rgb_img)

if __name__ == '__main__':
    main()
