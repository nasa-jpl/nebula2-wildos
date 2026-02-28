import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge
import cv2
from sensor_msgs.msg import CompressedImage

import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as F
from PIL import Image


# dinov2
from transformers import AutoImageProcessor, AutoModel

import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from sklearn.decomposition import PCA
from skimage import color

class DinoV2PCA(Node):

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
        device = "cuda"

        # dinov2
        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        self.model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)
        self.model.eval()
        self.model.requires_grad_(False)

        self.pca = PCA(n_components=3)

        self.device = device

        self.get_logger().info('Finished initializing models!')

        self.clbk_cntr = 0

    def visualize_preprocessing(self, inputs):
        # visualize preprocessing
        inputs_viz = inputs['pixel_values'].squeeze(0).permute(1, 2, 0).cpu().numpy()
        inputs_viz = (inputs_viz - inputs_viz.min()) / (inputs_viz.max() - inputs_viz.min()) * 255
        inputs_viz = inputs_viz.astype(np.uint8)
        inputs_viz = cv2.cvtColor(inputs_viz, cv2.COLOR_RGB2BGR)

        return inputs_viz

    def pe_visualization(self, features, gaussian_blur=False, lch=False):
        """
        Visualization of features as proposed in PerceptionEncoder: https://arxiv.org/pdf/2504.13181
        """
        # gaussian blur
        if gaussian_blur:
            features = features.reshape(16, 16, 768)
            blurred_features = F.gaussian_blur(torch.tensor(features), kernel_size=(3, 3), sigma=(1.0, 1.0)).numpy()
            features = features*0.5 + blurred_features*0.5
            features = features.reshape(-1, 768)


        self.pca.fit(features)
        pca_features = self.pca.transform(features)
        pca_features = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())

        if lch:
            # Map first 3 PCA components to L, C, and h in LCh color space
            # Ensure pca_features has at least 3 components
            L = pca_features[:, 0] * 100              # Lightness in [0, 100]
            C = pca_features[:, 1] * 100              # Chroma typically in [0, 100+]
            h = pca_features[:, 2] * 360              # Hue in degrees [0, 360]

            # Convert LCh to Lab first
            a = C * np.cos(np.radians(h))
            b = C * np.sin(np.radians(h))
            lab = np.stack([L, a, b], axis=-1)

            # Convert Lab to RGB
            # Reshape to 2D spatial layout before color conversion
            lab_image = lab.reshape(16, 16, 3)
            rgb_image = color.lab2rgb(lab_image)

            rgb_image_8bit = (rgb_image * 255).clip(0, 255).astype(np.uint8)
        else:
            rgb_image_8bit = (pca_features * 255).reshape(16, 16, 3).astype(np.uint8)

        return rgb_image_8bit
    

    def pca_visualization(self, features):
        # PCA
        self.pca.fit(features)
        pca_features = self.pca.transform(features)
        pca_features = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())
        pca_features = (pca_features * 255).reshape(16, 16, 3).astype(np.uint8)

        return pca_features
    
    def create_ui(self, img_rgb, img_pr, feats):
        fig = plt.figure(figsize=(12,6))
        fig.suptitle("DINOv2 Features", fontsize=16)

        # 1. Image 
        ax = fig.add_subplot(121)
        plt.title(f"Current Frame")
        ax.imshow(img_rgb)
        plt.xticks([])
        plt.yticks([])

        # 2. Preprocessed Image
        ax = fig.add_subplot(222)
        plt.title("Center Cropped Frame")
        ax.imshow(img_pr)
        plt.xticks([])
        plt.yticks([])


        # 3. Features
        ax = fig.add_subplot(224)
        plt.title("PCA DINO Feats")
        ax.imshow(feats)
        plt.xticks([])
        plt.yticks([])

        # convert plot to image
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)

        cv2.imshow("window", data)
        cv2.waitKey(1)

        plt.close()
        

    def listener_callback(self, msg):
        self.clbk_cntr += 1

        if self.clbk_cntr % 10 != 0:
            return
        
        # Convert ROS Image message to OpenCV image
        current_frame = self.br.compressed_imgmsg_to_cv2(msg)
        current_frame = cv2.flip(current_frame, 0)
        
        # Convert OpenCV BGR image to RGB
        img_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
        
        inputs = self.processor(images=img_rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        features = outputs.last_hidden_state[0] # (257, 768)
        features = features[1:] # (256, 768)
        features = features.cpu().numpy()

        # viz_features = self.pe_visualization(features, gaussian_blur=True)
        viz_features = self.pca_visualization(features)

        # Plot 
        self.create_ui(img_rgb, self.visualize_preprocessing(inputs), viz_features)


def main(args=None):
    rclpy.init(args=args)

    dino_sub = DinoV2PCA()
    rclpy.spin(dino_sub)

    dino_sub.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
