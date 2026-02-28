import rclpy
from rclpy.node import Node
from rclpy.time import Time

from visualization_msgs.msg import Marker
from sensor_msgs.msg import CompressedImage, Image as ImageMsg, CameraInfo, PointCloud2
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.duration import Duration
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from omegaconf import OmegaConf
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image

from triangulation3d.camera_data import Camera
from triangulation3d.particle_generator import ParticleGenerator
from triangulation3d.bbox_generator import BoundingBoxGenerator
from triangulation3d.triangulator import Triangulator
from image_geometry import PinholeCameraModel
from visual_navigation.third_party.nvidia_radio.hubconf import radio_model
from visual_navigation.third_party.nvidia_radio.radio.pamr import PAMR

import cv2
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
matplotlib.use("Agg")

from sklearn.decomposition import PCA
from skimage import color


class MessageBuffer:
    def __init__(self, max_size=10):
        self.max_size = max_size
        self.buffer = []

    def add_msg(self, msg: dict, timestamp):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)  # Remove the oldest image
        self.buffer.append((msg, timestamp))

    def get_oldest_msg(self) -> dict:
        if not self.buffer:
            return None
        return self.buffer[0][0]

    def get_closest_msg(self, timestamp):
        if not self.buffer:
            return None
        closest_msg = min(self.buffer, key=lambda x: abs(x[1] - timestamp))
        return closest_msg[0]
    
    def pop_oldest_msg(self):
        if self.buffer:
            self.buffer.pop(0)
    

class RADIOTriangulator(Node):
    default_config = {
        "max_cameras": 350,  # Maximum number of cameras to initialize
        "create_box": True,  # Whether to create a box for triangulation
        "particle_generator_config": {},
    }

    def __init__(self, config: OmegaConf=OmegaConf.create()):
        super().__init__('radio_triangulator')

        # Used to convert between ROS and OpenCV images
        self.br = CvBridge()

        # Triangulation parameters and initializations
        config = OmegaConf.merge(OmegaConf.create(self.default_config), config)
        self.cameras = []
        self.max_cameras = config.max_cameras
        self.create_box = config.create_box

        np.random.seed(0)
        self.particle_generator = ParticleGenerator(config.particle_generator_config)
        self.triangulator = Triangulator()
        self.triangulated_position = None

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
        self.text_queries = ["house"]
        adaptor = model.adaptors[self.adaptor_version]
        tokens = adaptor.tokenizer(self.text_queries).to(self.device)
        self.text_feats = adaptor.encode_text(tokens, normalize=True)

        assert len(self.text_queries) == 1 # Currently only supports single text query for triangulation

        # Visualization
        self.cmap = 'inferno'  # ['viridis', 'plasma', 'inferno', 'magma', 'seismic']
        self.viz_threshold = 0.1

        self.pixel_level_seg = False
        self.pamr = PAMR(
            num_iter=10,
            dilations=[1, 2],
        )

        self.get_logger().info('Finished initializing models!')

        self.clbk_cntr = 0

        # Subscribers and Publishers
        self.img_sub = Subscriber(self, ImageMsg, '/spot1/realsense/front/color/image_raw')
        self.camera_info_sub = Subscriber(self, CameraInfo, '/spot1/realsense/front/color/camera_info')

        self.ts = ApproximateTimeSynchronizer([self.img_sub, self.camera_info_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.listener_callback)
        self.msg_buffer = MessageBuffer(max_size=5)

        self.tf_buffer = Buffer(cache_time=Duration(seconds=10))
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

        self.timer = self.create_timer(0.05, self.check_tf_exists)

        self.point_cloud_publisher = self.create_publisher(
            PointCloud2,
            'particles',
            10
        )
        self.radio_det_publisher = self.create_publisher(
            ImageMsg,
            'radio_detection',
            10
        )

    def localize_query(self, spatial_features, img_rgb):
        """
        Localizes the text query in the spatial features.
        
        Args:
            spatial_features: Tensor (1, C, H, W), spatial features from the model
        """
        num_queries = len(self.text_queries)

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
            text_sim_spatial = text_sim_spatial[0]
        else:
            text_sim_spatial = text_sim_spatial.cpu().numpy().transpose(1, 2, 0)  # Shape: (H, W, num_queries)
            text_sim_spatial = cv2.resize(
                text_sim_spatial, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_NEAREST
            )

        # Binary mask for visualization
        binary_mask = (text_sim_spatial > self.viz_threshold).astype(np.uint8)

        # Get the bounding box for the text query
        bbox = cv2.boundingRect(binary_mask)

        return text_sim_spatial, binary_mask, bbox

    def create_heatmap_ui(self, img_rgb, text_sim_spatial, binary_mask, bbox):
        """
        - Left: Original image
        - Right: Similarity Heatmap and Binary Mask for the text query

        Args:
            spatial_features: Tensor (1, C, H, W), spatial features from the model
            img_rgb: RGB image as a numpy array (H, W, 3)
            text_sim_spatial: Similarity map for the text query (H, W)
            binary_mask: Binary mask for the text query (H, W)
            bbox: Bounding box for the text query (x, y, width, height)
        """
        fig = plt.figure(figsize=(18, 10))
        
        # Left: Original Image
        ax_img = fig.add_subplot(1, 2, 1)
        ax_img.set_title("Original Image", fontsize=14)
        ax_img.imshow(img_rgb)
        ax_img.axis('off')

        # Left: Original Image
        # ax_img = fig.add_subplot(2, 2, 3)
        # ax_img.set_title("Original Image", fontsize=14)
        # ax_img.imshow(np.rot90(img_rgb, k=2))  # Rotate the image 180 degrees
        # ax_img.axis('off')

        # Colormap configuration
        cmap = self.cmap
        vmin, vmax = 0.0, 0.2

        ax = fig.add_subplot(2, 2, 2)
        im = ax.imshow(text_sim_spatial, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"Similarity Heatmap: {self.text_queries[0]}", fontsize=12)
        ax.axis('off')

        # Add bbox to the binary mask if it exists
        if np.sum(binary_mask) != 0:
            x,y,w,h = bbox
            binary_mask = cv2.rectangle(binary_mask, (x, y), (x + w, y + h), color=1, thickness=2)

        ax = fig.add_subplot(2, 2, 4)
        ax.imshow(binary_mask, cmap='gray')
        ax.set_title(f"Binary Mask: {self.text_queries[0]}", fontsize=12)
        ax.axis('off')

        # Optional: add a single colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])  # [left, bottom, width, height]
        fig.colorbar(im, cax=cbar_ax)

        # Convert plot to image
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)

        img_msg = self.br.cv2_to_imgmsg(data, encoding='rgb8')
        self.radio_det_publisher.publish(img_msg)

        plt.close()
        

    def listener_callback(self, img_msg, cam_info_msg):
        self.clbk_cntr += 1

        # if self.clbk_cntr % 10 != 0:
        #     return
        self.msg_buffer.add_msg(
            {"img_msg": img_msg, "cam_info_msg": cam_info_msg}, 
            img_msg.header.stamp.sec + img_msg.header.stamp.nanosec * 1e-9
        )


    def check_tf_exists(self):
        # if len(self.cameras) >= self.max_cameras:
        #     self.timer.cancel()
        #     return

        if self.msg_buffer.buffer:
            oldest_msg = self.msg_buffer.get_oldest_msg()
            try:
                tf_oldest_msg = self.tf_buffer.lookup_transform(
                    'spot1/map', 
                    oldest_msg["img_msg"].header.frame_id, 
                    oldest_msg["img_msg"].header.stamp, 
                    timeout=Duration(seconds=0)
                )
                self.get_logger().info(f"TF found for {oldest_msg['img_msg'].header.frame_id} at time {oldest_msg['img_msg'].header.stamp}")
                self.do_processing(oldest_msg, tf_oldest_msg)
            except Exception as e:
                self.get_logger().info(f"TF not found for {oldest_msg['img_msg'].header.frame_id} at time {oldest_msg['img_msg'].header.stamp}: {e}")
                return

    def do_processing(self, msg, tf_msg):

        img_msg = msg["img_msg"]
        cam_info_msg = msg["cam_info_msg"]

        # Convert ROS Image message to OpenCV image
        current_frame = self.br.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        current_frame = np.rot90(current_frame, k=2)
        
        # Convert OpenCV BGR image to RGB
        img_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb).convert('RGB')
        
        # Preprocess the image
        x = pil_to_tensor(img_pil).to(dtype=torch.float32, device=self.device)
        x.div_(255.0)  # RADIO expects the input values to be between 0 and 1
        x = x.unsqueeze(0) # Add a batch dimension

        nearest_res = self.model.get_nearest_supported_resolution(*x.shape[-2:])
        x = F.interpolate(x, nearest_res, mode='bilinear', align_corners=False)
        # print(f"Input image shape: {img_rgb.shape}, Nearest supported resolution: {nearest_res}")

        summary, spatial_features = self.model(x, feature_fmt='NCHW')[self.adaptor_version]

        # flip the spatial features to match the image orientation
        # spatial_features = torch.flip(spatial_features, dims=[-1, -2])

        # Localize the text query in the spatial features
        text_sim_spatial, binary_mask, bbox = self.localize_query(spatial_features, img_rgb)
        self.create_heatmap_ui(img_rgb, text_sim_spatial, binary_mask, bbox)

        # Triangulation if object is detected
        if np.sum(binary_mask) == 0:
            return
        
        if len(self.cameras) < self.max_cameras:
            x,y,w,h = bbox
            bbox = np.array([x,y,x+w,y+h])  # Convert to (x_min, y_min, x_max, y_max)

            camera = Camera(
                camera_info = cam_info_msg,
                camera_tf = tf_msg,
                bounding_box=bbox,
                object_mask=binary_mask,
                image = img_msg,
            )
            camera = BoundingBoxGenerator.generate_ray_from_bbox(camera)

            camera = self.particle_generator.generate_particles(
                camera,
                use_mask=False,
                pcl_frame_id='spot1/map',
            )

            self.cameras.append(camera)
        
            if len(self.cameras) >= 2:
                self.triangulated_position = self.triangulator.triangulate(self.cameras)
                self.publish_triangulated_marker()
                self.get_logger().info(f"Triangulated position: {self.triangulated_position}")

        # Combine the points from all the cameras into a single PointCloud message
        combined_pcl_msg = self.triangulator.combine_points(self.cameras, pcl_frame_id='spot1/map')
        self.point_cloud_publisher.publish(combined_pcl_msg)
        if self.triangulated_position is not None:
            self.publish_triangulated_marker()

    def publish_triangulated_marker(self):
        """
        Publish a marker for the triangulated position.
        """
        marker = self.triangulator.get_triangulated_marker(self.triangulated_position, marker_frame_id='spot1/map')
        marker.header.stamp = self.get_clock().now().to_msg()

        # Publish the triangulated position
        triangulated_pub = self.create_publisher(Marker, 'triangulated_marker', 10)
        triangulated_pub.publish(marker)
        

def main(args=None):
    rclpy.init(args=args)

    dino_sub = RADIOTriangulator()
    rclpy.spin(dino_sub)

    dino_sub.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
