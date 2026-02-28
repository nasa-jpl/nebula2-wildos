import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge
import cv2
from sensor_msgs.msg import CompressedImage
from ament_index_python.packages import get_package_share_directory

import argparse
import os

import numpy as np
import json
import torch
import torchvision
from PIL import Image


# segment anything
from segment_anything import (
    sam_model_registry,
    SamAutomaticMaskGenerator
) 
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

import cv2
import numpy as np
import matplotlib.pyplot as plt


class SegmentAnythingSubscriber(Node):

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

        # segment anything - model does not fit in memory
        """
        sam_checkpoint = "third_party/Grounded-Segment-Anything/sam_vit_h_4b8939.pth"
        sam = sam_model_registry["default"](checkpoint=sam_checkpoint)
        sam.to(device=device)
        self.sam_detector = SamAutomaticMaskGenerator(sam)
        """

        # segment anything 2
        pkg_path = get_package_share_directory('img_vlms')
        sam2_checkpoint = os.path.join(
            pkg_path,
            "third_party/Grounded-SAM-2/checkpoints/sam2.1_hiera_small.pt"
        )
        sam2_cfg = os.path.join(
            pkg_path,
            "third_party/Grounded-SAM-2/configs/sam2.1_hiera_s.yaml"
        )
        sam2_cfg = f'/{sam2_cfg}'

        sam2_model = build_sam2(sam2_cfg, sam2_checkpoint, device=device, apply_postprocessing=True)
        self.sam2_detector = SAM2AutomaticMaskGenerator(sam2_model)

        self.device = device

        self.get_logger().info('Finished initializing models!')

        self.clbk_cntr = 0

    def listener_callback(self, msg):
        self.clbk_cntr += 1
        self.get_logger().info(f"Frame ID: {msg.header.frame_id}")

        if self.clbk_cntr % 10 != 0:
            return
        
        # Convert ROS Image message to OpenCV image
        current_frame = self.br.compressed_imgmsg_to_cv2(msg)
        current_frame = cv2.flip(current_frame, 0)
        
        # Convert OpenCV BGR image to RGB
        img_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)

        # masks = self.sam_detector.generate(img_rgb)
        masks = self.sam2_detector.generate(img_rgb)

        for mask in masks:
            color = np.concatenate([np.random.random(3)], axis=0) * 255
            color = color.astype(np.uint8)
            current_frame[mask["segmentation"]] = current_frame[mask["segmentation"]] * 0.4 + color * 0.6
        
        # Display image with bounding boxes
        cv2.imshow("VLM_img", current_frame)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)

    sam_sub = SegmentAnythingSubscriber()
    rclpy.spin(sam_sub)

    sam_sub.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
