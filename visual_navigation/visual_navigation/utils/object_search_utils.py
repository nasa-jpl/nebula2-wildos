import cv2
from typing import Dict, List
import numpy as np
import torch
import torch.nn.functional as F

from std_msgs.msg import UInt8MultiArray, MultiArrayDimension
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import CameraInfo
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from object_search_msgs.msg import ObjectMaskWithTf


def localize_query(
    text_feats: torch.Tensor,
    spatial_feats: torch.Tensor,
    orig_img_shape: tuple,
    pixel_level_seg: bool,
    mask_threshold: float
):
    """
    Localizes the text query in the spatial features.
    
    :param text_feats: (num_queries, D) Text feature vectors.
    :param spatial_feats: (B, D, H, W) Spatial feature map from the image.
    :param orig_img_shape: (H, W) Original image shape.
    :param pixel_level_seg: Whether to perform pixel-level segmentation.
    :param mask_threshold: Threshold for converting similarity maps to binary masks.

    :return: similarity_map, binary_mask
    """
    # Normalize features
    spatial_feats = spatial_feats / spatial_feats.norm(dim=1, keepdim=True)

    # Compute similarity maps
    text_sim_spatial = torch.einsum('qc,bchw->bqhw', text_feats, spatial_feats)  # Shape: (B, num_queries, H, W)
    
    # Resize similarity maps to match the original image size
    if pixel_level_seg:
        interp_mode = 'bilinear'
    else:
        interp_mode = 'nearest'
    text_sim_spatial = F.interpolate(
        text_sim_spatial, size=(orig_img_shape[0], orig_img_shape[1]), mode=interp_mode
    ).cpu().numpy()  # Shape: (B, num_queries, H, W)

    # Binary mask
    binary_mask = (text_sim_spatial > mask_threshold).astype(np.uint8)

    return text_sim_spatial, binary_mask

def convert_maskmsg_to_multiarray(mask_msg: np.ndarray) -> UInt8MultiArray:
    """
    Converts a binary mask to a UInt8MultiArray message.

    :param mask_msg: (B, 1, H, W) Binary mask of the detected object.
    :return: UInt8MultiArray message.
    """
    multiarray_msg = UInt8MultiArray()
    multiarray_msg.data = mask_msg.flatten().tolist()
    multiarray_msg.layout.data_offset = 0
    b,c,h,w = mask_msg.shape
    multiarray_msg.layout.dim = [
        MultiArrayDimension(label='batch', size=b, stride=b * c * h * w),
        MultiArrayDimension(label='channel', size=c, stride=c * h * w),
        MultiArrayDimension(label='height', size=h, stride=h * w),
        MultiArrayDimension(label='width', size=w, stride=w),
    ]
    return multiarray_msg

def get_objectmask_msg(
    binary_mask: np.ndarray,
    cam_inverted: bool,
    odom_msg: Odometry,
    tf_data: List[TransformStamped],
    cam_info_msgs: List[CameraInfo],
) -> ObjectMaskWithTf:
    """
    Converts the binary mask and associated data into an ObjectMaskWithTf message.

    :param binary_mask: (3, 1, H, W) Binary mask of the detected object.
    :param cam_inverted: Whether the camera is inverted.
    :param odom_msg: Odometry message for the robot's pose.
    :param tf_data: List of TF data for frame transformations from camera to odom.
    :param cam_info_msgs: List of CameraInfo messages for the cameras.

    :return: ObjectMaskWithTf message.
    """        
    
    if cam_inverted:
        binary_mask = np.rot90(binary_mask, k=2, axes=(2, 3))

    obj_mask_msg = ObjectMaskWithTf()
    obj_mask_msg.header = odom_msg.header

    obj_mask_msg.odom = odom_msg
    obj_mask_msg.cam_infos = cam_info_msgs
    obj_mask_msg.object_mask = convert_maskmsg_to_multiarray(binary_mask)
    obj_mask_msg.cam_transforms = TFMessage(transforms=tf_data)

    return obj_mask_msg