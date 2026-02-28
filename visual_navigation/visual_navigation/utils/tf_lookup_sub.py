from dataclasses import dataclass
from typing import List, Dict
from abc import ABC, abstractmethod

import rclpy
from rclpy.qos import ReliabilityPolicy
from rclpy.qos import DurabilityPolicy
from rclpy.qos import HistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.node import Node
from rclpy.duration import Duration

from builtin_interfaces.msg import Time
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from omegaconf import OmegaConf
import numpy as np
from scipy.spatial.transform import Rotation as R

from img_vlms.utils.buffer import MessageBuffer

@dataclass
class TFEdge:
    source_frame: str
    target_frame: str

class TFLookupSubscriber(Node, ABC):

    default_tflookup_config = {
        "buffer_size": 1,       # number of messages
        "cache_time": 10,       # seconds
        "timer_duration": 0.5,  # seconds
        "lookup_timeout": 0,     # seconds
        "qos_history_depth": 100,  # depth for QoS profile
        "wait_for_oldest": False,  # whether to wait when buffer is full
        "clear_buffer_on_process": False,  # whether to clear buffer after processing
        "spin_thread": False,     # whether to spin tf listener in a separate thread
    }

    def __init__(
        self, node_name: str, config: OmegaConf=OmegaConf.create()
    ):
        super().__init__(node_name)
        config = OmegaConf.merge(OmegaConf.create(self.default_tflookup_config), config)

        self.msg_buffer = MessageBuffer(max_size=config.buffer_size, wait_for_oldest=config.wait_for_oldest)

        self.tf_buffer = Buffer(cache_time=Duration(seconds=config.cache_time))
        self.tf_listener = TransformListener(
            self.tf_buffer, self,
            qos = QoSProfile(
                depth=config.qos_history_depth,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST,
            ),
            spin_thread=config.spin_thread
        )

        self.oldest_time_processed = None
        self.timer = None
        self.timer_duration = config.timer_duration
        self.lookup_timeout = config.lookup_timeout
        self.clear_buffer_on_process = config.clear_buffer_on_process

        self._required_transforms: Dict[str, TFEdge] = {}

    @property
    def required_transforms(self):
        return self._required_transforms

    @required_transforms.setter
    def required_transforms(self, transforms: Dict[str, TFEdge]):
        if not self._required_transforms:
            self._required_transforms = transforms
        else:
            self.get_logger().warn(f"Required transforms already set. {self.__class__.__name__} tried to set it again.")

    def start_timer(self):
        if self.timer is None:
            self.timer = self.create_timer(self.timer_duration, self.check_tf_exists)
        else:
            self.get_logger().warn(f"Timer already initialized. {self.__class__.__name__} tried to initialize it again.")

    def check_tf_exists(self):
        if self.msg_buffer.buffer:
            found_one_valid_ts = False
            found_invalid_after_valid = False
            valid_tfs = None
            valid_msg = None
            valid_ts = None

            for old_msg, old_msg_stamp, old_msg_tm in self.msg_buffer.buffer:
                tfs = {}
                for edge_name, edge in self._required_transforms.items():
                    try:
                        tf_oldest_msg = self.tf_buffer.lookup_transform(
                            edge.target_frame,
                            edge.source_frame, 
                            old_msg_stamp, 
                            timeout=Duration(seconds=self.lookup_timeout)
                        )
                        tfs[edge_name] = tf_oldest_msg

                    except Exception as e:
                        self.get_logger().info(f"TF not found for {edge.source_frame} at time {old_msg_stamp}: {e}")
                        if not found_one_valid_ts:
                            return
                        else:
                            found_invalid_after_valid = True
                            break
                if found_invalid_after_valid:
                    break
                found_one_valid_ts = True
                valid_tfs = tfs.copy()
                valid_msg = old_msg
                valid_ts = old_msg_tm
                break
            
            if self.oldest_time_processed is None or self.oldest_time_processed < valid_ts:
                self.oldest_time_processed = valid_ts
                self.get_logger().info(f"TF found for camera frames at time {valid_ts}")
                self.do_processing(valid_msg, valid_tfs)
                if self.clear_buffer_on_process:
                    self.msg_buffer.clear()
            else:
                self.get_logger().warn(f"Already processed TF for time {valid_ts}, skipping processing.")
                self.msg_buffer.pop_oldest_msg()
        else:
            self.get_logger().warn("Message buffer is empty, waiting for messages...")

    def fetch_cam_intrinsics_extrinsics(self, cam_info, tf_world_from_cam):
        """
        Fetch camera intrinsics and extrinsics using the CameraInfo message.
        """
        K = np.array(cam_info.k).reshape(3, 3)

        R_wc = R.from_quat([
            tf_world_from_cam.transform.rotation.x,
            tf_world_from_cam.transform.rotation.y,
            tf_world_from_cam.transform.rotation.z,
            tf_world_from_cam.transform.rotation.w
        ]).as_matrix()
        t_wc = np.array([
            tf_world_from_cam.transform.translation.x,
            tf_world_from_cam.transform.translation.y,
            tf_world_from_cam.transform.translation.z
        ]).reshape(3, 1)

        frame_id = cam_info.header.frame_id
        if frame_id.startswith("/"):
            frame_id = frame_id[1:]
        return {
            "K": K,
            "height": cam_info.height,
            "width": cam_info.width,
            "R_wc": R_wc,
            "t_wc": t_wc,
            "frame_id": frame_id,
        }

    @abstractmethod
    def do_processing(self, msg: Dict, tfs: List):
        raise NotImplementedError("This method should be overridden by subclasses.")