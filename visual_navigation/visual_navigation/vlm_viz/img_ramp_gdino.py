import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge
import cv2
from sensor_msgs.msg import CompressedImage
from ament_index_python.packages import get_package_share_directory

import os
import torch
import torchvision
from PIL import Image

# Grounding DINO
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Recognize Anything Model & Tag2Text
from ram.models import ram_plus
from ram import inference_ram_openset
import torchvision.transforms as TS



class RamPlusPlusSubscriber(Node):

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
        pkg_path = get_package_share_directory('img_vlms')

        # grounding dino
        config_file = os.path.join(
            pkg_path,
            "third_party/Grounded-Segment-Anything/configs/GroundingDINO_SwinT_OGC.py"
        )
        grounded_checkpoint = os.path.join(
            pkg_path,
            "third_party/Grounded-Segment-Anything/checkpoints/groundingdino_swint_ogc.pth"
        )
        device = "cuda"
        dino_model = self.load_model(config_file, grounded_checkpoint, device=device)
        self.dino_model = dino_model.to(device)
        self.box_threshold = 0.25
        self.text_threshold = 0.2
        self.iou_threshold = 0.5

        # ram
        self.color_transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        normalize = TS.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self.ram_transform = TS.Compose(
            [
                TS.Resize((384, 384)),
                TS.ToTensor(), normalize
            ]
        )
        ram_checkpoint=os.path.join(
            pkg_path,
            "third_party/Grounded-Segment-Anything/checkpoints/ram_plus_swin_large_14m.pth"
        )
        ram_model = ram_plus(
            pretrained=ram_checkpoint,
            image_size=384,
            vit='swin_l'
        )
        ram_model.eval()
        self.ram_model = ram_model.to(device)

        self.device = device

        self.get_logger().info('Finished initializing models!')

        self.clbk_cntr = 0

    def load_model(self, model_config_path, model_checkpoint_path, device):
        args = SLConfig.fromfile(model_config_path)
        args.device = device
        model = build_model(args)
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
        load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        print(load_res)
        _ = model.eval()
        return model
    
    def get_grounding_output(self, image, caption):
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."

        image = image.to(self.device)
        with torch.no_grad():
            outputs = self.dino_model(image[None], captions=[caption])
        logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
        logits.shape[0]

        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > self.box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        logits_filt.shape[0]

        # get phrase
        tokenlizer = self.dino_model.tokenizer
        tokenized = tokenlizer(caption)

        # build pred
        pred_phrases = []
        scores = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > self.text_threshold, tokenized, tokenlizer)
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            scores.append(logit.max().item())

        return boxes_filt, torch.Tensor(scores), pred_phrases

    def listener_callback(self, msg):
        self.clbk_cntr += 1
        self.get_logger().info(f"Frame ID: {msg.header.frame_id}")

        if self.clbk_cntr % 10 != 0:
            return
        
        # Convert ROS Image message to OpenCV image
        current_frame = self.br.compressed_imgmsg_to_cv2(msg)
        current_frame = cv2.flip(current_frame, 0)

        # Convert OpenCV image to PIL image
        img_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb).convert("RGB")
        img, _ = self.color_transform(img_pil, None) # CHW

        ram_img = img_pil.resize((384, 384))
        ram_img = self.ram_transform(ram_img).unsqueeze(0).to(self.device)
        ram_res = inference_ram_openset(ram_img, self.ram_model)
        tags = ram_res.replace(' |', ',')
        self.get_logger().info(f"Tags: {tags}")

        # run grounding dino model
        boxes_filt, scores, pred_phrases = self.get_grounding_output(img, tags)

        size = img_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        # use NMS to handle overlapped boxes
        print(f"Before NMS: {boxes_filt.shape[0]} boxes")
        nms_idx = torchvision.ops.nms(boxes_filt, scores, self.iou_threshold).numpy().tolist()
        boxes_filt = boxes_filt[nms_idx]
        pred_phrases = [pred_phrases[idx] for idx in nms_idx]
        print(f"After NMS: {boxes_filt.shape[0]} boxes")

        # pad image for boxes on the border
        pad = 20
        current_frame = cv2.copyMakeBorder(current_frame, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        # draw boxes
        for box, label in zip(boxes_filt, pred_phrases):
            box = box.int().numpy()
            cv2.rectangle(current_frame, (box[0] + pad, box[1] + pad), (box[2] + pad, box[3] + pad), (0, 255, 0), 2)
            cv2.putText(current_frame, label, (box[0] + pad, box[1] + pad - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display image with bounding boxes
        cv2.imshow("VLM_img", current_frame)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)

    img_ramp = RamPlusPlusSubscriber()
    rclpy.spin(img_ramp)

    img_ramp.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
