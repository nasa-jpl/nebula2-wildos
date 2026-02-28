import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge
import cv2
from sensor_msgs.msg import CompressedImage
from ament_index_python.packages import get_package_share_directory

import os
import numpy as np
import json
import torch
import torchvision
from PIL import Image
# import litellm

# Grounding DINO
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Recognize Anything Model & Tag2Text
from ram.models import ram
from ram import inference_ram
import torchvision.transforms as TS



class RamSubscriber(Node):

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
            "third_party/Grounded-Segment-Anything/checkpoints/ram_swin_large_14m.pth"
        )
        ram_model = ram(
            pretrained=ram_checkpoint,
            image_size=384,
            vit='swin_l'
        )
        ram_model.eval()
        self.ram_model = ram_model.to(device)

        # segment anything
        use_sam = False
        use_sam_hq = False
        sam_checkpoint = "/home/scarecrow/jpl_nebula2_ws/src/img_vlms/Grounded-Segment-Anything/sam_vit_h_4b8939.pth"
        sam_hq_checkpoint = None
        if use_sam_hq:
            print("Initialize SAM-HQ Predictor")
            sam_predictor = SamPredictor(build_sam_hq(checkpoint=sam_hq_checkpoint).to(device))
        elif use_sam:
            sam_predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))
        else:
            sam_predictor = None
        self.sam_predictor = sam_predictor
        
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

        del image, outputs

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

        if self.sam_predictor is not None:
            self.sam_predictor.set_image(img_rgb)

        ram_img = img_pil.resize((384, 384))
        ram_img = self.ram_transform(ram_img).unsqueeze(0).to(self.device)
        ram_res = inference_ram(ram_img, self.ram_model)
        tags = ram_res[0].replace(' |', ',')
        self.get_logger().info(f"Tags: {tags}")
        del ram_res, ram_img

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

        # sam inference
        masks = None
        if self.sam_predictor is not None:
            transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_filt, img_rgb.shape[:2]).to(self.device)

            masks, _, _ = self.sam_predictor.predict_torch(
                point_coords = None,
                point_labels = None,
                boxes = transformed_boxes,
                multimask_output = False,
            )
            del transformed_boxes

        # draw boxes
        for box, label in zip(boxes_filt, pred_phrases):
            box = box.int().numpy()
            cv2.rectangle(current_frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(current_frame, label, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # draw masks
        if masks is not None:
            for mask in masks:
                color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
                h, w = mask.shape[-2:]
                mask_img = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
                mask_img = mask_img.astype(np.uint8)
                current_frame += mask_img

        # Display image with bounding boxes
        cv2.imshow("VLM_img", current_frame)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)

    img_ram = RamSubscriber()
    rclpy.spin(img_ram)

    img_ram.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
