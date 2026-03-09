import cv2
import os
import json
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm

# ==== COLORS ====
BBOX_COLORS = [
    (0.73, 0.67, 0.25),
    (0.15, 0.59, 0.98),
    (0.56, 0.692, 0.195),
    (0.368, 0.507, 0.71),
    (0.881, 0.611, 0.142),
    (0.923, 0.386, 0.209),
    (0.528, 0.471, 0.701),
    (0.772, 0.432, 0.102),
    (0.572, 0.586, 0.0),
]
BBOX_COLORS = np.array(BBOX_COLORS) * 255

class FrontierAnnotator:
    """
    A class for annotating images with frontier boundaries.
    """
    
    def __init__(
        self,
        root_dir,
        scale=0.5,
        resume=True,
        view=False,
        resume_index=None
    ):
        # Initialize directories and parameters
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'RGB_rectified')
        self.boundary_dir = os.path.join(root_dir, 'SAM_boundaries')
        self.output_dir = os.path.join(root_dir, 'annotations')
        os.makedirs(self.output_dir, exist_ok=True)

        # display parameters
        self.scale = scale  # display scale
        self.resume = resume
        self.view = view
        self.resume_index = resume_index
        self.image_ext = '.png'
        self.overlay_key = ord('b')  # show boundaries
        self.label_switch_key = ord('n')  # switch to next label
        self.save_key = ord('s')  # save annotations
        self.reset_key = ord('r')  # reset current boxes

        self.labels = ['frontier', 'image_boundary_frontier']

        # annotation state
        self.current_label_index = 0
        self.start_point = None
        self.end_point = None
        self.boxes = []
        self.drawing = False

    def get_output_file(self, image_name):
        return os.path.join(self.output_dir, image_name.replace(self.image_ext, '.json').replace('rect', 'annotation'))

    def load_existing_annotations(self, image_name):
        path = self.get_output_file(image_name)
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        else:
            return []

    def save_annotations(self, image_name, data):
        with open(self.get_output_file(image_name), 'w') as f:
            json.dump(data, f, indent=2)

    def draw_all_boxes(self, img, annotations):
        for i in range(len(annotations)):
            box = annotations[i]['start'] + annotations[i]['end']
            label = annotations[i]['label']
            x1, y1, x2, y2 = [int(i * self.scale) for i in box]
            color = BBOX_COLORS[self.labels.index(label) % len(BBOX_COLORS)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_point = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_point = (x, y)
            self.boxes.append({
                "label": self.labels[self.current_label_index],
                "start": [self.start_point[0] / self.scale, self.start_point[1] / self.scale],
                "end": [self.end_point[0] / self.scale, self.end_point[1] / self.scale]
            })
            self.end_point = None

    def reset_state(self):
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.boxes = []

    def annotate_image(self, image_name):

        image_path = os.path.join(self.image_dir, image_name)
        boundary_path = os.path.join(self.boundary_dir, image_name.replace('rect', 'bound').replace(self.image_ext, '.png'))

        img = cv2.imread(image_path)
        boundary = cv2.imread(boundary_path, cv2.IMREAD_UNCHANGED) // 255
        canvas = cv2.resize(img, (0, 0), fx=self.scale, fy=self.scale)
        resized_boundary = cv2.resize(boundary, (canvas.shape[1], canvas.shape[0]))

        annotations = self.load_existing_annotations(image_name)

        cv2.namedWindow('Annotator')
        cv2.setMouseCallback('Annotator', self.mouse_callback)

        key = -1
        show_boundary = False

        while True:
            current_drawing = canvas.copy()

            # Overlay boundary
            if show_boundary:
                current_drawing[resized_boundary > 0] = [255, 255, 255]  # White overlay

            self.draw_all_boxes(current_drawing, annotations + self.boxes)
            cv2.putText(current_drawing, f'Current Label: {self.labels[self.current_label_index]}. Image: {self.img_idx}/{self.total_imgs-1}', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if self.drawing:
                if self.start_point and self.end_point:
                    cv2.rectangle(current_drawing, self.start_point, self.end_point, BBOX_COLORS[self.current_label_index], 2)

            cv2.imshow('Annotator', current_drawing)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                return 'exit'
            elif key == 32 or self.view:  # SPACE
                self.reset_state()
                return 'skip'
            elif key == self.label_switch_key:
                self.current_label_index = (self.current_label_index + 1) % len(self.labels)
            elif key == self.overlay_key:
                show_boundary = not show_boundary
            elif key == self.reset_key:
                self.reset_state()
                annotations = []

            # Press Enter to move to next image
            if key == self.save_key or key == 13:
                if self.boxes:
                    annotations += self.boxes
                self.reset_state()
                self.save_annotations(image_name, annotations)
                return 'next'

    def get_image_list(self):
        return sorted([f for f in os.listdir(self.image_dir) if f.endswith(self.image_ext)])

    def get_last_done_index(self, image_list):
        for i, img_name in enumerate(image_list):
            if not os.path.exists(self.get_output_file(img_name)):
                return i
        return len(image_list)
    
    def run(self):
        images = self.get_image_list()
        self.img_idx = 0
        self.total_imgs = len(images)

        if self.resume:
            if self.resume_index is None:
                start_index = self.get_last_done_index(images)
            else:
                start_index = self.resume_index
            if start_index == len(images):
                print("All images have been annotated.")
                return
        else:
            start_index = 0
        for i in tqdm(range(start_index, len(images))):
            self.img_idx = i
            result = self.annotate_image(images[i])
            if result == 'exit':
                break
            elif result == 'skip':
                continue

if __name__ == '__main__':
    parser = ArgumentParser(description="Frontier Annotation Tool")
    parser.add_argument('--resume_index', type=int, default=None, help="Index to resume annotation from")
    parser.add_argument('--view', action='store_true', help="View mode (no editing)")
    args = parser.parse_args()
    annotator = FrontierAnnotator(
        root_dir='/home/$USER/data/grand_tour_selected',
        resume_index=args.resume_index,
        view=args.view
    )
    annotator.run()
