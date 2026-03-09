import cv2
import os
import json
from pathlib import Path
import shutil
from collections import defaultdict
import numpy as np

# ---- Configuration ----
ROOT_DIR = '/home/$USER/data'
SRC_DIR = os.path.join(ROOT_DIR, 'grand_tour_data')
DEST_DIR = os.path.join(ROOT_DIR, 'grand_tour_selected', 'RGB_frames')
METADATA_FILE = os.path.join(DEST_DIR, 'metadata.json')
FRAME_SKIP = 10  # Number of frames to skip between selections

# ---- Scenes ----
SELECTED_SCENES = [
    "release_2024-11-03-07-57-34",  # Jungfraujoch-snow-2
    "release_2024-11-03-08-17-23",  # Jungfraujoch-snow-3
    "release_2024-11-03-13-51-43",  # Jungfraujoch - Eigergletscher Down
    "release_2024-11-03-13-59-54",  # Jungfraujoch - Eigergletscher Up
    "release_2024-11-04-10-57-34",  # Grindelwald - Village 1
    "release_2024-11-04-12-55-59",  # Grindelwald - Canyon 1
    "release_2024-11-04-13-07-13",  # Grindelwald - Canyon 2
    "release_2024-11-04-16-05-00",  # Grindelwald - Burglauenen 1
    "release_2024-11-11-12-07-40",  # Pilatus - Hike 1
    "release_2024-11-11-12-42-47",  # Pilatus - Hike 2
    "release_2024-11-11-14-29-44",  # Pilatus - Fraekmuentegg Forest 1
    "release_2024-11-14-11-17-02",  # Hoenggerberg - Forest 1
    "release_2024-11-14-12-01-26",  # Hoenggerberg - Forest 2
    "release_2024-11-14-13-45-37",  # Heap Testsite 1
    "release_2024-11-14-14-36-02",  # Käferberg - Forest 1
    "release_2024-11-14-16-04-09",  # Käferberg - Forest 3
    "release_2024-11-15-10-16-35",  # Triemli - Biketrail 1
    "release_2024-11-15-11-37-15",  # Albisgütli - Forest 2
    "release_2024-11-15-12-06-03",  # Albisgütli - Forest 3
    "release_2024-11-15-14-43-52",  # Leinbach - Forest 2
    "release_2024-11-18-12-05-01",  # ARCHE - Industrial
    "release_2024-11-18-15-46-05",  # ARCHE - Train
    "release_2024-12-03-13-26-40",  # SBB - Train Depot
    "release_2024-12-09-11-53-11",  # Construction - Outdoor
]

# ---- Calibration Parameters ----
DISTORTION_MODEL = "equidistant"
D = np.array(
        [-0.06226555154591874, 0.006984942920386333, -0.005291335660179726, 0.001455018149071658]
    )
K = np.array(
        [[984.8643239835407, 0.0, 946.7086445278437],
        [0.0, 984.5008837495056, 636.9616008492758],
        [0.0, 0.0, 1.0]]
    )

def rectify_fisheye(image, K, D):
    h, w = image.shape[:2]
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w, h), np.eye(3), balance=0)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2)
    undistorted = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted

# Create destination directory if needed
os.makedirs(DEST_DIR, exist_ok=True)

# Load image paths
all_image_paths = sorted(Path(SRC_DIR).rglob("*.png"))
image_paths = defaultdict(list)
for idx, img in enumerate(all_image_paths):
    if idx % FRAME_SKIP != 0:
        continue
    for scene in SELECTED_SCENES:
        if scene in str(img):
            image_paths[scene].append(str(img))
total_images = sum(len(paths) for paths in image_paths.values())

# Load or initialize metadata
if os.path.exists(METADATA_FILE):
    with open(METADATA_FILE, 'r') as f:
        metadata = json.load(f)
else:
    metadata = {}

# Track state
current_scene_index = 0
current_index = 0
selected_index = len(metadata)
exit_flag = False

# Main loop
for scene_idx, scene in enumerate(SELECTED_SCENES):
    if exit_flag:
        break
    current_index = sum(len(image_paths[s]) for s in SELECTED_SCENES[:scene_idx])
    if scene not in image_paths:
        print(f"No images found for scene: {scene}")
        continue
    print(f"Processing scene: {scene}")
    for current_scene_index in range(len(image_paths[scene])):
        img_path = image_paths[scene][current_scene_index]
        img = cv2.imread(img_path)

        if img is None:
            print(f"Failed to load image: {img_path}")
            current_index += 1
            continue
        
        # rectified_img = rectify_fisheye(img, K, D)
        # display = rectified_img.copy()
        display = img.copy()
        # resize to half the resolution
        display = cv2.resize(display, (display.shape[1]//2, display.shape[0]//2))
        cv2.putText(
            display,
            f"Scene: {current_scene_index+1}/{len(image_paths[scene])} Current: {current_index + 1}/{total_images}, Selected: {selected_index}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Image Selector", display)
        key = cv2.waitKey(0)

        if key == 27:  # ESC to quit
            exit_flag = True
            break
        elif key == 83 or key == 2555904:  # Right arrow
            current_index += 1
        elif key == 81 or key == 2424832:  # Left arrow
            # Save image to new directory
            filename = f"rgb_{selected_index:05d}.png"
            dest_path = os.path.join(DEST_DIR, filename)
            shutil.copy(img_path, dest_path)
            # cv2.imwrite(dest_path, rectified_img)

            # Update metadata
            metadata[str(selected_index)] = "/".join(str(img_path).split("/")[-3:])
            selected_index += 1

            print(f"Selected image saved: {dest_path}")
            current_index += 1

            # Save metadata after each selection
            with open(METADATA_FILE, 'w') as f:
                json.dump(metadata, f, indent=4)

        elif key == 32:  # Space to skip scene
            break
            

cv2.destroyAllWindows()
print("Selection process finished.")
