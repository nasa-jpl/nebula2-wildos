# VLMs on Spot
This package is a testbed for different VLMs on the camera feed from a Spot robot.

## Supported Models
- [Grounded-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything)
    Query the image using a text prompt to get the segmentation masks. RAM is used for grounding the text prompt to the image 
    and get "text tags" for the image - objects appeared in the image. The text tags are used as prompts for Grounded-SAM to get the segmentation masks.
    - [Recognize-Anything (RAM)](https://github.com/OPPOMKLab/recognize-anything)
    - [Grounding-DINO](https://github.com/IDEA-Research/GroundingDINO)
- [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2)


## Installation
In your ros2 workspace, clone this repository:
```bash
git clone git@hydra.robotics.caltech.edu:nebula2/sandbox/planning/img_vlms.git
```

To install the dependencies in the `nebula` docker, switch to the `img_vlms` branch of [core_docker](https://hydra.robotics.caltech.edu/nebula2/integration/core/core_docker/-/tree/img_vlms?ref_type=heads) and build the docker image:
```bash
run_docker_nebula2.py --image_tag nebula2-developer:amd64
```

To install third-party repositories, run the following command in the `img_vlms` directory:
```bash
git submodule init --recursive
git submodule update

cd img_vlms/third_party

# SAM
cd Grounded-Segment-Anything && mkdir checkpoints && cd checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
wget https://huggingface.co/spaces/xinyu1205/Tag2Text/resolve/main/ram_swin_large_14m.pth
wget https://huggingface.co/xinyu1205/recognize-anything-plus-model/blob/main/ram_plus_swin_large_14m.pth
wget https://huggingface.co/spaces/xinyu1205/Tag2Text/resolve/main/tag2text_swin_14m.pth

# Grounded-SAM-2
cd ../../cd Grounded-SAM-2 && cd checkpoints
bash download_ckpts.sh
cd ../ && cd gdino_checkpoints
bash download_ckpts.sh

# Grounding-DINO
# for the first time(during installation), run the following command to fix the .cu file
cd ../../Grounded-Segment-Anything/GroundingDINO/groundingdino/models/GroundingDINO/csrc/MsDeformAttn/
sed -i 's/value.type()/value.scalar_type()/g' ms_deform_attn_cuda.cu
sed -i 's/value.scalar_type().is_cuda()/value.is_cuda()/g' 
```

## Usage
The following must be run in the `third_party` directory of the `img_vlms` package every time you start a new docker container.
This installs the thrid_party packages in the PYTHONPATH of the docker container, so they can be imported directly without needing to
use relative imports in the code.
```bash
# run everytime you start a new docker container
cd ../../
bash install_deps.sh
```

To play the bag file with the Spot camera feed, run the following command:
```bash
ros2 bag play ../bags/spot1-20250417-153615U/ --remap /spot1/tf:=/tf /spot1/tf_static:=/tf_static
```

To run RADIO, run the following command:
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ros2 run img_vlms img_radio
```

To run SAM-2 on the Spot camera feed, run the following command:
```bash
ros2 run img_vlms img_sam
```

To run Grounding-DINO with RAM/RAM++ run:
```bash
ros2 run img_vlms img_ramp_gdino
```

To visualize DINOv2 PCA features, run:
```bash
ros2 run img_vlms img_dinov2_pca
```