<div align="center">

# WildOS: Open-Vocabulary Object Search in the Wild

[Hardik Shah](https://hardik01shah.github.io/)<sup>1,2</sup>,
[Erica Tevere](https://www-robotics.jpl.nasa.gov/who-we-are/people/erica-tevere/)<sup>1</sup>,
[Deegan Atha](https://www.linkedin.com/in/deeganatha/)<sup>3</sup>,
[Marcel Kaufmann](http://www.kaufmann.space/)<sup>1</sup>,
[Shehryar Khattak](https://www.linkedin.com/in/shehryar-khattak)<sup>3</sup>,
[Manthan Patel](https://manthan99.github.io/)<sup>2</sup>,
[Marco Hutter](https://rsl.ethz.ch/the-lab/people/person-detail.MTIxOTEx.TGlzdC8yNDQxLC0xNDI1MTk1NzM1.html)<sup>2</sup>,
[Jonas Frey](https://jonasfrey96.github.io/)<sup>2,4,5</sup>,
[Patrick Spieler](https://www-robotics.jpl.nasa.gov/who-we-are/people/patrick-spieler/)<sup>1</sup>

<sup>1</sup>Jet Propulsion Laboratory (JPL), NASA &nbsp;&nbsp;
<sup>2</sup>Robotics Systems Lab, ETH Zurich &nbsp;&nbsp;
<sup>3</sup>Field AI (Work done while at JPL)

<sup>4</sup>Stanford University &nbsp;&nbsp;
<sup>5</sup>University of California, Berkeley

[![arXiv](https://img.shields.io/badge/arXiv-2602.19308-b31b1b.svg)](https://arxiv.org/abs/2602.19308)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://leggedrobotics.github.io/wildos/)
[![Videos](https://img.shields.io/badge/Experiment-Videos-red?logo=youtube)](https://www.youtube.com/playlist?list=PLE-BQwvVGf8HjidqjQSML1E4tP20daDNS)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97-Dataset-yellow)](https://huggingface.co/datasets/leggedrobotics/wildos)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

</div>

---

<div align="center">
  <img src="assets/teaser.svg" alt="WildOS Teaser" width="90%"/>
</div>

## Abstract

Autonomous navigation in complex, unstructured outdoor environments requires robots to operate over long ranges without prior maps and limited depth sensing. In such settings, relying solely on geometric frontiers for exploration is often insufficient; the ability to reason semantically about *where to go* and *what is safe to traverse* is crucial for robust, efficient exploration.

This work presents **WildOS**, a unified system for long-range, open-vocabulary object search that combines safe geometric exploration with semantic visual reasoning. WildOS builds a sparse navigation graph to maintain spatial memory, while utilizing a foundation-model-based vision module, **ExploRFM**, to score frontier nodes of the graph. ExploRFM simultaneously predicts traversability, visual frontiers, and object similarity in image space, enabling real-time, onboard semantic navigation tasks. The resulting vision-scored graph enables the robot to explore semantically meaningful directions while ensuring geometric safety.

Furthermore, we introduce a **particle-filter-based method for coarse localization** of the open-vocabulary target query, that estimates candidate goal positions beyond the robot's immediate depth horizon, enabling effective planning toward distant goals. Extensive closed-loop field experiments across diverse off-road and urban terrains demonstrate that WildOS enables robust navigation, significantly outperforming purely geometric and purely vision-based baselines in both efficiency and autonomy.

---

## Repository Structure

```
wildos/
├── nvidia_radio/          # Modified RADIO backbone with NACLIP + SigLIP2 alignment
├── explorfm/              # ExploRFM model (inference): frontiers, traversability, object similarity
├── explorfm_trainer/      # Training pipeline for ExploRFM heads (Lightning + Hydra)
├── visual_navigation/     # ROS 2 navigation: WildOS, baselines (LRN, ImgFrontierNav)
├── triangulation3d/       # Particle-filter-based 3D object triangulation
├── graphnav_planner/      # Graph-based path planner (C++)
├── graphnav_msgs/         # ROS 2 message definitions for navigation graph
├── object_search_msgs/    # ROS 2 message definitions for object search
├── gps_visualization/     # GPS path visualization (ROS 2 C++)
└── ckpts/                 # Model checkpoints
```

Each package has its own README with additional details. See the [Component Overview](#component-overview) section below.

---

## Installation

### Prerequisites

- **ROS 2 Jazzy** (tested)
- **Python >= 3.10**
- **CUDA-capable GPU** (ExploRFM trained on *NVIDIA GeForce RTX 4090*, deployed on *NVIDIA Jetson AGX Orin* GPU)

### 1. Create a Virtual Environment

```bash
python3 -m venv wildos_venv
source wildos_venv/bin/activate
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Local Packages

```bash
pip install -e ./nvidia_radio
pip install -e ./explorfm
```

### 4. Build ROS 2 Packages

```bash
# From your colcon workspace (with this repo cloned/symlinked into src/)
colcon build --packages-select graphnav_msgs object_search_msgs gps_visualization graphnav_planner triangulation3d visual_navigation
source install/setup.bash
```

> **Note**: WildOS was deployed inside a Docker container during field experiments. The dependencies above can be replicated in a virtual environment for development.

---

## Checkpoints

Pre-trained head checkpoints are included in `ckpts/`:

| Checkpoint | Description |
|---|---|
| `ckpts/frontier_head.ckpt` | Visual frontier prediction head |
| `ckpts/trav_head.ckpt` | Traversability prediction head |

### Download Backbone & Adaptor Weights

1. **C-RADIOv3-B backbone** — download to `ckpts/`:
   ```bash
   # Download from: https://huggingface.co/nvidia/C-RADIOv3-B/blob/main/c-radio_v3-b_half.pth.tar
   wget -P ckpts/ https://huggingface.co/nvidia/C-RADIOv3-B/resolve/main/c-radio_v3-b_half.pth.tar
   ```

2. **SigLIP2 adaptor** — download to `ckpts/siglip2/`:
   ```bash
   mkdir -p ckpts/siglip2
   # Download all files from: https://huggingface.co/google/siglip2-so400m-patch16-naflex/tree/main
   ```

> **Path configuration**: All nodes in `visual_navigation` expect the `ckpts/` folder to be at `Path.home() / ckpts`.

---

## Quick Start: Deployment

### Launch WildOS (Full Pipeline)

```bash
# Launch WildOS with open-vocabulary object search
ros2 launch visual_navigation wildos_launch.py ns:=spot1 do_object_search:=true

# Launch the graph planner
ros2 launch graphnav_planner graphnav_planner.launch.yml ns:=spot1
```

### Launch Baselines

```bash
# Image Frontier Navigation baseline
ros2 launch visual_navigation imgfrontier_nav_launch.py ns:=spot1 do_object_search:=true

# LRN baseline
ros2 launch visual_navigation lrn_launch.py ns:=spot1 do_object_search:=false
```

### Standalone Tools

```bash
# Standalone ExploRFM triangulation (for testing, with teleoperation)
ros2 launch visual_navigation explorfm_triangulation_launch.py robot_namespace:=spot1

# Visualize ExploRFM outputs (debugging)
ros2 run visual_navigation viz_net
```

> All experiment videos are available on [YouTube](https://www.youtube.com/playlist?list=PLE-BQwvVGf8HjidqjQSML1E4tP20daDNS).

### Required External Components

The following packages must be running alongside WildOS:

- [**Elevation Mapping CuPy**](https://github.com/leggedrobotics/elevation_mapping_cupy) — GPU based local 2.5D mapping
- [**DLIO**](https://github.com/vectr-ucla/direct_lidar_inertial_odometry) — LiDAR-inertial odometry
- [**Nav2**](https://github.com/ros-navigation/navigation2) — local planning and control
- **Graph Construction** - code will be released in a future update.

---

## Component Overview

| Package | Description | Details |
|---|---|---|
| [`nvidia_radio/`](nvidia_radio/) | Modified [RADIO](https://github.com/NVlabs/RADIO) backbone with NACLIP + SigLIP2 language alignment | [README](nvidia_radio/README.md) |
| [`explorfm/`](explorfm/) | ExploRFM model — predicts traversability, visual frontiers, and object similarity | [README](explorfm/README.md) |
| [`explorfm_trainer/`](explorfm_trainer/) | Lightning + Hydra training pipeline for ExploRFM heads | [README](explorfm_trainer/README.md) |
| [`visual_navigation/`](visual_navigation/) | ROS 2 navigation: WildOS pipeline, baselines (LRN, ImgFrontierNav), scoring, triangulation | [README](visual_navigation/README.md) |
| [`triangulation3d/`](triangulation3d/) | Particle-filter-based 3D object triangulation | [README](triangulation3d/README.md) |
| [`graphnav_planner/`](graphnav_planner/) | C++ graph-based path planner | — |
| [`graphnav_msgs/`](graphnav_msgs/) | ROS 2 message definitions for navigation graph | — |
| [`object_search_msgs/`](object_search_msgs/) | ROS 2 message definitions for object search | — |
| [`gps_visualization/`](gps_visualization/) | GPS path visualization (ROS 2 C++) | — |


---

## Citation

If you find this work useful, please cite:

```bibtex
@misc{shah2026wildosopenvocabularyobjectsearch,
      title={WildOS: Open-Vocabulary Object Search in the Wild}, 
      author={Hardik Shah and Erica Tevere and Deegan Atha and Marcel Kaufmann and Shehryar Khattak and Manthan Patel and Marco Hutter and Jonas Frey and Patrick Spieler},
      year={2026},
      eprint={2602.19308},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2602.19308}, 
}
```

---

## Acknowledgements

We thank the authors of the following works for open-sourcing their code:

- [NVIDIA RADIO](https://github.com/NVlabs/RADIO)
- [RayFronts](https://github.com/RayFronts/RayFronts)
- [NACLIP](https://github.com/sinahmr/NACLIP)
- [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template)

We also thank the authors of [LRN](https://arxiv.org/abs/2504.13149) for sharing their code, which was helpful in setting up the baseline.

---

## License

This project is released under the [Apache 2.0 License](LICENSE).
