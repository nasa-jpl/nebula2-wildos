# Triangulation3D — Particle-Filter-Based 3D Object Triangulation

ROS 2 package for coarse 3D localization of open-vocabulary target objects beyond the robot's depth horizon using a particle-filter-based approach.

## Overview

When the target object is detected in multiple camera views (via ExploRFM), this module estimates its 3D position by:

1. **Particle sampling** — Randomly sampling candidate 3D positions given an object detection in image space
2. **Multi-view triangulation** — Fusing multiple object hypotheses (projected particles) across camera views to converge on the target location

## Key Files

| File | Description |
|---|---|
| `triangulator.py` | Core triangulation logic using multiple object hypotheses and projected particles |
| `particle_generator.py` | Particle sampling from object detections |

## Usage

### Multi-Camera Triangulation

To visualize multi-camera triangulation, where each camera position is randomly generated, run in separate terminals:

```bash
ros2 run triangulation3d triangulation_visualizer
```

The `triangulation_visualizer` node will publish random camera positions and the object in space, along with the triangulated position and the particles projected from each camera which can be visualized in RViz:

![triangulation_rviz](assets/multicam_viz.png)

### Single Camera Triangulation with Teleoperation

To teleoperate the camera and triangulate the detected object, run in separate terminals:

```bash
ros2 run triangulation3d teleop_triangulation
ros2 run triangulation3d teleop_twist_keyboard
```

In the terminal running `teleop_twist_keyboard`, use the following keys to control the camera:

| Key | Action |
|---|---|
| `w` / `s` | Move forward / backward |
| `a` / `d` | Move left / right |
| `q` / `e` | Move up / down |
| `p` / `l` | Pitch up / down |
| `o` / `k` | Roll right / left |
| `i` / `j` | Yaw right / left |

Press `Ctrl+C` to stop the teleoperation.

![teleop_rviz](assets/teleop_viz.png)

## Integration with WildOS

During WildOS deployment, triangulation is handled by `visual_navigation/explorfm_triangulation/obj_mask_triangulation.py`, which receives object masks from the WildOS navigation node and uses the triangulation logic from this package to estimate goal positions.
