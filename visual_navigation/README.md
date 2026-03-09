# Visual Navigation

ROS 2 navigation package for WildOS and baseline implementations. This package contains the main navigation pipelines, scoring logic, and utility nodes.

## Modules

| Module | Description |
|---|---|
| `wildos/` | **WildOS** — Full navigation pipeline with ExploRFM inference, graph scoring, and object search |
| `explorfm_triangulation/` | Particle-filter-based object triangulation nodes |
| `imgfrontier_nav/` | Image frontier navigation baseline (vision-only, no geometry) |
| `lrn/` | [LRN](https://arxiv.org/abs/2504.13149) baseline (vision-only, no geometry) |
| `geofrontier_nav/` | Geometric frontier navigation with fixed goal scoring |
| `gps/` | GPS visualization and metric logging nodes |
| `utils/` | Shared scoring and utility functions |

## Key Files

| File | Description |
|---|---|
| `wildos/nav.py` | WildOS main node — runs ExploRFM inference and publishes scored navigation graph |
| `wildos/goalagnostic_scoring.py` | Goal-agnostic frontier scoring combining traversability and frontier predictions |
| `utils/scoring.py` | Graph scoring utilities shared across navigation methods |
| `explorfm_triangulation/obj_mask_triangulation.py` | Object mask triangulation (used during WildOS deployment) |
| `explorfm_triangulation/explorfm_triangulator.py` | Standalone ExploRFM triangulation node (for testing) |
| `imgfrontier_nav/viz_net.py` | ExploRFM output visualization (debugging tool) |

> See the [main README](../README.md) for launch commands and deployment instructions.

## Configuration

YAML config files for each exectuable are in `configs/`:

| Config | Used By |
|---|---|
| `wildos_nav_conf.yaml` | WildOS navigation |
| `imgfrontier_nav_conf.yaml` | Image frontier baseline |
| `lrn_nav_conf.yaml` | LRN baseline |
| `geofrontier_nav_conf.yaml` | Geometric frontier navigation |
| `explorfm_triangulator_conf.yaml` | Standalone ExploRFM triangulation |
| `triangulation3d_objsearch_conf.yaml` | Object search triangulation |

## Method Details

### WildOS
WildOS scores frontier nodes of the navigation graph using ExploRFM predictions. The scoring combines traversability (is it safe?), visual frontier confidence (where to explore?), and object similarity (does it match the query?). When `do_object_search` is enabled, the `obj_mask_triangulation` node is automatically launched to estimate coarse goal positions using a particle filter.

### Image Frontier Navigation (Baseline)
Assumes a single geometric frontier at the center-bottom pixel of each camera image. Projects a path from the bottom-center pixel to the chosen visual frontier using the depth image and sends a goal at `lookahead_dist` along the projected path to the local planner.

### LRN (Baseline)
A purely vision-based baseline that does not use geometric information for exploration. It scores angular bins around the robot using visual frontier scores and the goal heading.
