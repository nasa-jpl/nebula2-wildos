# ExploRFM Trainer

Training pipeline for ExploRFM heads, built on [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template).

This module trains the lightweight CNN deconvolution heads that sit on top of the frozen RADIO backbone for:
- **Traversability prediction** (trained on RUGD)
- **Visual frontier prediction** (trained on WildOS Frontiers / GrandTour)

## Datasets

| Dataset | Usage | Download |
|---|---|---|
| [RUGD](http://rugd.vision/) | Traversability head training | [http://rugd.vision/](http://rugd.vision/) |
| [WildOS Frontiers](https://huggingface.co/datasets/leggedrobotics/wildos) | Frontier head training (built on [GrandTour](https://grand-tour.leggedrobotics.com/)) | [HuggingFace](https://huggingface.co/datasets/leggedrobotics/wildos) |
| [GOOSE-Ex](https://goose-dataset.de/) | Traversability ablation (not used in WildOS) | [https://goose-dataset.de/](https://goose-dataset.de/) |

## Configuration

Before training, update the paths in your config files:

```yaml
# configs/paths/default.yaml or in the experiment config
paths:
  root_dir: /path/to/explorfm_trainer    # path to this folder
  data_dir: /path/to/data               # folder containing RUGD and WildOS Frontiers datasets
```

### Key Experiment Configs

| Config | Description |
|---|---|
| `configs/experiment/gtour_radio_cnn_new.yaml` | **Frontier head** training |
| `configs/experiment/rugd_radio_cnn.yaml` | **Traversability head** training |
| `configs/experiment/rugd_radio_cnn_adaptor.yaml` | Traversability with adaptor variant (*ablation*) |
| `configs/experiment/gooseex_radio_cnn.yaml` | Traversability training using GOOSE-Ex traversability (*ablation*) |

## Training

```bash
cd explorfm_trainer

# Train frontier head
python -u src/train.py experiment=gtour_radio_cnn_new

# Train traversability head
python -u src/train.py experiment=rugd_radio_cnn

# Resume from checkpoint
python -u src/train.py experiment=gtour_radio_cnn_new ckpt_path="/path/to/checkpoint.ckpt"
```

## Evaluation

```bash
cd explorfm_trainer

# Evaluate traversability head (edit configs/evaluation/radio_ovts.yaml for paths)
python -u src/eval.py evaluation=radio_ovts
```

> **Note**: `radio_ovts` refers to **Open-Vocabulary Traversability Segmentation** — a training-free ablation that queries the language-aligned RADIO backbone with a fixed set of "safe" and "risky" text labels to produce traversability maps. See [`src/models/components/radio_ovts.py`](src/models/components/radio_ovts.py) for the implementation.

## Annotation Tools

The `annotation/` directory contains tools used to build the WildOS Frontiers dataset from the [GrandTour](https://grand-tour.leggedrobotics.com/) dataset:

1. **[`select-images.py`](annotation/select-images.py)** — Interactive image selector for curating RGB training frames from GrandTour scenes. Use arrow keys to navigate and select images.

2. **[`auto_mask_gen_gtour.py`](annotation/auto_mask_gen_gtour.py)** — Generates SAM2 segmentation boundaries for the selected images. These boundaries were used in an ablation where only the SAM2 segments *within* human-annotated bounding boxes served as frontier labels (instead of full boxes). Full bounding boxes worked better and were used in the final model.

3. **[`bbox_annotate.py`](annotation/bbox_annotate.py)** — Interactive bounding box annotation tool for labeling visual frontiers. Supports multiple label types. Controls:
   - **Draw**: Click and drag to create a bounding box
   - **`n`**: Switch label type
   - **`b`**: Toggle SAM2 boundary overlay
   - **`r`**: Reset current annotations
   - **`s`** / **Enter**: Save and move to next image
   - **Space**: Skip image

## Project Structure

```
explorfm_trainer/
├── configs/                  # Hydra configuration files
│   ├── experiment/           # Experiment-specific configs
│   ├── evaluation/           # Evaluation configs
│   ├── data/                 # Datamodule configs
│   ├── model/                # Model configs
│   ├── paths/                # Path configs (edit these first!)
│   └── ...
├── src/
│   ├── train.py              # Training entry point
│   ├── eval.py               # Evaluation entry point
│   ├── data/                 # Dataset and datamodule implementations
│   │   ├── rugd_traversability_datamodule.py
│   │   ├── grandtour_frontiers_datamodule.py
│   │   └── goose-ex_traversability_datamodule.py
│   └── models/               # Model definitions
│       ├── binary_segmentation_module.py
│       ├── frontier_segmentation_module.py
│       └── components/       # Model building blocks
└── annotation/               # Annotation tools
    ├── auto_mask_gen_gtour.py
    ├── bbox_annotate.py
    └── select-images.py
```
