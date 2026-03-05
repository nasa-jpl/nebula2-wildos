#!/bin/bash

#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=trainer.out

source ~/.bashrc

conda activate radio
cd ..
# python -u src/train.py experiment=rugd_radio_cnn
# python -u src/train.py experiment=rugd_radio_cnn_adaptor
# python -u src/train.py experiment=gtour_radio_cnn_bbox
# python -u src/train.py experiment=gtour_radio_cnn ckpt_path="/cluster/home/$USER/scratch/explorfm_trainer/logs/train/runs/2025-09-14_05-03-19/checkpoints/epoch_097.ckpt"
# python -u src/train.py experiment=gtour_radio_cnn