#!/bin/bash

#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=eval2.out

source ~/.bashrc

conda activate radio
cd ..
python -u src/eval.py evaluation=radio_ovts
# python -u src/eval.py evaluation=radio_ovts_clip