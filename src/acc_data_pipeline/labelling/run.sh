#!/bin/bash
# """Handoff note: SLURM wrapper for the label-distribution counter. It scans the execution-ready CSV and
# writes raw original_domain label frequencies used before taxonomy normalization."""
#SBATCH --job-name=distribution_label_count
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --exclude=master
#SBATCH --time=0-01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=/home/minjikim/minji_link/code/benchmark/src/acc_data_pipeline/labelling/logs/%x_%j.out
#SBATCH --error=/home/minjikim/minji_link/code/benchmark/src/acc_data_pipeline/labelling/logs/%x_%j.err

source ~/.bashrc
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate agent

export CUDA_VISIBLE_DEVICES=""

cd /home/minjikim/minji_link/code/benchmark/src/acc_data_pipeline/labelling

python distribusion.py
