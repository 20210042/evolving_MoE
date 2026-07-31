#!/bin/bash
#SBATCH --job-name=lbox_router_test_hs
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/lbox_router/%x.%j.log
#SBATCH --error=logs/lbox_router/%x.%j.log

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
source ~/data/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-MoE}"
export PYTHONPATH="$REPO/src"
mkdir -p logs/lbox_router results/embed_viz_test

python scripts/extract_hidden_states.py \
    --dataset lbox \
    --split test \
    --batch 16 \
    --max_len 1024
