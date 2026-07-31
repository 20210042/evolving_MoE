#!/bin/bash
#SBATCH --job-name=lbox_task_router
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/lbox_router/%x.%j.log
#SBATCH --error=logs/lbox_router/%x.%j.log

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
cd "$REPO"
source ~/data/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-MoE}"
export PYTHONPATH="$REPO/src:$REPO/scripts"
mkdir -p logs/lbox_router "$OUTPUT_DIR"

python scripts/lbox_router/train_lbox_task_prior_router.py \
    --output-dir "$OUTPUT_DIR" \
    --hidden 512 \
    --epochs 120 \
    --dropout 0.3 \
    --weight-decay 1e-2 \
    --learning-rate 1e-3 \
    --batch-size 256 \
    --seed 42
