#!/bin/bash
#SBATCH --job-name=lbox_router_feat
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/lbox_router/%x.%A_%a.log
#SBATCH --error=logs/lbox_router/%x.%A_%a.log

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
source ~/data/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-MoE}"
export PYTHONPATH="$REPO/src"
mkdir -p logs/lbox_router results/embed_viz_test

case "${SLURM_ARRAY_TASK_ID}" in
    0) python scripts/extract_hidden_states.py --dataset lbox --split train --batch 16 --max_len 1024 ;;
    1) python scripts/lbox_router/extract_router_encoder_embeddings.py --split train --batch 256 ;;
    *) echo "ERROR: unsupported array index ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac
