#!/bin/bash
#SBATCH --job-name=expert_tsne
#SBATCH --partition=gpu
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/expert_tsne.%j.log
#SBATCH --error=logs/expert_tsne.%j.log

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
mkdir -p logs results/embed_viz_test results/expert_embedding_tsne /tmp/matplotlib-${SLURM_JOB_ID}
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

/data6/jongbinwon/.conda/envs/MoE/bin/python scripts/extract_sni_test_hidden_states.py \
  --batch-size 16 \
  --max-length 1024

/data6/jongbinwon/.conda/envs/RRAG/bin/python scripts/plot_4x1_expert_embedding_tsne.py
