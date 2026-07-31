#!/bin/bash
#SBATCH --job-name=summarize_lbox_router
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/lbox_router/%x.%j.log
#SBATCH --error=logs/lbox_router/%x.%j.log

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
RESULTS_ROOT="${RESULTS_ROOT:?RESULTS_ROOT is required}"
cd "$REPO"
source ~/data/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-MoE}"
python scripts/lbox_router/summarize_lbox_router_baselines.py --results-root "$RESULTS_ROOT"
