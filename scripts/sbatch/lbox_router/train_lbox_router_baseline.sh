#!/bin/bash
#SBATCH --job-name=train_lbox_router
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/lbox_router/%x.%A_%a.log
#SBATCH --error=logs/lbox_router/%x.%A_%a.log

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
RESULTS_ROOT="${RESULTS_ROOT:?RESULTS_ROOT is required}"
cd "$REPO"
source ~/data/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-MoE}"
mkdir -p logs/lbox_router "$RESULTS_ROOT/routers"

case "${SLURM_ARRAY_TASK_ID}" in
    0) BANK=low7_high8; FEATURE=hs_mean ;;
    1) BANK=low7_high8; FEATURE=encoder ;;
    2) BANK=low5_high6; FEATURE=hs_mean ;;
    3) BANK=low5_high6; FEATURE=encoder ;;
    *) echo "ERROR: unsupported array index ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

python scripts/lbox_router/train_lbox_router_baseline.py \
    --bank-config configs/lbox_router/lbox_router_banks.json \
    --bank "$BANK" \
    --feature "$FEATURE" \
    --output-dir "$RESULTS_ROOT/routers/${BANK}_${FEATURE}" \
    --hidden 512 \
    --epochs 120 \
    --dropout 0.3 \
    --weight-decay 1e-2 \
    --learning-rate 1e-3 \
    --batch-size 256 \
    --seeds 42 1 7
