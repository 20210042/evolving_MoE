#!/bin/bash
#SBATCH --job-name=eval_gpt5mini_bigmath
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"

source ~/.bashrc
source ~/data/miniconda3/etc/profile.d/conda.sh
conda activate MoE

export PYTHONPATH="$REPO/src"


MODEL="gpt-5-mini-2025-08-07"
RUN_NAME="eval_${MODEL}_bigmath_reasoning"

echo "=== 모델: ${RUN_NAME} ==="

srun --chdir="$REPO" python "$REPO/src/evaluate.py" \
    --model_name_or_path "${MODEL}" \
    --test_dataset bigmath \
    --data_ratio 1.0 \
    --inference_mode api \
    --max_new_tokens 16384 \
    --enable_thinking \
    --output_dir "results/${RUN_NAME}" \
    --wandb_run_name "${RUN_NAME}" \
    --wandb_project "gpt5mini-bigmath" \
    --seed 42

echo "=== 완료: ${RUN_NAME} → results/${RUN_NAME} ==="
