#!/bin/bash
#SBATCH --job-name=eval_vanilla_gemma4_numina_cot
#SBATCH --gres=gpu:PRO6000:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log

set -euo pipefail

export VLLM_USE_FLASHINFER_SAMPLER=0

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"

source ~/.bashrc
source ~/data/miniconda3/etc/profile.d/conda.sh
conda activate MoE

export PYTHONPATH="$REPO/src"

RUN_NAME="vanilla_gemma4_numina_cot"

echo "=== vanilla gemma-4 평가 시작 (numina_cot) ==="

srun --chdir="$REPO" python "$REPO/src/evaluate.py" \
    --model_name_or_path "google/gemma-4-31B-it" \
    --test_dataset numina_cot \
    --data_ratio 1.0 \
    --inference_mode vllm \
    --max_model_len 32768 \
    --max_new_tokens 16384 \
    --temperature 0.0 \
    --output_dir "results/${RUN_NAME}" \
    --wandb_run_name "${RUN_NAME}" \
    --wandb_project "gemma4-31b" \
    --seed 42

echo "=== 평가 완료 → results/${RUN_NAME} ==="
