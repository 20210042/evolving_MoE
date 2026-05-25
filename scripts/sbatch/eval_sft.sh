#!/bin/bash
#SBATCH --job-name=eval_sft
#SBATCH --nodelist=n02
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"

source /data6/jongbinwon/miniconda3/etc/profile.d/conda.sh
conda activate MoE

export PYTHONPATH="$REPO/src"

MODEL="${MODEL:?MODEL 환경변수를 지정하세요. e.g. MODEL=checkpoints/my_sft}"
DATASET="${DATASET:-bigmath}"
SPLIT="${SPLIT:-test}"
CATEGORIES="${CATEGORIES:-}"
RUN_NAME="${RUN_NAME:-eval_${DATASET}_$(basename $MODEL)_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-results/eval_sft}"
WANDB_PROJECT="${WANDB_PROJECT:-sft_eval}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
DATA_RATIO="${DATA_RATIO:-1.0}"

CATEGORIES_FLAG=""
if [ -n "${CATEGORIES}" ]; then
    CATEGORIES_FLAG="--categories ${CATEGORIES}"
fi

srun --chdir="$REPO" python "$REPO/src/evaluate.py" \
    --model "${MODEL}" \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --max_tokens "${MAX_TOKENS}" \
    --temperature 0.0 \
    --tp_size 1 \
    --data_ratio "${DATA_RATIO}" \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_run_name "${RUN_NAME}" \
    --output_dir "${OUTPUT_DIR}" \
    ${CATEGORIES_FLAG}

echo "=== 평가 완료. 결과: ${OUTPUT_DIR} ==="
