#!/bin/bash
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=48:00:00
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

MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"
LORA_PATH="${LORA_PATH:-__NONE__}"
RUN_NAME="${RUN_NAME:-eval_$(basename "${LORA_PATH}")}"
TEST_DATASET="${TEST_DATASET:-numina_cot}"
DATA_RATIO="${DATA_RATIO:-1.0}"
INFERENCE_MODE="${INFERENCE_MODE:-vllm}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16384}"
TEMPERATURE="${TEMPERATURE:-0.0}"
OUTPUT_DIR="${OUTPUT_DIR:-results}"
WANDB_PROJECT="${WANDB_PROJECT:-eval_numina_cot}"
SEED="${SEED:-42}"
ENABLE_THINKING="${ENABLE_THINKING:-false}"

LORA_FLAG=()
if [ "${LORA_PATH}" != "__NONE__" ] && [ -n "${LORA_PATH}" ]; then
    LORA_FLAG=(--finetuned_lora_path "${LORA_PATH}")
fi

THINKING_FLAG=()
if [ "${ENABLE_THINKING}" = "true" ] || [ "${ENABLE_THINKING}" = "1" ]; then
    THINKING_FLAG=(--enable_thinking)
fi

echo "=== 평가 시작: ${RUN_NAME} ==="
echo "MODEL_NAME=${MODEL_NAME}"
echo "LORA_PATH=${LORA_PATH}"
echo "TEST_DATASET=${TEST_DATASET}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unknown}"

srun --chdir="$REPO" python "$REPO/src/evaluate.py" \
    --model_name_or_path "${MODEL_NAME}" \
    "${LORA_FLAG[@]}" \
    --test_dataset "${TEST_DATASET}" \
    --data_ratio "${DATA_RATIO}" \
    --inference_mode "${INFERENCE_MODE}" \
    --max_model_len "${MAX_MODEL_LEN}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --output_dir "${OUTPUT_DIR}" \
    --wandb_run_name "${RUN_NAME}" \
    --wandb_project "${WANDB_PROJECT}" \
    "${THINKING_FLAG[@]}" \
    --seed "${SEED}"

echo "=== 평가 완료: ${RUN_NAME} → ${OUTPUT_DIR} ==="
