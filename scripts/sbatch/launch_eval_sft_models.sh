#!/bin/bash
set -euo pipefail

# Launch one SLURM job per model/LoRA so evaluations can run in parallel.
# Usage:
#   bash scripts/sbatch/launch_eval_sft_models.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKER="${SCRIPT_DIR}/eval_sft_model.sh"

MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"
TEST_DATASET="${TEST_DATASET:-numina_cot}"
DATA_RATIO="${DATA_RATIO:-1.0}"
INFERENCE_MODE="${INFERENCE_MODE:-vllm}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16384}"
TEMPERATURE="${TEMPERATURE:-0.0}"
WANDB_PROJECT="${WANDB_PROJECT:-eval_numina_cot}"
WANDB_ENTITY="${WANDB_ENTITY:-jongbin-kr-skiml_moe}"
SEED="${SEED:-42}"
OUTPUT_DIR="${OUTPUT_DIR:-results/llama3_numina_cot_more}"
ENABLE_THINKING="${ENABLE_THINKING:-false}"

LORA_PATHS=(
    "Jongbin-kr/llama3_NuminaCoT_more_calculus"
    "Jongbin-kr/llama3_NuminaCoT_more_combinatorics"
    "Jongbin-kr/llama3_NuminaCoT_more_number_theory"
    "Jongbin-kr/llama3_NuminaCoT_more_geometry"
    "Jongbin-kr/llama3_NuminaCoT_more_algebra"
)

if [ ! -f "${WORKER}" ]; then
    echo "ERROR: worker script not found: ${WORKER}" >&2
    exit 1
fi

for LORA_PATH in "${LORA_PATHS[@]}"; do
    if [ -z "${LORA_PATH}" ]; then
        RUN_NAME="vanilla_llama3"
        LORA_EXPORT="__NONE__"
    else
        RUN_NAME="$(basename "${LORA_PATH}")"
        LORA_EXPORT="${LORA_PATH}"
    fi

    JOB_NAME="eval_${RUN_NAME}"
    echo "Submitting ${JOB_NAME}"
    sbatch \
        --job-name="${JOB_NAME}" \
        --export=ALL,REPO="${REPO}",MODEL_NAME="${MODEL_NAME}",LORA_PATH="${LORA_EXPORT}",RUN_NAME="${RUN_NAME}",TEST_DATASET="${TEST_DATASET}",DATA_RATIO="${DATA_RATIO}",INFERENCE_MODE="${INFERENCE_MODE}",MAX_MODEL_LEN="${MAX_MODEL_LEN}",MAX_NEW_TOKENS="${MAX_NEW_TOKENS}",TEMPERATURE="${TEMPERATURE}",OUTPUT_DIR="${OUTPUT_DIR}",WANDB_PROJECT="${WANDB_PROJECT}",WANDB_ENTITY="${WANDB_ENTITY}",SEED="${SEED}",ENABLE_THINKING="${ENABLE_THINKING}" \
        "${WORKER}"
done
