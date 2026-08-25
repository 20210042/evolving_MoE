#!/bin/bash
# Archived experiment launcher. The reusable evaluator is scripts/sbatch/eval_sft_model.sh.

set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
cd "${REPO}"

DATA_DIR="${DATA_DIR:-export/acc_seed20211004}"
WANDB_ENTITY="${WANDB_ENTITY:-jongbin-kr-skiml_moe}"
WANDB_PROJECT="${WANDB_PROJECT:-acc-seed20211004-vanilla-eval}"
OUTPUT_DIR="${OUTPUT_DIR:-results/acc/seed20211004/vanilla}"
EVAL_SCRIPT="${REPO}/scripts/sbatch/eval_sft_model.sh"

if [ ! -f "${DATA_DIR}/acc_test.jsonl" ]; then
    echo "ERROR: test dataset not found: ${DATA_DIR}/acc_test.jsonl" >&2
    exit 1
fi

mkdir -p logs "${OUTPUT_DIR}"

submit_eval() {
    local job_name="$1"
    local model_name="$2"
    local run_name="$3"
    local memory="$4"

    env \
        REPO="${REPO}" \
        MODEL_NAME="${model_name}" \
        LORA_PATH=__NONE__ \
        RUN_NAME="${run_name}" \
        TEST_DATASET=acc SPLIT=test DATA_DIR="${DATA_DIR}" DATA_RATIO=1.0 \
        INFERENCE_MODE=vllm MAX_MODEL_LEN=16384 MAX_NEW_TOKENS=8192 \
        TEMPERATURE=0.0 OUTPUT_DIR="${OUTPUT_DIR}" SEED=20211004 \
        ENABLE_THINKING=false USE_CATEGORY_PROMPT=false \
        WANDB_ENTITY="${WANDB_ENTITY}" WANDB_PROJECT="${WANDB_PROJECT}" \
        sbatch --parsable --job-name="${job_name}" --mem="${memory}" --export=ALL \
        "${EVAL_SCRIPT}"
}

LLAMA_JOB="$(submit_eval \
    acc_vanilla_llama3_8b \
    meta-llama/Llama-3.1-8B-Instruct \
    acc_seed20211004_vanilla_llama3_8b \
    32G)"

GEMMA_JOB="$(submit_eval \
    acc_vanilla_gemma4_a4b \
    google/gemma-4-26B-A4B-it \
    acc_seed20211004_vanilla_gemma4_26b_a4b \
    64G)"

echo "submitted vanilla llama3-8b eval: ${LLAMA_JOB}"
echo "submitted vanilla gemma4-26b-a4b eval: ${GEMMA_JOB}"
