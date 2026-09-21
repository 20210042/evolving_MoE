#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKER="${SCRIPT_DIR}/eval_sft_model.sh"
OUTPUT_DIR="${REPO}/results/recent_sft_best_test"
WANDB_PROJECT="eval_recent_sft_best_test"

SNI_MODEL="${REPO}/merged_models/sni_moe_best_checkpoint-4000"
LBOX_MODEL="${REPO}/merged_models/lbox_moe_best_checkpoint-4200"

for path in "${WORKER}" "${SNI_MODEL}/config.json" \
    "${SNI_MODEL}/model.safetensors.index.json" "${LBOX_MODEL}/config.json" \
    "${LBOX_MODEL}/model.safetensors.index.json" \
    "${REPO}/data/sni_split/sni_test.jsonl" "${REPO}/export/lbox/lbox_test.jsonl"; do
    if [ ! -f "${path}" ]; then
        echo "ERROR: required file not found: ${path}" >&2
        exit 1
    fi
done

mkdir -p "${OUTPUT_DIR}" "${REPO}/logs"

submit_eval() {
    local job_name="$1"
    local run_name="$2"
    local model_path="$3"
    local dataset="$4"
    local data_dir="$5"
    local max_model_len="$6"
    local max_new_tokens="$7"

    sbatch --parsable \
        --job-name="${job_name}" \
        --mem=128G \
        --time=48:00:00 \
        --export=ALL,REPO="${REPO}",MODEL_NAME="${model_path}",LORA_PATH=__NONE__,RUN_NAME="${run_name}",TEST_DATASET="${dataset}",SPLIT=test,DATA_DIR="${data_dir}",DATA_RATIO=1.0,INFERENCE_MODE=hf,MAX_MODEL_LEN="${max_model_len}",MAX_NEW_TOKENS="${max_new_tokens}",TEMPERATURE=0.0,OUTPUT_DIR="${OUTPUT_DIR}",WANDB_PROJECT="${WANDB_PROJECT}",SEED=42,ENABLE_THINKING=false,USE_CATEGORY_PROMPT=false,PROMPT_SYSTEM=baseline \
        "${WORKER}"
}

SNI_JOB_ID="$(submit_eval eval_sni_moe_hf eval_sni_moe_best_test_hf "${SNI_MODEL}" sni "${REPO}/data/sni_split" 32768 2048)"
LBOX_JOB_ID="$(submit_eval eval_lbox_moe_hf eval_lbox_moe_best_test_hf "${LBOX_MODEL}" lbox "${REPO}/export/lbox" 8192 128)"

echo "SNI_JOB_ID=${SNI_JOB_ID}"
echo "LBOX_JOB_ID=${LBOX_JOB_ID}"
