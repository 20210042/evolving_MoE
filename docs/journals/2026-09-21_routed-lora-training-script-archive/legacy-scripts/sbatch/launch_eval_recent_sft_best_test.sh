#!/bin/bash
set -euo pipefail

# Evaluate the three recent best FFN-LoRA SFT checkpoints on their matching test sets.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKER="${SCRIPT_DIR}/eval_sft_model.sh"
OUTPUT_DIR="${REPO}/results/recent_sft_best_test"
WANDB_PROJECT="eval_recent_sft_best_test"

DENSE_SNI_LORA="${REPO}/checkpoints/sft_sni_llama31_8b_ffn_lora/checkpoint-4350"
MOE_SNI_LORA="${REPO}/checkpoints/sft_sni_llama31_8b_4x1_moe_ffn_lora/checkpoint-4000"
MOE_LBOX_LORA="/data6/jongbinwon/pytorch_upcycling/outputs/lbox-ffn-lora-sft-5ep-top1/checkpoint-4200"

for path in "${WORKER}" "${DENSE_SNI_LORA}/adapter_config.json" \
    "${MOE_SNI_LORA}/adapter_config.json" "${MOE_LBOX_LORA}/adapter_config.json" \
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
    local model_name="$3"
    local lora_path="$4"
    local dataset="$5"
    local data_dir="$6"
    local max_model_len="$7"
    local max_new_tokens="$8"

    sbatch \
        --job-name="${job_name}" \
        --mem=64G \
        --time=24:00:00 \
        --export=ALL,REPO="${REPO}",MODEL_NAME="${model_name}",LORA_PATH="${lora_path}",RUN_NAME="${run_name}",TEST_DATASET="${dataset}",SPLIT=test,DATA_DIR="${data_dir}",DATA_RATIO=1.0,INFERENCE_MODE=vllm,MAX_MODEL_LEN="${max_model_len}",MAX_NEW_TOKENS="${max_new_tokens}",TEMPERATURE=0.0,OUTPUT_DIR="${OUTPUT_DIR}",WANDB_PROJECT="${WANDB_PROJECT}",SEED=42,ENABLE_THINKING=false,USE_CATEGORY_PROMPT=false,PROMPT_SYSTEM=baseline \
        "${WORKER}"
}

submit_eval \
    eval_sni_dense_best \
    eval_sni_dense_best_test \
    meta-llama/Llama-3.1-8B-Instruct \
    "${DENSE_SNI_LORA}" \
    sni \
    "${REPO}/data/sni_split" \
    32768 \
    2048

submit_eval \
    eval_sni_moe_best \
    eval_sni_moe_best_test \
    Jongbin-kr/llama-3.1-8b-instruct-4x1-moe \
    "${MOE_SNI_LORA}" \
    sni \
    "${REPO}/data/sni_split" \
    32768 \
    2048

submit_eval \
    eval_lbox_moe_best \
    eval_lbox_moe_best_test \
    Jongbin-kr/llama-3.1-8b-instruct-4x1-moe \
    "${MOE_LBOX_LORA}" \
    lbox \
    "${REPO}/export/lbox" \
    8192 \
    128
