#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${SCRIPT_DIR}/train_sft_sni_ffn_lora.sh}"

sbatch --job-name=sft_sni_llama8b \
    --export=ALL,MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct,RUN_NAME=sft_sni_llama31_8b_ffn_lora,HUB_MODEL_ID=Jongbin-kr/llama-3.1-8b-instruct-sni-ffn-lora \
    "$TRAIN_SCRIPT"

sbatch --job-name=sft_sni_moe4x1 \
    --export=ALL,MODEL_NAME=Jongbin-kr/llama-3.1-8b-instruct-4x1-moe,RUN_NAME=sft_sni_llama31_8b_4x1_moe_ffn_lora,HUB_MODEL_ID=Jongbin-kr/llama-3.1-8b-instruct-4x1-moe-sni-ffn-lora \
    "$TRAIN_SCRIPT"
