#!/bin/bash
# Submit one dense/generalist LoRA trained on all 11,097 ACC train examples.
# This reuses train_sft_by_expert.sh with EXPERT_ID=shared, which disables the
# per-expert solved filter and persona prompt while preserving the common SFT stack.

set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
cd "${REPO}"

SOURCE_JSONL="${SOURCE_JSONL:-export/acc_seed20211004/acc_train.jsonl}"
LABEL_PACKAGE="${LABEL_PACKAGE:-export/acc_binning_seed20211004_persona}"
DATA_DIR="${DATA_DIR:-$(dirname "${SOURCE_JSONL}")}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-${DATA_DIR}}"
EVAL_SPLIT="${EVAL_SPLIT:-validation}"
WANDB_ENTITY="${WANDB_ENTITY:-jongbin-kr-skiml_moe}"
WANDB_PROJECT="${WANDB_PROJECT:-acc-seed20211004-persona-sft}"

for REQUIRED in \
    "${SOURCE_JSONL}" \
    "${LABEL_PACKAGE}/binning_labels.jsonl" \
    "${LABEL_PACKAGE}/agent_mapping.json" \
    "${EVAL_DATA_DIR}/acc_${EVAL_SPLIT}.jsonl"; do
    if [ ! -f "${REQUIRED}" ]; then
        echo "ERROR: required file not found: ${REQUIRED}" >&2
        exit 1
    fi
done

JOB_ID="$(env \
    LABEL_PACKAGE="${LABEL_PACKAGE}" \
    EXPERT_ID=shared \
    SOURCE_JSONL="${SOURCE_JSONL}" \
    DATA_DIR="${DATA_DIR}" \
    EVAL_DATA_DIR="${EVAL_DATA_DIR}" \
    EVAL_SPLIT="${EVAL_SPLIT}" \
    MIN_N_SOLVED= MAX_N_SOLVED= MAX_TRAIN_SAMPLES= \
    COMMON_CORE_SIZE=0 USE_EXPERT_SYSTEM_PROMPT=false \
    MAX_LENGTH="${MAX_LENGTH:-3072}" GRAD_CKPT="${GRAD_CKPT:-true}" \
    NUM_TRAIN_EPOCHS=5 EVAL_STEPS=500 SAVE_STEPS=500 \
    NPROC_PER_NODE=1 REMAINING_EXPERTS= \
    RUN_NAME=acc_seed20211004_dense_all11097 \
    OUTPUT_DIR=checkpoints/dense_sft/acc_seed20211004_all \
    PUSH_TO_HUB=true \
    HUB_MODEL_ID="${HUB_MODEL_ID:-Jongbin-kr/llama3-8b_acc-seed20211004-dense-all11097}" \
    HUB_PRIVATE_REPO="${HUB_PRIVATE_REPO:-false}" \
    WANDB_ENTITY="${WANDB_ENTITY}" WANDB_PROJECT="${WANDB_PROJECT}" \
    sbatch --parsable --job-name=acc_dense_all11097 --export=ALL \
    "${REPO}/scripts/sbatch/train_sft_by_expert.sh")"

echo "submitted dense SFT: ${JOB_ID}"
