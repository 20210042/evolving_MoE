#!/bin/bash
# Build the seed20211004 ACC package and submit LUCA + 11 specialist LoRA jobs in parallel.
# SOURCE_JSONL must contain the same ids as binning_train_full.binned.jsonl and must
# include execution-verified `solution` or `ground_truth` fields.
#
# Usage:
#   SOURCE_JSONL=export/acc_v2/sft/acc_train.jsonl \
#     bash scripts/sbatch/launch_acc_seed20211004_persona_sft.sh

set -euo pipefail

JOURNAL_CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "${JOURNAL_CONFIG_DIR}/../../../.." && pwd)}"
cd "${REPO}"

SOURCE_JSONL="${SOURCE_JSONL:?Set SOURCE_JSONL to the ACC train JSONL with verified solutions}"
ASSET_DIR="${ASSET_DIR:-results/acc/seed20211004}"
LABEL_PACKAGE="${LABEL_PACKAGE:-export/acc_binning_seed20211004_persona}"
if [ ! -f "${SOURCE_JSONL}" ]; then
    echo "ERROR: SOURCE_JSONL does not exist: ${SOURCE_JSONL}" >&2
    exit 1
fi

python "${JOURNAL_CONFIG_DIR}/../scripts/build_sft_label_package.py" \
    --roster "${ASSET_DIR}/roster_final.json" \
    --binned "${ASSET_DIR}/binning_train_full.binned.jsonl" \
    --source-jsonl "${SOURCE_JSONL}" \
    --output-dir "${LABEL_PACKAGE}" \
    --dataset acc

export LABEL_PACKAGE
export DATA_DIR="${DATA_DIR:-$(dirname "${SOURCE_JSONL}")}"
export EVAL_DATA_DIR="${EVAL_DATA_DIR:-${DATA_DIR}}"
export EVAL_SPLIT="${EVAL_SPLIT:-validation}"
export MAX_LENGTH="${MAX_LENGTH:-3072}"
export GRAD_CKPT="${GRAD_CKPT:-true}"
if [ ! -f "${EVAL_DATA_DIR}/acc_${EVAL_SPLIT}.jsonl" ]; then
    echo "ERROR: eval split not found: ${EVAL_DATA_DIR}/acc_${EVAL_SPLIT}.jsonl" >&2
    echo "Set EVAL_DATA_DIR and EVAL_SPLIT to a held-out ACC split." >&2
    exit 1
fi

TRAIN_SCRIPT="${REPO}/scripts/sbatch/train_sft_by_expert.sh"
HUB_ORG="${HUB_ORG:-Jongbin-kr}"
WANDB_PROJECT="${WANDB_PROJECT:-acc-seed20211004-persona-sft}"
WANDB_ENTITY="${WANDB_ENTITY:-cvar_ddpo}"
COMMON_CORE_SEED="${COMMON_CORE_SEED:-20211004}"
SPECIALISTS="${SPECIALISTS:-c_46087 c_10367 c_17316 c_4998 c_34728 c_63819 c_50585 c_16428 c_30658 c_56276 c_56422}"

COMMON_ENV=(
    LABEL_PACKAGE="${LABEL_PACKAGE}"
    DATA_DIR="${DATA_DIR}"
    EVAL_DATA_DIR="${EVAL_DATA_DIR}"
    EVAL_SPLIT="${EVAL_SPLIT}"
    MAX_LENGTH="${MAX_LENGTH}"
    GRAD_CKPT="${GRAD_CKPT}"
    NUM_TRAIN_EPOCHS=5
    EVAL_STEPS=100
    SAVE_STEPS=100
    PUSH_TO_HUB=true
    HUB_ORG="${HUB_ORG}"
    HUB_PRIVATE_REPO="${HUB_PRIVATE_REPO:-false}"
    WANDB_PROJECT="${WANDB_PROJECT}"
    WANDB_ENTITY="${WANDB_ENTITY}"
    NPROC_PER_NODE=1
    REMAINING_EXPERTS=
)

echo "=== LUCA generalist: all-pass 1000, baseline prompt ==="
LUCA_JOB_ID="$(env "${COMMON_ENV[@]}" \
    EXPERT_ID=luca \
    MIN_N_SOLVED=12 MAX_N_SOLVED=12 MAX_TRAIN_SAMPLES=1000 \
    COMMON_CORE_SIZE=0 USE_EXPERT_SYSTEM_PROMPT=false \
    RUN_NAME=acc_seed20211004_luca_allpass1000 \
    OUTPUT_DIR=checkpoints/expert_sft/acc_seed20211004/luca_allpass1000 \
    HUB_MODEL_ID="${HUB_ORG}/evolving-moe-acc-seed20211004-luca-allpass1000" \
    sbatch --parsable --job-name=acc_luca_allpass1000 --export=ALL "${TRAIN_SCRIPT}")"
echo "submitted luca: ${LUCA_JOB_ID}"

echo "=== Specialists: solved n<=8 + identical all-pass core 200, persona prompt ==="
for EXPERT in ${SPECIALISTS}; do
    JOB_ID="$(env "${COMMON_ENV[@]}" \
        EXPERT_ID="${EXPERT}" \
        MIN_N_SOLVED= MAX_N_SOLVED=8 MAX_TRAIN_SAMPLES= \
        COMMON_CORE_SIZE=200 COMMON_CORE_N_SOLVED=12 COMMON_CORE_SEED="${COMMON_CORE_SEED}" \
        USE_EXPERT_SYSTEM_PROMPT=true \
        RUN_NAME="acc_seed20211004_${EXPERT}_cap8_core200" \
        OUTPUT_DIR="checkpoints/expert_sft/acc_seed20211004/cap8_core200/${EXPERT}" \
        HUB_MODEL_ID="${HUB_ORG}/evolving-moe-acc-seed20211004-${EXPERT}-cap8-core200" \
        sbatch --parsable --job-name="acc_${EXPERT}_c8core" --export=ALL "${TRAIN_SCRIPT}")"
    echo "submitted ${EXPERT}: ${JOB_ID}"
done
