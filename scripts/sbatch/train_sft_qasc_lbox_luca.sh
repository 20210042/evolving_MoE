#!/bin/bash
#SBATCH --job-name=sft_llama3_qasc_luca
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/qasc_lbox_sft/%x.%j.log
#SBATCH --error=logs/qasc_lbox_sft/%x.%j.log

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"

source ~/.bashrc
source ~/data/miniconda3/etc/profile.d/conda.sh
conda activate MoE

export PYTHONPATH="$REPO/src"

DATASET="${DATASET:-qasc}"
DATA_DIR="${DATA_DIR:-export/${DATASET}}"
PROMPT_SYSTEM="${PROMPT_SYSTEM:-luca}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
if [[ "$DATASET" == "lbox" ]]; then
    EVAL_SPLIT="${EVAL_SPLIT:-valid}"
else
    EVAL_SPLIT="${EVAL_SPLIT:-validation}"
fi

RUN_NAME="${RUN_NAME:-sft_llama3_${DATASET}_${PROMPT_SYSTEM}_full}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/${RUN_NAME}}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${SLURM_GPUS_ON_NODE:-1}}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
EVAL_STEPS="${EVAL_STEPS:-1000}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
DATA_RATIO="${DATA_RATIO:-1.0}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
MAX_STEPS="${MAX_STEPS:-}"
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"
HUB_MODEL_ID="${HUB_MODEL_ID:-}"
LABEL_PACKAGE="${LABEL_PACKAGE:-}"
SOURCE_JSONL="${SOURCE_JSONL:-}"

# Dataset selection modes for roster-labelled SFT:
#   roster_expert_with_unanimous_sampling: EXPERT_ID's solved set, sample unanimous rows.
#   roster_expert_with_low_consensus_solved: EXPERT_ID's solved rows with low n_solved.
#   shared_with_high_consensus_solved:      high n_solved rows; EXPERT_ID is ignored.
EXPERT_DATA_MODE="${EXPERT_DATA_MODE:-roster_expert_with_unanimous_sampling}"
EXPERT_ID="${EXPERT_ID:-}"

# roster_expert_with_unanimous_sampling mode only.
UNANIMOUS_SOLVED_SAMPLING_RATIO="${UNANIMOUS_SOLVED_SAMPLING_RATIO:-1.0}"

# low/high consensus modes only. The LBox 10-expert setup uses 7 and 8.
LOW_CONSENSUS_MAX_SOLVED="${LOW_CONSENSUS_MAX_SOLVED:-7}"
HIGH_CONSENSUS_MIN_SOLVED="${HIGH_CONSENSUS_MIN_SOLVED:-8}"

EXTRA_TRAIN_ARGS=()
if [[ -n "${DEEPSPEED_CONFIG}" ]]; then
    EXTRA_TRAIN_ARGS+=(--deepspeed "${DEEPSPEED_CONFIG}")
fi
if [[ -n "${MAX_STEPS}" ]]; then
    EXTRA_TRAIN_ARGS+=(--max_steps "${MAX_STEPS}")
    EXTRA_TRAIN_ARGS+=(--load_best_model_at_end false)
fi
if [[ -n "${LABEL_PACKAGE}" ]]; then
    EXTRA_TRAIN_ARGS+=(--label_package "${LABEL_PACKAGE}")
    if [[ -n "${SOURCE_JSONL}" ]]; then
        EXTRA_TRAIN_ARGS+=(--source_jsonl "${SOURCE_JSONL}")
    fi
    EXTRA_TRAIN_ARGS+=(--expert_data_mode "${EXPERT_DATA_MODE}")
    case "${EXPERT_DATA_MODE}" in
        roster_expert_with_unanimous_sampling)
            EXTRA_TRAIN_ARGS+=(--expert_id "${EXPERT_ID}")
            EXTRA_TRAIN_ARGS+=(--unanimous_solved_sampling_ratio "${UNANIMOUS_SOLVED_SAMPLING_RATIO}")
            ;;
        roster_expert_with_low_consensus_solved)
            EXTRA_TRAIN_ARGS+=(--expert_id "${EXPERT_ID}")
            EXTRA_TRAIN_ARGS+=(--low_consensus_max_solved "${LOW_CONSENSUS_MAX_SOLVED}")
            ;;
        shared_with_high_consensus_solved)
            EXTRA_TRAIN_ARGS+=(--high_consensus_min_solved "${HIGH_CONSENSUS_MIN_SOLVED}")
            ;;
        *)
            echo "ERROR: unknown EXPERT_DATA_MODE=${EXPERT_DATA_MODE}" >&2
            exit 2
            ;;
    esac
fi
if [[ "${PUSH_TO_HUB}" == "true" || "${PUSH_TO_HUB}" == "True" ]]; then
    EXTRA_TRAIN_ARGS+=(--push_to_hub True)
    if [[ -n "${HUB_MODEL_ID}" ]]; then
        EXTRA_TRAIN_ARGS+=(--hub_model_id "${HUB_MODEL_ID}")
    fi
fi

mkdir -p logs/qasc_lbox_sft

echo "=== QASC/LBox LUCA SFT 시작 ==="
echo "=== dataset=${DATASET} prompt_system=${PROMPT_SYSTEM} train=${TRAIN_SPLIT} eval=${EVAL_SPLIT} data_dir=${DATA_DIR} ==="
echo "=== run=${RUN_NAME} output=${OUTPUT_DIR} ==="
echo "=== label_package=${LABEL_PACKAGE:-none} expert=${EXPERT_ID:-none} mode=${EXPERT_DATA_MODE} low_max=${LOW_CONSENSUS_MAX_SOLVED} high_min=${HIGH_CONSENSUS_MIN_SOLVED} ==="
echo "=== GPUs=${NPROC_PER_NODE} GA=${GRADIENT_ACCUMULATION_STEPS} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} ==="

srun --ntasks=1 --gpus-per-task="${NPROC_PER_NODE}" --chdir="$REPO" \
    torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" "$REPO/src/train_sft.py" \
    --model_name_or_path "meta-llama/Llama-3.1-8B-Instruct" \
    --dtype bfloat16 \
    --attn_implementation sdpa \
    --train_dataset "${DATASET}" \
    --eval_dataset "${DATASET}" \
    --train_split "${TRAIN_SPLIT}" \
    --eval_split "${EVAL_SPLIT}" \
    --data_dir "${DATA_DIR}" \
    --eval_data_dir "${DATA_DIR}" \
    --data_ratio "${DATA_RATIO}" \
    --prompt_system "${PROMPT_SYSTEM}" \
    --luca_roster_path configs/roster_init.json \
    --seed 42 \
    --train_sft_with_lora true \
    --sft_lora_rank 16 \
    --sft_lora_alpha 32 \
    --sft_lora_dropout 0.05 \
    --output_dir "${OUTPUT_DIR}" \
    --run_name "${RUN_NAME}" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --learning_rate 2e-5 \
    --lr_scheduler_type cosine \
    --bf16 true \
    --logging_steps 10 \
    --eval_strategy steps \
    --eval_steps "${EVAL_STEPS}" \
    --save_strategy steps \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit 2 \
    --load_best_model_at_end true \
    --metric_for_best_model eval_loss \
    --report_to wandb \
    --wandb_project sft_dense \
    "${EXTRA_TRAIN_ARGS[@]}"

echo "=== QASC/LBox LUCA SFT 완료. 체크포인트: ${OUTPUT_DIR} ==="
