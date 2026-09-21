#!/bin/bash
# Train one LBox evolved-roster specialist or the shared high-consensus model.
# The resulting adapter is FFN-only and is compatible with routed_lora_ffn.py.
#SBATCH --job-name=lbox_roster_ffn5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x.%A_%a.log
#SBATCH --error=logs/%x.%A_%a.log

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
source ~/data/miniconda3/etc/profile.d/conda.sh
conda activate MoE

export PYTHONPATH="$REPO/src"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python -c 'from huggingface_hub import get_token; assert get_token(), "Hugging Face authentication is required. Run: hf auth login"'
python -c 'import wandb; assert wandb.api.api_key, "Weights & Biases authentication is required. Run: wandb login"'

MANIFEST="${LBOX_JOB_MANIFEST:?Set LBOX_JOB_MANIFEST}"
TASK_INDEX="${SLURM_ARRAY_TASK_ID:?This worker must run as an array}"
ROW="$(sed -n "$((TASK_INDEX + 1))p" "$MANIFEST")"
if [ -z "$ROW" ]; then
    echo "ERROR: no manifest row for array index $TASK_INDEX" >&2
    exit 2
fi
# The manifest has four fields: role, expert id, slug, and display name.
# Consume the fourth field separately so NICKNAME remains a valid Hub slug.
IFS='|' read -r ROLE EXPERT_ID NICKNAME _DISPLAY_NAME <<< "$ROW"

MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"
PACKAGE_DIR="${LBOX_PACKAGE_DIR:-$REPO/results/lbox_binning_seed20210311}"
SOURCE_JSONL="${LBOX_SOURCE_JSONL:-$REPO/export/lbox/lbox_train.jsonl}"
DATA_DIR="${LBOX_DATA_DIR:-$REPO/export/lbox}"

if [ "$ROLE" = "generalist" ]; then
    MODE="shared_with_high_consensus_solved"
    EXPERT_ID_ARG=()
    NICKNAME="generalist"
    RUN_NAME="${RUN_NAME:-sft_llama31_8b_lbox_generalist_ffn_only_5ep_eval500}"
    HUB_MODEL_ID="${HUB_MODEL_ID:-Jongbin-kr/llama-3.1-8b-instruct_lbox-generalist_ffn-only}"
else
    MODE="roster_expert_with_low_consensus_solved"
    EXPERT_ID_ARG=(--expert_id "$EXPERT_ID")
    NICKNAME="${NICKNAME:?missing roster nickname slug}"
    RUN_NAME="${RUN_NAME:-sft_llama31_8b_lbox_${NICKNAME}_ffn_only_5ep_eval500}"
    HUB_MODEL_ID="${HUB_MODEL_ID:-Jongbin-kr/llama-3.1-8b-instruct_lbox-${NICKNAME}_ffn-only}"
fi

for required in "$PACKAGE_DIR/binning_labels.jsonl" "$PACKAGE_DIR/agent_mapping.json" "$SOURCE_JSONL" "$DATA_DIR"; do
    if [ ! -e "$required" ]; then
        echo "ERROR: missing LBox roster input: $required" >&2
        exit 2
    fi
done

OUTPUT_DIR="${OUTPUT_DIR:-$REPO/checkpoints/$RUN_NAME}"
echo "=== LBox roster FFN-only LoRA SFT: role=$ROLE expert=$EXPERT_ID ==="
echo "PACKAGE_DIR=$PACKAGE_DIR"
echo "MODE=$MODE low<=7 high>=8"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "HUB_MODEL_ID=$HUB_MODEL_ID"
echo "LoRA targets=gate_proj,up_proj,down_proj; completion_only_loss=true"

srun --ntasks=1 --gpus-per-task=1 --chdir="$REPO" \
    python "$REPO/src/train_sft.py" \
    --model_name_or_path "$MODEL_NAME" \
    --trust_remote_code true \
    --dtype bfloat16 \
    --attn_implementation sdpa \
    --train_dataset lbox \
    --eval_dataset lbox \
    --train_split train \
    --eval_split valid \
    --data_dir "$DATA_DIR" \
    --eval_data_dir "$DATA_DIR" \
    --label_package "$PACKAGE_DIR" \
    --source_jsonl "$SOURCE_JSONL" \
    --expert_data_mode "$MODE" \
    "${EXPERT_ID_ARG[@]}" \
    --low_consensus_max_solved 7 \
    --high_consensus_min_solved 8 \
    --data_ratio 1.0 \
    --seed 42 \
    --train_sft_with_lora true \
    --sft_lora_rank 16 \
    --sft_lora_alpha 32 \
    --sft_lora_dropout 0.05 \
    --sft_lora_target_modules gate_proj,up_proj,down_proj \
    --output_dir "$OUTPUT_DIR" \
    --run_name "$RUN_NAME" \
    --num_train_epochs 5 \
    --max_length 3072 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing true \
    --learning_rate 2e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --bf16 true \
    --tf32 true \
    --logging_steps 10 \
    --eval_strategy steps \
    --eval_steps 500 \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 3 \
    --load_best_model_at_end true \
    --metric_for_best_model eval_loss \
    --greater_is_better false \
    --completion_only_loss true \
    --report_to wandb \
    --wandb_project sft_dense_lbox_roster_ffn_only \
    --push_to_hub true \
    --hub_strategy every_save \
    --hub_model_id "$HUB_MODEL_ID"

echo "=== Complete; loaded best model saved and pushed: $HUB_MODEL_ID ==="
