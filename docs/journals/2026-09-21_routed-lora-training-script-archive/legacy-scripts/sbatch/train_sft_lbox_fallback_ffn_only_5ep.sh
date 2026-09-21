#!/bin/bash
# Train one prebuilt LBox fallback expert with FFN-only LoRA SFT.
# This script is intended to run as an array worker.
#SBATCH --job-name=lbox_fallback_ffn5
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
export WANDB_ENTITY="${WANDB_ENTITY:-cvar_ddpo}"
export WANDB_PROJECT="${WANDB_PROJECT:-sft_dense_lbox_roster_ffn_only}"

python -c 'from huggingface_hub import get_token; assert get_token(), "Hugging Face authentication is required"'
python -c 'import wandb; assert wandb.api.api_key, "W&B authentication is required"'

PACKAGE_DIR="${LBOX_FALLBACK_PACKAGE_DIR:-$REPO/export/lbox_moe_seed20210311}"
MANIFEST="${LBOX_FALLBACK_JOB_MANIFEST:?Set LBOX_FALLBACK_JOB_MANIFEST}"
TASK_INDEX="${SLURM_ARRAY_TASK_ID:?This worker must run as an array}"
ROW="$(sed -n "$((TASK_INDEX + 1))p" "$MANIFEST")"
if [ -z "$ROW" ]; then
    echo "ERROR: no manifest row for array index $TASK_INDEX" >&2
    exit 2
fi
IFS='|' read -r TRAIN_REL NICKNAME DISPLAY_NAME <<< "$ROW"

TRAIN_JSONL="$PACKAGE_DIR/$TRAIN_REL"
EVAL_JSONL="$PACKAGE_DIR/valid.jsonl"
RUN_NAME="${RUN_NAME:-sft_llama31_8b_lbox_${NICKNAME}_fallback_ffn_only_5ep_eval500}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO/checkpoints/$RUN_NAME}"
HUB_MODEL_ID="${HUB_MODEL_ID:-Jongbin-kr/llama-3.1-8b-instruct_lbox-${NICKNAME}_fallback_ffn-only}"

for required in "$TRAIN_JSONL" "$EVAL_JSONL"; do
    if [ ! -f "$required" ]; then
        echo "ERROR: missing fallback package input: $required" >&2
        exit 2
    fi
done

echo "=== LBox fallback FFN-only SFT ==="
echo "display_name=$DISPLAY_NAME train=$TRAIN_JSONL eval=$EVAL_JSONL"
echo "output=$OUTPUT_DIR hub=$HUB_MODEL_ID"
echo "LoRA targets=gate_proj,up_proj,down_proj; completion_only_loss=true; epochs=5"

srun --ntasks=1 --gpus-per-task=1 --chdir="$REPO" \
    python "$REPO/src/train_sft.py" \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --trust_remote_code true \
    --dtype bfloat16 \
    --attn_implementation sdpa \
    --train_dataset lbox \
    --eval_dataset lbox \
    --train_split train \
    --eval_split valid \
    --data_dir "$PACKAGE_DIR" \
    --eval_data_dir "$PACKAGE_DIR" \
    --prebuilt_train_jsonl "$TRAIN_JSONL" \
    --prebuilt_eval_jsonl "$EVAL_JSONL" \
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
    --wandb_project "$WANDB_PROJECT" \
    --wandb_entity cvar_ddpo \
    --push_to_hub true \
    --hub_strategy every_save \
    --hub_model_id "$HUB_MODEL_ID"

echo "=== Complete; best fallback FFN-only model saved and pushed: $HUB_MODEL_ID ==="
