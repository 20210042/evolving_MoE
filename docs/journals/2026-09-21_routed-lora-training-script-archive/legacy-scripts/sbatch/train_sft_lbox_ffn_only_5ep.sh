#!/bin/bash
#SBATCH --job-name=sft_lbox_ffn5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log

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

LBOX_KIND="${LBOX_KIND:?Set LBOX_KIND to task or category}"
LBOX_NAME="${LBOX_NAME:?Set LBOX_NAME to the task/category slug}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"
RUN_NAME="${RUN_NAME:-sft_llama31_8b_lbox_${LBOX_NAME}_ffn_only_5ep}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO/checkpoints/$RUN_NAME}"
HUB_MODEL_ID="${HUB_MODEL_ID:-Jongbin-kr/llama-3.1-8b-instruct_lbox-${LBOX_NAME//_/-}_ffn-only}"

DATA_ARGS=()
case "$LBOX_KIND" in
    task)
        DATA_DIR="$REPO/export/lbox_tasks/$LBOX_NAME"
        DATA_ARGS=(--data_dir "$DATA_DIR")
        ;;
    category)
        DATA_DIR="$REPO/export/lbox"
        TAGS_PATH="$REPO/results/lbox_legal_category_tags/gemma4_a4b_family_patent_merged/lbox_train_legal_categories.jsonl"
        DATA_ARGS=(
            --data_dir "$DATA_DIR"
            --legal_category_tags_path "$TAGS_PATH"
            --legal_category "$LBOX_NAME"
        )
        ;;
    *)
        echo "ERROR: unsupported LBOX_KIND=$LBOX_KIND (expected task or category)" >&2
        exit 2
        ;;
esac

if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: data directory does not exist: $DATA_DIR" >&2
    exit 2
fi
if [ "$LBOX_KIND" = category ] && [ ! -f "$TAGS_PATH" ]; then
    echo "ERROR: legal-category tags do not exist: $TAGS_PATH" >&2
    exit 2
fi

echo "=== LBox FFN-only LoRA SFT: $LBOX_NAME ($LBOX_KIND) ==="
echo "MODEL_NAME=$MODEL_NAME"
echo "DATA_DIR=$DATA_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "HUB_MODEL_ID=$HUB_MODEL_ID"
echo "LoRA targets=gate_proj,up_proj,down_proj"
echo "completion_only_loss=true"
echo "GPUs=${SLURM_GPUS_ON_NODE:-1}, CPUs=${SLURM_CPUS_PER_TASK:-2}, memory=${SLURM_MEM_PER_NODE:-unknown}MB"

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
    "${DATA_ARGS[@]}" \
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
    --wandb_project sft_dense_lbox_ffn_only \
    --push_to_hub true \
    --hub_strategy every_save \
    --hub_model_id "$HUB_MODEL_ID"

echo "=== Complete; loaded best model saved and pushed: $HUB_MODEL_ID ==="
