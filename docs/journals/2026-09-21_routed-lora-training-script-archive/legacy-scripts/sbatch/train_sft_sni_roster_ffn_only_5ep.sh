#!/bin/bash
#SBATCH --job-name=sni_roster_ffn5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=2-00:00:00
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

if [ -n "${SNI_JOB_MANIFEST:-}" ]; then
    if [ ! -f "$SNI_JOB_MANIFEST" ]; then
        echo "ERROR: SNI job manifest does not exist: $SNI_JOB_MANIFEST" >&2
        exit 2
    fi
    task_index="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required with SNI_JOB_MANIFEST}"
    manifest_row="$(sed -n "$((task_index + 1))p" "$SNI_JOB_MANIFEST")"
    if [ -z "$manifest_row" ]; then
        echo "ERROR: no manifest row for array index $task_index" >&2
        exit 2
    fi
    IFS='|' read -r _source_rows _arm EXPERT_NICKNAME TRAIN_FILE MAX_SOLVED <<< "$manifest_row"
    if [ "$MAX_SOLVED" -ge 0 ]; then
        NSOLVED_PATH="${NSOLVED_PATH:?Set NSOLVED_PATH for filtered expert rows}"
    else
        NSOLVED_PATH=""
    fi
fi

TRAIN_FILE="${TRAIN_FILE:?Set TRAIN_FILE to one roster expert/shared JSONL}"
EXPERT_NICKNAME="${EXPERT_NICKNAME:?Set EXPERT_NICKNAME}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"
MAX_SOLVED="${MAX_SOLVED:--1}"
NSOLVED_PATH="${NSOLVED_PATH:-}"
RUN_NAME="${RUN_NAME:-sft_llama31_8b_sni_${EXPERT_NICKNAME}_ffn_only_5ep}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO/checkpoints/$RUN_NAME}"
HUB_MODEL_ID="${HUB_MODEL_ID:-Jongbin-kr/llama-3.1-8b-instruct_SNI-${EXPERT_NICKNAME}_ffn-only}"

if [ ! -f "$TRAIN_FILE" ]; then
    echo "ERROR: roster train file does not exist: $TRAIN_FILE" >&2
    exit 2
fi

FILTER_ARGS=(--sni_roster_max_solved "$MAX_SOLVED")
if [ -n "$NSOLVED_PATH" ]; then
    if [ ! -f "$NSOLVED_PATH" ]; then
        echo "ERROR: n_solved reference does not exist: $NSOLVED_PATH" >&2
        exit 2
    fi
    FILTER_ARGS+=(--sni_roster_n_solved_path "$NSOLVED_PATH")
fi

echo "=== SNI roster FFN-only LoRA SFT: $EXPERT_NICKNAME ==="
echo "MODEL_NAME=$MODEL_NAME"
echo "TRAIN_FILE=$TRAIN_FILE"
echo "MAX_SOLVED=$MAX_SOLVED"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "HUB_MODEL_ID=$HUB_MODEL_ID"
echo "LoRA targets=gate_proj,up_proj,down_proj"
echo "completion_only_loss=true"

srun --ntasks=1 --gpus-per-task=1 --chdir="$REPO" \
    python "$REPO/src/train_sft.py" \
    --model_name_or_path "$MODEL_NAME" \
    --trust_remote_code true \
    --dtype bfloat16 \
    --attn_implementation sdpa \
    --train_dataset sni_roster \
    --eval_dataset sni_roster \
    --eval_split valid \
    --sni_roster_train_path "$TRAIN_FILE" \
    --sni_roster_eval_data_dir "$REPO/data/sni_split" \
    "${FILTER_ARGS[@]}" \
    --sni_roster_eval_ratio 0.1 \
    --sni_roster_eval_max_samples 512 \
    --seed 42 \
    --train_sft_with_lora true \
    --sft_lora_rank 16 \
    --sft_lora_alpha 32 \
    --sft_lora_dropout 0.05 \
    --sft_lora_target_modules gate_proj,up_proj,down_proj \
    --output_dir "$OUTPUT_DIR" \
    --run_name "$RUN_NAME" \
    --num_train_epochs 5 \
    --max_length 16384 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing true \
    --learning_rate 2e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --bf16 true \
    --tf32 true \
    --logging_steps 10 \
    --eval_strategy steps \
    --eval_steps 100 \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 3 \
    --load_best_model_at_end true \
    --metric_for_best_model eval_loss \
    --greater_is_better false \
    --completion_only_loss true \
    --report_to wandb \
    --wandb_project sft_dense_sni_roster_ffn_only \
    --push_to_hub true \
    --hub_strategy every_save \
    --hub_model_id "$HUB_MODEL_ID"

echo "=== Complete; loaded best model saved and pushed: $HUB_MODEL_ID ==="
