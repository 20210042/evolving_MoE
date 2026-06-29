#!/bin/bash
# """Handoff note: SLURM job for training one all-critic ACC algorithm LoRA adapter.
# It builds the local ACC dataset if needed, then launches `src/train_sft_algorithm.py` on the full
# balanced train split using the Llama base model and writes checkpoints under `checkpoints/`."""
#SBATCH --job-name=sft_acc_all_alg
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
mkdir -p "$REPO/logs"

source ~/.bashrc
if [ -f "$HOME/data/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/data/miniconda3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    echo "ERROR: conda not found"
    exit 1
fi
conda activate MoE

export PYTHONPATH="$REPO/src"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export HF_HOME="/home/minjikim/minji_link/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE"

DATA_DIR="${DATA_DIR:-$REPO/data/acc_algorithm}"
if [ ! -f "$DATA_DIR/acc_algorithm_train.jsonl" ]; then
    echo "=== ACC algorithm data not found. Building: $DATA_DIR ==="
    python "$REPO/scripts/build_acc_sft_dataset_algorithm.py" --output-dir "$DATA_DIR"
fi

RUN_NAME="${RUN_NAME:-sft_acc_algorithm_all_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/${RUN_NAME}}"
PUSH_TO_HUB="${PUSH_TO_HUB:-False}"
HUB_MODEL_ID="${HUB_MODEL_ID:-meta-llama/Llama-3.1-8B-Instruct}"

srun --chdir="$REPO" python "$REPO/src/train_sft_algorithm.py" \
    --model_name_or_path "meta-llama/Llama-3.1-8B-Instruct" \
    --dtype bfloat16 \
    --attn_implementation eager \
    --device_map auto \
    --train_dataset acc_algorithm \
    --eval_dataset acc_algorithm \
    --data_dir "$DATA_DIR" \
    --data_ratio 1.0 \
    --seed 42 \
    --train_sft_with_lora true \
    --sft_lora_rank 16 \
    --sft_lora_alpha 32 \
    --sft_lora_dropout 0.05 \
    --sft_lora_target_modules all-linear \
    --output_dir "$OUTPUT_DIR" \
    --run_name "$RUN_NAME" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --lr_scheduler_type cosine \
    --bf16 true \
    --logging_steps 10 \
    --eval_strategy steps \
    --eval_steps 200 \
    --save_strategy steps \
    --save_steps 200 \
    --save_total_limit 3 \
    --load_best_model_at_end true \
    --metric_for_best_model eval_loss \
    --report_to none \
    --wandb_project sft_dense \
    --push_to_hub "$PUSH_TO_HUB" \
    --hub_model_id "$HUB_MODEL_ID"

echo "=== ACC all-critic SFT complete: ${OUTPUT_DIR} ==="
