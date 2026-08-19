#!/bin/bash
#SBATCH --job-name=train_mergoo_router_ag
#SBATCH --gres=gpu:A6000:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"

source ~/.bashrc
source ~/data/miniconda3/etc/profile.d/conda.sh
conda activate MoE_mergoo_a6000

export PYTHONPATH="$REPO/src"

MODEL_PATH="${MODEL_PATH:-checkpoints/mergoo_lora_moe_algebra_geometry_top1}"
RUN_NAME="${RUN_NAME:-router_algebra_geometry_top2_numina_2k}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/${RUN_NAME}}"

python "$REPO/src/train_mergoo_router.py" \
    --model_name_or_path "$MODEL_PATH" \
    --dtype bfloat16 \
    --train_dataset numina_cot \
    --eval_dataset numina_cot \
    --categories Algebra Geometry \
    --data_ratio 1.0 \
    --max_train_samples 2048 \
    --max_eval_samples 128 \
    --max_seq_length 1024 \
    --output_dir "$OUTPUT_DIR" \
    --run_name "$RUN_NAME" \
    --seed 42 \
    --do_train true \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --bf16 true \
    --logging_steps 10 \
    --eval_strategy no \
    --save_strategy no \
    --report_to none \
    --router_num_experts_per_tok_for_training 2 \
    --save_full_model false
