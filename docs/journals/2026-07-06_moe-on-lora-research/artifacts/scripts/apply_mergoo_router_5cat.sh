#!/bin/bash
#SBATCH --job-name=apply_mergoo_router_5cat
#SBATCH --gres=gpu:A6000:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"

source ~/.bashrc
source ~/data/miniconda3/etc/profile.d/conda.sh
conda activate MoE_mergoo_a6000

MODEL_PATH="${MODEL_PATH:-checkpoints/mergoo_lora_moe_5cat_top2}"
ROUTER_STATE="${ROUTER_STATE:-checkpoints/router_5cat_top2_numina_5k/router_model.safetensors}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/mergoo_lora_moe_5cat_top2_router_trained}"

python "$REPO/scripts/apply_mergoo_router.py" \
    --model_name_or_path "$MODEL_PATH" \
    --router_state "$ROUTER_STATE" \
    --output_dir "$OUTPUT_DIR" \
    --dtype bfloat16 \
    --max_seq_length_for_load 1024 \
    --num_experts_per_tok 2 \
    --max_shard_size 9GB
