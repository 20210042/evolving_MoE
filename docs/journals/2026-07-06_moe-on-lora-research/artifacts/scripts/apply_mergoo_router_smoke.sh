#!/bin/bash
#SBATCH --job-name=apply_mergoo_router_smoke
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

python "$REPO/scripts/apply_mergoo_router.py" \
    --model_name_or_path checkpoints/mergoo_lora_moe_algebra_geometry_top1 \
    --router_state checkpoints/router_smoke_algebra_geometry_top1/router_model.safetensors \
    --output_dir checkpoints/mergoo_lora_moe_algebra_geometry_top2_router_smoke \
    --dtype bfloat16 \
    --max_seq_length_for_load 1024 \
    --num_experts_per_tok 2 \
    --max_shard_size 9GB
