#!/bin/bash
#SBATCH --job-name=compose_lora_moe_smoke
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"

source ~/.bashrc
source ~/data/miniconda3/etc/profile.d/conda.sh
conda activate MoE_mergoo

export PYTHONPATH="$REPO/src"

BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/mergoo_lora_moe_algebra_geometry_top1}"
ALGEBRA_LORA="${ALGEBRA_LORA:-checkpoints/sft_llama3_numina_cot_algebra}"
GEOMETRY_LORA="${GEOMETRY_LORA:-checkpoints/sft_llama3_numina_cot_geometry}"
NUM_EXPERTS_PER_TOK="${NUM_EXPERTS_PER_TOK:-1}"

python "$REPO/scripts/compose_lora_moe.py" \
    --base_model "$BASE_MODEL" \
    --expert "algebra=${ALGEBRA_LORA}" \
    --expert "geometry=${GEOMETRY_LORA}" \
    --num_experts_per_tok "$NUM_EXPERTS_PER_TOK" \
    --output_dir "$OUTPUT_DIR" \
    --dtype bfloat16 \
    --device_map auto
