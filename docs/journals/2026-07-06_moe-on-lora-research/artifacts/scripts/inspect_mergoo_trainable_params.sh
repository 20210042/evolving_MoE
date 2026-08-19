#!/bin/bash
#SBATCH --job-name=inspect_mergoo_trainable
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"

source ~/.bashrc
source ~/data/miniconda3/etc/profile.d/conda.sh
conda activate MoE_mergoo

export PYTHONPATH="$REPO/src"

CHECKPOINT="${CHECKPOINT:-checkpoints/mergoo_lora_moe_algebra_geometry_top1}"
OUTPUT_JSON="${OUTPUT_JSON:-results/mergoo_trainable_params/algebra_geometry_top1.json}"

python "$REPO/scripts/inspect_mergoo_trainable_params.py" \
    --checkpoint "$CHECKPOINT" \
    --dtype bfloat16 \
    --device_map cpu \
    --rope_compat linear \
    --write_json "$OUTPUT_JSON"
