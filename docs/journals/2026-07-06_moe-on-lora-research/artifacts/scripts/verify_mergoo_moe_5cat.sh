#!/bin/bash
#SBATCH --job-name=verify_mergoo_moe_5cat
#SBATCH --gres=gpu:A6000:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"

source ~/.bashrc
source ~/data/miniconda3/etc/profile.d/conda.sh
conda activate MoE_mergoo_a6000

CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/mergoo_lora_moe_5cat_top2_router_trained}"

python "$REPO/scripts/verify_mergoo_moe_checkpoint.py" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --dtype bfloat16 \
    --max_seq_length_for_load 1024 \
    --write_json "$CHECKPOINT_DIR/verification_summary.json"
