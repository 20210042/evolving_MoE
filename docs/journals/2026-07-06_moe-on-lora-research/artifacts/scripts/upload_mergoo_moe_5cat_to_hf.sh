#!/bin/bash
#SBATCH --job-name=upload_mergoo_moe_5cat
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"

source ~/.bashrc
source ~/data/miniconda3/etc/profile.d/conda.sh
conda activate MoE_mergoo_a6000

: "${HF_REPO_ID:?Set HF_REPO_ID, for example HF_REPO_ID=username/mergoo-lora-moe-5cat-top2-router-trained}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/mergoo_lora_moe_5cat_top2_router_trained}"

python "$REPO/scripts/upload_mergoo_moe_to_hf.py" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --repo_id "$HF_REPO_ID" \
    ${HF_PRIVATE_REPO:+--private}
