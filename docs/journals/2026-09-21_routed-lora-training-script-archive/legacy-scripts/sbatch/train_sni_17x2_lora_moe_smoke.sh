#!/bin/bash
#SBATCH --job-name=sni_17x2_lora_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:PRO6000:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
source ~/data/miniconda3/etc/profile.d/conda.sh
conda activate MoE

export PYTHONPATH="$REPO:$REPO/src"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_MODE=disabled

torchrun --standalone --nproc_per_node=2 "$REPO/scripts/train_sni_17x2_lora_moe.py" \
    --package-dir "$REPO/data/sni_moe_rosters/sni_moe_seed20212003" \
    --output-dir "$REPO/checkpoints/sni_17x2_lora_moe_smoke" \
    --max-length 1024 \
    --max-steps 20 \
    --gradient-accumulation-steps 2 \
    --router-warmup-steps 5 \
    --eval-per-specialist 1 \
    --eval-generalist 8 \
    --eval-steps 10 \
    --save-steps 10 \
    --logging-steps 1 \
    --run-name sni_17x2_lora_moe_smoke
