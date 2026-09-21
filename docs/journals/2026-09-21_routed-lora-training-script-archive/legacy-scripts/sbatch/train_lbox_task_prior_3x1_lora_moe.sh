#!/bin/bash
# Train LBox task-prior (civil case name / criminal charge / statute) top-1 MoE.
#SBATCH --job-name=lbox_taskprior_3x1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:PRO6000:2
# Avoid the node that terminated the previous run with a CUDA launch timeout.
#SBATCH --exclude=n04
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log

set -euo pipefail
REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
source ~/data/miniconda3/etc/profile.d/conda.sh
conda activate MoE
export PYTHONPATH="$REPO/src:$REPO"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_PROJECT="${WANDB_PROJECT:-lbox-lora-moe}"

python -c 'from huggingface_hub import get_token; assert get_token(), "Hugging Face authentication is required"'
python -c 'import wandb; assert wandb.api.api_key, "W&B authentication is required"'

SOURCE_JSONL="${SOURCE_JSONL:-$REPO/export/lbox/lbox_train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO/checkpoints/lbox_task_prior_3x1_lora_moe_5ep_retry}"
HUB_MODEL_ID="${HUB_MODEL_ID:-Jongbin-kr/llama-3.1-8b-instruct_LBox-task-prior-3x1-lora-moe}"

torchrun --standalone --nproc_per_node=2 "$REPO/scripts/train_lbox_task_prior_3x1_lora_moe.py" \
  --source-jsonl "$SOURCE_JSONL" --output-dir "$OUTPUT_DIR" --hub-model-id "$HUB_MODEL_ID" \
  --epochs 5 --top-k 1 --eval-steps 500 --save-steps 500 \
  --router-warmup-steps 500 --gradient-accumulation-steps 8 --run-name lbox_task_prior_3x1_lora_moe_5ep
