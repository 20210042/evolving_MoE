#!/bin/bash
# Fallback-aware 10-specialist + 1-generalist top-2 routed LoRA-MoE on LBox.
#SBATCH --job-name=lbox_11x2_clustered_moe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:PRO6000:2
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
export WANDB_ENTITY="${WANDB_ENTITY:-cvar_ddpo}"

python -c 'from huggingface_hub import get_token; assert get_token(), "Hugging Face authentication is required"'
python -c 'import wandb; assert wandb.api.api_key, "W&B authentication is required"'

PACKAGE_DIR="${PACKAGE_DIR:-$REPO/results/lbox_binning_seed20210311}"
ASSIGNMENT_PACKAGE_DIR="${ASSIGNMENT_PACKAGE_DIR:-$REPO/export/lbox_moe_seed20210311}"
SOURCE_JSONL="${SOURCE_JSONL:-$REPO/export/lbox/lbox_train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO/checkpoints/lbox_11x2_clustered_fallback_lora_moe_5ep}"
HUB_MODEL_ID="${HUB_MODEL_ID:-Jongbin-kr/llama-3.1-8b-instruct_LBox-10x2-clustered-fallback-plus-shared-lora-moe}"
MODEL_REVISION="${MODEL_REVISION:-0e9e39f249a16976918f6564b8830bc894c89659}"

test ! -e "$OUTPUT_DIR/checkpoint-1"

torchrun --standalone --nproc_per_node=2 "$REPO/scripts/train_lbox_11x2_fallback_lora_moe.py" \
  --package-dir "$PACKAGE_DIR" --assignment-package-dir "$ASSIGNMENT_PACKAGE_DIR" \
  --source-jsonl "$SOURCE_JSONL" --output-dir "$OUTPUT_DIR" \
  --hub-model-id "$HUB_MODEL_ID" --model-revision "$MODEL_REVISION" \
  --epochs 5 --max-length 2048 --eval-steps 500 --save-steps 500 \
  --router-warmup-steps 500 --gradient-accumulation-steps 8 \
  --run-name lbox_11x2_clustered_fallback_lora_moe_5ep
