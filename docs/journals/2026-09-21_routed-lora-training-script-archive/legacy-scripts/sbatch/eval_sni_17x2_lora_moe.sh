#!/bin/bash
# Evaluate the custom 17-expert top-2 routed LoRA-MoE on SNI test.
#SBATCH --job-name=eval_sni_17x2_moe
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
export PYTHONPATH="$REPO:$REPO/src"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CHECKPOINT="${CHECKPOINT:-$REPO/checkpoints/sni_17x2_lora_moe/checkpoint-5200}"
PACKAGE_DIR="${PACKAGE_DIR:-$REPO/data/sni_moe_rosters/sni_moe_seed20212003}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO/results/sni_17x2_lora_moe_test_checkpoint5200}"

python -c 'from huggingface_hub import get_token; assert get_token(), "Hugging Face authentication is required"'
torchrun --standalone --nproc_per_node=2 "$REPO/scripts/eval_sni_17x2_lora_moe.py" \
  --checkpoint "$CHECKPOINT" \
  --package-dir "$PACKAGE_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --max-input-length 8192 \
  --max-new-tokens 2048 \
  --batch-size 1 \
  --collapse-threshold 0.90 \
  --seed 42
