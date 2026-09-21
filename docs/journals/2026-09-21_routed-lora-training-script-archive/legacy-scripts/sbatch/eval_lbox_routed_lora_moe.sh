#!/bin/bash
# Evaluate one completed routed-LoRA LBox model on the local test split.
#SBATCH --job-name=lbox_routed_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G
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

: "${CHECKPOINT:?Set CHECKPOINT to the best routed-LoRA checkpoint}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for predictions and summary}"
PACKAGE_DIR="${PACKAGE_DIR:-$REPO/results/lbox_binning_seed20210311}"
TEST_JSONL="${TEST_JSONL:-$REPO/export/lbox/lbox_test.jsonl}"
MODEL_REVISION="${MODEL_REVISION:-0e9e39f249a16976918f6564b8830bc894c89659}"

python -c 'from huggingface_hub import get_token; assert get_token(), "Hugging Face authentication is required"'
python "$REPO/scripts/eval_lbox_routed_lora_moe.py" \
  --checkpoint "$CHECKPOINT" --package-dir "$PACKAGE_DIR" --test-jsonl "$TEST_JSONL" \
  --output-dir "$OUTPUT_DIR" --model-revision "$MODEL_REVISION" \
  --max-input-length 8192 --max-new-tokens 256
