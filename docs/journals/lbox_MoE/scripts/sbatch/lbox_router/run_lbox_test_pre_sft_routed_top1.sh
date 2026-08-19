#!/bin/bash
#SBATCH --job-name=lbox_pre_sft_routed_test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --exclude=n05
#SBATCH --output=logs/lbox_router/%x.%j.log
#SBATCH --error=logs/lbox_router/%x.%j.log

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
ROSTER="${ROSTER:-results/lbox_test_roster_binning/20260730_135227/roster_pre_sft.json}"
RESULTS_ROOT="${RESULTS_ROOT:-results/lbox_pre_sft_routed_test/seed20210311}"
OUTPUT="$RESULTS_ROOT/inference_test_routed_top1.jsonl"

cd "$REPO"
source /home/jongbinwon/data/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-evolving_moe}"

export PYTHONPATH="$REPO/src"
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_DISABLE_FLASHINFER=1
export HF_HOME="${HF_HOME:-/data6/jongbinwon/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"

mkdir -p logs/lbox_router "$RESULTS_ROOT"
test -f "$ROSTER"

echo "=== LBox pre-SFT persona-roster routed top-1 test ==="
echo "roster=$ROSTER output=$OUTPUT"

srun --chdir="$REPO" python scripts/run_inference.py \
  --config configs/lbox_router/lbox_eval_a4b_test_binning.yaml \
  --dataset lbox \
  --split test \
  --data_dir export/lbox \
  --pipeline evolved \
  --roster_path "$ROSTER" \
  --seed 20210311 \
  --infer_batch_size 64 \
  --ignore_test_ids \
  --output_file "$OUTPUT"

python scripts/score_outputs.py \
  --input "$OUTPUT" \
  --dataset lbox \
  --split test \
  --data_dir export/lbox

echo "=== Routed top-1 test complete: $OUTPUT ==="
