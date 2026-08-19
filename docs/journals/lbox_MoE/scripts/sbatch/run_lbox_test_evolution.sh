#!/bin/bash
#SBATCH --job-name=mae_evolve_lbox_test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=n05
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
SEED="${SEED:-20210312}"
CONFIG="${CONFIG:-configs/lbox_test_full_seed20210312.yaml}"
RESULTS_DIR="${RESULTS_DIR:-results/lbox_test/seed${SEED}}"
RUN_ID="${RUN_ID:-lbox_test/seed${SEED}}"

cd "$REPO"
source /home/jongbinwon/data/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-evolving_moe}"

export PYTHONPATH="$REPO/src"
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_DISABLE_FLASHINFER=1
export HF_HOME="${HF_HOME:-/data6/jongbinwon/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"

mkdir -p logs "$RESULTS_DIR"

echo "=== LBox test evolution ==="
echo "repo=$REPO config=$CONFIG seed=$SEED"
echo "results=$RESULTS_DIR run_id=$RUN_ID"

python scripts/run_evolution.py \
  --config "$CONFIG" \
  --seed "$SEED" \
  --roster_path "$RESULTS_DIR/roster_final.json" \
  --results_dir "$RESULTS_DIR" \
  --run_id "$RUN_ID"

echo "=== Evolution complete: $RESULTS_DIR/roster_final.json ==="
