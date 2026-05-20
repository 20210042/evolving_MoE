#!/bin/bash
#SBATCH --job-name=mae_evolve_mbpp_train
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err

set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO"
mkdir -p logs

CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
# shellcheck source=/dev/null
source "$CONDA_SH"
conda activate "${CONDA_ENV:-MoE}"

export PYTHONPATH="$REPO/src"
DATA_ROOT="${DATA_DIR:-$REPO/data}"
SEED="${SEED:-20210042}"
MODEL="${MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"

RESULTS_DIR="results/mbpp/seed${SEED}"

RESUME_FLAG=""
if [ "${RESUME:-false}" = "true" ] || [ "${1:-}" = "--resume" ]; then
    RESUME_FLAG="--resume"
    echo "=== Resuming Evolution from Checkpoint ==="
fi

python scripts/run_evolution.py \
    --config configs/mbpp_train.yaml \
    --model "${MODEL}" \
    --data_dir "${DATA_ROOT}" \
    --seed "${SEED}" \
    --roster_path "${RESULTS_DIR}/roster_final.json" \
    --results_dir "${RESULTS_DIR}" \
    --run_id "mbpp/seed${SEED}" \
    ${RESUME_FLAG}

echo "=== Evolution done. Results under ${RESULTS_DIR}/ ==="
