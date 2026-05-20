#!/bin/bash
#SBATCH --job-name=mae_infer_mbpp
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=12:00:00
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

ROSTER="$REPO/results/mbpp/seed${SEED}/roster_final.json"

if [ ! -f "${ROSTER}" ]; then
    echo "ERROR: roster not found at ${ROSTER}"
    echo "Run eval_mbpp_train_evolution.sh first."
    exit 1
fi

echo "=== [1/4] Inference: MBPP test ==="
python scripts/run_inference.py \
    --dataset mbpp \
    --split test \
    --seed "${SEED}" \
    --model "${MODEL}" \
    --roster_path "${ROSTER}" \
    --output_file "results/mbpp/seed${SEED}/inference_test.jsonl" \
    --data_dir "${DATA_ROOT}"

echo "=== [2/4] Inference: HumanEval ==="
python scripts/run_inference.py \
    --dataset humaneval \
    --split test \
    --seed "${SEED}" \
    --model "${MODEL}" \
    --roster_path "${ROSTER}" \
    --output_file "results/humaneval/seed${SEED}/inference.jsonl" \
    --data_dir "${DATA_ROOT}"

echo "=== [3/4] Score: MBPP test ==="
python scripts/score_outputs.py \
    --input "results/mbpp/seed${SEED}/inference_test.jsonl" \
    --dataset mbpp \
    --split test \
    --data_dir "${DATA_ROOT}"

echo "=== [4/4] Score: HumanEval ==="
python scripts/score_outputs.py \
    --input "results/humaneval/seed${SEED}/inference.jsonl" \
    --dataset humaneval \
    --split test \
    --data_dir "${DATA_ROOT}"

echo "=== Done. Results under results/{mbpp,humaneval}/seed${SEED}/ ==="
