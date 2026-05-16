#!/bin/bash
#SBATCH --job-name=smoke_test_release
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err

set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
source /data5/jaehoonjeong/miniconda3/etc/profile.d/conda.sh
conda activate pro6000

export PYTHONPATH="$REPO/src"

echo "Starting Smoke Test: Evolution (tiny run; may require GPU + LCB path)"
python scripts/run_evolution.py \
    --config configs/mbpp.yaml \
    --model Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --batch_size 2 \
    --train_size 4 \
    --epochs 1 \
    --seed 0 \
    --run_id smoke \
    --roster_path results/smoke_roster.json \
    --data_dir "${DATA_DIR:-/home/jaehoonjeong/data/MultiAgent}"

echo "Starting Smoke Test: Inference (optional; requires evolved roster)"
python scripts/run_inference.py \
    --dataset mbpp \
    --model Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --roster_path results/smoke_roster.json \
    --output_file results/smoke_eval.jsonl \
    --data_dir "${DATA_DIR:-/home/jaehoonjeong/data/MultiAgent}" \
    || true

echo "Smoke Test script finished (inference may be skipped if no GPU)."
