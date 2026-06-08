#!/bin/bash
#SBATCH --job-name=mae_eval_bigmath
#SBATCH --gres=gpu:PRO6000:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/home/jaehoonjeong/data/MetaAgentEvolution_Release/logs/%x.%j.out
#SBATCH --error=/home/jaehoonjeong/data/MetaAgentEvolution_Release/logs/%x.%j.err

# BigMath eval: MoE evolved roster + luca baseline, 500 test problems

set -euo pipefail

REPO="${REPO:-/home/jaehoonjeong/data/MetaAgentEvolution_Release}"
source "${REPO}/scripts/sbatch/common_bigmath.sh"
setup_job_env

RESULTS_DIR="results/bigmath/seed${SEED}"
EVAL_DIR="${RESULTS_DIR}/eval"
ROSTER_PATH="${RESULTS_DIR}/roster_final.json"
LUCA_ROSTER="${REPO}/configs/roster_init.json"

mkdir -p "${EVAL_DIR}"

if [ ! -f "${ROSTER_PATH}" ]; then
    echo "ERROR: roster_final.json not found at ${ROSTER_PATH}"
    exit 1
fi

echo "=== Eval seed${SEED}: MoE evolved roster ==="
python scripts/run_inference.py \
    --config configs/bigmath_train_nothink.yaml \
    --dataset bigmath --split test \
    --pipeline evolved \
    --roster_path "${ROSTER_PATH}" \
    --seed "${SEED}" \
    --max_items 500 \
    --output_file "${EVAL_DIR}/inference_moe.jsonl"

python scripts/score_outputs.py \
    --input "${EVAL_DIR}/inference_moe.jsonl" \
    --dataset bigmath --split test

echo "=== Eval seed${SEED}: luca baseline ==="
python scripts/run_inference.py \
    --config configs/bigmath_train_nothink.yaml \
    --dataset bigmath --split test \
    --pipeline evolved \
    --roster_path "${LUCA_ROSTER}" \
    --seed "${SEED}" \
    --max_items 500 \
    --output_file "${EVAL_DIR}/inference_luca.jsonl"

python scripts/score_outputs.py \
    --input "${EVAL_DIR}/inference_luca.jsonl" \
    --dataset bigmath --split test

echo "=== Eval done. Results in ${EVAL_DIR}/ ==="
