#!/bin/bash
#SBATCH --job-name=mae_eval_qasc_routed
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --exclude=n05
#SBATCH --output=/home/jaehoonjeong/data/MetaAgentEvolution_Release/logs/%x.%j.out
#SBATCH --error=/home/jaehoonjeong/data/MetaAgentEvolution_Release/logs/%x.%j.err

# QASC validation routed eval: final roster, router picks 1 of N.
# Env: SEED, EVAL_SIZE, EVAL_CONFIG, DATA_DIR, OUT_TAG.

set -euo pipefail

REPO="${REPO:-/home/jaehoonjeong/data/MetaAgentEvolution_Release}"
# shellcheck source=scripts/sbatch/common_bigmath.sh
source "${REPO}/scripts/sbatch/common_bigmath.sh"
setup_job_env
print_experiment_config

DATASET="qasc"
EVAL_CONFIG="${EVAL_CONFIG:-configs/qasc_eval_a4b.yaml}"
DATA_DIR="${DATA_DIR:-export/qasc}"
FINAL_ROSTER="${RESULTS_DIR}/roster_final.json"
EVAL_SIZE="${EVAL_SIZE:-926}"
SPLIT="${SPLIT:-validation}"
OUT_TAG="${OUT_TAG:-final}"
OUT="${RESULTS_DIR}/inference_${SPLIT}_routed_${OUT_TAG}.jsonl"

echo "=== qasc routed eval: final roster (${FINAL_ROSTER}) split=${SPLIT} size=${EVAL_SIZE} out=${OUT} ==="
[ -f "${FINAL_ROSTER}" ] || { echo "ERROR: final roster not found at ${FINAL_ROSTER}"; exit 1; }

rm -f "${OUT}"
python scripts/run_inference.py \
    --config "${EVAL_CONFIG}" \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --seed "${SEED}" \
    --data_dir "${DATA_DIR}" \
    --pipeline evolved \
    --roster_path "${FINAL_ROSTER}" \
    --max_items "${EVAL_SIZE}" \
    --output_file "${OUT}"

python scripts/score_outputs.py \
    --input "${OUT}" \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --data_dir "${DATA_DIR}"

echo "=== qasc routed eval done ==="
