#!/bin/bash
#SBATCH --job-name=mae_eval_lbox_routed
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --exclude=n05
#SBATCH --output=/home/jaehoonjeong/data/MetaAgentEvolution_Release/logs/%x.%j.out
#SBATCH --error=/home/jaehoonjeong/data/MetaAgentEvolution_Release/logs/%x.%j.err

# LBox Open valid routed eval:
#   (1) top-1 routing
#   (2) top-2 routing + union scoring
# Env: SEED, EVAL_SIZE, EVAL_CONFIG, EVAL_CONFIG_TOP2, DATA_DIR.

set -euo pipefail

REPO="${REPO:-/home/jaehoonjeong/data/MetaAgentEvolution_Release}"
SEED="${SEED:-20210301}"
DATASET="${DATASET:-lbox}"
# shellcheck source=scripts/sbatch/common_bigmath.sh
source "${REPO}/scripts/sbatch/common_bigmath.sh"
setup_job_env
print_experiment_config

EVAL_CONFIG="${EVAL_CONFIG:-configs/lbox_eval_a4b.yaml}"
EVAL_CONFIG_TOP2="${EVAL_CONFIG_TOP2:-configs/lbox_eval_a4b_top2.yaml}"
DATA_DIR="${DATA_DIR:-export/lbox}"
FINAL_ROSTER="${RESULTS_DIR}/roster_final.json"
EVAL_SIZE="${EVAL_SIZE:-7651}"
SPLIT="${SPLIT:-valid}"
OUT_TOP1="${RESULTS_DIR}/inference_${SPLIT}_routed_final.jsonl"
OUT_TOP2="${RESULTS_DIR}/inference_${SPLIT}_routed_top2.jsonl"

echo "=== lbox routed eval: final roster (${FINAL_ROSTER}) split=${SPLIT} size=${EVAL_SIZE} ==="
[ -f "${FINAL_ROSTER}" ] || { echo "ERROR: final roster not found at ${FINAL_ROSTER}"; exit 1; }

echo "=========================================================================="
echo "=== [1/2] routed top-1 ==="
echo "=========================================================================="
rm -f "${OUT_TOP1}"
python scripts/run_inference.py \
    --config "${EVAL_CONFIG}" \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --seed "${SEED}" \
    --data_dir "${DATA_DIR}" \
    --pipeline evolved \
    --roster_path "${FINAL_ROSTER}" \
    --max_items "${EVAL_SIZE}" \
    --output_file "${OUT_TOP1}"

python scripts/score_outputs.py \
    --input "${OUT_TOP1}" \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --data_dir "${DATA_DIR}"

echo "=========================================================================="
echo "=== [2/2] routed top-2 union ==="
echo "=========================================================================="
rm -f "${OUT_TOP2}"
python scripts/run_inference.py \
    --config "${EVAL_CONFIG_TOP2}" \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --seed "${SEED}" \
    --data_dir "${DATA_DIR}" \
    --pipeline evolved \
    --roster_path "${FINAL_ROSTER}" \
    --max_items "${EVAL_SIZE}" \
    --output_file "${OUT_TOP2}"

python scripts/score_outputs_topk.py \
    --input "${OUT_TOP2}" \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --data_dir "${DATA_DIR}"

echo "=== lbox routed eval done (top-1 + top-2) ==="
