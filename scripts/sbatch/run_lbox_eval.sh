#!/bin/bash
#SBATCH --job-name=mae_eval_lbox
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --exclude=n05
#SBATCH --output=/home/jaehoonjeong/data/MetaAgentEvolution_Release/logs/%x.%j.out
#SBATCH --error=/home/jaehoonjeong/data/MetaAgentEvolution_Release/logs/%x.%j.err

# LBox Open valid eval:
#   (1) LUCA baseline routed pass@1
#   (2) final roster UB via binning union
# Env: SEED, EVAL_SIZE, EVAL_CONFIG, LUCA_ROSTER, DATA_DIR.

set -euo pipefail

REPO="${REPO:-/home/jaehoonjeong/data/MetaAgentEvolution_Release}"
# shellcheck source=scripts/sbatch/common_bigmath.sh
source "${REPO}/scripts/sbatch/common_bigmath.sh"
setup_job_env
print_experiment_config

DATASET="lbox"
EVAL_CONFIG="${EVAL_CONFIG:-configs/lbox_eval_a4b.yaml}"
LUCA_ROSTER="${LUCA_ROSTER:-configs/roster_init.json}"
DATA_DIR="${DATA_DIR:-export/lbox}"
FINAL_ROSTER="${RESULTS_DIR}/roster_final.json"
EVAL_SIZE="${EVAL_SIZE:-500}"
SPLIT="${SPLIT:-valid}"

echo "=== lbox eval: config=${EVAL_CONFIG} split=${SPLIT} size=${EVAL_SIZE} data_dir=${DATA_DIR} ==="
echo "=== LUCA=${LUCA_ROSTER}  FINAL=${FINAL_ROSTER} ==="

echo "=========================================================================="
echo "=== [1/2] LUCA baseline (single-agent routed) ==="
echo "=========================================================================="
rm -f "${RESULTS_DIR}/inference_${SPLIT}_luca.jsonl"
python scripts/run_inference.py \
    --config "${EVAL_CONFIG}" \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --seed "${SEED}" \
    --data_dir "${DATA_DIR}" \
    --pipeline evolved \
    --roster_path "${LUCA_ROSTER}" \
    --max_items "${EVAL_SIZE}" \
    --output_file "${RESULTS_DIR}/inference_${SPLIT}_luca.jsonl"

python scripts/score_outputs.py \
    --input "${RESULTS_DIR}/inference_${SPLIT}_luca.jsonl" \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --data_dir "${DATA_DIR}"

echo "=========================================================================="
echo "=== [2/2] Final roster UB (binning, all experts solve every problem) ==="
echo "=========================================================================="
[ -f "${FINAL_ROSTER}" ] || { echo "ERROR: final roster not found at ${FINAL_ROSTER}"; exit 1; }
rm -f "${RESULTS_DIR}/inference_${SPLIT}_binning_final.jsonl"
python scripts/run_inference.py \
    --config "${EVAL_CONFIG}" \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --seed "${SEED}" \
    --data_dir "${DATA_DIR}" \
    --pipeline binning \
    --roster_path "${FINAL_ROSTER}" \
    --max_items "${EVAL_SIZE}" \
    --output_file "${RESULTS_DIR}/inference_${SPLIT}_binning_final.jsonl"

python scripts/score_binning.py \
    --input "${RESULTS_DIR}/inference_${SPLIT}_binning_final.jsonl" \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --data_dir "${DATA_DIR}"

echo "=== lbox eval done (LUCA baseline + final-roster UB) ==="
