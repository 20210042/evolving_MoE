#!/bin/bash
#SBATCH --job-name=mae_full_binning
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --exclude=n05
#SBATCH --output=/home/jaehoonjeong/data/MetaAgentEvolution_Release/logs/%x.%j.out
#SBATCH --error=/home/jaehoonjeong/data/MetaAgentEvolution_Release/logs/%x.%j.err

# Full split per-expert binning for QASC/LBox train-labeling or labeled eval splits.
# Produces:
#   ${OUT}
#   ${OUT%.jsonl}.binned.jsonl
#   ${OUT%.jsonl}.binned.summary.json
#   ${OUT%.jsonl}.binned.agent_solves.json
#
# Required env: DATASET, SEED, SPLIT, EVAL_CONFIG
# Optional env : DATA_DIR, ROSTER, EVAL_SIZE, IBS, OUT_TAG, IGNORE_TEST_IDS

set -euo pipefail

REPO="${REPO:-/home/jaehoonjeong/data/MetaAgentEvolution_Release}"
: "${DATASET:?Set DATASET, e.g. qasc or lbox}"
: "${SEED:?Set SEED, e.g. 20210201 or 20210301}"
: "${SPLIT:?Set SPLIT, e.g. validation or valid}"
: "${EVAL_CONFIG:?Set EVAL_CONFIG, e.g. configs/qasc_eval_a4b.yaml}"

# shellcheck source=scripts/sbatch/common_bigmath.sh
source "${REPO}/scripts/sbatch/common_bigmath.sh"
setup_job_env
print_experiment_config

DATA_DIR="${DATA_DIR:-export/${DATASET}}"
ROSTER="${ROSTER:-${RESULTS_DIR}/roster_final.json}"
IBS="${IBS:-64}"
OUT_TAG="${OUT_TAG:-full}"
IGNORE_TEST_IDS="${IGNORE_TEST_IDS:-1}"
OUT="${RESULTS_DIR}/binning_${SPLIT}_${OUT_TAG}.jsonl"

if [ ! -f "${ROSTER}" ]; then
    echo "ERROR: roster not found at ${ROSTER}"
    exit 1
fi

echo "=== Full binning: dataset=${DATASET} split=${SPLIT} seed=${SEED} ==="
echo "=== config=${EVAL_CONFIG} roster=${ROSTER} data_dir=${DATA_DIR} size=${EVAL_SIZE:-all} infer_batch=${IBS} ==="

rm -f "${OUT}" "${OUT%.jsonl}.binned.jsonl" "${OUT%.jsonl}.binned.summary.json" "${OUT%.jsonl}.binned.agent_solves.json"

cmd=(
  python scripts/run_inference.py
  --config "${EVAL_CONFIG}"
  --dataset "${DATASET}"
  --split "${SPLIT}"
  --data_dir "${DATA_DIR}"
  --pipeline binning
  --roster_path "${ROSTER}"
  --seed "${SEED}"
  --infer_batch_size "${IBS}"
  --output_file "${OUT}"
)
if [ "${IGNORE_TEST_IDS}" = "1" ]; then
  cmd+=(--ignore_test_ids)
fi
if [ -n "${EVAL_SIZE:-}" ]; then
  cmd+=(--max_items "${EVAL_SIZE}")
fi
"${cmd[@]}"

python scripts/score_binning.py \
  --input "${OUT}" \
  --dataset "${DATASET}" \
  --split "${SPLIT}" \
  --data_dir "${DATA_DIR}"

python scripts/export_binning_solve_index.py \
  --input "${OUT%.jsonl}.binned.jsonl" \
  --dataset "${DATASET}" \
  --split "${SPLIT}" \
  --data_dir "${DATA_DIR}"

echo "=== Full binning done ==="
echo "Raw outputs : ${OUT}"
echo "Labels      : ${OUT%.jsonl}.binned.jsonl"
echo "Agent index : ${OUT%.jsonl}.binned.agent_solves.json"
