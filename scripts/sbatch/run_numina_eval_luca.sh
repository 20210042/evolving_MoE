#!/bin/bash
#SBATCH --job-name=mae_eval_luca
#SBATCH --gres=gpu:PRO6000:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/jaehoonjeong/data/MetaAgentEvolution_Release/logs/%x.%j.out
#SBATCH --error=/home/jaehoonjeong/data/MetaAgentEvolution_Release/logs/%x.%j.error

# LUCA single-agent baseline on the same held-out test set as the seed16 evals.
# Same backbone/config/seed -> identical 500 problems -> directly comparable to
# seed16 routed pass@1 and per-agent UB. Roster = generic LUCA only.
# Env: SEED, DATASET, EVAL_CONFIG, EVAL_SIZE, LUCA_ROSTER.

set -euo pipefail

REPO="${REPO:-/home/jaehoonjeong/data/MetaAgentEvolution_Release}"
# shellcheck source=scripts/sbatch/common_bigmath.sh
source "${REPO}/scripts/sbatch/common_bigmath.sh"
setup_job_env
print_experiment_config

EVAL_CONFIG="${EVAL_CONFIG:-configs/numina_train_seed16.yaml}"
LUCA_ROSTER="${LUCA_ROSTER:-configs/roster_init.json}"
echo "=== LUCA baseline eval: config=${EVAL_CONFIG} roster=${LUCA_ROSTER} ==="

rm -f "${RESULTS_DIR}/inference_test_luca.jsonl"
python scripts/run_inference.py \
    --config "${EVAL_CONFIG}" \
    --dataset "${DATASET}" \
    --split test \
    --seed "${SEED}" \
    --roster_path "${LUCA_ROSTER}" \
    --max_items "${EVAL_SIZE:-500}" \
    --output_file "${RESULTS_DIR}/inference_test_luca.jsonl"

python scripts/score_outputs.py \
    --input "${RESULTS_DIR}/inference_test_luca.jsonl" \
    --dataset "${DATASET}" \
    --split test

echo "=== LUCA baseline eval done ==="
