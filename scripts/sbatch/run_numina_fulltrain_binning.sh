#!/bin/bash
#SBATCH --job-name=mae_binning
#SBATCH --gres=gpu:PRO6000:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/home/jaehoonjeong/data/MetaAgentEvolution_Release/logs/%x.%j.out
#SBATCH --error=/home/jaehoonjeong/data/MetaAgentEvolution_Release/logs/%x.%j.err

# Full-train per-expert binning: every roster expert solves every train problem
# (NO routing), then score → per-problem who-solved-what labels = training signal
# for the weight-level MoE.
#
# One model load, one batched pass over (problems × experts) via --pipeline binning.
# Env: SEED, DATASET, EVAL_CONFIG (a4b), EVAL_SIZE (full train), ROSTER.

set -euo pipefail

REPO="${REPO:-/home/jaehoonjeong/data/MetaAgentEvolution_Release}"
# shellcheck source=scripts/sbatch/common_bigmath.sh
source "${REPO}/scripts/sbatch/common_bigmath.sh"
setup_job_env

EVAL_CONFIG="${EVAL_CONFIG:-configs/numina_eval_a4b.yaml}"
ROSTER="${ROSTER:-${RESULTS_DIR}/roster_final.json}"
IBS="${IBS:-256}"
OUT="${RESULTS_DIR}/binning_train.jsonl"

if [ ! -f "${ROSTER}" ]; then
    echo "ERROR: roster not found at ${ROSTER}"
    exit 1
fi

echo "=== Full-train binning seed${SEED} | config=${EVAL_CONFIG} | roster=${ROSTER} ==="
echo "=== dataset=${DATASET} split=train size=${EVAL_SIZE:-all} infer_batch=${IBS} ==="

# 재발방지: 옛 출력 있으면 run_inference resume가 생성 skip → stale 재사용. 항상 새로 생성.
rm -f "${OUT}"
python scripts/run_inference.py \
    --config "${EVAL_CONFIG}" \
    --dataset "${DATASET}" --split train \
    --pipeline binning \
    --roster_path "${ROSTER}" \
    --seed "${SEED}" --max_items "${EVAL_SIZE:-62185}" \
    --infer_batch_size "${IBS}" \
    --output_file "${OUT}"

echo "=== Scoring per-expert solves ==="
python scripts/score_binning.py \
    --input "${OUT}" \
    --dataset "${DATASET}" --split train

echo "=== Binning done. Labels: ${OUT%.jsonl}.binned.jsonl | summary: ${OUT%.jsonl}.binned.summary.json ==="
