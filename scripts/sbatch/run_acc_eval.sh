#!/bin/bash
#SBATCH --job-name=mae_eval_acc
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/jaehoonjeong/data/MetaAgentEvolution_Release/logs/%x.%j.out
#SBATCH --error=/home/jaehoonjeong/data/MetaAgentEvolution_Release/logs/%x.%j.err

# 코딩 도메인(acc) 홀드아웃 평가 — 1 GPU(tp=1).
#   (1) LUCA baseline: 단독 에이전트 routed pass@1 (score_outputs).
#   (2) 최종 로스터 UB: --pipeline binning(전원 각자 풀이) → union UB (score_binning).
# acc_test.jsonl 없음 → split=train으로 acc_train.jsonl 로드 후 test_ids.json(홀드아웃)으로 자동 필터.
# Env: SEED, EVAL_SIZE, EVAL_CONFIG, LUCA_ROSTER, DATA_DIR.

set -euo pipefail

REPO="${REPO:-/home/jaehoonjeong/data/MetaAgentEvolution_Release}"
# shellcheck source=scripts/sbatch/common_bigmath.sh
source "${REPO}/scripts/sbatch/common_bigmath.sh"
setup_job_env
print_experiment_config

DATASET="acc"
EVAL_CONFIG="${EVAL_CONFIG:-configs/acc_eval_a4b.yaml}"
LUCA_ROSTER="${LUCA_ROSTER:-configs/roster_init.json}"
DATA_DIR="${DATA_DIR:-export/acc_selfconsistent}"
FINAL_ROSTER="${RESULTS_DIR}/roster_final.json"
EVAL_SIZE="${EVAL_SIZE:-500}"

echo "=== acc eval: config=${EVAL_CONFIG} size=${EVAL_SIZE} data_dir=${DATA_DIR} ==="
echo "=== LUCA=${LUCA_ROSTER}  FINAL=${FINAL_ROSTER} ==="

# ---------- (1) LUCA baseline (routed pass@1) ----------
echo "=========================================================================="
echo "=== [1/2] LUCA baseline (evolved routing, single agent) ==="
echo "=========================================================================="
rm -f "${RESULTS_DIR}/inference_test_luca.jsonl"
python scripts/run_inference.py \
    --config "${EVAL_CONFIG}" \
    --dataset "${DATASET}" \
    --split train \
    --seed "${SEED}" \
    --data_dir "${DATA_DIR}" \
    --pipeline evolved \
    --roster_path "${LUCA_ROSTER}" \
    --max_items "${EVAL_SIZE}" \
    --output_file "${RESULTS_DIR}/inference_test_luca.jsonl"

python scripts/score_outputs.py \
    --input "${RESULTS_DIR}/inference_test_luca.jsonl" \
    --dataset "${DATASET}" \
    --split train \
    --data_dir "${DATA_DIR}"

# ---------- (2) 최종 로스터 UB (binning, all experts) ----------
echo "=========================================================================="
echo "=== [2/2] Final roster UB (binning, all experts solve every problem) ==="
echo "=========================================================================="
if [ ! -f "${FINAL_ROSTER}" ]; then
    echo "ERROR: final roster not found at ${FINAL_ROSTER}"
    exit 1
fi
rm -f "${RESULTS_DIR}/inference_test_binning_final.jsonl"
python scripts/run_inference.py \
    --config "${EVAL_CONFIG}" \
    --dataset "${DATASET}" \
    --split train \
    --seed "${SEED}" \
    --data_dir "${DATA_DIR}" \
    --pipeline binning \
    --roster_path "${FINAL_ROSTER}" \
    --max_items "${EVAL_SIZE}" \
    --output_file "${RESULTS_DIR}/inference_test_binning_final.jsonl"

python scripts/score_binning.py \
    --input "${RESULTS_DIR}/inference_test_binning_final.jsonl" \
    --dataset "${DATASET}" \
    --split train \
    --data_dir "${DATA_DIR}"

echo "=== acc eval done (LUCA baseline + final-roster UB) ==="
