#!/bin/bash
#SBATCH --job-name=mae_eval_acc_top2
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/jaehoonjeong/data/MetaAgentEvolution_Release/logs/%x.%j.out
#SBATCH --error=/home/jaehoonjeong/data/MetaAgentEvolution_Release/logs/%x.%j.err

# 코딩(acc) top-k routing 평가 — 라우터가 k명 선택 → 각 생성 → union 채점(둘 중 통과=정답).
# Env: SEED, EVAL_SIZE, EVAL_CONFIG(top_k 지정), DATA_DIR, OUT_TAG.
set -euo pipefail
REPO="${REPO:-/home/jaehoonjeong/data/MetaAgentEvolution_Release}"
# shellcheck source=scripts/sbatch/common_bigmath.sh
source "${REPO}/scripts/sbatch/common_bigmath.sh"
setup_job_env
print_experiment_config

DATASET="acc"
EVAL_CONFIG="${EVAL_CONFIG:-configs/acc_eval_a4b_top2.yaml}"
DATA_DIR="${DATA_DIR:-export/acc_selfconsistent}"
FINAL_ROSTER="${RESULTS_DIR}/roster_final.json"
EVAL_SIZE="${EVAL_SIZE:-500}"
OUT_TAG="${OUT_TAG:-top2}"
OUT="${RESULTS_DIR}/inference_test_routed_${OUT_TAG}.jsonl"

echo "=== acc top-k eval: config=${EVAL_CONFIG} size=${EVAL_SIZE} out=${OUT} ==="
[ -f "${FINAL_ROSTER}" ] || { echo "ERROR: final roster not found at ${FINAL_ROSTER}"; exit 1; }

rm -f "${OUT}"
python scripts/run_inference.py \
    --config "${EVAL_CONFIG}" \
    --dataset "${DATASET}" \
    --split train \
    --seed "${SEED}" \
    --data_dir "${DATA_DIR}" \
    --pipeline evolved \
    --roster_path "${FINAL_ROSTER}" \
    --max_items "${EVAL_SIZE}" \
    --output_file "${OUT}"

python scripts/score_outputs_topk.py \
    --input "${OUT}" \
    --dataset "${DATASET}" \
    --split train \
    --data_dir "${DATA_DIR}"

echo "=== acc top-k eval done ==="