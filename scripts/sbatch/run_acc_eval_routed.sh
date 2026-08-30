#!/bin/bash
#SBATCH --job-name=mae_eval_acc_routed
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/home/jaehoonjeong/data/MetaAgentEvolution_Release/logs/%x.%j.out
#SBATCH --error=/home/jaehoonjeong/data/MetaAgentEvolution_Release/logs/%x.%j.err

# 코딩 도메인(acc) — 최종 로스터 evolved routing(LLM 라우터가 N명 중 1명 선택) pass@1.
#   run_acc_eval.sh(LUCA baseline + binning UB)와 짝. 같은 홀드아웃 → 직접 비교.
# Env: SEED, EVAL_SIZE, EVAL_CONFIG, DATA_DIR, SPLIT, IGNORE_TEST_IDS, OUT_TAG.
#   v1은 test 파일이 없어 SPLIT=train + test_ids.json 필터를 썼다. acc_v2는 실제 test
#   split(751)이 있으므로 SPLIT=test IGNORE_TEST_IDS=1로 준다.

set -euo pipefail

REPO="${REPO:-/home/jaehoonjeong/data/MetaAgentEvolution_Release}"
# shellcheck source=scripts/sbatch/common_bigmath.sh
source "${REPO}/scripts/sbatch/common_bigmath.sh"
setup_job_env
print_experiment_config

DATASET="acc"
EVAL_CONFIG="${EVAL_CONFIG:-configs/acc_eval_a4b.yaml}"
DATA_DIR="${DATA_DIR:-export/acc_selfconsistent}"
FINAL_ROSTER="${RESULTS_DIR}/roster_final.json"
EVAL_SIZE="${EVAL_SIZE:-500}"
OUT_TAG="${OUT_TAG:-final}"                 # 출력 분리(v1=final, v2=routerv2 등) → A/B 보존
OUT="${RESULTS_DIR}/inference_test_routed_${OUT_TAG}.jsonl"

echo "=== acc routed eval: final roster (${FINAL_ROSTER}) size=${EVAL_SIZE} config=${EVAL_CONFIG} out=${OUT} ==="
if [ ! -f "${FINAL_ROSTER}" ]; then
    echo "ERROR: final roster not found at ${FINAL_ROSTER}"
    exit 1
fi

SPLIT="${SPLIT:-train}"
cmd=(
  python scripts/run_inference.py
  --config "${EVAL_CONFIG}"
  --dataset "${DATASET}"
  --split "${SPLIT}"
  --seed "${SEED}"
  --data_dir "${DATA_DIR}"
  --pipeline evolved
  --roster_path "${FINAL_ROSTER}"
  --max_items "${EVAL_SIZE}"
  --output_file "${OUT}"
)
if [ "${IGNORE_TEST_IDS:-0}" = "1" ]; then
    cmd+=(--ignore_test_ids)
fi

rm -f "${OUT}"
"${cmd[@]}"

python scripts/score_outputs.py \
    --input "${OUT}" \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --data_dir "${DATA_DIR}"

echo "=== acc routed eval done (final roster, router picks 1 of N) ==="