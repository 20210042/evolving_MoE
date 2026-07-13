#!/bin/bash
# 최종 로스터 전원 per-expert SFT 일괄 제출.
# 기본은 직렬 체인(afterany): 앞 잡이 끝나야 다음 시작 → 동시 GPU 사용은 잡 1개분(2장)뿐.
# 사용: [MAX_N_SOLVED=10] [EXPERT_IDS="c_33055 luca"] [SERIAL=0] ./launch_sft_by_experts.sh [LABEL_PACKAGE]
#   LABEL_PACKAGE 기본값: export/qasc_binning_seed20210211
#   EXPERT_IDS 생략 시 agent_mapping.json의 로스터 전원. SERIAL=0이면 전부 병렬 제출.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${SCRIPT_DIR}/train_sft_by_expert.sh}"

LABEL_PACKAGE="${1:-${LABEL_PACKAGE:-export/qasc_binning_seed20210211}}"
MAX_N_SOLVED="${MAX_N_SOLVED:-}"
SERIAL="${SERIAL:-1}"

if [ -z "${EXPERT_IDS:-}" ]; then
    EXPERT_IDS="$(python3 -c "import json,sys; print(' '.join(json.load(open(sys.argv[1]))))" "${LABEL_PACKAGE}/agent_mapping.json")"
fi

echo "=== package: ${LABEL_PACKAGE} / max_n_solved: ${MAX_N_SOLVED:-none} / serial: ${SERIAL} ==="
echo "=== experts: ${EXPERT_IDS} ==="

PREV_JOB_ID=""
for EXPERT_ID in ${EXPERT_IDS}; do
    DEP_FLAG=()
    if [ "${SERIAL}" = "1" ] && [ -n "${PREV_JOB_ID}" ]; then
        DEP_FLAG=(--dependency="afterany:${PREV_JOB_ID}")
    fi
    JOB_ID="$(sbatch --parsable --job-name="sft_expert_${EXPERT_ID}" \
        "${DEP_FLAG[@]}" \
        --export=ALL,LABEL_PACKAGE="${LABEL_PACKAGE}",EXPERT_ID="${EXPERT_ID}",MAX_N_SOLVED="${MAX_N_SOLVED}" \
        "${TRAIN_SCRIPT}")"
    echo "--- submitted expert: ${EXPERT_ID} (job ${JOB_ID}${PREV_JOB_ID:+, after ${PREV_JOB_ID}})"
    PREV_JOB_ID="${JOB_ID}"
done
