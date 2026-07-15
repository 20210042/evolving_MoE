#!/bin/bash
# 최종 로스터 전원 per-expert SFT 일괄 제출.
# 기본은 rolling chain: 첫 expert 잡 1개만 제출하고, 각 잡이 시작 시점에 다음 잡을
# 스스로 제출(afterany) → 큐에는 항상 러닝 1개 + 대기 1개만 보이고 GPU도 잡 1개분만 사용.
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

if [ "${SERIAL}" = "1" ]; then
    set -- ${EXPERT_IDS}
    FIRST="$1"; shift
    REST="$*"
    JOB_ID="$(LABEL_PACKAGE="${LABEL_PACKAGE}" EXPERT_ID="${FIRST}" MAX_N_SOLVED="${MAX_N_SOLVED}" \
        REMAINING_EXPERTS="${REST}" \
        sbatch --parsable --job-name="sft_expert_${FIRST}" --export=ALL "${TRAIN_SCRIPT}")"
    echo "--- submitted expert: ${FIRST} (job ${JOB_ID}); 나머지 $#개는 rolling chain으로 자동 제출: ${REST:-없음}"
else
    for EXPERT_ID in ${EXPERT_IDS}; do
        JOB_ID="$(LABEL_PACKAGE="${LABEL_PACKAGE}" EXPERT_ID="${EXPERT_ID}" MAX_N_SOLVED="${MAX_N_SOLVED}" \
            sbatch --parsable --job-name="sft_expert_${EXPERT_ID}" --export=ALL "${TRAIN_SCRIPT}")"
        echo "--- submitted expert: ${EXPERT_ID} (job ${JOB_ID})"
    done
fi
