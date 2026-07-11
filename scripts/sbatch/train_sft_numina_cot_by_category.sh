#!/bin/bash
# 사용법: bash scripts/sbatch/train_sft_numina_cot_by_category.sh
# 카테고리별로 sbatch job을 각각 제출 (5개 job)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKER="${SCRIPT_DIR}/train_sft_numina_cot_single_category.sh"

CATEGORIES=("Algebra" "Geometry" "Number Theory" "Combinatorics" "Calculus")

for CATEGORY in "${CATEGORIES[@]}"; do
    CATEGORY_SLUG="${CATEGORY// /_}"
    RUN_NAME="sft_gemma4_numina_cot_${CATEGORY_SLUG}"

    JOB_ID=$(sbatch --parsable \
        --job-name="${RUN_NAME}" \
        --export=ALL,CATEGORY="${CATEGORY}",RUN_NAME="${RUN_NAME}" \
        "${WORKER}")

    echo "제출됨: [${CATEGORY}] → job ${JOB_ID}  (log: logs/${RUN_NAME}.${JOB_ID}.log)"
done
