#!/bin/bash
set -euo pipefail
JOURNAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO="${REPO:-$(cd "${JOURNAL_DIR}/../../.." && pwd)}"; cd "$REPO"
mkdir -p logs results/acc/seed20211004/sft_oracle/parts
EXPERTS=(luca c_46087 c_10367 c_17316 c_4998 c_34728 c_63819 c_50585 c_16428 c_30658 c_56276 c_56422)
JOBS=()
for EXPERT in "${EXPERTS[@]}"; do
  JOB="$(EXPERT_ID="$EXPERT" sbatch --parsable --job-name="acc_ub_${EXPERT}" --export=ALL "${JOURNAL_DIR}/scripts/sbatch/infer_acc_single_sft_expert.sh")"
  JOBS+=("$JOB")
  echo "$EXPERT: $JOB"
done
DEPENDENCY="$(IFS=:; echo "${JOBS[*]}")"
AGG_JOB="$(sbatch --parsable --dependency="afterok:${DEPENDENCY}" --export=ALL "${JOURNAL_DIR}/scripts/sbatch/aggregate_acc_sft_oracle.sh")"
echo "oracle aggregate (afterok all): $AGG_JOB"
