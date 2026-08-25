#!/bin/bash
set -euo pipefail
JOURNAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO="${REPO:-$(cd "${JOURNAL_DIR}/../../.." && pwd)}"; cd "$REPO"
mkdir -p logs checkpoints/router/acc_seed20211004_soft12 results/acc/seed20211004/router_merge
TRAIN_JOB="$(sbatch --parsable --export=ALL "${JOURNAL_DIR}/scripts/sbatch/train_acc_soft_router.sh")"
INFER_JOB="$(sbatch --parsable --dependency="afterok:${TRAIN_JOB}" --export=ALL "${JOURNAL_DIR}/scripts/sbatch/infer_acc_soft_router_merge.sh")"
echo "router train: ${TRAIN_JOB}"
echo "soft-12 parameter-merge inference (afterok): ${INFER_JOB}"
