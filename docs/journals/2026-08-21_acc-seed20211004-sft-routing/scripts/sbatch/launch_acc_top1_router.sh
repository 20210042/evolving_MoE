#!/bin/bash
set -euo pipefail
JOURNAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO="${REPO:-$(cd "${JOURNAL_DIR}/../../.." && pwd)}"; cd "$REPO"
mkdir -p logs checkpoints/router/acc_seed20211004_top1_set results/acc/seed20211004/router_top1
TRAIN_JOB="$(sbatch --parsable --export=ALL "${JOURNAL_DIR}/scripts/sbatch/train_acc_top1_router.sh")"
INFER_JOB="$(sbatch --parsable --dependency="afterok:${TRAIN_JOB}" --export=ALL "${JOURNAL_DIR}/scripts/sbatch/infer_acc_top1_router.sh")"
echo "top-1 router train: ${TRAIN_JOB}"
echo "exclusive top-1 inference (afterok): ${INFER_JOB}"
