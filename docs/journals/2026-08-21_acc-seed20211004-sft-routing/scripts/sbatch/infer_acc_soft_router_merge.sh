#!/bin/bash
#SBATCH --job-name=acc_soft12_merge
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log
set -euo pipefail
JOURNAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO="${REPO:-$(cd "${JOURNAL_DIR}/../../.." && pwd)}"; cd "$REPO"
source ~/.bashrc; source ~/data/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-evolving_moe}"
export PYTHONPATH="$REPO/src:${JOURNAL_DIR}/scripts/router"
python "${JOURNAL_DIR}/scripts/router/infer_acc_soft_router_merge.py" \
  --router-dir "${ROUTER_DIR:-checkpoints/router/acc_seed20211004_soft12}" \
  --output "${OUTPUT:-results/acc/seed20211004/router_merge/soft12_test.jsonl}" \
  --wandb-entity "${WANDB_ENTITY:-jongbin-kr-skiml_moe}" \
  --wandb-project "${WANDB_PROJECT:-acc-seed20211004-soft-router}" \
  ${INFER_ARGS:-}
