#!/bin/bash
#SBATCH --job-name=acc_sft_oracle
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log
set -euo pipefail
JOURNAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO="${REPO:-$(cd "${JOURNAL_DIR}/../../.." && pwd)}"; cd "$REPO"
source ~/.bashrc; source ~/data/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-evolving_moe}"
export PYTHONPATH="$REPO/src:${JOURNAL_DIR}/scripts/router"
if [ -z "${EXPERT_ID:-}" ]; then
  echo "ERROR: EXPERT_ID is required" >&2
  exit 1
fi
python "${JOURNAL_DIR}/scripts/router/infer_acc_single_sft_expert.py" \
  --expert "$EXPERT_ID" \
  --wandb-entity "${WANDB_ENTITY:-jongbin-kr-skiml_moe}" \
  --wandb-project "${WANDB_PROJECT:-acc-seed20211004-sft-oracle}" \
  ${INFER_ARGS:-}
