#!/bin/bash
#SBATCH --job-name=acc_sft_oracle_agg
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log
set -euo pipefail
JOURNAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO="${REPO:-$(cd "${JOURNAL_DIR}/../../.." && pwd)}"; cd "$REPO"
source ~/.bashrc; source ~/data/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-evolving_moe}"
export PYTHONPATH="$REPO/src:${JOURNAL_DIR}/scripts/router"
python "${JOURNAL_DIR}/scripts/router/aggregate_acc_sft_oracle.py"
