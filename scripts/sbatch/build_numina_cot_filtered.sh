#!/bin/bash
#SBATCH --job-name=build_numina_cot_filtered
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"

source ~/.bashrc
source ~/data/miniconda3/etc/profile.d/conda.sh
conda activate MoE

export PYTHONPATH="$REPO/src"

echo "=== build_numina_cot_filtered 시작 ==="
srun --chdir="$REPO" python "$REPO/scripts/build_numina_cot_filtered.py"
echo "=== 완료 ==="
