#!/bin/bash
#SBATCH --job-name=route_matrix
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source ~/.bashrc; source ~/data/miniconda3/etc/profile.d/conda.sh; conda activate evolving_moe
export HF_HOME=/data5/jaehoonjeong/.cache/huggingface
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src"
python scripts/route_eval_matrix.py
echo "=== route_matrix 완료 ==="
