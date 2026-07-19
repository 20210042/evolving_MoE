#!/bin/bash
#SBATCH --job-name=deploy_sweep
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source ~/.bashrc; source ~/data/miniconda3/etc/profile.d/conda.sh; conda activate evolving_moe
export HF_HOME=/data5/jaehoonjeong/.cache/huggingface
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src"
python scripts/moe_deploy_sweep.py
echo "=== deploy_sweep 완료 ==="
