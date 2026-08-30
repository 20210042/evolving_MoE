#!/bin/bash
#SBATCH --job-name=router_sweep_acc_emb
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --exclude=n05
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source ~/.bashrc
source ~/data/miniconda3/etc/profile.d/conda.sh
conda activate evolving_moe
export HF_HOME=/data5/jaehoonjeong/.cache/huggingface
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src"
python scripts/router_sweep_max_nsolved.py --dataset acc --feature emb
echo "=== acc router emb(embeddinggemma) 스윕(학습곡선 포함) 완료 ==="
