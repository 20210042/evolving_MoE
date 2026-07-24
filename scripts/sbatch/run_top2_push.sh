#!/bin/bash
#SBATCH --job-name=top2_push
#SBATCH --gres=gpu:4090:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source ~/.bashrc; source ~/data/miniconda3/etc/profile.d/conda.sh; conda activate pro6000
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
python scripts/router_top2_push.py --dataset "${DATASET:-qasc}"
