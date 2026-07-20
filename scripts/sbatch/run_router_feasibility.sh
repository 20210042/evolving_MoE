#!/bin/bash
#SBATCH --job-name=router_feas
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source ~/.bashrc; source ~/data/miniconda3/etc/profile.d/conda.sh; conda activate evolving_moe
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
echo "===== emb ====="; python scripts/router_feasibility.py --feat emb
echo "===== hs_last ====="; python scripts/router_feasibility.py --feat hs_last
echo "===== hs_mean ====="; python scripts/router_feasibility.py --feat hs_mean
