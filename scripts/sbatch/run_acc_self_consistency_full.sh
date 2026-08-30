#!/bin/bash
#SBATCH --job-name=router_self_consistency_acc_full
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --exclude=n05
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source ~/.bashrc
source ~/data/miniconda3/etc/profile.d/conda.sh
conda activate evolving_moe
export HF_HOME=/data5/jaehoonjeong/.cache/huggingface
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src"
python scripts/router_self_consistency.py --dataset acc --n_problems 40 --k 5 \
    --experts c_12606 c_18757 c_23373 c_28885 c_40681 c_429 c_49191 c_54530 c_6483 c_9948 luca \
    --out results/acc/router_self_consistency_full.md
echo "=== acc router self-consistency(11명 전체) 체크 완료 ==="
