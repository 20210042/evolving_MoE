#!/bin/bash
#SBATCH --job-name=moe_verify
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
echo "===== Phase2: route_top2_picks ====="
python scripts/route_top2_picks.py
echo "===== Phase3 앵커 (weights 1,0 → 병합==단일 재현) limit50 ====="
python scripts/moe_merge_infer.py --weights 1,0 --limit 50
echo "===== Phase3 스모크 (linear 0.5) limit50 ====="
python scripts/moe_merge_infer.py --weights 0.5,0.5 --limit 50
echo "=== moe_verify 완료 ==="
