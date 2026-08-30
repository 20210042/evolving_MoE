#!/bin/bash
#SBATCH --job-name=acc_soft_war_screen
# GPU 미사용(numpy·yaml만 쓴다) → --gres 없이 제출해 GPU 쿼터를 물지 않는다.
#SBATCH --exclude=master
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log

# REPORT_task3_task4.md soft-WAR 사전 검증 (scripts/soft_war_screen.py).
#   기존 산출물만 읽는다 — 모델 로드·신규 생성 없음.
# 사용: sbatch scripts/sbatch/run_acc_soft_war_screen.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source ~/.bashrc
source ~/data/miniconda3/etc/profile.d/conda.sh
conda activate evolving_moe
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src"
export HF_HOME="${HF_HOME:-/data5/jaehoonjeong/.cache/huggingface}"

python scripts/soft_war_screen.py
echo "=== done ==="
