#!/bin/bash
#SBATCH --job-name=acc_routed_bucket
# GPU 미사용(코드 실행 채점만) → --gres 없이 제출.
#SBATCH --exclude=master
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log

# 라우팅 산출물을 문항별로 재채점해 버킷별로 집계 (scripts/routed_bucket_report.py).
# Env: INPUT, BINNED, LABEL

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source ~/.bashrc
source ~/data/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-evolving_moe}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src"
export HF_HOME="${HF_HOME:-/data5/jaehoonjeong/.cache/huggingface}"

python scripts/routed_bucket_report.py \
    --input "${INPUT:-results/acc/seed20211004/inference_test_routed_20211004_greedy.jsonl}" \
    --binned "${BINNED:-results/acc/seed20211004/binning_test_full.binned.jsonl}" \
    --label "${LABEL:-Evolved Roster (LLM top-1 routing)}"
echo "=== done ==="
