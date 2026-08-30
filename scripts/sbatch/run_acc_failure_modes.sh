#!/bin/bash
#SBATCH --job-name=acc_failure_modes
# GPU 미사용(코드 실행 채점만) → --gres 없이 제출해 GPU 쿼터를 물지 않는다.
#SBATCH --exclude=master
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log

# expert별 실패유형 분포(scripts/failure_mode_by_expert.py).
#   기존 binning 생성물을 재실행해 status를 살린다 — 신규 생성 없음.
# Env: INPUT(기본 seed20211004 test), DATASET, SPLIT, DATA_DIR

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source ~/.bashrc
source ~/data/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-evolving_moe}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src"
export HF_HOME="${HF_HOME:-/data5/jaehoonjeong/.cache/huggingface}"

INPUT="${INPUT:-results/acc/seed20211004/binning_test_full.jsonl}"
echo "=== failure modes: ${INPUT} ==="
python scripts/failure_mode_by_expert.py \
    --input "${INPUT}" \
    --dataset "${DATASET:-acc}" \
    --split "${SPLIT:-test}" \
    --data_dir "${DATA_DIR:-export/acc_v2}"
echo "=== done ==="
