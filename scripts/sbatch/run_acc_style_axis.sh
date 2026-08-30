#!/bin/bash
#SBATCH --job-name=acc_style_axis
# GPU 미사용(텍스트 특징 + 순열검정) → --gres 없이 제출해 GPU 쿼터를 물지 않는다.
#SBATCH --exclude=master
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log

# 명제 A: persona가 출력분포를 계통적으로 옮기는가 (scripts/output_style_axis.py).
#   INPUTS의 각 binning jsonl에 대해 문제 내 expert 라벨 순열검정을 돌린다.
#   기본값 = greedy 로스터(기존) + T=0.7 로스터 + T=0.7 LUCA×12(시드 기준선).
# Env: INPUTS (공백 구분 목록)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source ~/.bashrc
source ~/data/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-evolving_moe}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src"
export HF_HOME="${HF_HOME:-/data5/jaehoonjeong/.cache/huggingface}"

R="results/acc/seed20211004"
INPUTS="${INPUTS:-${R}/binning_test_full.jsonl ${R}/binning_test_roster12_sampled.jsonl ${R}/binning_test_luca12_sampled.jsonl}"

for f in ${INPUTS}; do
    echo "=== style axis: ${f} ==="
    python scripts/output_style_axis.py --input "${f}"
done
echo "=== done ==="
