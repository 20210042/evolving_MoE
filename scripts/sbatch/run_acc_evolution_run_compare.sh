#!/bin/bash
#SBATCH --job-name=acc_evolution_run_compare
#SBATCH --exclude=master
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log

# 세 진화 런(20210111 hard-WAR / 20211001 soft_linear,장치없음 / 20211002 soft_partial,장치복원)
# 궤적 비교 + 실행 로그 기반 인접 스텝 재현성 (scripts/evolution_run_compare.py).
#   기존 산출물만 읽는다 — 모델 로드·신규 생성 없음.

set -euo pipefail
REPO="${REPO:-/home/jaehoonjeong/data/MetaAgentEvolution_Release}"
cd "${REPO}"
source ~/.bashrc
source ~/data/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-evolving_moe}"
export PYTHONPATH="${REPO}/src"
export HF_HOME="${HF_HOME:-/data5/jaehoonjeong/.cache/huggingface}"

python scripts/evolution_run_compare.py
echo "=== done ==="
