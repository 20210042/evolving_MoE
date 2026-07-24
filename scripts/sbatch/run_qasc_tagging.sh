#!/bin/bash
#SBATCH --job-name=qasc_llm_tags
#SBATCH --gres=gpu:PRO6000:1
# n05: torch/CUDA는 정상이나 vllm 기동이 segfault(2회 재현, 2026-07-16) → vllm 잡만 제외
#SBATCH --exclude=n05
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log

# QASC train 전체를 gemma(a4b)로 과학 과목 태깅 → embed_expert_viz의 LLM prior 패널.
# 출력: results/embed_viz/qasc_llm_tags.json (resume 지원)

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"

source ~/.bashrc
source ~/data/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-evolving_moe}"

export HF_HOME="${HF_HOME:-/data5/jaehoonjeong/.cache/huggingface}"
export PYTHONPATH="$REPO/src"
export PYTHONUNBUFFERED=1
# flashinfer는 이 클러스터에서 깨짐 — 기존 vllm 잡 표준(common_bigmath.sh)과 동일하게 비활성화
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_DISABLE_FLASHINFER=1

# vllm은 srun 스텝 환경(SLURM_PROCID 등)과 충돌해 segfault — 기존 vllm 잡들처럼 직접 실행
python scripts/tag_qasc_topics.py

echo "=== qasc_llm_tags 완료 ==="
