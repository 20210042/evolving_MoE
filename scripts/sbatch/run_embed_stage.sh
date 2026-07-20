#!/bin/bash
#SBATCH --job-name=embed_stage
# A6000(n01)·4090(master)은 드라이버가 낡아 torch 2.11 CUDA 초기화 실패 → PRO6000 사용
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log

# 임베딩 1회성 단계: QASC/TACO/LBOX 전체를 embeddinggemma-300m로 임베딩해
# results/embed_viz/<ds>_emb.npy + <ds>_ids.json 저장 (id 일치 시 재실행 스킵).

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"

source ~/.bashrc
source ~/data/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-evolving_moe}"

export HF_HOME="${HF_HOME:-/data5/jaehoonjeong/.cache/huggingface}"

srun --ntasks=1 --gpus-per-task=1 --chdir="$REPO" \
    python scripts/embed_expert_viz.py --stage embed \
    --batch "${BATCH:-512}" --datasets ${DATASETS:-qasc taco lbox}

echo "=== embed_stage 완료: results/embed_viz/*_emb.npy ==="