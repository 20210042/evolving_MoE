#!/bin/bash
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/home/jaehoonjeong/data/MetaAgentEvolution_Release/logs/%x.%j.log
#SBATCH --error=/home/jaehoonjeong/data/MetaAgentEvolution_Release/logs/%x.%j.log
set -euo pipefail
cd "${REPO:-/home/jaehoonjeong/data/MetaAgentEvolution_Release}"
source ~/.bashrc; source ~/data/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-evolving_moe}"
export PYTHONPATH="$PWD/src"
# LBox 로스터 10명 → τ = N-2 = 8 (QASC seed20210211의 cap10 = 12-2 와 같은 규칙).
# 센트로이드 밴드 tc는 SNI(τ=10 → tc=5)와 같은 비율로 τ의 절반.
# ⚠️ LBox에는 teacher hs_mean이 없어 --emb 로 768차원 embed_viz 임베딩을 쓴다.
python scripts/sni_build_split.py \
  --dataset lbox --emb \
  --tau "${TAU:-8}" --tau_c "${TAU_C:-4}" \
  --roster results/lbox/seed20210311/roster_final.json \
  --out    export/lbox_split_seed20210311/split.jsonl \
  --report results/lbox/split_build.md
