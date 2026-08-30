#!/bin/bash
# 정본 라벨 패키지 — K=3 중 **rep 0**을 승격(사용자 결정 2026-08-28). 다운스트림은 이걸 쓴다.
# rep 1·2는 ANOVA 오차항 전용이고 라벨에 관여하지 않는다.
# Usage: sbatch --dependency=afterok:<binning_job> scripts/sbatch/run_sni_label_package.sh
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/home/jaehoonjeong/data/MetaAgentEvolution_Release/logs/%x.%j.log
#SBATCH --error=/home/jaehoonjeong/data/MetaAgentEvolution_Release/logs/%x.%j.log

set -euo pipefail
REPO="${REPO:-/home/jaehoonjeong/data/MetaAgentEvolution_Release}"
cd "${REPO}"
source ~/.bashrc
source ~/data/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-evolving_moe}"
export PYTHONPATH="${REPO}/src"
export HF_HOME="${HF_HOME:-/data5/jaehoonjeong/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/data5/jaehoonjeong/.cache/huggingface}"

INDIR="${INDIR:-results/sni/binning_seed20212001}"
ROSTER="${ROSTER:-results/sni/seed20212001/roster_final.json}"

for SPLIT in train valid test; do
  echo "=== ${SPLIT} · 정본(rep 0) ==="
  python scripts/sni_build_label_package.py \
      --raw "${INDIR}/${SPLIT}_raw.jsonl" \
      --roster "${ROSTER}" \
      --rule rep --rep 0 \
      --out "${INDIR}/${SPLIT}"

  echo "=== ${SPLIT} · 부차(majority 2/3) — 대조용, 다운스트림 아님 ==="
  python scripts/sni_build_label_package.py \
      --raw "${INDIR}/${SPLIT}_raw.jsonl" \
      --roster "${ROSTER}" \
      --rule majority \
      --out "${INDIR}/${SPLIT}_majority"
done
echo "=== done ==="
ls -l "${INDIR}"/*.binned.*
