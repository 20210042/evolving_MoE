#!/bin/bash
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/home/jaehoonjeong/data/MetaAgentEvolution_Release/logs/%x.%j.log
#SBATCH --error=/home/jaehoonjeong/data/MetaAgentEvolution_Release/logs/%x.%j.log
set -euo pipefail
REPO="${REPO:-/home/jaehoonjeong/data/MetaAgentEvolution_Release}"
source "${REPO}/scripts/sbatch/common_bigmath.sh"
setup_job_env

# 전문화 검정 (A 재현성 · B excess · C 승자예측 · D 길이반응(연속) · E 배정이득).
# 발명한 축 없음. 생성 0회 — 기존 official_labels.npz만 접는다.
python scripts/sni_specialization_test.py --metric em \
    --out docs/REPORT_sni_specialization_em.md
python scripts/sni_specialization_test.py --metric rouge \
    --out docs/REPORT_sni_specialization_rouge.md
echo "=== done ==="
