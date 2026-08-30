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

# 회수율 분해 — 실현 / 라벨 오라클 / 문제 오라클 사다리 + 승자 분산의 태스크 간·내 분해.
python scripts/sni_recovery_gap.py --metric em    --out docs/REPORT_sni_recovery_em.md
python scripts/sni_recovery_gap.py --metric rouge --out docs/REPORT_sni_recovery_rouge.md
echo "=== done ==="
