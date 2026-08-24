#!/bin/bash
# SNI 프로브 v2 축 단위 검정. GPU는 안 쓰지만 로그인 노드에서
# 652만 레코드(1.5GB)를 돌리지 않기 위해 SLURM으로 보낸다.
# Usage: sbatch --job-name=sni_readout scripts/sbatch/run_sni_readout.sh
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
export PYTHONHASHSEED=0

python scripts/sni_hard_error_rate.py \
    --raw results/sni/probe_v2_raw.jsonl \
    --data export/sni_v2/sni_all.jsonl \
    --roster configs/roster_sni_probe_v2.json \
    --out docs/REPORT_hard_error_rate.md
echo "=== done ==="
