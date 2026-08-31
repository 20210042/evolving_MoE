#!/bin/bash
#SBATCH --gres=gpu:4090:1
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
# 축 하나당 잡 하나. AXIS=category | sni_domain
AXIS="${AXIS:?AXIS=category 또는 sni_domain 을 지정하세요}"
case "$AXIS" in
  category)   TAG=cat ;;
  sni_domain) TAG=dom ;;
  *) echo "알 수 없는 AXIS: $AXIS" >&2; exit 1 ;;
esac
python scripts/sni_build_split_human.py --axis "$AXIS" --feat "${FEAT:-hs_mean}"
python scripts/sni_export_moe_package.py --arm human \
  --split "export/sni_split_human_${TAG}/split.jsonl" \
  --out   "export/sni_moe_human_${TAG}"
