#!/bin/bash
# 라우터 입력 특징 추출 — 문제 쪽(train/test) + 전문가 프로파일 쪽(로스터 16명).
#   문제: system=중립+정의 / user=예시2+Input (실제 생성과 같은 조립, 페르소나 자리만 중립)
#   전문가: 각자의 system prompt 한 개씩
# max_len 4096 = 스모크 236203 실측(잘리는 문제 0.45~0.62%).
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ML="${MAX_LEN:-4096}"
ROSTER="${ROSTER:-results/sni/seed20212003/roster_final.json}"

echo "=== 전문가 프로파일 16명 ==="
python scripts/extract_expert_profiles.py --dataset sni --roster "${ROSTER}" --max_len "${ML}"

for SP in test train; do
  echo "=== 문제 특징: ${SP} (max_len=${ML}) ==="
  python scripts/extract_hidden_states.py --dataset sni --split "${SP}" \
      --batch "${BATCH:-8}" --max_len "${ML}"
done
echo "=== done ==="
ls -la results/embed_viz_test/sni_*
