#!/bin/bash
#SBATCH --job-name=acc_inter_sft
# GPU 미사용(작은 행렬 + 순열) → --gres 없이 제출.
#SBATCH --exclude=master
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log

# 학습된(per-expert LoRA) expert들 사이에 문제×expert 궁합이 생겼는가.
#   프롬프트 층(gemma 페르소나)에서는 0이었다 — 적용 층(가중치)에서도 0인지 확인.
#   기존 산출물만 읽는다: results/acc/seed20210111_v2/ablation/*.binned.jsonl (test751)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source ~/.bashrc
source ~/data/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-evolving_moe}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src"
export HF_HOME="${HF_HOME:-/data5/jaehoonjeong/.cache/huggingface}"

A="results/acc/seed20210111_v2/ablation"
O="results/acc/seed20210111_v2/ablation/interaction"
mkdir -p "${O}"

for COND in evolved_fewshot evolved rnd hp; do
  # contested = 만장일치 제외 → 상한은 로스터 크기-1 (hp만 6명)
  if [ "${COND}" = "hp" ]; then HI=5; else HI=11; fi
  for FEAT in onehot emb; do
    for TAG in all contested; do
      if [ "${TAG}" = "all" ]; then BAND=""; else BAND="1,${HI}"; fi
      echo "=== ${COND} / ${FEAT} / ${TAG} ==="
      python scripts/interaction_lowrank_test.py \
        --binned "${A}/inference_test751_${COND}.binned.jsonl" \
        --feature "${FEAT}" --split cell --ranks 1,2,4,8 \
        --feat_npy results/embed_viz_test/acc_test_emb.npy \
        --feat_ids results/embed_viz_test/acc_test_emb_ids.json \
        --band "${BAND}" \
        --out "${O}/${COND}_${FEAT}_${TAG}.md"
    done
  done
done
echo "=== done -> ${O} ==="
