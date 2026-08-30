#!/bin/bash
# 저랭크 이중선형 상호작용 검정 (scripts/interaction_lowrank_test.py).
#   기존 자산만 씀: binning_train_full 행렬 + acc_train 임베딩/hidden-state → 신규 생성 0.
#   모델 로드가 없어 가볍다. 자원은 분석 잡 표준(PRO6000×1, CPU2, 32G)에 맞춤.
# Usage:
#   sbatch --job-name=acc_inter_emb scripts/sbatch/run_acc_interaction_lowrank.sh
#   FEATURE=hs_last SPLIT=problem OUT=results/acc/interaction_lowrank_hs_problem.md \
#     sbatch --job-name=acc_inter_hs scripts/sbatch/run_acc_interaction_lowrank.sh
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --exclude=n05
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

FEATURE="${FEATURE:-emb}"
SPLIT="${SPLIT:-cell}"
RANKS="${RANKS:-1,2,4,8}"
POS_CONTROL="${POS_CONTROL:-0}"
BAND="${BAND:-}"       # 예: BAND=1,5 → 11명 중 1~5명만 푼 어려운 문제만   # >0 이면 인위적 상호작용을 심고 검출되는지 확인(파이프라인 검증)
BINNED="${BINNED:-results/acc/seed20210111/binning_train_full.binned.jsonl}"
FEAT_NPY="${FEAT_NPY:-}"   # 기본 FEATS 경로 대신 쓸 특징 파일(예: seed20211004 전체 11,097 임베딩)
FEAT_IDS="${FEAT_IDS:-}"
OUT="${OUT:-results/acc/interaction_lowrank_${FEATURE}_${SPLIT}.md}"

echo "=== interaction lowrank: feature=${FEATURE} split=${SPLIT} ranks=${RANKS} pc=${POS_CONTROL} binned=${BINNED} feat=${FEAT_NPY:-default} out=${OUT} ==="
python scripts/interaction_lowrank_test.py \
    --binned "${BINNED}" \
    --feature "${FEATURE}" --split "${SPLIT}" --ranks "${RANKS}" \
    --feat_npy "${FEAT_NPY}" --feat_ids "${FEAT_IDS}" \
    --positive_control "${POS_CONTROL}" --band "${BAND}" --out "${OUT}"
echo "=== done -> ${OUT} ==="
