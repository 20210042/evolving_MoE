#!/bin/bash
# 진화 로스터(seed20212001 최종 9명)로 **train/valid/test 전량 라벨링**.
#   각 expert가 모든 문제를 독립으로 풀고 "누가 맞췄나"를 기록한다(라우팅 없음).
#
# 새 파이프라인을 만들지 않는다 — `evo_multisample_pilot.py`가 이미 같은 일을 하고,
# SNI 프롬프트(system=페르소나+정의 / user=pos example 2건+Input)를 진화와 동일하게 조립한다.
#   ⚠️ `--pipeline binning`(BinningPipeline)은 쓰지 않는다: build_expert_prompt에
#      definition/positive_examples/answer_line을 넘기지 않아 SNI에서 프롬프트가 달라진다.
#
# 디코딩·모델·max_tokens는 진화와 같은 config에서 읽는다(configs/sni_train_seed20212001.yaml).
# GPU 2장은 그 config의 vllm.tp_size=2 때문이고, 진화 잡(232380/235687)과 동일한 자원이다.
#
# Usage: K=3 sbatch --job-name=sni_binning scripts/sbatch/run_sni_binning_v4.sh
#SBATCH --gres=gpu:PRO6000:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
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
# ⚠️ HF 캐시 고정 — 안 걸면 워커가 홈 캐시를 보고 26B 가중치를 허브에서 새로 받다가 죽는다
# (job 235728 실패 원인). common_bigmath.sh와 같은 값.
export HF_HOME="${HF_HOME:-/data5/jaehoonjeong/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/data5/jaehoonjeong/.cache/huggingface}"
# vllm 필수 플래그 (flashinfer가 이 클러스터 Blackwell에서 엔진 init 크래시)
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_DISABLE_FLASHINFER=1

K="${K:-3}"
CFG="${CFG:-configs/sni_train_seed20212001.yaml}"
DATA_DIR="${DATA_DIR:-export/sni_v4}"
ROSTER="${ROSTER:-results/sni/seed20212001/roster_final.json}"
OUTDIR="${OUTDIR:-results/sni/binning_seed20212001}"
mkdir -p "${OUTDIR}"

# split별 전량 (사전 선별 없음). 크기는 export/sni_v4 실측.
run_split () {
  local SPLIT="$1" N="$2"
  echo "=== ${SPLIT}: ${N}문제 × 9명 × K=${K} ==="
  python scripts/evo_multisample_pilot.py \
      --config "${CFG}" \
      --dataset sni \
      --data_dir "${DATA_DIR}" \
      --split "${SPLIT}" \
      --roster_path "${ROSTER}" \
      --arms persona \
      --n_problems "${N}" \
      --k "${K}" \
      --gen_chunk 2000 \
      --score_workers 1 \
      --seed 0 \
      --out "${OUTDIR}/${SPLIT}.md" \
      --raw_out "${OUTDIR}/${SPLIT}_raw.jsonl"
}

# SPLITS 미지정이면 기존 동작(train valid test) 그대로.
#   진화 직후 평가만 원하면 SPLITS="test".
declare -A SPLIT_N=( [train]=69588 [valid]=8699 [test]=8699 )
for SP in ${SPLITS:-train valid test}; do
  run_split "${SP}" "${SPLIT_N[$SP]}"
done

echo "=== done ==="
wc -l "${OUTDIR}"/*_raw.jsonl
