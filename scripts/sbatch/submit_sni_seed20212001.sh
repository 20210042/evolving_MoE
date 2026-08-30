#!/bin/bash
# SNI 첫 진화 제출 — seed20212001. 설계 근거는 configs/sni_train_seed20212001.yaml 주석 참고.
#   1에폭 = 69,588문제 / batch 50 = 1,392스텝.
#   소요 추정 13~19시간 (파일럿 job 232084 실측 57.8 gen/s · 총 ~615,000 생성).
#   자원은 vllm tp_size=2라 GPU 2장 필요. 시간 48h.
#   ⚠️ TRAIN_SIZE/BATCH_SIZE 환경변수는 로그 출력용이고 실제 값은 config YAML에서 읽는다.
# Usage: bash scripts/sbatch/submit_sni_seed20212001.sh
#        RESUME=true bash scripts/sbatch/submit_sni_seed20212001.sh   # 중단 후 이어 돌리기
set -euo pipefail
REPO="${REPO:-/home/jaehoonjeong/data/MetaAgentEvolution_Release}"

SEED=20212001
DATASET="sni"
EVOL_CFG="configs/sni_train_seed20212001.yaml"
TRAIN_SIZE=69588
MAX_EPOCHS=1
BATCH_SIZE=50

GPUS=2
CPUS=4
MEM=64G
TIME=48:00:00

EVOL_JOB=$(SEED=${SEED} DATASET=${DATASET} TRAIN_SIZE=${TRAIN_SIZE} MAX_EPOCHS=${MAX_EPOCHS} BATCH_SIZE=${BATCH_SIZE} EVOL_CONFIG=${EVOL_CFG} RESUME="${RESUME:-false}" \
  sbatch --parsable --job-name=mae_evolve_sni \
    --gres=gpu:PRO6000:${GPUS} --cpus-per-task=${CPUS} --mem=${MEM} --time=${TIME} \
    --output="${REPO}/logs/%x.%j.out" --error="${REPO}/logs/%x.%j.err" \
    "${REPO}/scripts/sbatch/run_math_evolution.sh")
echo "Submitted sni seed${SEED}: ${EVOL_JOB}  (GPU=${GPUS} CPU=${CPUS} batch=${BATCH_SIZE} train=${TRAIN_SIZE})"
echo "  results -> results/sni/seed${SEED}/"
