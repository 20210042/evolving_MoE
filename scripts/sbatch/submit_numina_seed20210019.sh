#!/bin/bash
# seed20210019: 게이트 제거 로직 수정(windowed deletion) — seed17(실패유형 scout)과 프롬프트 동일,
#   게이트만 누적 unique-rate 윈도로 통일(W=30, floor=0, cooldown=15). config = numina_train_seed19.yaml.
# Usage: bash submit_numina_seed20210019.sh [--resume]
set -euo pipefail
REPO="${REPO:-/home/jaehoonjeong/data/MetaAgentEvolution_Release}"

# ── 실험 파라미터 ───────────────────────────────────────────────
SEED=20210019
DATASET="numina_cot"
EVOL_CFG="configs/numina_train_seed19.yaml"
TRAIN_SIZE=62185
MAX_EPOCHS=1
BATCH_SIZE=25

# ── SLURM 자원 (수정은 여기서만) ────────────────────────────────
GPUS=2          # PRO6000 장수
CPUS=4          # cpus-per-task
MEM=64G         # 시스템 RAM
TIME=48:00:00
N_RESUME=1      # batch25 진화 ~54h 추정 → main(48h)+resume(48h)=96h. 1개면 충분(체인 cascade 모호성 회피)

# ────────────────────────────────────────────────────────────────
submit_one() {  # $1=resume_flag  $2=dependency_jobid(optional)
    local rflag="$1" dep="$2" depopt=""
    # afternotok: 직전 잡이 실패/타임아웃(TIMEOUT=failed)일 때만 실행. 정상 완료(COMPLETED) 시엔 실행 안 됨.
    [ -n "$dep" ] && depopt="--dependency=afternotok:${dep}"
    SEED=${SEED} DATASET=${DATASET} TRAIN_SIZE=${TRAIN_SIZE} MAX_EPOCHS=${MAX_EPOCHS} BATCH_SIZE=${BATCH_SIZE} EVOL_CONFIG=${EVOL_CFG} \
      sbatch --parsable --job-name=mae_evolve_seed19 ${depopt} \
        --gres=gpu:PRO6000:${GPUS} --cpus-per-task=${CPUS} --mem=${MEM} --time=${TIME} \
        --output="${REPO}/logs/%x.%j.out" --error="${REPO}/logs/%x.%j.err" \
        "${REPO}/scripts/sbatch/run_math_evolution.sh" ${rflag}
}

# main 잡 (fresh). 인자로 --resume 주면 첫 잡도 resume로.
FIRST_FLAG=""
if [ "${1:-}" = "--resume" ]; then FIRST_FLAG="--resume"; fi
EVOL_JOB=$(submit_one "${FIRST_FLAG}" "")
echo "Submitted evolution seed${SEED}: ${EVOL_JOB}  (GPU=${GPUS} CPU=${CPUS} MEM=${MEM} TIME=${TIME})"

# resume 체인: 직전 잡이 타임아웃/실패로 끝났을 때만(afternotok) --resume 으로 이어감.
# 본잡이 48h 안에 정상 완료하면 체인 전체가 자동 취소됨(낭비 없음).
PREV=${EVOL_JOB}
for i in $(seq 1 ${N_RESUME}); do
    RJOB=$(submit_one "--resume" "${PREV}")
    echo "  + resume #${i}: ${RJOB}  (afternotok:${PREV})"
    PREV=${RJOB}
done
