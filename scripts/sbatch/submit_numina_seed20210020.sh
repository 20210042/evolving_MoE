#!/bin/bash
# seed20210020: SATURATED RUN — add_only(Phase-2 delete 봉쇄), topic 기본형, batch 100.
#   로스터가 단조증가해 커지므로 스텝당 생성비용↑ → seed16보다 훨씬 오래 걸림(대략 2배+).
#   N_RESUME=2 체인(main 48h + resume 48h×2 = 최대 144h). config = numina_train_seed20.yaml.
# Usage: bash submit_numina_seed20210020.sh [--resume]
set -euo pipefail
REPO="${REPO:-/home/jaehoonjeong/data/MetaAgentEvolution_Release}"

# ── 실험 파라미터 ───────────────────────────────────────────────
SEED=20210020
DATASET="numina_cot"
EVOL_CFG="configs/numina_train_seed20.yaml"
TRAIN_SIZE=62185
MAX_EPOCHS=1
BATCH_SIZE=100

# ── SLURM 자원 (수정은 여기서만) ────────────────────────────────
GPUS=2          # PRO6000 장수
CPUS=4          # cpus-per-task
MEM=64G         # 시스템 RAM
TIME=48:00:00
N_RESUME=2      # 포화 로스터가 커서 스텝당 느림 → 48h 초과 예상. main+resume×2.

# ────────────────────────────────────────────────────────────────
submit_one() {  # $1=resume_flag  $2=dependency_jobid(optional)
    local rflag="$1" dep="$2" depopt="" exclopt=""
    [ -n "$dep" ] && depopt="--dependency=afternotok:${dep}"
    [ -n "${EXCLUDE:-}" ] && exclopt="--exclude=${EXCLUDE}"   # e.g. EXCLUDE=n05 (전력문제 노드 제외)
    SEED=${SEED} DATASET=${DATASET} TRAIN_SIZE=${TRAIN_SIZE} MAX_EPOCHS=${MAX_EPOCHS} BATCH_SIZE=${BATCH_SIZE} EVOL_CONFIG=${EVOL_CFG} \
      sbatch --parsable --job-name=mae_evolve_seed20 ${depopt} ${exclopt} \
        --gres=gpu:PRO6000:${GPUS} --cpus-per-task=${CPUS} --mem=${MEM} --time=${TIME} \
        --output="${REPO}/logs/%x.%j.out" --error="${REPO}/logs/%x.%j.err" \
        "${REPO}/scripts/sbatch/run_math_evolution.sh" ${rflag}
}

FIRST_FLAG=""
if [ "${1:-}" = "--resume" ]; then FIRST_FLAG="--resume"; fi
EVOL_JOB=$(submit_one "${FIRST_FLAG}" "")
echo "Submitted saturated seed${SEED}: ${EVOL_JOB}  (GPU=${GPUS} CPU=${CPUS} MEM=${MEM} TIME=${TIME})"

PREV=${EVOL_JOB}
for i in $(seq 1 ${N_RESUME}); do
    RJOB=$(submit_one "--resume" "${PREV}")
    echo "  + resume #${i}: ${RJOB}  (afternotok:${PREV})"
    PREV=${RJOB}
done
