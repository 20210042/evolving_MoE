#!/bin/bash
#SBATCH --job-name=acc_abl_eval
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log

# 코딩(TACO/acc) 4조건 배포평가 — 조건별 per-expert 실생성 → 코드실행 채점.
# QASC용 run_ablation_eval.sh의 acc 판. 3단계 배포스윕은 이번 범위 밖(핵심 4셀만).
#
# 홀드아웃은 acc_eval_clean290.jsonl — acc_validation 500 중 problem_id가 학습에
# 노출되지 않은 290문제. 원본 500은 42%가 train과 problem_id를 공유해(행 단위 tail
# split 버그) 조건 간 비교가 Dense 쪽으로 편향된다.
#
# 사용: COND=evolved|rnd|hp|dense sbatch scripts/sbatch/run_acc_ablation_eval.sh
# 옵션: SMOKE=N (앞 N문제만), BATCH, MAX_NEW, MAX_LEN

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source ~/.bashrc
source ~/data/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-evolving_moe}"
export HF_HOME="${HF_HOME:-/data5/jaehoonjeong/.cache/huggingface}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src"
# 배치 생성은 KV 캐시가 스텝마다 늘어 단편화가 심하다. 앞선 OOM에서 예약-미사용이
# 18.6GiB였다(할당 74GiB / 한계 95GiB). expandable_segments가 그 대부분을 회수한다.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

COND="${COND:?COND=evolved|rnd|hp|dense 필요}"
case "$COND" in
  evolved) CKPT="checkpoints/expert_sft/acc_seed20210111_cap9"; LABEL="Evolved MoE (cap9)" ;;
  rnd)     CKPT="checkpoints/expert_sft/acc_seedrnd_cap9";       LABEL="Random-partition MoE" ;;
  hp)      CKPT="checkpoints/expert_sft/acc_seedhp_cap9";        LABEL="Human-prior MoE" ;;
  dense)   CKPT="checkpoints/dense_sft/acc_seed20210111";        LABEL="Dense SFT (anchor)" ;;
  *) echo "unknown COND=$COND" >&2; exit 1 ;;
esac

SRC="${SRC:-export/acc/acc_eval_clean290.jsonl}"
RES="results/acc/seed20210111/ablation"
BATCH="${BATCH:-8}"
# 참조 솔루션 타깃은 p50 ~100 / p90 ~330 토큰이다. 6144는 붕괴한 생성이 상한까지
# 달리면서 OOM과 28시간 런타임을 만든 값이지 필요한 값이 아니었다.
MAX_NEW="${MAX_NEW:-2048}"
MAX_LEN="${MAX_LEN:-4096}"
REP_PEN="${REP_PEN:-1.05}"        # 주석 무한루프 차단 (진화 경로가 쓰던 값)
RESUME="${RESUME:-1}"             # 전문가 단위 부분 저장 재사용

TAG="$COND"
LIMIT_FLAG=()
if [ -n "${SMOKE:-}" ]; then
    TAG="${COND}_smoke"
    LIMIT_FLAG=(--limit "${SMOKE}")
fi
[ "${RESUME}" = "1" ] && LIMIT_FLAG+=(--resume)
OUT="${RES}/inference_clean290_${TAG}.jsonl"

echo "=== [1/2] 생성: ${LABEL} (${CKPT}) tag=${TAG} ==="
python scripts/generate_lora_binning.py \
    --dataset acc \
    --split clean290 \
    --src "${SRC}" \
    --out_dir "${RES}" \
    --ckpt "${CKPT}" \
    --tag "${TAG}" \
    --batch "${BATCH}" \
    --max_len "${MAX_LEN}" \
    --max_new_tokens "${MAX_NEW}" \
    --repetition_penalty "${REP_PEN}" \
    "${LIMIT_FLAG[@]}"

echo "=== [2/2] 코드실행 채점 (solve 매트릭스) ==="
# ref(test_cases/eval_spec)는 acc_validation.jsonl에서 id로 붙는다 — clean290은 그 부분집합.
python scripts/score_binning.py \
    --input "${OUT}" \
    --dataset acc \
    --split validation \
    --data_dir export/acc

echo "=== acc_abl_eval 완료: ${COND} -> ${OUT%.jsonl}.binned.jsonl ==="
