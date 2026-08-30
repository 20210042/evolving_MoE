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
# v2 재빌드본: 홀드아웃은 export/acc_v2/acc_test.jsonl (751문제, problem_id 단위로
# train과 완전 disjoint — 누수 0 재검증됨). 타깃은 검증된 참조 솔루션이라 붕괴 없음.
#
# 사용: COND=evolved|rnd|hp|dense|evolved_fewshot sbatch scripts/sbatch/run_acc_ablation_eval.sh
# 옵션: SMOKE=N (앞 N문제만), BATCH, MAX_NEW, MAX_LEN, SRC, EVAL_SPLIT, DATA_DIR

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
  evolved) CKPT="checkpoints/expert_sft/acc_seed20210111_v2_cap9"; LABEL="Evolved MoE (cap9)" ;;
  rnd)     CKPT="checkpoints/expert_sft/acc_seedrnd_v2_cap9";       LABEL="Random-partition MoE" ;;
  hp)      CKPT="checkpoints/expert_sft/acc_seedhp_v2_cap9";        LABEL="Human-prior MoE" ;;
  dense)   CKPT="checkpoints/dense_sft/acc_seed20210111_v2";        LABEL="Dense SFT (anchor)" ;;
  evolved_fewshot) CKPT="checkpoints/expert_sft/acc_seed20210111_v2_cap9_fewshot"; LABEL="Evolved MoE (cap9+persona+fewshot, §11)" ;;
  *) echo "unknown COND=$COND" >&2; exit 1 ;;
esac

# evolved_fewshot: 학습 때와 동일한 persona+few-shot을 추론에도 적용(train/inference 프롬프트
# 일치). 다른 COND는 빈 배열 = 기존 build_baseline_prompt 경로 그대로.
ROSTER_FLAG=()
if [ "$COND" = "evolved_fewshot" ]; then
    ROSTER_FLAG=(
        --roster_path "results/acc/seed20210111/roster_final.json"
        --label_package "export/acc_binning_seed20210111_v2"
        --max_n_solved 9
        --n_fewshot 2
    )
fi

SRC="${SRC:-export/acc_v2/acc_test.jsonl}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
DATA_DIR="${DATA_DIR:-export/acc_v2}"
RES="results/acc/seed20210111_v2/ablation"
# 생성기는 파일명을 inference_<split>_<tag>.jsonl로 짓는다. 채점이 찾는 이름과
# 어긋나면 생성만 끝나고 채점이 죽으므로 한 변수로 묶는다.
SPLIT_TAG="${SPLIT_TAG:-test751}"
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
OUT="${RES}/inference_${SPLIT_TAG}_${TAG}.jsonl"

echo "=== [1/2] 생성: ${LABEL} (${CKPT}) tag=${TAG} ==="
python scripts/generate_lora_binning.py \
    --dataset acc \
    --split "${SPLIT_TAG}" \
    --src "${SRC}" \
    --out_dir "${RES}" \
    --ckpt "${CKPT}" \
    --tag "${TAG}" \
    --batch "${BATCH}" \
    --max_len "${MAX_LEN}" \
    --max_new_tokens "${MAX_NEW}" \
    --repetition_penalty "${REP_PEN}" \
    "${LIMIT_FLAG[@]}" \
    "${ROSTER_FLAG[@]}"

echo "=== [2/2] 코드실행 채점 (solve 매트릭스) ==="
# ref(test_cases/eval_spec)는 acc_test.jsonl에서 id로 붙는다 (홀드아웃 그 자체).
python scripts/score_binning.py \
    --input "${OUT}" \
    --dataset acc \
    --split "${EVAL_SPLIT}" \
    --data_dir "${DATA_DIR}"

echo "=== acc_abl_eval 완료: ${COND} -> ${OUT%.jsonl}.binned.jsonl ==="
