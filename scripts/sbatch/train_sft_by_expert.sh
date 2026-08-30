#!/bin/bash
#SBATCH --job-name=sft_llama3_expert
#SBATCH --gres=gpu:PRO6000:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log

# per-expert SFT (합의 2026-07-13): system prompt는 baseline GEN으로 통일,
# expert별로 binning 라벨(per_expert==1)에 따라 보는 데이터만 다름.
# 사용: LABEL_PACKAGE=export/qasc_binning_seed20210211 EXPERT_ID=c_33055 [MAX_N_SOLVED=10] sbatch $0

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"

source ~/.bashrc
source ~/data/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-evolving_moe}"

export PYTHONPATH="$REPO/src"
# HF cache는 /data5 고정 (home quota 작음; ~/.bashrc의 HF_HOME은 비인터랙티브 셸에서 미적용)
export HF_HOME="${HF_HOME:-/data5/jaehoonjeong/.cache/huggingface}"
# train_sft.py 기본 entity는 종빈 팀(jongbin-kr-skiml_moe) — 멤버가 아니면 wandb init 실패
export WANDB_ENTITY="${WANDB_ENTITY:-wjdwongs1234-none}"

LABEL_PACKAGE="${LABEL_PACKAGE:-}"
EXPERT_ID="${EXPERT_ID:-}"
if [ -z "${LABEL_PACKAGE}" ] || [ -z "${EXPERT_ID}" ]; then
    echo "ERROR: LABEL_PACKAGE and EXPERT_ID are required." >&2
    echo "e.g. LABEL_PACKAGE=export/qasc_binning_seed20210211 EXPERT_ID=c_33055 sbatch $0" >&2
    exit 1
fi
MAX_N_SOLVED="${MAX_N_SOLVED:-}"
MIN_N_SOLVED="${MIN_N_SOLVED:-}"   # shared 어댑터(EXPERT_ID=shared)용: n_solved>=이 값만 학습

# export/<domain>_binning_seed<seed> → domain / seed
PKG_BASE="$(basename "${LABEL_PACKAGE}")"
DOMAIN="${PKG_BASE%%_binning_*}"
BIN_SEED="${PKG_BASE##*_seed}"

# lbox는 valid, qasc는 validation split 파일명을 사용
case "${DOMAIN}" in
    lbox) EVAL_SPLIT="${EVAL_SPLIT:-valid}" ;;
    *)    EVAL_SPLIT="${EVAL_SPLIT:-validation}" ;;
esac

CAP_SUFFIX=""
CAP_FLAG=()
if [ -n "${MAX_N_SOLVED}" ]; then
    CAP_SUFFIX="_cap${MAX_N_SOLVED}"
    CAP_FLAG=(--max_n_solved "${MAX_N_SOLVED}")
fi
if [ -n "${MIN_N_SOLVED}" ]; then
    CAP_SUFFIX="${CAP_SUFFIX}_min${MIN_N_SOLVED}"
    CAP_FLAG+=(--min_n_solved "${MIN_N_SOLVED}")
fi
# 출력 경로/run_name에 붙는 임의 접미사(예: "_fewshot") — 기존 체크포인트를 안 덮어쓰고
# 나란히 두기 위한 것. 미설정 시 기존 동작 그대로.
CAP_SUFFIX="${CAP_SUFFIX}${EXTRA_SUFFIX:-}"
# 코딩(acc)처럼 시퀀스가 긴 도메인용. 미설정 시 TRL 기본(1024) 유지 = QASC 기존 동작 불변.
LEN_FLAG=()
if [ -n "${MAX_LENGTH:-}" ]; then
    LEN_FLAG=(--max_length "${MAX_LENGTH}")
fi
# 긴 시퀀스에서 메모리 확보용(미설정 시 기존 동작 유지)
GC_FLAG=()
if [ -n "${GRAD_CKPT:-}" ]; then
    GC_FLAG=(--gradient_checkpointing "${GRAD_CKPT}")
fi
# 학습셋은 label_package의 source_train_jsonl에서 오지만 per-epoch eval은 data_dir에서
# 온다. 코퍼스를 새로 만든 경우(export/acc_v2) 여기로 홀드아웃을 지정해야 옛 누수
# validation으로 평가하지 않는다. 미설정 시 기존 동작 유지.
EVAL_DIR_FLAG=()
if [ -n "${EVAL_DATA_DIR:-}" ]; then
    EVAL_DIR_FLAG=(--eval_data_dir "${EVAL_DATA_DIR}")
fi
# 2026-07-26 persona+few-shot 확장(§11 파일럿). 미설정 시(기존 동작, Random/Human-prior
# 포함) build_baseline_prompt로 통일 — 이 플래그를 안 쓰면 완전히 예전과 같다.
ROSTER_FLAG=()
if [ -n "${ROSTER_PATH:-}" ]; then
    ROSTER_FLAG=(--roster_path "${ROSTER_PATH}" --n_fewshot "${N_FEWSHOT:-2}")
fi

RUN_NAME="${RUN_NAME:-sft_llama3_${DOMAIN}_seed${BIN_SEED}${CAP_SUFFIX}_${EXPERT_ID}}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/expert_sft/${DOMAIN}_seed${BIN_SEED}${CAP_SUFFIX}/${EXPERT_ID}}"
DEEPSPEED_CONFIG="$REPO/configs/deepspeed_zero3.json"
NPROC_PER_NODE="${NPROC_PER_NODE:-${SLURM_GPUS_ON_NODE:-2}}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"

# rolling chain: 시작 시점에 다음 expert 잡 하나만 제출(afterany:자기 자신).
# 전체를 미리 걸어두지 않으므로 큐에는 항상 러닝 1개 + 대기 1개만 보인다.
if [ -n "${REMAINING_EXPERTS:-}" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
    read -r NEXT_EXPERT NEXT_REMAINING <<< "${REMAINING_EXPERTS}"
    echo "=== rolling chain: 다음 expert ${NEXT_EXPERT} 제출 (afterany:${SLURM_JOB_ID}) ==="
    EXPERT_ID="${NEXT_EXPERT}" REMAINING_EXPERTS="${NEXT_REMAINING:-}" RUN_NAME="" OUTPUT_DIR="" \
        sbatch --job-name="sft_expert_${NEXT_EXPERT}" \
        --dependency="afterany:${SLURM_JOB_ID}" \
        --export=ALL \
        "${REPO}/scripts/sbatch/train_sft_by_expert.sh"
fi

echo "=== expert SFT 시작: ${DOMAIN}/${EXPERT_ID} (max_n_solved=${MAX_N_SOLVED:-none}) / 출력: ${OUTPUT_DIR} ==="
echo "=== GPUs: ${NPROC_PER_NODE}, GA: ${GRADIENT_ACCUMULATION_STEPS}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} ==="

srun --ntasks=1 --gpus-per-task="${NPROC_PER_NODE}" --chdir="$REPO" \
    torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" "$REPO/src/train_sft.py" \
    --model_name_or_path "meta-llama/Llama-3.1-8B-Instruct" \
    --dtype bfloat16 \
    --attn_implementation sdpa \
    --label_package "${LABEL_PACKAGE}" \
    --expert_id "${EXPERT_ID}" \
    "${CAP_FLAG[@]}" \
    "${LEN_FLAG[@]}" \
    "${GC_FLAG[@]}" \
    "${ROSTER_FLAG[@]}" \
    --eval_dataset "${DOMAIN}" \
    --eval_split "${EVAL_SPLIT}" \
    --data_dir "export/${DOMAIN}" \
    "${EVAL_DIR_FLAG[@]}" \
    --data_ratio 1.0 \
    --seed 42 \
    --train_sft_with_lora true \
    --sft_lora_rank 16 \
    --sft_lora_alpha 32 \
    --sft_lora_dropout 0.05 \
    --output_dir "${OUTPUT_DIR}" \
    --run_name "${RUN_NAME}" \
    --num_train_epochs 7 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --learning_rate 2e-5 \
    --lr_scheduler_type cosine \
    --bf16 true \
    --deepspeed "${DEEPSPEED_CONFIG}" \
    --logging_steps 10 \
    --eval_strategy steps \
    --eval_steps 100 \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 3 \
    --load_best_model_at_end true \
    --metric_for_best_model eval_loss \
    --report_to wandb \
    --wandb_project "${WANDB_PROJECT:-sft_expert}" \
    --push_to_hub "${PUSH_TO_HUB:-False}"

echo "=== expert SFT 완료: ${EXPERT_ID}. 체크포인트: ${OUTPUT_DIR} ==="
