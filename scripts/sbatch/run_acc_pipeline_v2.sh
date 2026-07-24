#!/bin/bash
#SBATCH --job-name=acc_pipe_v2
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log

# 재빌드(run_acc_rebuild.sh) 뒤에 붙는 조립 단계 — 분할 → SFT 자산 → 라벨패키지 →
# 4조건 학습 체인 제출까지 한 번에 간다. GPU는 쓰지 않지만 클러스터 표준 스펙을 따른다.
#
# 사용: sbatch --dependency=afterok:<rebuild_jobid> scripts/sbatch/run_acc_pipeline_v2.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source ~/.bashrc
source ~/data/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-evolving_moe}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src"

CORPUS="${CORPUS:-export/acc_v2}"
VAL="${VAL:-750}"
TEST="${TEST:-750}"
PKG="${PKG:-export/acc_binning_seed20210111_v2}"
# 실측: prompt+참조솔루션 토큰 p95=1523 p99=2202 → 3072면 잘림 0.2%.
# (에이전트 코드 타깃 시절의 8192는 붕괴한 장문 타깃 때문이었다)
export MAX_LENGTH="${MAX_LENGTH:-3072}"
export GRAD_CKPT="${GRAD_CKPT:-true}"
export EVAL_DATA_DIR="${EVAL_DATA_DIR:-${CORPUS}/sft}"

echo "=== [1/4] problem_id 단위 분할 (val=${VAL} test=${TEST}) ==="
python scripts/split_acc_problems.py "${CORPUS}/acc_all_verified.jsonl" \
    --out_dir "${CORPUS}" --val "${VAL}" --test "${TEST}" \
    --prefer_train "${CORPUS}/labeled_problem_ids.json"

echo "=== [2/4] SFT 자산 + evolved 라벨패키지 ==="
python scripts/build_acc_sft_assets.py --corpus_dir "${CORPUS}" --pkg "${PKG}"

echo "=== [3/4] random / human-prior 라벨패키지 (v2 볼륨에 count-match) ==="
python scripts/build_ablation_packages.py --dataset acc \
    --evolved "${PKG}/binning_labels.jsonl" \
    --src "${CORPUS}/sft/acc_train.jsonl" \
    --suffix _v2

echo "=== [4/4] 4조건 학습 체인 제출 ==="
LAUNCH=scripts/sbatch/launch_sft_by_experts.sh

# 조건4 Evolved: 로스터 11명(cap9) 롤링체인 + shared(min10)는 같은 로스터 디렉터리로
MAX_N_SOLVED=9 "${LAUNCH}" "${PKG}"
MIN_N_SOLVED=10 EXPERT_ID=shared \
    OUTPUT_DIR="checkpoints/expert_sft/acc_seed20210111_v2_cap9/shared" \
    RUN_NAME="sft_llama3_acc_seed20210111_v2_shared" \
    LABEL_PACKAGE="${PKG}" \
    sbatch --job-name=sft_expert_shared --export=ALL scripts/sbatch/train_sft_by_expert.sh

# 조건2 Random / 조건3 Human-prior
MAX_N_SOLVED=9 "${LAUNCH}" export/acc_binning_seedrnd_v2
MAX_N_SOLVED=9 "${LAUNCH}" export/acc_binning_seedhp_v2

# 조건1 Dense: 라벨 전체(밴드 없음) 단일 어댑터
EXPERT_ID=shared LABEL_PACKAGE="${PKG}" \
    OUTPUT_DIR="checkpoints/dense_sft/acc_seed20210111_v2" \
    RUN_NAME="sft_llama3_acc_seed20210111_v2_dense" \
    sbatch --job-name=sft_dense_v2 --export=ALL scripts/sbatch/train_sft_by_expert.sh

echo "=== 제출 완료 ==="
squeue -u "$USER" -o "%.10i %.28j %.8T %.10M %R"
