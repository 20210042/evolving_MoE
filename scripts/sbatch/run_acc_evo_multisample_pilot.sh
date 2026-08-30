#!/bin/bash
# 진화 레짐 다회시행 파일럿 (scripts/evo_multisample_pilot.py).
#   자원/환경은 acc 진화 잡(submit_acc_seed20210101_smoke.sh + common_bigmath.sh)과 동일하게 맞춤:
#   gemma-4-26B-A4B-it tp_size=2 → PRO6000×2, CPU 4, 64G. vllm 잡이라 FLASHINFER 비활성 2플래그 필수.
# Usage:
#   sbatch --job-name=acc_evo_ms_smoke scripts/sbatch/run_acc_evo_multisample_pilot.sh   # 기본=스모크
#   N_PROBLEMS=50 K=5 OUT=results/acc/evo_multisample_pilot.md \
#     sbatch --job-name=acc_evo_ms_pilot scripts/sbatch/run_acc_evo_multisample_pilot.sh
#SBATCH --gres=gpu:PRO6000:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/jaehoonjeong/data/MetaAgentEvolution_Release/logs/%x.%j.log
#SBATCH --error=/home/jaehoonjeong/data/MetaAgentEvolution_Release/logs/%x.%j.log

set -euo pipefail
REPO="${REPO:-/home/jaehoonjeong/data/MetaAgentEvolution_Release}"
cd "${REPO}"

source ~/.bashrc
source ~/data/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-evolving_moe}"
export PYTHONPATH="${REPO}/src"
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_DISABLE_FLASHINFER=1
export HF_HOME="${HF_HOME:-/data5/jaehoonjeong/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/data5/jaehoonjeong/.cache/huggingface}"

N_PROBLEMS="${N_PROBLEMS:-5}"
K="${K:-2}"
ARMS="${ARMS:-persona,persona_fewshot}"
OUT="${OUT:-results/acc/evo_multisample_pilot_smoke.md}"
EXTRA="${EXTRA:-}"   # 예: "--problem_ids ....json --prior_solver ....json"

echo "=== evo multisample pilot: n_problems=${N_PROBLEMS} k=${K} arms=${ARMS} out=${OUT} extra=${EXTRA} ==="
# shellcheck disable=SC2086
python scripts/evo_multisample_pilot.py \
    --n_problems "${N_PROBLEMS}" \
    --k "${K}" \
    --arms "${ARMS}" \
    --out "${OUT}" \
    ${EXTRA}
echo "=== done -> ${OUT} ==="
