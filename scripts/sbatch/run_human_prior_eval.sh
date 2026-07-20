#!/bin/bash
#SBATCH --job-name=human_prior_eval
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --exclude=n01,n02,master
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log

# Human-prior roster eval without changing run_inference.py.
#
# Usage examples:
#   DOMAIN=lbox sbatch scripts/sbatch/run_human_prior_eval.sh
#   DOMAIN=qasc sbatch scripts/sbatch/run_human_prior_eval.sh
#   DOMAIN=taco sbatch scripts/sbatch/run_human_prior_eval.sh
#
# Optional env overrides: REPO, CONDA_SH, CONDA_ENV, SPLIT, DATA_DIR, EVAL_CONFIG,
# ROSTER, OUT, EVAL_SIZE, SEED.

set -euo pipefail

REPO="${REPO:-/home/jongbinwon/data/evolving_MoE}"
CONDA_SH="${CONDA_SH:-/home/jongbinwon/data/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-evolving_moe}"
DOMAIN="${DOMAIN:-lbox}"
SEED="${SEED:-20260716}"

cd "${REPO}"
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

export PYTHONPATH="${REPO}/src"
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_DISABLE_FLASHINFER=1
export HF_HOME="${HF_HOME:-/data6/jongbinwon/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

case "${DOMAIN}" in
  lbox)
    DATASET="${DATASET:-lbox}"
    SPLIT="${SPLIT:-test}"
    DATA_DIR="${DATA_DIR:-export/lbox}"
    EVAL_CONFIG="${EVAL_CONFIG:-configs/lbox_eval_a4b.yaml}"
    ROSTER="${ROSTER:-configs/lbox_human_prior_roster.json}"
    OUT="${OUT:-results/lbox_human_prior/inference_${SPLIT}.jsonl}"
    SCORE_SCRIPT="${SCORE_SCRIPT:-scripts/score_outputs.py}"
    ;;
  qasc)
    DATASET="${DATASET:-qasc}"
    SPLIT="${SPLIT:-validation}"
    DATA_DIR="${DATA_DIR:-export/qasc}"
    EVAL_CONFIG="${EVAL_CONFIG:-configs/qasc_eval_a4b.yaml}"
    ROSTER="${ROSTER:-configs/qasc_human_prior_roster.json}"
    OUT="${OUT:-results/qasc_human_prior/inference_${SPLIT}.jsonl}"
    SCORE_SCRIPT="${SCORE_SCRIPT:-scripts/score_outputs.py}"
    ;;
  taco|acc)
    DATASET="${DATASET:-acc}"
    SPLIT="${SPLIT:-test}"
    DATA_DIR="${DATA_DIR:-export/acc_taco_official}"
    EVAL_CONFIG="${EVAL_CONFIG:-configs/acc_eval_a4b.yaml}"
    ROSTER="${ROSTER:-configs/acc_taco_human_prior_roster.json}"
    OUT="${OUT:-results/acc_taco_human_prior/inference_${SPLIT}.jsonl}"
    SCORE_SCRIPT="${SCORE_SCRIPT:-scripts/score_outputs.py}"
    ;;
  *)
    echo "ERROR: unknown DOMAIN='${DOMAIN}' (expected lbox, qasc, taco)"
    exit 2
    ;;
esac

if [ ! -f "${ROSTER}" ]; then
  echo "ERROR: roster not found: ${ROSTER}"
  exit 1
fi

mkdir -p "$(dirname "${OUT}")"
rm -f "${OUT}"

echo "=== human-prior eval ==="
echo "domain=${DOMAIN} dataset=${DATASET} split=${SPLIT}"
echo "config=${EVAL_CONFIG}"
echo "data_dir=${DATA_DIR}"
echo "roster=${ROSTER}"
echo "out=${OUT}"
echo "eval_size=${EVAL_SIZE:-all}"
echo "hf_home=${HF_HOME} offline=${HF_HUB_OFFLINE}"
echo "node=${SLURMD_NODENAME:-unknown} gpus=${SLURM_GPUS_ON_NODE:-unknown}"

cmd=(
  python scripts/run_inference.py
  --config "${EVAL_CONFIG}"
  --dataset "${DATASET}"
  --split "${SPLIT}"
  --seed "${SEED}"
  --data_dir "${DATA_DIR}"
  --pipeline evolved
  --roster_path "${ROSTER}"
  --output_file "${OUT}"
)
if [ -n "${EVAL_SIZE:-}" ]; then
  cmd+=(--max_items "${EVAL_SIZE}")
fi
"${cmd[@]}"

python "${SCORE_SCRIPT}" \
  --input "${OUT}" \
  --dataset "${DATASET}" \
  --split "${SPLIT}" \
  --data_dir "${DATA_DIR}"

echo "=== human-prior eval done ==="
