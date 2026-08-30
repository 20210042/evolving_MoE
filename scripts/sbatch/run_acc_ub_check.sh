#!/bin/bash
#SBATCH --job-name=acc_ub_check
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log

# 코딩(acc) v2 위에서 BinningPipeline+persona+gemma UB 최초 측정. QASC와 동일 질문:
# 샘플링 없이(greedy)도 union UB가 유지되는가 — QASC는 유지됐다(94.60→94.82).
# 사용: MODE=sampled|greedy sbatch scripts/sbatch/run_acc_ub_check.sh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source ~/.bashrc; source ~/data/miniconda3/etc/profile.d/conda.sh; conda activate evolving_moe
export HF_HOME="${HF_HOME:-/data5/jaehoonjeong/.cache/huggingface}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src"

MODE="${MODE:?MODE=sampled|greedy 필요}"
CONFIG="configs/acc_eval_a4b_v2_${MODE}.yaml"
ROSTER="results/acc/seed20210111/roster_final.json"
RESDIR="results/acc/seed20210111_v2/ub_check"
mkdir -p "${RESDIR}"
MAX_ITEMS="${MAX_ITEMS:-751}"
TAG="${MODE}"
[ "${MAX_ITEMS}" != "751" ] && TAG="${MODE}_smoke"
OUT="${RESDIR}/inference_test_binning_${TAG}.jsonl"

echo "=== [1/2] ${MODE} 생성: 코딩 로스터 UB (persona+gemma, v2) ==="
python scripts/run_inference.py \
    --config "${CONFIG}" \
    --dataset acc \
    --split test \
    --seed 20210111 \
    --data_dir export/acc_v2 \
    --pipeline binning \
    --roster_path "${ROSTER}" \
    --max_items "${MAX_ITEMS}" \
    --output_file "${OUT}"

echo "=== [2/2] 채점 ==="
python scripts/score_binning.py \
    --input "${OUT}" \
    --dataset acc --split test --data_dir export/acc_v2

echo "=== acc_ub_check(${MODE}) 완료 -> ${OUT%.jsonl}.binned.jsonl ==="
