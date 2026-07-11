#!/bin/bash
#SBATCH --job-name=mae_probe_tput
#SBATCH --gres=gpu:PRO6000:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/jaehoonjeong/data/MetaAgentEvolution_Release/logs/%x.%j.out
#SBATCH --error=/home/jaehoonjeong/data/MetaAgentEvolution_Release/logs/%x.%j.err

# Timed throughput probe: 1 expert × PROBE_N train problems with optimized batching.
# Measures real gen/sec to size the full-train per-expert sweep accurately.
set -euo pipefail
REPO="${REPO:-/home/jaehoonjeong/data/MetaAgentEvolution_Release}"
source "${REPO}/scripts/sbatch/common_bigmath.sh"
setup_job_env

EVAL_CONFIG="${EVAL_CONFIG:-configs/numina_train_seed16.yaml}"   # tp=2, A4B
ROSTER="${ROSTER:-results/numina_cot/seed20210016/ub_eval/roster_c_49174.json}"
PROBE_N="${PROBE_N:-2000}"
IBS="${IBS:-256}"
OUT="results/numina_cot/seed20210016/_probe_inf.jsonl"

rm -f "${OUT}"
echo "=== PROBE: N=${PROBE_N} infer_batch_size=${IBS} config=${EVAL_CONFIG} ==="
T0=$(date +%s)
python scripts/run_inference.py \
    --config "${EVAL_CONFIG}" \
    --dataset numina_cot --split train \
    --pipeline evolved \
    --roster_path "${ROSTER}" \
    --seed 20210016 --max_items "${PROBE_N}" \
    --infer_batch_size "${IBS}" \
    --output_file "${OUT}"
T1=$(date +%s)
DT=$((T1-T0))
echo "=== PROBE DONE: ${PROBE_N} problems in ${DT}s = $(python3 -c "print(f'{${PROBE_N}/${DT}:.2f}')") prob/s ==="
echo "=== 62185 환산: $(python3 -c "print(f'{62185/(${PROBE_N}/${DT})/3600:.1f}')")h / expert (incl. model load in DT) ==="
