#!/bin/bash
#SBATCH --job-name=mae_ub_eval
#SBATCH --gres=gpu:PRO6000:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/jaehoonjeong/data/MetaAgentEvolution_Release/logs/%x.%j.out
#SBATCH --error=/home/jaehoonjeong/data/MetaAgentEvolution_Release/logs/%x.%j.err

# Per-agent test UB from the final roster.
# Splits roster_final.json into single-agent rosters at runtime, runs each agent
# solo on the held-out test set, scores. Union across agents = oracle UB
# (compute union post-hoc from the inference_<id>.jsonl files).
#
# Env: SEED (required), EVAL_CONFIG (yaml; sets gen thinking + tp_size).
# --seed ${SEED} -> same held-out 500 as the MoE/epoch evals (comparable UB).

set -euo pipefail

REPO="${REPO:-/home/jaehoonjeong/data/MetaAgentEvolution_Release}"
source "${REPO}/scripts/sbatch/common_bigmath.sh"
setup_job_env

EVAL_CONFIG="${EVAL_CONFIG:-configs/bigmath_train_nothink.yaml}"
RESULTS_DIR="results/${DATASET}/seed${SEED}"
ROSTER_FINAL="${RESULTS_DIR}/roster_final.json"
OUT="${RESULTS_DIR}/ub_eval"

if [ ! -f "${ROSTER_FINAL}" ]; then
    echo "ERROR: roster_final.json not found at ${ROSTER_FINAL}"
    exit 1
fi

mkdir -p "${OUT}"
echo "=== UB eval seed${SEED} | config=${EVAL_CONFIG} ==="

# Split roster_final into single-agent rosters; print the agent ids.
PIDS=$(python - "${ROSTER_FINAL}" "${OUT}" <<'PY'
import json, sys, pathlib
final_path, out_dir = sys.argv[1], pathlib.Path(sys.argv[2])
roster = json.load(open(final_path))
ids = []
for agent in roster:
    aid = agent.get("id")
    if not aid:
        continue
    ids.append(aid)
    json.dump([agent], open(out_dir / f"roster_{aid}.json", "w"), ensure_ascii=False, indent=2)
print(" ".join(ids))
PY
)
echo "=== agents: ${PIDS} ==="

for PID in ${PIDS}; do
    echo "=== Running: ${PID} ==="
    # 재발방지: 옛 출력 있으면 run_inference resume가 생성 skip → stale 재사용. 항상 새로 생성.
    rm -f "${OUT}/inference_${PID}.jsonl"
    python scripts/run_inference.py \
        --config "${EVAL_CONFIG}" \
        --dataset "${DATASET}" --split test \
        --pipeline evolved \
        --roster_path "${OUT}/roster_${PID}.json" \
        --seed "${SEED}" --max_items "${EVAL_SIZE:-500}" \
        --output_file "${OUT}/inference_${PID}.jsonl"
    python scripts/score_outputs.py \
        --input "${OUT}/inference_${PID}.jsonl" \
        --dataset "${DATASET}" --split test
done

echo "=== UB eval done. Per-agent scores in ${OUT}/ (compute union post-hoc). ==="
