#!/bin/bash
#SBATCH --job-name=mae_baselines
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/home/jaehoonjeong/data/MetaAgentEvolution_Release/logs/%x.%j.out
#SBATCH --error=/home/jaehoonjeong/data/MetaAgentEvolution_Release/logs/%x.%j.err

# Step 3/3 — Baselines (init_persona / raw / self-refine) on MBPP test + HumanEval

set -euo pipefail

REPO="${REPO:-/home/jaehoonjeong/data/MetaAgentEvolution_Release}"
# shellcheck source=scripts/sbatch/common_mbpp.sh
source "${REPO}/scripts/sbatch/common_mbpp.sh"
setup_job_env
print_experiment_config

OUT_BASE="${REPO}/results/mbpp/baselines"
OUT_HE="${REPO}/results/humaneval/baselines"
mkdir -p "${OUT_BASE}" "${OUT_HE}"

run_and_score() {
    local PIPELINE="$1"
    local DATASET="$2"
    local OUT_FILE="$3"
    local ROSTER="${4:-}"

    echo ""
    echo "======================================================================"
    echo "=== Pipeline: ${PIPELINE} | Dataset: ${DATASET} ==="
    echo "======================================================================"

    ROSTER_ARG=""
    if [ -n "${ROSTER}" ]; then
        ROSTER_ARG="--roster_path ${ROSTER}"
    fi

    python scripts/run_inference.py \
        --pipeline "${PIPELINE}" \
        --dataset "${DATASET}" \
        --split test \
        --seed "${SEED}" \
        --max_refine_iters "${MAX_REFINE_ITERS}" \
        --output_file "${OUT_FILE}" \
        ${ROSTER_ARG}

    python scripts/score_outputs.py \
        --input "${OUT_FILE}" \
        --dataset "${DATASET}" \
        --split test
}

echo "=== [1/6] MBPP | Init persona (5 domain critics + GMRoutingPipeline) ==="
run_and_score evolved mbpp "${OUT_BASE}/init_persona_mbpp.jsonl" "${INIT_ROSTER}"

echo "=== [2/6] MBPP | Raw (1-pass) ==="
run_and_score raw mbpp "${OUT_BASE}/raw_mbpp.jsonl"

echo "=== [3/6] MBPP | Self-Refine (${MAX_REFINE_ITERS} iters) ==="
run_and_score self-refine mbpp "${OUT_BASE}/selfrefine_mbpp.jsonl"

echo "=== [4/6] HumanEval | Init persona ==="
run_and_score evolved humaneval "${OUT_HE}/init_persona_humaneval.jsonl" "${INIT_ROSTER}"

echo "=== [5/6] HumanEval | Raw (1-pass) ==="
run_and_score raw humaneval "${OUT_HE}/raw_humaneval.jsonl"

echo "=== [6/6] HumanEval | Self-Refine (${MAX_REFINE_ITERS} iters) ==="
run_and_score self-refine humaneval "${OUT_HE}/selfrefine_humaneval.jsonl"

echo ""
echo "=== Baseline summary ==="
for f in "${OUT_BASE}"/*.score.json "${OUT_HE}"/*.score.json; do
    [ -f "${f}" ] || continue
    echo "--- ${f} ---"
    python3 -c "
import json, pathlib
d = json.loads(pathlib.Path('${f}').read_text())
print(f'  {d[\"input\"].split(\"/\")[-1]}: Pass@1={d[\"pass_at_1\"]:.2f}% ({d[\"passed\"]}/{d[\"total\"]})')
"
done

echo "=== Baselines done ==="
