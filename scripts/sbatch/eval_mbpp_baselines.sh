#!/bin/bash
#SBATCH --job-name=mae_baselines
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/home/jaehoonjeong/data/MetaAgentEvolution_Release/logs/%x.%j.out
#SBATCH --error=/home/jaehoonjeong/data/MetaAgentEvolution_Release/logs/%x.%j.err

# ─────────────────────────────────────────────────────────────────
#  MBPP Baseline Experiments
#  3 conditions × 2 datasets = 6 inference + 6 scoring runs
#
#  Conditions:
#    1) init_persona  — 3 generalist personas + GMRoutingPipeline (step-0)
#    2) raw           — 1-pass, no persona, no refinement
#    3) self_refine   — generate → neutral critic → refine (2 iters)
#
#  Datasets: MBPP Test (500) | HumanEval (164)
# ─────────────────────────────────────────────────────────────────

set -euo pipefail

REPO=/home/jaehoonjeong/data/MetaAgentEvolution_Release
cd "$REPO"

source /data5/jaehoonjeong/miniconda3/etc/profile.d/conda.sh
conda activate pro6000

export PYTHONPATH="$REPO/src"
DATA_ROOT="${DATA_DIR:-/home/jaehoonjeong/data/MultiAgent/Data}"
SEED="${SEED:-20210042}"
MODEL="${MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"
MAX_REFINE="${MAX_REFINE_ITERS:-2}"

INIT_ROSTER="$REPO/configs/roster_init.json"
OUT_BASE="$REPO/results/mbpp/baselines"
OUT_HE="$REPO/results/humaneval/baselines"

mkdir -p "$OUT_BASE" "$OUT_HE"

run_and_score() {
    local PIPELINE="$1"   # evolved | raw | self-refine
    local DATASET="$2"    # mbpp | humaneval
    local OUT_FILE="$3"
    local ROSTER="${4:-}"  # optional, only for evolved

    echo ""
    echo "======================================================================"
    echo "=== Pipeline: $PIPELINE | Dataset: $DATASET ==="
    echo "======================================================================"

    ROSTER_ARG=""
    if [ -n "$ROSTER" ]; then
        ROSTER_ARG="--roster_path $ROSTER"
    fi

    python scripts/run_inference.py \
        --pipeline "$PIPELINE" \
        --dataset "$DATASET" \
        --split test \
        --seed "$SEED" \
        --model "$MODEL" \
        --max_refine_iters "$MAX_REFINE" \
        --output_file "$OUT_FILE" \
        --data_dir "$DATA_ROOT" \
        $ROSTER_ARG

    python scripts/score_outputs.py \
        --input "$OUT_FILE" \
        --dataset "$DATASET" \
        --split test \
        --data_dir "$DATA_ROOT"
}

# ── MBPP Test ──────────────────────────────────────────────────────────────
echo "=== [1/6] MBPP | Step-0 Init Persona (3 generalists + GMRoutingPipeline) ==="
run_and_score evolved mbpp "$OUT_BASE/init_persona_mbpp.jsonl" "$INIT_ROSTER"

echo "=== [2/6] MBPP | Raw (1-pass) ==="
run_and_score raw mbpp "$OUT_BASE/raw_mbpp.jsonl"

echo "=== [3/6] MBPP | Self-Refine (${MAX_REFINE} iters, no persona) ==="
run_and_score self-refine mbpp "$OUT_BASE/selfrefine_mbpp.jsonl"

# ── HumanEval ─────────────────────────────────────────────────────────────
echo "=== [4/6] HumanEval | Step-0 Init Persona ==="
run_and_score evolved humaneval "$OUT_HE/init_persona_humaneval.jsonl" "$INIT_ROSTER"

echo "=== [5/6] HumanEval | Raw (1-pass) ==="
run_and_score raw humaneval "$OUT_HE/raw_humaneval.jsonl"

echo "=== [6/6] HumanEval | Self-Refine (${MAX_REFINE} iters) ==="
run_and_score self-refine humaneval "$OUT_HE/selfrefine_humaneval.jsonl"

# ── Final summary ──────────────────────────────────────────────────────────
echo ""
echo "======================================================================"
echo "=== ALL BASELINE RUNS COMPLETE — Summary ==="
echo "======================================================================"
for f in "$OUT_BASE"/*.score.json "$OUT_HE"/*.score.json; do
    [ -f "$f" ] || continue
    echo "--- $f ---"
    python3 -c "
import json, pathlib
d = json.loads(pathlib.Path('$f').read_text())
print(f'  {d[\"input\"].split(\"/\")[-1]}: Pass@1={d[\"pass_at_1\"]:.2f}% ({d[\"passed\"]}/{d[\"total\"]})')
"
done

echo "=== Done. ==="
