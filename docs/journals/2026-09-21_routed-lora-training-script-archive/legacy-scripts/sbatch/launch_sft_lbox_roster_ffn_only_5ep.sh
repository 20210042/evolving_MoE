#!/bin/bash
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
WORKER="$REPO/scripts/sbatch/train_sft_lbox_roster_ffn_only_5ep.sh"
PACKAGE_DIR="${LBOX_PACKAGE_DIR:-$REPO/results/lbox_binning_seed20210311}"
MANIFEST="${LBOX_JOB_MANIFEST:-$REPO/data/lbox_roster_sft_jobs.tsv}"
MAX_CONCURRENT="${MAX_CONCURRENT:-4}"

mkdir -p "$REPO/logs" "$(dirname "$MANIFEST")"
if [ ! -x "$WORKER" ]; then
    chmod +x "$WORKER"
fi
if [ ! -f "$PACKAGE_DIR/agent_mapping.json" ] || [ ! -f "$PACKAGE_DIR/binning_labels.jsonl" ]; then
    echo "ERROR: missing LBox binning package under $PACKAGE_DIR" >&2
    exit 2
fi

mapfile -t EXPERTS < <(python - "$PACKAGE_DIR/agent_mapping.json" <<'PY'
import json, re, sys
mapping = json.load(open(sys.argv[1], encoding="utf-8"))
for expert_id, metadata in mapping.items():
    name = str(metadata.get("name") or expert_id).strip()
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    print(f"{expert_id}|{slug}|{name}")
PY
)
if [ "${#EXPERTS[@]}" -ne 10 ]; then
    echo "ERROR: expected 10 roster experts, found ${#EXPERTS[@]}" >&2
    exit 2
fi

tmp_manifest="$(mktemp "${MANIFEST}.XXXXXX")"
{
    echo "generalist|generalist|generalist|Shared High-Consensus Generalist"
    for entry in "${EXPERTS[@]}"; do
        IFS='|' read -r expert_id slug display_name <<< "$entry"
        echo "expert|$expert_id|$slug|$display_name"
    done
} > "$tmp_manifest"
mv "$tmp_manifest" "$MANIFEST"

if [[ ! "$MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: MAX_CONCURRENT must be a positive integer" >&2
    exit 2
fi

echo "=== LBox roster FFN-only SFT jobs ==="
nl -ba "$MANIFEST"
submission="$(sbatch --parsable \
    --array="0-$((${#EXPERTS[@]}))%$MAX_CONCURRENT" \
    --export="ALL,REPO=$REPO,LBOX_PACKAGE_DIR=$PACKAGE_DIR,LBOX_JOB_MANIFEST=$MANIFEST" \
    "$WORKER")"
job_id="${submission%%;*}"
if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
    echo "ERROR: could not parse sbatch output: $submission" >&2
    exit 1
fi
echo "Submitted LBox roster SFT array $job_id (11 tasks, max concurrent=$MAX_CONCURRENT)."
echo "ARRAY_JOB_ID=$job_id"
echo "MANIFEST=$MANIFEST"
