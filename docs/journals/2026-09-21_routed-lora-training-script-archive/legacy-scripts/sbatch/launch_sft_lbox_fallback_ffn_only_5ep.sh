#!/bin/bash
# Submit one FFN-only SFT job for each materialized fallback expert.
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PACKAGE_DIR="${LBOX_FALLBACK_PACKAGE_DIR:-$REPO/export/lbox_moe_seed20210311}"
MANIFEST="${LBOX_FALLBACK_JOB_MANIFEST:-$REPO/data/lbox_fallback_ffn_sft_jobs.tsv}"
WORKER="$REPO/scripts/sbatch/train_sft_lbox_fallback_ffn_only_5ep.sh"
MAX_CONCURRENT="${MAX_CONCURRENT:-2}"

mkdir -p "$REPO/logs" "$(dirname "$MANIFEST")"
if [ ! -x "$WORKER" ]; then chmod +x "$WORKER"; fi
if [ ! -f "$PACKAGE_DIR/manifest.json" ]; then
    echo "ERROR: missing fallback package manifest: $PACKAGE_DIR/manifest.json" >&2
    exit 2
fi

python - "$PACKAGE_DIR/manifest.json" "$MANIFEST" <<'PY'
import json, re, sys
from pathlib import Path

manifest_path, out_path = sys.argv[1:]
manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
experts = manifest.get("experts", [])
if len(experts) != 11 or manifest.get("n_experts_routed") != 10:
    raise SystemExit(f"unexpected fallback package expert count: {len(experts)}")
rows = []
for item in experts:
    idx = int(item["index"])
    expert_id = str(item["id"])
    name = str(item["name"])
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    if expert_id == "__shared__":
        rel = "train/shared.jsonl"
        slug = "shared-generalist"
    else:
        rel = f"train/expert_{idx:02d}_{expert_id}.jsonl"
    rows.append(f"{rel}|{slug}|{name}")
Path(out_path).write_text("\n".join(rows) + "\n", encoding="utf-8")
PY

if [[ ! "$MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: MAX_CONCURRENT must be a positive integer" >&2
    exit 2
fi

echo "=== LBox fallback FFN-only SFT jobs ==="
nl -ba "$MANIFEST"
submission="$(sbatch --parsable \
    --array="0-10%$MAX_CONCURRENT" \
    --export="ALL,REPO=$REPO,LBOX_FALLBACK_PACKAGE_DIR=$PACKAGE_DIR,LBOX_FALLBACK_JOB_MANIFEST=$MANIFEST" \
    "$WORKER")"
job_id="${submission%%;*}"
if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
    echo "ERROR: could not parse sbatch output: $submission" >&2
    exit 1
fi
echo "Submitted LBox fallback FFN-only array $job_id (11 tasks, max concurrent=$MAX_CONCURRENT)."
echo "ARRAY_JOB_ID=$job_id"
echo "MANIFEST=$MANIFEST"
