#!/bin/bash
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
WORKER="$REPO/scripts/sbatch/train_sft_sni_roster_ffn_only_5ep.sh"
DATA_ROOT="${DATA_ROOT:-$REPO/data/sni_moe_rosters}"
NSOLVED_PATH="${NSOLVED_PATH:-$DATA_ROOT/export/sni_split_seed20212003/split.jsonl}"
MAX_CONCURRENT="${MAX_CONCURRENT:-4}"
MANIFEST="${MANIFEST:-$DATA_ROOT/sni_sft_jobs.tsv}"

mkdir -p "$REPO/logs"

if [ ! -f "$WORKER" ] || [ ! -f "$NSOLVED_PATH" ]; then
    echo "ERROR: missing worker or n_solved reference" >&2
    exit 2
fi

entries=()
ours_dir="$DATA_ROOT/sni_moe_seed20212003/train"
shared_file="$ours_dir/shared.jsonl"
entries+=("$(wc -l < "$shared_file")|generalist|all-pass-generalist|$shared_file|-1")

for arm_spec in \
    "ours:sni_moe_seed20212003" \
    "random:sni_moe_random" \
    "category:sni_moe_human_cat" \
    "domain:sni_moe_human_dom"; do
    arm="${arm_spec%%:*}"
    directory="${arm_spec#*:}"
    for train_file in "$DATA_ROOT/$directory"/train/expert_*.jsonl; do
        base="$(basename "$train_file" .jsonl)"
        expert_id="${base#expert_??_}"
        entries+=("$(wc -l < "$train_file")|$arm|$arm-$expert_id|$train_file|10")
    done
done

if [ "${#entries[@]}" -ne 65 ]; then
    echo "ERROR: expected 65 jobs (64 experts + generalist), found ${#entries[@]}" >&2
    exit 2
fi

if [[ ! "$MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: MAX_CONCURRENT must be a positive integer" >&2
    exit 2
fi

manifest_tmp="$(mktemp "$DATA_ROOT/.sni_sft_jobs.XXXXXX")"
printf '%s\n' "${entries[@]}" | sort -t'|' -k1,1nr > "$manifest_tmp"
mv "$manifest_tmp" "$MANIFEST"

index=0
while IFS='|' read -r source_rows arm nickname train_file max_solved; do
    echo "$index arm=$arm expert=$nickname source_rows=$source_rows max_solved=$max_solved hub=Jongbin-kr/llama-3.1-8b-instruct_SNI-${nickname}_ffn-only"
    index=$((index + 1))
done < "$MANIFEST"

submission="$({ sbatch --parsable \
    --array="0-$((${#entries[@]} - 1))%$MAX_CONCURRENT" \
    --export="ALL,REPO=$REPO,SNI_JOB_MANIFEST=$MANIFEST,NSOLVED_PATH=$NSOLVED_PATH" \
    "$WORKER"; } 2>&1)" || {
    echo "ERROR: array submission failed: $submission" >&2
    exit 1
}
job_id="${submission%%;*}"
if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
    echo "ERROR: could not parse array job id: $submission" >&2
    exit 1
fi

echo "Submitted array job $job_id with ${#entries[@]} independent one-GPU tasks, max concurrent=$MAX_CONCURRENT."
echo "ARRAY_JOB_ID=$job_id"
echo "MANIFEST=$MANIFEST"
