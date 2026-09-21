#!/bin/bash
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
WORKER="$REPO/scripts/sbatch/train_sft_lbox_ffn_only_5ep.sh"

mkdir -p "$REPO/logs"

if [ ! -f "$WORKER" ]; then
    echo "ERROR: worker script not found: $WORKER" >&2
    exit 2
fi

KINDS=(
    task
    task
    task
    category
    category
    category
    category
    category
    category
    category
    category
)

NAMES=(
    casename_civil
    casename_criminal
    statute
    civil_property_obligation
    civil_family_inheritance
    criminal_property
    criminal_non_property
    admin_traffic
    admin_labor
    admin_other
    family_patent_special
)

START_INDEX="${START_INDEX:-0}"
INITIAL_DEPENDENCY="${INITIAL_DEPENDENCY:-}"
NUM_CHAINS="${NUM_CHAINS:-1}"
if [[ ! "$START_INDEX" =~ ^[0-9]+$ ]] || [ "$START_INDEX" -ge "${#NAMES[@]}" ]; then
    echo "ERROR: START_INDEX must be between 0 and $((${#NAMES[@]} - 1))" >&2
    exit 2
fi
if [ -n "$INITIAL_DEPENDENCY" ] && [[ ! "$INITIAL_DEPENDENCY" =~ ^[0-9]+$ ]]; then
    echo "ERROR: INITIAL_DEPENDENCY must be a numeric Slurm job id" >&2
    exit 2
fi
if [[ ! "$NUM_CHAINS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: NUM_CHAINS must be a positive integer" >&2
    exit 2
fi

lane_dependencies=()
for ((lane = 0; lane < NUM_CHAINS; lane++)); do
    lane_dependencies[$lane]="$INITIAL_DEPENDENCY"
done
job_ids=()

for i in "${!NAMES[@]}"; do
    if [ "$i" -lt "$START_INDEX" ]; then
        continue
    fi
    kind="${KINDS[$i]}"
    name="${NAMES[$i]}"
    slug="${name//_/-}"
    run_name="sft_llama31_8b_lbox_${name}_ffn_only_5ep"
    hub_model_id="Jongbin-kr/llama-3.1-8b-instruct_lbox-${slug}_ffn-only"
    job_name="lbox_ffn_${name}"
    lane=$(((i - START_INDEX) % NUM_CHAINS))
    previous_job_id="${lane_dependencies[$lane]}"

    dependency_args=()
    if [ -n "$previous_job_id" ]; then
        dependency_args=(--dependency="afterany:$previous_job_id")
    fi

    submission="$({ sbatch --parsable \
        --job-name="$job_name" \
        "${dependency_args[@]}" \
        --export="ALL,REPO=$REPO,LBOX_KIND=$kind,LBOX_NAME=$name,RUN_NAME=$run_name,HUB_MODEL_ID=$hub_model_id" \
        "$WORKER"; } 2>&1)" || {
            echo "ERROR: sbatch failed for $name: $submission" >&2
            exit 1
        }
    job_id="${submission%%;*}"
    if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
        echo "ERROR: could not parse job id for $name: $submission" >&2
        exit 1
    fi

    job_ids+=("$job_id")
    echo "$job_id  lane=$((lane + 1))  $kind  $name  dependency=${previous_job_id:-none}  hub=$hub_model_id"
    lane_dependencies[$lane]="$job_id"
done

echo "Submitted ${#job_ids[@]} jobs across $NUM_CHAINS afterany chain(s) from index $START_INDEX."
echo "JOB_IDS=${job_ids[*]}"
echo "CHAIN_TAILS=${lane_dependencies[*]}"
