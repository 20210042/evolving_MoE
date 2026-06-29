#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${SCRIPT_DIR}/train_sft_by_category.sh}"

CATEGORIES=(
    # "Algebra"
    # "Geometry"
    # "Number Theory"
    "Combinatorics"
    "Calculus"
)

for CATEGORY in "${CATEGORIES[@]}"; do
    echo "=== Submitting SFT job for category: ${CATEGORY} ==="
    sbatch --export=ALL,CATEGORY="${CATEGORY}" "${TRAIN_SCRIPT}"
done
