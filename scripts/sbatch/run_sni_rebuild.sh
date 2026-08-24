#!/bin/bash
# SNI export v3 재빌드 + 공식형식 프롬프트 실물 덤프.
#   v3에서 바뀐 것: Positive/Negative Examples 보존 (공식 표준은 --num_pos_examples 2).
#   프롬프트도 공식 Tk-Instruct 형식(Now complete the following example - / Input: / Output: ).
# Usage: sbatch --job-name=sni_rebuild scripts/sbatch/run_sni_rebuild.sh
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/home/jaehoonjeong/data/MetaAgentEvolution_Release/logs/%x.%j.log
#SBATCH --error=/home/jaehoonjeong/data/MetaAgentEvolution_Release/logs/%x.%j.log

set -euo pipefail
REPO="${REPO:-/home/jaehoonjeong/data/MetaAgentEvolution_Release}"
cd "${REPO}"
source ~/.bashrc
source ~/data/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-evolving_moe}"
export PYTHONPATH="${REPO}/src"

python scripts/build_sni_export.py \
    --out export/sni_v3 \
    --audit-out results/sni/answer_lines_v3.md

python scripts/sni_context_filter.py \
    --data export/sni_v3/sni_all.jsonl \
    --roster configs/roster_sni_probe_v2.json \
    --out results/sni/excluded_over_context_v3.json

python scripts/evo_multisample_pilot.py \
    --config configs/sni_probe_v2.yaml \
    --dataset sni --data_dir export/sni_v3 --split all \
    --roster_path configs/roster_sni_probe_v2.json \
    --arms persona --n_problems 200 --dry_run \
    --preview_out results/sni/prompts_preview_v3.md
echo "=== done ==="
