#!/bin/bash
# SNI export v4 — 진화 런에 실제로 쓸 데이터.
#   v3에서 바뀐 것 (둘 다 사용자 결정):
#     1) answer_line 제거 → 프롬프트가 공식 Tk-Instruct 그대로
#        (정의 = system, user = pos example 2건 + "Now complete..." + Input/Output).
#     2) 컨텍스트 초과분을 데이터 단계에서 제외 + train/valid/test 8:1:1 분할.
#   ⚠️ v3까지의 초과 목록(61건)은 sni_context_filter가 v2 프롬프트로 조립해 잰 값이라
#      pos example이 빠져 있었다 — 과소평가다. 여기서 실제 프롬프트로 다시 잰다.
#   분할은 인스턴스 단위 무작위다(태스크 단위 분할은 채택하지 않기로 결정).
#   로더가 sni_{train,valid,test}.jsonl을 집는다. 공식 category-disjoint split은
#   official_{train,test}.jsonl로만 남는다(우리 용도엔 못 쓴다).
# Usage: sbatch --job-name=sni_rebuild_v4 scripts/sbatch/run_sni_rebuild_v4.sh
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

OUT="${OUT:-export/sni_v4}"
EXCL="${EXCL:-results/sni/excluded_over_context_v4.json}"
SPLIT="${SPLIT:-8:1:1}"

echo "=== [1/4] 초과 목록을 재려면 먼저 전수를 한 번 만든다 (분할·제외 없이) ==="
python scripts/build_sni_export.py \
    --out "${OUT}" \
    --no-answer-line \
    --audit-out results/sni/answer_lines_v4.md

echo "=== [2/4] 실제 프롬프트(공식 형식)로 컨텍스트 초과분 측정 ==="
python scripts/sni_context_filter.py \
    --data "${OUT}/sni_all.jsonl" \
    --roster configs/roster_sni_probe_v2.json \
    --out "${EXCL}"

echo "=== [3/4] 초과분 제외 + ${SPLIT} 분할로 재빌드 ==="
python scripts/build_sni_export.py \
    --out "${OUT}" \
    --no-answer-line \
    --exclude-ids "${EXCL}" \
    --split "${SPLIT}" \
    --audit-out results/sni/answer_lines_v4.md

echo "=== [4/4] 프롬프트 실물 덤프 (승인 전에는 진화 런에 쓰지 않는다) ==="
python scripts/evo_multisample_pilot.py \
    --config configs/sni_probe_v2.yaml \
    --dataset sni --data_dir "${OUT}" --split train \
    --roster_path configs/roster_sni_probe_v2.json \
    --arms persona --n_problems 200 --dry_run \
    --preview_out results/sni/prompts_preview_v4.md

echo "=== done -> ${OUT} ==="
wc -l "${OUT}"/sni_*.jsonl
