#!/bin/bash
# SNI 프로브 — 진화 없이 축 기반 고정 로스터로 "프롬프트만으로 결과가 갈리는 축이 있나"를 본다.
#   로스터: category 12 + domain 10 + LUCA = 23명 (configs/roster_sni_probe.json)
#   문제  : 구역별 균등 600문제 (scripts/sni_probe_sample.py 산출), 각 item에 category/sni_domain 라벨
#   생성량: 23 × 600 × K=3 = 41,400
#   자원  : run_acc_evo_multisample_pilot.sh와 동일(gemma-4-26B-A4B-it tp_size=2 → PRO6000×2).
#           시간만 48h — 생성량이 acc 파일럿의 19배라 12h로 잡을 근거가 없다.
#   vllm 잡이라 FLASHINFER 비활성 2플래그 필수.
# Usage:
#   sbatch --job-name=sni_probe scripts/sbatch/run_sni_probe.sh
#SBATCH --gres=gpu:PRO6000:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
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
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_DISABLE_FLASHINFER=1
export HF_HOME="${HF_HOME:-/data5/jaehoonjeong/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/data5/jaehoonjeong/.cache/huggingface}"

K="${K:-3}"
IDS="${IDS:-results/sni/probe_problem_ids.json}"
ROSTER="${ROSTER:-configs/roster_sni_probe.json}"
OUT="${OUT:-results/sni/probe.md}"
RAW="${RAW:-results/sni/probe_raw.jsonl}"
RESUME="${RESUME:-}"   # 중단 시: RESUME="--resume_raw results/sni/probe_raw.jsonl"

mkdir -p results/sni
echo "=== SNI probe: roster=${ROSTER} ids=${IDS} k=${K} out=${OUT} ==="
# shellcheck disable=SC2086
python scripts/evo_multisample_pilot.py \
    --dataset sni \
    --data_dir export/sni \
    --split all \
    --roster_path "${ROSTER}" \
    --problem_ids "${IDS}" \
    --arms persona \
    --k "${K}" \
    --out "${OUT}" \
    --raw_out "${RAW}" \
    ${RESUME}
echo "=== done -> ${OUT} ==="
