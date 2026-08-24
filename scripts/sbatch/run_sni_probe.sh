#!/bin/bash
# SNI 프로브 v2 — "어떤 축으로 자른 로스터가 출력에 어떤 변화를 주는가"
#   설계: docs/PLAN_sni_probe_v2.md  (v1이 무효인 이유: docs/REFLECTION_sni_probe.md)
#   대상: SNI 전수 87,089건. **표집 규칙 없음** — 자르지 않으므로 대상 풀을 구성할 여지가 없다.
#   로스터: luca 1 + category 상위 12 + domain 상위 12 = 25명 (configs/roster_sni_probe_v2.json)
#   생성량: 25 × 87,089 × K=3 = 6,531,675
#   프롬프트: system = 페르소나 + 태스크 정의 / user = answer_line + 입력
#            (v1은 정의를 user에 둬 페르소나가 묻혔다 → job 229352 무효)
#   gen_chunk: 문제 2,000개씩 흘려보낸다(=50k 프롬프트/청크). 전수를 한 리스트에 쌓으면 터진다.
#   제외: 컨텍스트(16,384) 초과 61건(0.070%, CUAD 계약서 전문 3개 task) —
#         results/sni/excluded_over_context.json. 범위 판단이 아니라 모델 한계다(job 229520 실패 원인).
#   max_tokens 8192 → 4096 (configs/sni_probe_v2.yaml). gold 최대 2,147토큰이라 잘리는 건 0건.
#   score_workers=1: SNI 채점은 문자열 비교라 프로세스풀 피클링이 순손해다.
#   자원: gemma-4-26B-A4B-it tp_size=2 → PRO6000×2. vllm 잡이라 FLASHINFER 비활성 2플래그 필수.
# Usage:
#   sbatch --job-name=sni_probe_v2 scripts/sbatch/run_sni_probe.sh
#   # 중단 후 이어 돌리기 (완결된 (arm,rep) 패스만 재사용):
#   RESUME="--resume_raw results/sni/probe_v2_raw.jsonl" sbatch ... scripts/sbatch/run_sni_probe.sh
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
CONFIG="${CONFIG:-configs/sni_probe_v2.yaml}"
EXCLUDE="${EXCLUDE:-results/sni/excluded_over_context.json}"
DATA_DIR="${DATA_DIR:-export/sni_v2}"
ROSTER="${ROSTER:-configs/roster_sni_probe_v2.json}"
N_PROBLEMS="${N_PROBLEMS:-87089}"     # 전수. min(n, len(data))라 상한이면 전부 들어간다
GEN_CHUNK="${GEN_CHUNK:-2000}"
OUT="${OUT:-results/sni/probe_v2.md}"
RAW="${RAW:-results/sni/probe_v2_raw.jsonl}"
RESUME="${RESUME:-}"

mkdir -p results/sni
echo "=== SNI probe v2: roster=${ROSTER} data=${DATA_DIR} n=${N_PROBLEMS} k=${K} chunk=${GEN_CHUNK} ==="
# shellcheck disable=SC2086
python scripts/evo_multisample_pilot.py \
    --config "${CONFIG}" \
    --dataset sni \
    --data_dir "${DATA_DIR}" \
    --split all \
    --roster_path "${ROSTER}" \
    --arms persona \
    --n_problems "${N_PROBLEMS}" \
    --gen_chunk "${GEN_CHUNK}" \
    --exclude_ids "${EXCLUDE}" \
    --score_workers 1 \
    --k "${K}" \
    --out "${OUT}" \
    --raw_out "${RAW}" \
    ${RESUME}
echo "=== done -> ${OUT} ==="
