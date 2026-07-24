#!/bin/bash
#SBATCH --job-name=acc_rebuild
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log

# acc(TACO) 데이터 재빌드 — 원본 3 split을 합쳐 problem_id 단위로 dedupe하고,
# 각 문제의 canonical ref(refs[0])를 우리 러너로 실행검증해 solution으로 박는다.
# 실패하면 다음 known-correct ref로 폴백. GPU 불필요(코드 실행만).
#
# 왜 다시 만드는가:
#   ① 기존 export/acc_selfconsistent가 reference_solutions를 버려서, 하위 SFT 빌더가
#      "참조 솔루션 없음"으로 오판 → 에이전트 생성코드(주석 27%·pass 스텁 849개)를
#      학습 타깃으로 승격 → 모델이 주석 루프로 붕괴.
#   ② 원본 split은 critic별 행 확장이라 problem_id가 split을 가로지른다(3,636건).
#      행 단위로 자른 홀드아웃은 42% 누수됐다. dedupe 후 problem_id 단위로 다시 나눈다.
#
# 리소스: QOS 상한이 사용자당 cpu=8이라 다른 잡이 돌 때도 들어가도록 2코어로 잡았다.
# 코어가 비면 그만큼 빨라진다 — sbatch -c 8 --mem=64G 로 덮어쓰면 된다.
# 사용: sbatch scripts/sbatch/run_acc_rebuild.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source ~/.bashrc
source ~/data/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-evolving_moe}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src"

SRC="${SRC:-/data5/jaehoonjeong/.cache/huggingface/hub/datasets--QuantCat--Algorithm-Dataset-filtered/snapshots/136cf2bb8dcb8fa6a0611c4237db5703012dc505}"
OUTDIR="${OUTDIR:-export/acc_v2}"
OUT="${OUTDIR}/acc_all_verified.jsonl"
mkdir -p "${OUTDIR}"

echo "=== acc 재빌드: 원본 3 split -> dedupe -> ref 실행검증 ==="
python scripts/build_acc_selfconsistent.py \
    "${SRC}/acc_algorithm_train.jsonl" \
    "${SRC}/acc_algorithm_validation.jsonl" \
    "${SRC}/acc_algorithm_test.jsonl" \
    -o "${OUT}" \
    --workers "${SLURM_CPUS_PER_TASK:-8}" \
    --dedupe-problem-id \
    --ref-fallback \
    --keep-solution

echo "=== 완료 -> ${OUT} ==="
wc -l "${OUT}"
