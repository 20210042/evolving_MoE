#!/bin/bash
#SBATCH --job-name=acc_deploy_top2
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log

# 과업 3번 코딩 실행 — top-2 어댑터 0.5/0.5 병합 배포 평가.
# 자원/환경은 run_acc_deploy_top1.sh와 동일(같은 모델·같은 데이터). 조건별 top-1 수치는 비교용.
# 사용: COND=hp|rnd|evolved_fewshot DENSE_ACC=15.05 sbatch $0

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source ~/.bashrc
source ~/data/miniconda3/etc/profile.d/conda.sh
conda activate evolving_moe
export HF_HOME="${HF_HOME:-/data5/jaehoonjeong/.cache/huggingface}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

COND="${COND:?COND=hp|rnd|evolved_fewshot 필요}"
DENSE_ACC="${DENSE_ACC:?DENSE_ACC(dense SFT pass@1, %%) 필요}"
RES="results/acc/seed20210111_v2/ablation"

ROSTER_FLAG=()
case "$COND" in
  hp)     CKPT="checkpoints/expert_sft/acc_seedhp_v2_cap9";  LABEL="Human-prior MoE"; TOP1=14.65 ;;
  rnd)    CKPT="checkpoints/expert_sft/acc_seedrnd_v2_cap9"; LABEL="Random-partition MoE"; TOP1=13.85 ;;
  evolved_fewshot)
    CKPT="checkpoints/expert_sft/acc_seed20210111_v2_cap9_fewshot"
    LABEL="Evolved MoE (persona+fewshot)"
    TOP1=14.25
    ROSTER_FLAG=(--roster_path results/acc/seed20210111/roster_final.json
                 --label_package export/acc_binning_seed20210111_v2
                 --max_n_solved 9 --n_fewshot 2)
    ;;
  *) echo "unknown COND=$COND" >&2; exit 1 ;;
esac
BINNED="${RES}/inference_test751_${COND}.binned.jsonl"

echo "=== top-2 병합 배포 평가: ${LABEL} (${CKPT}) ==="
python scripts/moe_deploy_top2_merge.py \
    --dataset acc \
    --ckpt "${CKPT}" \
    --binned "${BINNED}" \
    --dense_acc "${DENSE_ACC}" \
    --top1_acc "${TOP1}" \
    --label "${LABEL}" \
    --out "${RES}/deploy_top2_${COND}.md" \
    "${ROSTER_FLAG[@]}"

echo "=== 완료: ${COND} ==="
