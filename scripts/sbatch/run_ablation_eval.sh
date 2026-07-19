#!/bin/bash
#SBATCH --job-name=abl_eval
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log
# 조건별 배포평가 파이프라인: 생성 → solve매트릭스 → 0.5병합 배포스윕
# 사용: COND=hp|rnd sbatch scripts/sbatch/run_ablation_eval.sh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source ~/.bashrc; source ~/data/miniconda3/etc/profile.d/conda.sh; conda activate evolving_moe
export HF_HOME=/data5/jaehoonjeong/.cache/huggingface
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src"

COND="${COND:?COND=hp|rnd 필요}"
case "$COND" in
  hp)  CKPT="checkpoints/expert_sft/qasc_seedhp_cap10";  LABEL="Human-prior MoE" ;;
  rnd) CKPT="checkpoints/expert_sft/qasc_seedrnd_cap10"; LABEL="Random-partition MoE" ;;
  cap7) CKPT="checkpoints/expert_sft/qasc_seed20210211_cap7"; LABEL="Evolved MoE (cap7 재분할)" ;;
  *) echo "unknown COND=$COND"; exit 1 ;;
esac
TAG="${COND}moe"
RES="results/qasc/seed20210211"

echo "=== [1/3] 생성: $LABEL ($CKPT) ==="
python scripts/generate_lora_binning.py --split validation --ckpt "$CKPT" --tag "$TAG"

echo "=== [2/3] solve 매트릭스 채점 ==="
python scripts/score_binning.py --input "$RES/inference_validation_${TAG}.jsonl" \
    --dataset qasc --split validation --data_dir export/qasc

echo "=== [3/3] 0.5병합 배포 스윕 ==="
python scripts/moe_deploy_sweep.py \
    --ckpt "$CKPT" \
    --binned "$RES/inference_validation_${TAG}.binned.jsonl" \
    --conf "results/embed_viz_test/qasc_validation_${TAG}_conf.npz" \
    --out "$RES/deploy_sweep_${COND}.md" \
    --label "$LABEL"

echo "=== abl_eval 완료: $COND ==="
