#!/bin/bash
#SBATCH --job-name=acc_fewshot_retrain
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log

# §11 확장: Evolved(cap9) 로스터 11명을 persona(roster_final.json 실제 system_prompt) +
# 자기소재 few-shot 2개로 재학습. Random/Human-prior는 손대지 않는다(기존 체크포인트 유지).
# 이 sbatch 자체는 GPU 안 씀 — rolling chain의 첫 expert 잡 하나만 제출하고 끝난다.
#
# 사용: sbatch scripts/sbatch/run_acc_evolved_fewshot_retrain.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source ~/.bashrc; source ~/data/miniconda3/etc/profile.d/conda.sh; conda activate evolving_moe
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src"

export PKG="export/acc_binning_seed20210111_v2"
export MAX_LENGTH="3072"
export GRAD_CKPT="true"
export EVAL_DATA_DIR="export/acc_v2/sft"
export ROSTER_PATH="results/acc/seed20210111/roster_final.json"
export N_FEWSHOT="2"
export EXTRA_SUFFIX="_fewshot"
export MAX_N_SOLVED="9"

echo "=== Evolved(cap9)+few-shot 재학습 체인 제출 ==="
scripts/sbatch/launch_sft_by_experts.sh "${PKG}"
echo "=== 제출 완료 (rolling chain — 11명 순차) ==="
squeue -u "$USER" -o "%.10i %.28j %.8T %.10M %R"
