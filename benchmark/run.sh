#!/bin/bash
#SBATCH --job-name=acc_data_pipeline
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --exclude=master
#SBATCH --time=0-12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/minjikim/minji_link/code/benchmark/logs/%x_%j.out
#SBATCH --error=/home/minjikim/minji_link/code/benchmark/logs/%x_%j.err

set -euo pipefail

source ~/.bashrc
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate agent || true
elif [ -f /data/minjikim/miniconda3/etc/profile.d/conda.sh ]; then
  source /data/minjikim/miniconda3/etc/profile.d/conda.sh
  conda activate agent || true
fi

export CUDA_VISIBLE_DEVICES=""

cd /home/minjikim/minji_link/code/benchmark
BENCHMARK_ROOT="$(pwd -P)"
mkdir -p data/processed data/reports data/intermediate logs

RAW_ROOT="${RAW_ROOT:-/home/minjikim/data/raw}"
if [ ! -d "$RAW_ROOT" ] && [ -d /data/minjikim/raw ]; then
  RAW_ROOT=/data/minjikim/raw
fi
if [ ! -d "$RAW_ROOT" ] && [ -d /home/minjikim/minji_link/raw ]; then
  RAW_ROOT=/home/minjikim/minji_link/raw
fi

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export TACO_MAX_TESTS="${TACO_MAX_TESTS:-8}"
export TACO_MAX_SOLUTIONS="${TACO_MAX_SOLUTIONS:-3}"
export LIVECODEBENCH_MAX_PUBLIC_TESTS="${LIVECODEBENCH_MAX_PUBLIC_TESTS:-8}"
export LIVECODEBENCH_MAX_PRIVATE_TESTS="${LIVECODEBENCH_MAX_PRIVATE_TESTS:-8}"

CODECONTESTS_REPO="${CODECONTESTS_REPO:-/home/minjikim/minji_link/external/code_contests}"
CODECONTESTS_REPO="$(readlink -f "$CODECONTESTS_REPO" 2>/dev/null || echo "$CODECONTESTS_REPO")"
CODECONTESTS_RAW_DIR="$RAW_ROOT/dm-code_contests"
CODECONTESTS_EXPORT_JSONL="${CODECONTESTS_EXPORT_JSONL:-$BENCHMARK_ROOT/data/intermediate/codecontests_export.jsonl}"
CODECONTESTS_EXPORTER="$CODECONTESTS_REPO/bazel-bin/tools/export_codecontests_jsonl"
CODECONTESTS_CC="${CODECONTESTS_CC:-/data/minjikim/.conda/envs/agent/bin/x86_64-conda-linux-gnu-clang}"
CODECONTESTS_CXX="${CODECONTESTS_CXX:-/data/minjikim/.conda/envs/agent/bin/x86_64-conda-linux-gnu-clang++}"

if [ -d "$CODECONTESTS_REPO" ] && [ -d "$CODECONTESTS_RAW_DIR" ] && command -v bazel >/dev/null 2>&1; then
  if [ -x "$CODECONTESTS_CC" ] && [ -x "$CODECONTESTS_CXX" ]; then
    export CC="$CODECONTESTS_CC"
    export CXX="$CODECONTESTS_CXX"
  fi
  if [ ! -x "$CODECONTESTS_EXPORTER" ]; then
    (
      cd "$CODECONTESTS_REPO"
      bazel \
        --host_jvm_args=--add-opens=java.base/java.lang=ALL-UNNAMED \
        --output_user_root=/data/minjikim/.cache/bazel \
        build \
        --repo_env=CC="${CC:-$CODECONTESTS_CC}" \
        --repo_env=CXX="${CXX:-$CODECONTESTS_CXX}" \
        --host_linkopt=-fuse-ld=lld \
        --linkopt=-fuse-ld=lld \
        --host_cxxopt=-include \
        --host_cxxopt=cstdint \
        --cxxopt=-include \
        --cxxopt=cstdint \
        --package_path="%workspace%:$BENCHMARK_ROOT" \
        //tools:export_codecontests_jsonl
    )
  fi
  if [ "${FORCE_CODECONTESTS_EXPORT:-0}" = "1" ] || [ ! -s "$CODECONTESTS_EXPORT_JSONL" ]; then
    "$CODECONTESTS_EXPORTER" \
      --output "$CODECONTESTS_EXPORT_JSONL" \
      --max-public-tests "${CODECONTESTS_MAX_PUBLIC_TESTS:-8}" \
      --max-private-tests "${CODECONTESTS_MAX_PRIVATE_TESTS:-8}" \
      --max-generated-tests "${CODECONTESTS_MAX_GENERATED_TESTS:-0}" \
      --max-solutions "${CODECONTESTS_MAX_SOLUTIONS:-3}" \
      --max-incorrect-solutions "${CODECONTESTS_MAX_INCORRECT_SOLUTIONS:-0}" \
      "$CODECONTESTS_RAW_DIR"/code_contests_valid.riegeli \
      "$CODECONTESTS_RAW_DIR"/code_contests_test.riegeli \
      "$CODECONTESTS_RAW_DIR"/code_contests_train.riegeli-*
  fi
  export CODECONTESTS_EXPORT_JSONL
fi

python -m acc_data_pipeline.cli.load \
  --raw-root "$RAW_ROOT" \
  --datasets apps codecontests taco livecodebench \
  --output data/processed/01_unified_raw.jsonl \
  --report data/reports/load_report.json

python -m acc_data_pipeline.cli.filter \
  --input data/processed/01_unified_raw.jsonl \
  --output data/processed/02_algorithmic_filtered.jsonl \
  --config configs/filter_config.yaml \
  --report data/reports/filter_report.json

python -m acc_data_pipeline.cli.dedup \
  --input data/processed/02_algorithmic_filtered.jsonl \
  --output data/processed/03_deduplicated.jsonl \
  --config configs/dedup_config.yaml \
  --report data/reports/dedup_report.json

python -m acc_data_pipeline.cli.prepare_execution \
  --input data/processed/03_deduplicated.jsonl \
  --output data/processed/04_execution_ready.jsonl \
  --config configs/execution_config.yaml \
  --report data/reports/eval_mode_report.json

python -m acc_data_pipeline.cli.validate \
  --input data/processed/04_execution_ready.jsonl \
  --report data/reports/schema_validation_report.json
