# ACC Algorithm SFT and Evaluation Guide

This repository contains an ACC algorithm-specific SFT/evaluation path on top of the original
`evolving_MoE` project. The current handoff is centered on the `acc_algorithm` dataset and the files
with an `_algorithm` suffix.

## What This Pipeline Does

The pipeline has four stages:

1. Build balanced ACC algorithm train/validation/test JSONL files from the final labeled benchmark CSV.
2. Train one all-critic LoRA adapter using all ACC algorithm examples.
3. Train five critic-specific LoRA adapters, one per critic category.
4. Evaluate vanilla base models and LoRA adapters with stdin/stdout execution scoring.

The main local entry point is:

```bash
bash run_algorithm.sh <action>
```

The convenience wrapper below does the same thing because it delegates to `run_algorithm.sh`:

```bash
bash run.sh <action>
```

Supported actions:

```text
data          build data/acc_algorithm/*.jsonl only
train_all     train one all-critic LoRA adapter
train_critics train five critic-specific LoRA adapters sequentially
eval          evaluate vanilla and discovered LoRA checkpoints
all           run data, train_all, train_critics, then eval
```

## Core Files

```text
README_algorithm.md
run.sh
run_algorithm.sh
environment.yml
scripts/build_acc_sft_dataset_algorithm.py
scripts/sbatch/train_sft_acc_all_algorithm.sh
scripts/sbatch/train_sft_acc_by_critic_algorithm.sh
scripts/sbatch/eval_sft_acc_models_algorithm.sh
scripts/sbatch/eval_sft_acc_models_algorithm_llama.sh
scripts/sbatch/eval_sft_acc_by_critic_algorithm_llama.sh
src/data/loader_algorithm.py
src/train_sft_algorithm.py
src/evaluate_algorithm.py
src/evaluation/code_exec_algorithm.py
src/evaluation/scorer_algorithm.py
src/utils/helpers.py
src/utils/llm.py
```

`NON_ALGORITHM_FILES.md` records which old files were removed and which compatibility files were kept.

## Environment Setup

Create the environment:

```bash
cd /home/minjikim/minji_link/evolving_MoE
conda env create -f environment.yml --name MoE
conda activate MoE
export PYTHONPATH=$PWD/src
```

If the environment already exists:

```bash
cd /home/minjikim/minji_link/evolving_MoE
conda activate MoE
export PYTHONPATH=$PWD/src
```

The local runner also sets these paths automatically:

```bash
export HF_HOME=/home/minjikim/minji_link/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HUB_CACHE
```

You may need Hugging Face authentication for gated models:

```bash
huggingface-cli login
```

## Input CSV

The dataset builder reads this labeled CSV from the benchmark pipeline:

```text
/home/minjikim/minji_link/code/benchmark/data/labelling/04_execution_ready_final_labels_local.csv
```

The file is read-only from this project. Important columns:

| CSV column | Used as |
|---|---|
| `problem_id` | Original normalized benchmark problem ID. |
| `problem` | Model instruction / problem statement. |
| `answer` | Reference solution source; converted into `ground_truth`. |
| `test_cases` | Normalized stdin/stdout tests for execution scoring. |
| `eval_spec` | Output comparison rules. |
| `normalized_labels` | Fine-grained label metadata. |
| `critic_categories` | Multi-critic expansion source. |
| `main_critic_category` | Fallback critic category. |
| `source`, `source_platform` | Dataset provenance metadata. |

## Critic Categories

The ACC algorithm dataset balances these five critic categories:

```text
Constructive Implementation
Quantitative Reasoning
State-Space Reasoning
Structured Data
Greedy Strategy
```

A problem with multiple critic categories is expanded into one example per critic. Example:

```text
problem_id = P1
critic_categories = [Structured Data, State-Space Reasoning]

P1__structured_data          -> category = Structured Data
P1__state_space_reasoning    -> category = State-Space Reasoning
```

The same original problem can appear under different critics. Within the same critic, the builder keeps
`problem_id` disjoint across train/validation/test.

## 1. Build the ACC Algorithm Dataset

Local command:

```bash
cd /home/minjikim/minji_link/evolving_MoE
bash run_algorithm.sh data
```

Equivalent direct command:

```bash
python scripts/build_acc_sft_dataset_algorithm.py \
  --output-dir data/acc_algorithm
```

Generated files:

```text
data/acc_algorithm/acc_algorithm_train.jsonl
data/acc_algorithm/acc_algorithm_validation.jsonl
data/acc_algorithm/acc_algorithm_test.jsonl
data/acc_algorithm/split_report.json
```

The training/evaluation scripts also auto-build this directory if the expected split file is missing.

## 2. Train One All-Critic LoRA Adapter

The all-critic run trains one LoRA adapter on the full balanced ACC algorithm train split.

Local command, using `run_algorithm.sh` defaults:

```bash
cd /home/minjikim/minji_link/evolving_MoE
CUDA_VISIBLE_DEVICES=0,1,2 bash run_algorithm.sh train_all
```

Current local defaults:

```text
MODEL_NAME=google/gemma-4-31B-it
DATA_DIR=data/acc_algorithm
LoRA rank=16, alpha=32, dropout=0.05, target_modules=all-linear
epochs=3, lr=2e-5, scheduler=cosine
output_dir=checkpoints/sft_acc_algorithm_all_<timestamp>
```

Override the base model or output name if needed:

```bash
MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct \
RUN_NAME=sft_acc_algorithm_all_llama \
OUTPUT_DIR=checkpoints/sft_acc_algorithm_all_llama \
bash run_algorithm.sh train_all
```

SLURM command:

```bash
cd /home/minjikim/minji_link/evolving_MoE
sbatch scripts/sbatch/train_sft_acc_all_algorithm.sh
```

Important: the current SLURM all-critic training wrapper directly uses
`meta-llama/Llama-3.1-8B-Instruct`, while the local runner defaults to `google/gemma-4-31B-it` unless
`MODEL_NAME` is overridden.

## 3. Train Five Critic-Specific LoRA Adapters

This runs five sequential trainings, each with `--categories <critic>`:

```text
Constructive Implementation
Quantitative Reasoning
State-Space Reasoning
Structured Data
Greedy Strategy
```

Local command:

```bash
cd /home/minjikim/minji_link/evolving_MoE
CUDA_VISIBLE_DEVICES=0,1,2 bash run_algorithm.sh train_critics
```

Expected checkpoint pattern:

```text
checkpoints/sft_acc_algorithm_constructive_implementation_<timestamp>/
checkpoints/sft_acc_algorithm_quantitative_reasoning_<timestamp>/
checkpoints/sft_acc_algorithm_state_space_reasoning_<timestamp>/
checkpoints/sft_acc_algorithm_structured_data_<timestamp>/
checkpoints/sft_acc_algorithm_greedy_strategy_<timestamp>/
```

SLURM command:

```bash
cd /home/minjikim/minji_link/evolving_MoE
sbatch scripts/sbatch/train_sft_acc_by_critic_algorithm.sh
```

Important: the current SLURM critic-specific training wrapper also directly uses
`meta-llama/Llama-3.1-8B-Instruct`. The local runner uses `MODEL_NAME`, defaulting to Gemma.

## 4. Evaluate Vanilla and LoRA Models

Evaluation uses `src/evaluate_algorithm.py`. For `acc_algorithm`, scoring is execution-based:

generated code -> temporary Python file -> stdin/stdout test cases -> `pass_score`.

Each evaluation directory contains:

```text
acc_algorithm_results.jsonl
acc_algorithm_summary.json
```

Evaluation supports resume mode by default. If a previous result JSONL exists, completed example IDs are
skipped and malformed/duplicate rows are repaired.

### 4.1 Local Evaluation: Vanilla Gemma + Latest LoRAs

Local command:

```bash
cd /home/minjikim/minji_link/evolving_MoE
CUDA_VISIBLE_DEVICES=0,1,2 bash run_algorithm.sh eval
```

What this evaluates:

1. Vanilla base model with no LoRA:

```text
results/acc_algorithm/eval_acc_algorithm_vanilla_gemma/
```

2. Latest matching all-critic LoRA checkpoint, if present:

```text
checkpoints/sft_acc_algorithm_all_*
```

3. Latest matching critic-specific LoRA checkpoints, if present:

```text
checkpoints/sft_acc_algorithm_constructive_implementation_*
checkpoints/sft_acc_algorithm_quantitative_reasoning_*
checkpoints/sft_acc_algorithm_state_space_reasoning_*
checkpoints/sft_acc_algorithm_structured_data_*
checkpoints/sft_acc_algorithm_greedy_strategy_*
```

The local evaluation wrapper uses `MODEL_NAME`, defaulting to:

```text
google/gemma-4-31B-it
```

It sets `--tensor_parallel_size 1` in the current script. If you need multi-GPU tensor parallel
evaluation, run `src/evaluate_algorithm.py` directly with a larger `--tensor_parallel_size`.

### 4.2 Direct Vanilla Evaluation

Gemma vanilla:

```bash
python src/evaluate_algorithm.py \
  --model_name_or_path google/gemma-4-31B-it \
  --test_dataset acc_algorithm \
  --data_dir data/acc_algorithm \
  --inference_mode vllm \
  --tensor_parallel_size 1 \
  --max_model_len 16384 \
  --max_new_tokens 8192 \
  --eval_batch_size 8 \
  --resume true \
  --output_dir results/acc_algorithm/eval_acc_algorithm_vanilla_gemma \
  --wandb_run_name eval_acc_algorithm_vanilla_gemma
```

Llama vanilla:

```bash
python src/evaluate_algorithm.py \
  --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
  --test_dataset acc_algorithm \
  --data_dir data/acc_algorithm \
  --inference_mode vllm \
  --tensor_parallel_size 1 \
  --max_model_len 16384 \
  --max_new_tokens 8192 \
  --eval_batch_size 8 \
  --resume true \
  --output_dir results/acc_algorithm/eval_acc_algorithm_vanilla_llama \
  --wandb_run_name eval_acc_algorithm_vanilla_llama
```

### 4.3 Direct All-Critic LoRA Evaluation

```bash
python src/evaluate_algorithm.py \
  --model_name_or_path google/gemma-4-31B-it \
  --finetuned_lora_path checkpoints/sft_acc_algorithm_all_<timestamp> \
  --test_dataset acc_algorithm \
  --data_dir data/acc_algorithm \
  --inference_mode vllm \
  --tensor_parallel_size 1 \
  --max_model_len 16384 \
  --max_new_tokens 8192 \
  --eval_batch_size 8 \
  --resume true \
  --output_dir results/acc_algorithm/eval_acc_algorithm_sft_acc_algorithm_all_<timestamp> \
  --wandb_run_name eval_acc_algorithm_sft_acc_algorithm_all_<timestamp>
```

Use the matching base model for the LoRA adapter. If the LoRA was trained on Llama, use
`--model_name_or_path meta-llama/Llama-3.1-8B-Instruct`.

### 4.4 Direct Critic-Specific LoRA Evaluation

Example for `Structured Data`:

```bash
python src/evaluate_algorithm.py \
  --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
  --finetuned_lora_path checkpoints/sft_acc_algorithm_structured_data_<timestamp> \
  --test_dataset acc_algorithm \
  --data_dir data/acc_algorithm \
  --inference_mode vllm \
  --tensor_parallel_size 1 \
  --max_model_len 16384 \
  --max_new_tokens 8192 \
  --eval_batch_size 8 \
  --resume true \
  --output_dir results/acc_algorithm/eval_acc_algorithm_llama_sft_acc_algorithm_structured_data_<timestamp> \
  --wandb_run_name eval_acc_algorithm_llama_sft_acc_algorithm_structured_data_<timestamp>
```

Repeat with the other critic-specific checkpoint directories.

### 4.5 SLURM Evaluation Wrappers

Evaluate Gemma vanilla plus latest Gemma/all/critic LoRA checkpoint patterns:

```bash
sbatch scripts/sbatch/eval_sft_acc_models_algorithm.sh
```

Evaluate Llama vanilla plus latest Llama/all/critic LoRA checkpoint patterns:

```bash
sbatch scripts/sbatch/eval_sft_acc_models_algorithm_llama.sh
```

Evaluate only Llama critic-specific LoRAs. By default this script does not include vanilla unless you
set `INCLUDE_VANILLA=true`:

```bash
INCLUDE_VANILLA=true sbatch scripts/sbatch/eval_sft_acc_by_critic_algorithm_llama.sh
```

Useful evaluation wrapper environment variables:

```text
DATA_DIR=...                 override data/acc_algorithm
SKIP_VANILLA=true            skip vanilla baseline in eval_sft_acc_models_*.sh
SKIP_DEFAULT_LORAS=true      do not auto-discover checkpoint patterns
EXTRA_LORA_PATHS="p1 p2"      evaluate extra adapter directories
WANDB_DISABLED=true          disable WandB logging
```

## 5. Run the Full Local Workflow

This builds data, trains all-critic LoRA, trains five critic-specific LoRAs, and evaluates discovered
checkpoints:

```bash
cd /home/minjikim/minji_link/evolving_MoE
CUDA_VISIBLE_DEVICES=0,1,2 bash run.sh all
```

Equivalent:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 bash run_algorithm.sh all
```

This can take a long time because it includes six training runs and evaluation.

## Large Local Files

The largest files currently under this repository are generated data/checkpoint artifacts, not source
code. They are useful locally but usually should not be pushed to GitHub.

```text
1083.2 MiB  data/acc_algorithm/acc_algorithm_train.jsonl
 934.7 MiB  checkpoints/sft_acc_algorithm_all_20260602_215632/checkpoint-4557/optimizer.pt
 934.7 MiB  checkpoints/sft_acc_algorithm_all_20260602_215632/checkpoint-4400/optimizer.pt
 934.7 MiB  checkpoints/sft_acc_algorithm_all_20260602_215632/checkpoint-4200/optimizer.pt
 509.9 MiB  checkpoints/sft_acc_algorithm_all_20260602_215632/checkpoint-4557/adapter_model.safetensors
 509.9 MiB  checkpoints/sft_acc_algorithm_all_20260602_215632/checkpoint-4400/adapter_model.safetensors
 509.9 MiB  checkpoints/sft_acc_algorithm_all_20260602_215632/checkpoint-4200/adapter_model.safetensors
 509.9 MiB  checkpoints/sft_acc_algorithm_all_20260602_215632/adapter_model.safetensors
 467.5 MiB  checkpoints/sft_acc_algorithm_all_20260602_215632/checkpoint-4557-vllm/adapter_model.safetensors
 320.4 MiB  checkpoints/sft_acc_algorithm_structured_data_20260616_204723/checkpoint-912/optimizer.pt
```

## Safety Notes

- Do not evaluate generated code on an untrusted shared machine without isolation. The ACC evaluator runs
  model-generated Python against test cases.
- Do not push `checkpoints/`, `results/`, `logs/`, or large `data/` files unless the handoff explicitly
  requires artifacts.
- `PUSH_TO_HUB` defaults to `False` in the local training runner. Set `PUSH_TO_HUB=True` only when model
  upload is intended.
- If an evaluation result file already exists, `--resume true` skips completed IDs. Use a fresh
  `--output_dir` or `--resume false` when you want a clean rerun.
