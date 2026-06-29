# Non-Algorithm Code Cleanup Notes

This file records the cleanup decision for the ACC algorithm-only handoff.
The goal was to remove only files that are not used by ACC algorithm SFT training or evaluation.

## Direct ACC Algorithm Path Kept

These files drive the current ACC algorithm workflow and were kept:

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
data/acc_algorithm/
```

## Compatibility Files Kept Deliberately

These files look like old non-algorithm code, but they are still needed for imports or compatibility in
the current package layout. Do not delete them unless the imports are refactored first.

```text
src/data/loader.py
src/evaluation/scorer.py
src/evaluation/code_exec.py
src/evaluation/lcb_score.py
src/evaluation/metrics.py
src/paths.py
```

Reasons:

- `src/data/__init__.py` imports `data.loader`, so deleting `src/data/loader.py` would break
  `from data.loader_algorithm import ...` package initialization.
- `src/evaluation/__init__.py` imports `evaluation.scorer`, so deleting `src/evaluation/scorer.py` or
  `src/evaluation/code_exec.py` would break evaluation package initialization.
- `src/evaluate_algorithm.py` imports `evaluation.metrics` for math-dataset compatibility.
- `src/evaluation/scorer_algorithm.py` imports `evaluation.lcb_score`, and `lcb_score.py` imports
  `src/paths.py`.

## Files Deleted as Not Used by ACC Algorithm Training/Evaluation

The following files were removed because `run_algorithm.sh`, the ACC SLURM wrappers, and the
`*_algorithm.py` modules do not call or import them:

```text
configs/base.yaml
configs/humaneval.yaml
configs/livecodebench.yaml
configs/mbpp.yaml
configs/mbpp_train.yaml
smoke_test.sh
scripts/analyze_evolution.py
scripts/build_big_math_filtered_and_leftover.py
scripts/run_evolution.py
scripts/run_inference.py
scripts/run_multi_seed.py
scripts/score_outputs.py
scripts/sbatch/eval_mbpp_baselines.sh
scripts/sbatch/eval_mbpp_inference.sh
scripts/sbatch/eval_mbpp_inference_epochs.sh
scripts/sbatch/eval_mbpp_train_evolution.sh
scripts/sbatch/eval_sft_models.sh
scripts/sbatch/train_sft_bigmath_all.sh
scripts/sbatch/train_sft_bigmath_by_category.sh
src/action_selector.py
src/agents/base.py
src/evaluate.py
src/orchestrator.py
src/pipelines/base_pipeline.py
src/pipelines/baselines.py
src/pipelines/routing_inference.py
src/prompts/baseline_prompts.py
src/prompts/coding.py
src/prompts/meta.py
src/prompts/qwen3_lcb.py
src/roster.py
src/scout.py
src/step_logger.py
src/train_sft.py
src/war.py
tests/test_action_selector.py
tests/test_roster.py
tests/test_scorer.py
```

## Generated Artifacts

These are not source-code dependencies for algorithm training/evaluation. Keep them locally only when
you need previous run outputs.

```text
checkpoints/
results/
logs/
```
