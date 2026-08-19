# 2026-07-06 Numina CoT LLM Judge Research Journal

## Provenance

- Repository: `/home/jongbinwon/data/evolving_MoE`
- Branch: `main`
- Commit used for LLM judge scripts: `a8339e35d3d30b01cda2cc2fd9401fa9d296d595`
- Commit title: `add scripts for LLM judge`
- Main implementation files in repo:
  - `scripts/llm_judge_numina.py`
  - `scripts/sbatch/llm_judge_numina.sh`

This journal folder intentionally does not duplicate code files. Reproduce the code state from the branch and commit above.

## Objective

Evaluate Numina CoT model responses with an LLM judge using the dataset `solution` field as the reference, rather than only the short `ground_truth` answer. The judge output was integrated as:

```text
updated combined_score = original combined_score OR llm_judge_score
```

The target result directory was:

```text
results/llama3_numina_cot_LUCA_LUCA
```

## Collected Files

This journal folder contains:

```text
journals/2026-07-06_numina_llm_judge/
  2026-07-06_numina_cot_llm_judge_research_journal.md
  logs/
    llm_judge_numina.202009.log
    llm_judge_numina.202604.log
    vllm_judge_202604.log
  results/
    original/
      12 original evaluation JSONL files
    llm_judge_outputs/
      12 *.llm_judge.jsonl files
      12 *.llm_judge.summary.json files
      llm_judge_aggregate.summary.json
      llm_judge_file_index.csv
```

The journal folder is about 772 MB because it includes both original and LLM-judged JSONL outputs.

## Implementation Summary

The committed judge implementation:

- accepts either a JSONL file or a directory of JSONL files;
- loads `Jongbin-kr/NuminaMath-CoT_filtered` with `split=test`;
- matches each record by `id` to retrieve the full dataset `solution`;
- sends the problem, candidate prediction, and reference solution to a local OpenAI-compatible vLLM endpoint;
- writes `*.llm_judge.jsonl` with:
  - `llm_judge_score`
  - `llm_judge_verdict`
  - `llm_judge_reason`
  - `combined_score_before_llm_judge`
  - updated `combined_score`
- writes per-file and aggregate summary JSON.

The Slurm runner:

- uses `gpu:PRO6000:2`;
- activates the `MoE` conda environment;
- pre-downloads `google/gemma-4-26B-A4B-it` through `huggingface_hub.snapshot_download`;
- starts a local vLLM server;
- runs the judge script;
- stops the vLLM server after completion.

## Debugging History

Several issues were identified and fixed before the successful run:

- The first implementation used `split=train`, which produced missing solution lookups. The correct split for these result ids was `test`.
- `google/gemma-4-26B-A4B-it` initially appeared stuck because the Hugging Face cache was incomplete.
- `huggingface-cli` resolved to `~/.local/bin` with Python 3.8 and failed due to missing `filelock`. The runner was changed to call `huggingface_hub.snapshot_download` from the active `MoE` Python.
- An unsupported vLLM flag, `--disable-log-requests`, was removed.
- The judge model needed 2-GPU tensor parallelism. The runner now defaults to `TENSOR_PARALLEL_SIZE=${SLURM_GPUS_ON_NODE:-2}` and `MAX_MODEL_LEN=8192`.
- LLM judge outputs were separated from original JSONL files under `llm_judge_outputs/`.

## Successful Runs

Single-file sanity run:

```text
Slurm job: 202009
Main log: logs/llm_judge_numina.202009.log
```

Full-directory run:

```text
Slurm job: 202604
Main log: logs/llm_judge_numina.202604.log
vLLM log: logs/vllm_judge_202604.log
```

The successful full run ended with:

```text
Done numina_cot_llama3_NuminaCoT_number_theory_200414.jsonl: 0.3408 (2118/6215)
Aggregate LLM-judge score: 0.3336 (24879/74580)
=== Numina LLM judge 완료 ===
```

The `srun ... CANCELLED/Killed` line in the vLLM log occurred during cleanup after the judge had finished and the server was intentionally stopped.

## Aggregate Results

Across all 12 files:

| Metric | Count | Score |
|---|---:|---:|
| Total judged | 74,580 | 100.00% |
| LLM judge correct | 24,879 | 33.36% |
| Original combined correct | 29,122 | 39.05% |
| Combined correct after OR with LLM judge | 30,825 | 41.33% |
| Additional correct from LLM judge OR | 1,703 | +2.28 pp |

The LLM judge OR rule recovered 1,703 additional examples compared with the previous combined metric.

## Per-File Results

| File | Original combined | LLM judge | Combined after OR | Delta |
|---|---:|---:|---:|---:|
| `numina_cot_Llama-3.1-8B-Instruct_200410.jsonl` | 40.68% | 38.95% | 44.39% | +3.72 pp |
| `numina_cot_llama3_NuminaCoT_algebra_200416.jsonl` | 38.52% | 32.15% | 40.58% | +2.06 pp |
| `numina_cot_llama3_NuminaCoT_all_200411.jsonl` | 41.37% | 34.98% | 43.31% | +1.95 pp |
| `numina_cot_llama3_NuminaCoT_calculus_200412.jsonl` | 37.18% | 31.09% | 39.37% | +2.19 pp |
| `numina_cot_llama3_NuminaCoT_combinatorics_200413.jsonl` | 39.42% | 33.69% | 41.82% | +2.40 pp |
| `numina_cot_llama3_NuminaCoT_geometry_200415.jsonl` | 38.84% | 33.24% | 40.74% | +1.90 pp |
| `numina_cot_llama3_NuminaCoT_more_algebra_200421.jsonl` | 38.36% | 31.60% | 40.26% | +1.90 pp |
| `numina_cot_llama3_NuminaCoT_more_calculus_200417.jsonl` | 36.17% | 30.33% | 38.26% | +2.09 pp |
| `numina_cot_llama3_NuminaCoT_more_combinatorics_200418.jsonl` | 39.87% | 33.95% | 42.20% | +2.33 pp |
| `numina_cot_llama3_NuminaCoT_more_geometry_200420.jsonl` | 39.03% | 32.61% | 41.34% | +2.30 pp |
| `numina_cot_llama3_NuminaCoT_more_number_theory_200419.jsonl` | 39.50% | 33.63% | 41.71% | +2.20 pp |
| `numina_cot_llama3_NuminaCoT_number_theory_200414.jsonl` | 39.63% | 34.08% | 42.00% | +2.37 pp |

## Interpretation

The LLM judge consistently increased the measured score, but the effect was modest rather than transformative:

```text
39.05% -> 41.33%
```

This suggests the existing metric was conservative for a meaningful subset of cases, likely due to formatting, symbolic equivalence, or parser limitations. The base Llama run benefited most, gaining +3.72 percentage points, while SFT variants typically gained about +1.9 to +2.4 points.

The ranking is not dramatically changed by the OR rule. The main value of the judge pass is recovering plausible correct solutions missed by parser-based or answer-extraction metrics.

## Caveats

- Judge model: `google/gemma-4-26B-A4B-it`.
- Serving stack: local vLLM, OpenAI-compatible chat completions.
- No manually labeled calibration set was created in this run.
- The OR rule increases recall and may introduce false positives.
- For reporting, it is safest to show both original combined and LLM-augmented combined scores.

## Next Steps

1. Manually audit 50-100 cases where `combined_score_before_llm_judge = 0` and `llm_judge_score = 1`.
2. Manually audit disagreement cases where existing metrics passed but the LLM judge failed.
3. Estimate false-positive and false-negative behavior of the judge.
4. Decide whether the final paper/report should use original metrics, LLM-augmented metrics, or both.
