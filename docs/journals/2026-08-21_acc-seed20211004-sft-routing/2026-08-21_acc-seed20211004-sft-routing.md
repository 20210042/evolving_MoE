# 2026-08-19–21 ACC seed20211004 SFT and Routing Research Journal

## Provenance

- Repository: `/data6/jongbinwon/evolving_MoE`
- Branch: `collab/acc-roster-binning-20211004`
- Base asset commit: `00b1d7b4178de27a6c29c4d612e28a15092e6893` (`collab: ACC roster and binning assets for LoRA training`)
- Scoring/evaluator fix: `7e218b7e853cc2f21615748e9ad31bbf622f448d` (`fix(acc-eval): normalize fenced code and forward local splits`)
- Experiment seed: `20211004`; model training seed: `42`
- Experiment execution: 2026-08-19 through 2026-08-21 (Asia/Seoul)
- Journal assembled: 2026-08-25

The scorer and reusable evaluation wrapper are identified by commit `7e218b7`. The SFT and routing programs were still uncommitted when this journal was assembled; their exact SHA-256 values are recorded in [`configs/source_sha256.txt`](configs/source_sha256.txt). On 2026-08-25, the experiment-specific SFT work was removed from the main working tree rather than committed because local `main` already contained overlapping, more complete SFT modes. The exact tracked-file delta was preserved as [`patches/acc_sft_training.patch`](patches/acc_sft_training.patch), while the untracked builder, regression test, and launchers were archived as runnable snapshots. The ACC-specific router, inference, oracle, Slurm, and regression-test files were also moved into this journal rather than added to the generic router stack already present on `main`. The seed-specific vanilla launcher was likewise archived as [`configs/launch_acc_seed20211004_vanilla_eval.sh`](configs/launch_acc_seed20211004_vanilla_eval.sh).

Relevant committed source paths:

- `src/train_sft.py`
- `src/evaluation/scorer.py`
- `scripts/sbatch/train_sft_by_expert.sh`

Archived experiment code:

- [Router training and inference programs](scripts/router/)
- [Router/oracle Slurm wrappers and launchers](scripts/sbatch/)
- [Router and oracle regression tests](tests/)

## Objective

This experiment tested whether an evolved 12-persona ACC roster could be converted into a trained mixture of LoRA experts and routed at inference time. The main questions were:

1. Can the final roster/binning assets supervise persona-specific SFT?
2. How much common all-pass data should be retained while specializing experts on problems solved by at most eight roster members?
3. Does normalized multi-label routing plus parameter merging work better than exclusive Top-1 routing?
4. How much accuracy is lost by the learned router relative to traversing all trained experts?
5. How do the trained mixture, dense all-data LoRA, and vanilla Llama/Gemma baselines compare on the held-out ACC test set?

## Artifacts

### Repository assets

- Final roster: `results/acc/seed20211004/roster_final.json`
- Train binning: `results/acc/seed20211004/binning_train_full.binned.jsonl`
- Test binning: `results/acc/seed20211004/binning_test_full.binned.jsonl`
- Train summary: `results/acc/seed20211004/binning_train_full.binned.summary.json`
- Test summary: `results/acc/seed20211004/binning_test_full.binned.summary.json`

### Journal copies and derived summaries

- [Top-1 inference summary](results/original/top1_router_summary.json)
- [Actual 12-SFT-expert oracle union](results/original/sft_oracle_union_summary.json)
- [Soft and Top-1 router training summary](results/derived/router_training_summary.json)
- [Dense LoRA training summary](results/derived/dense_training_summary.json)
- [Archived vanilla evaluation launcher](configs/launch_acc_seed20211004_vanilla_eval.sh)
- [ACC SFT tracked-file patch](patches/acc_sft_training.patch)
- [Archived SFT label-package builder](scripts/build_sft_label_package.py)
- [Archived label-package regression test](tests/test_build_sft_label_package.py)
- [Archived persona SFT launcher](configs/launch_acc_seed20211004_persona_sft.sh)
- [Archived dense SFT launcher](configs/launch_acc_seed20211004_dense_sft.sh)
- [Archived soft-router trainer](scripts/router/train_acc_soft_router.py)
- [Archived Top-1 set-mass router trainer](scripts/router/train_acc_top1_router.py)
- [Archived soft parameter-merge inference](scripts/router/infer_acc_soft_router_merge.py)
- [Archived exclusive Top-1 inference](scripts/router/infer_acc_top1_router.py)
- [Archived full expert traversal and oracle aggregation](scripts/router/aggregate_acc_sft_oracle.py)
- [Archived router/oracle Slurm programs](scripts/sbatch/)
- [Archived router/oracle tests](tests/)

The patch is relative to commit `7e218b7` and contains the experiment-time changes to `src/train_sft.py`, the generic expert Slurm wrappers, their SFT regression test, and the ACC asset README. It is an archival reproduction artifact, not a recommended replacement for the newer SFT implementation on `main`. The two archived SFT launchers retain their experiment arguments, with only repository/builder path resolution adapted to their deeper journal location.

Large raw predictions, checkpoints, embeddings, and W&B directories remain in their ignored original locations. They were not duplicated into the journal.

## Dataset and Binning

The authoritative evaluation labels are the repository's scored `.binned.jsonl` files, not the later attached raw `binning_test_full.jsonl`. The attachment contained 453 valid rows plus a truncated 454th row; those 453 IDs were a subset of the complete 751-ID scored file.

| Split | Problems | All Failed | Contested | All Passed | Original roster union |
|---|---:|---:|---:|---:|---:|
| Train | 11,097 | 1,800 | 2,839 | 6,458 | 9,297 (83.78%) |
| Test | 751 | 130 | 174 | 447 | 621 (82.69%) |

`n_solved` is the number of original roster experts that solved a problem. The test subsets used in aggregate reporting are:

- All Failed: `n_solved = 0`, 130 problems
- All Passed: `n_solved = 12`, 447 problems
- w/o All Failed: `n_solved > 0`, 621 problems

The per-expert solved sets overlap heavily because all 447 all-pass problems belong to every expert's slice.

## Evolved Roster

| ID | Nickname | Train solved | Test solved |
|---|---|---:|---:|
| `luca` | LUCA | 8,058 | 549 |
| `c_46087` | Combinatorialist | 8,136 | 546 |
| `c_10367` | GreedyOptimizer | 8,186 | 545 |
| `c_17316` | DynamicProgrammer | 8,195 | 547 |
| `c_4998` | SystemsArchitect | 8,146 | 546 |
| `c_34728` | TopologySpecialist | 8,151 | 553 |
| `c_63819` | StringAnalyst | 8,173 | 545 |
| `c_50585` | ComputationalGeometer | 8,138 | 552 |
| `c_16428` | ComplexityAnalyst | 8,141 | 554 |
| `c_30658` | NumberTheorist | 8,141 | 533 |
| `c_56276` | PermutationMathematician | 8,127 | 548 |
| `c_56422` | BacktrackingSpecialist | 8,186 | 552 |

The large individual solved counts include the 6,458 train all-pass problems shared by all experts. They are not disjoint partitions.

## SFT Method

### Specialist LoRAs

Each of the 11 specialists was trained on:

- examples the corresponding original roster expert solved;
- specialization band `1 <= n_solved <= 8` (the lower bound is implicit because the expert itself solved the row);
- the same seeded sample of 200 `n_solved = 12` all-pass common-core problems;
- its roster persona system prompt.

| Expert | Specialized (`n_solved <= 8`) | Common core | Total training rows |
|---|---:|---:|---:|
| Combinatorialist | 532 | 200 | 732 |
| GreedyOptimizer | 577 | 200 | 777 |
| DynamicProgrammer | 607 | 200 | 807 |
| SystemsArchitect | 542 | 200 | 742 |
| TopologySpecialist | 542 | 200 | 742 |
| StringAnalyst | 564 | 200 | 764 |
| ComputationalGeometer | 524 | 200 | 724 |
| ComplexityAnalyst | 530 | 200 | 730 |
| NumberTheorist | 561 | 200 | 761 |
| PermutationMathematician | 562 | 200 | 762 |
| BacktrackingSpecialist | 545 | 200 | 745 |

LUCA was treated as a baseline-prompt generalist rather than a persona specialist. It was trained on 1,000 seeded all-pass examples and did not receive a persona prompt.

Common configuration:

- Base: `meta-llama/Llama-3.1-8B-Instruct`
- Method: LoRA, rank 16, alpha 32, dropout 0.05
- Target: supervised reference `solution`, falling back to `ground_truth`
- Epochs: 5
- Learning rate: `2e-5`, cosine schedule
- Train batch: 2; gradient accumulation: 4
- Sequence length: 3,072
- Precision: BF16
- Hardware per job: 1× PRO6000, 2 CPUs, 32 GB RAM, 24-hour limit
- Evaluation/checkpointing: every 100 steps for specialists; at most three local checkpoints
- Selection: minimum `eval_loss`, restored with `load_best_model_at_end`
- Hub strategy: every save, followed by an explicit final push of the restored best adapter

The dataset passed to TRL had separate `prompt` and `completion` fields. Although the logged `SFTConfig` showed `completion_only_loss=None`, TRL automatically enables completion-only loss for prompt-completion datasets. User/system prompt tokens were therefore masked; `assistant_only_loss=False` did not disable completion masking.

Hub naming:

- LUCA: `Jongbin-kr/evolving-moe-acc-seed20211004-luca-allpass1000`
- Specialist template: `Jongbin-kr/evolving-moe-acc-seed20211004-<expert-id>-cap8-core200`

### Dense baseline

“Dense SFT” in the result table is a single all-data LoRA generalist, not full-parameter fine-tuning. It used all 11,097 train examples, the baseline prompt, and the same LoRA configuration for five epochs.

- Hub: `Jongbin-kr/llama3-8b_acc-seed20211004-dense-all11097`
- W&B: `jongbin-kr-skiml_moe/acc-seed20211004-persona-sft`, run `3pg1l8a5`
- Total steps: 6,940
- Train loss: 0.4895
- Best eval-loss checkpoint: step 2,500, approximately epoch 1.80
- Best eval loss: 0.5220
- Final step eval loss: 0.5482

The increase in eval loss after step 2,500 is evidence of overfitting, although the evaluated/pushed adapter was the restored step-2,500 best checkpoint rather than the final epoch-five weights.

## Router Methods

### Normalized multi-label soft router

For a row solved by `n` experts, each solver received target probability `1/n`. All-fail rows were excluded because there is no identifiable correct expert. The router used mean-pooled `google/embeddinggemma-300m` embeddings followed by a small MLP.

On the 621 solvable test rows, the learned router achieved:

- solver probability mass: 0.8815
- Top-1 solver hit: 0.8631
- KL loss: 0.2144

At inference, all 12 LoRA deltas were combined according to the router probabilities. The final implementation used PEFT `combination_type="cat"`, which represents `sum_i w_i B_i A_i` without the cross-adapter terms introduced by independently linearly mixing the A and B matrices.

### Exclusive Top-1 router

The Top-1 router was trained only on contested rows (`1 <= n_solved <= 11`). Its set loss maximized total probability mass assigned to any original solver rather than choosing an arbitrary single label. Inference selected exactly one LoRA adapter and generated exclusively from that expert.

On 174 contested test rows:

- solver probability mass: 0.5336
- Top-1 original-solver hit: 0.5402
- set loss: 3.0427
- entropy: 0.4769

Across all 751 test rows, the selected original roster member was a known solver for 541 problems (72.04%). This label-level routing result should not be confused with the selected trained SFT expert's execution accuracy, which was 14.65%.

### Trained-expert oracle

Each of the 12 trained adapters generated an answer for every test problem. A problem counted as an oracle success if at least one actual generated answer passed execution. This is an empirical 12-attempt union, not a deployable router and not the original roster upper bound.

## Evaluation Method

All methods were evaluated on the same 751-ID ACC test set using execution-based pass@1. Vanilla and dense evaluations used temperature 0, `max_model_len=16384`, and up to 8,192 new tokens. The archived launcher records the exact vanilla commands.

Models and methods:

- Vanilla Llama: `meta-llama/Llama-3.1-8B-Instruct`
- Vanilla Gemma: `google/gemma-4-26B-A4B-it`
- Dense: all-data Llama LoRA generalist
- Soft merge: normalized multi-label router plus weighted LoRA delta merge
- Top-1: set-loss router plus one exclusive trained LoRA expert
- Original roster Top-1: whether the same selected expert solved the original no-SFT roster output
- Oracle: union over all 12 actual SFT expert generations

W&B entity was `jongbin-kr-skiml_moe`. Projects included:

- `acc-seed20211004-persona-sft`
- `acc-seed20211004-vanilla-eval`
- `acc-seed20211004-soft-router`
- `acc-seed20211004-top1-router`
- `acc-seed20211004-sft-oracle`

## Slurm Runs

| Stage | Job IDs | Outcome |
|---|---|---|
| Initial specialist/LUCA SFT attempt | 227203–227214 | Failed/cancelled after foreign HF cache permission error |
| Successful LUCA and specialist SFT | 227215–227226 | Completed |
| Dense all-data LoRA | 227314 | Completed; best adapter pushed to Hub |
| Vanilla Llama/Gemma | 227324, 227325 | Generation completed; initial scores invalid due to fenced-code scorer bug |
| Soft router training | 227348 | Completed |
| Initial soft merge | 227349 | Superseded while correcting merge behavior |
| Vanilla CPU-only rescore | 227465, 227466 | Completed; 128 and 557 passes |
| `cat` merge smoke/full | 227464, 227477 | Smoke passed; full run stopped by repetition guard at 40 rows |
| Soft merge resume without guard | 227685 | Completed all 751 rows |
| Dense best-checkpoint evaluation | 227897 | Completed |
| Top-1 router training/inference | 227955, 227956 | Completed |
| Per-expert full traversals | 227961–227972 | All 12 completed with exit code 0 |
| Oracle aggregation | 227973 | Completed with exit code 0 |

## Results

Parentheses denote the relevant oracle upper bound: original roster union for the no-SFT roster row and actual 12-trained-expert traversal for the SFT Top-1 row.

| Dimension | Method | Overall [751] | All Failed [130] | All Passed [447] | w/o All Failed [621] |
|---|---|---:|---:|---:|---:|
| Solvability | Evolved Roster (Top-1), no SFT | 72.04 (82.69) | 0.00 (0.00) | 100.00 (100.00) | 87.12 (100.00) |
|  | Evolved MoE, soft router → parameter merge | 15.98 | 3.85 | 21.48 | 18.52 |
|  | Evolved MoE, exclusive Top-1 | 14.65 (25.17) | 2.31 (6.15) | 21.25 (33.33) | 17.23 (29.15) |
| Baseline | Vanilla Llama3-8B | 17.04 | 2.31 | 24.83 | 20.13 |
|  | Vanilla Gemma4-26B-A4B | 74.17 | 0.77 | 99.33 | 89.53 |
|  | Dense all-data LoRA | 15.58 | 6.92 | 21.25 | 17.39 |

An earlier manually assembled roster Top-1 value of 74.03% was incorrect. The saved 751-row routing artifact gives 541/751 = 72.04%.

### Individual trained SFT experts

| Expert | Solved | Overall pass@1 | Unique oracle contribution |
|---|---:|---:|---:|
| LUCA | 117 | 15.58 | 6 |
| Combinatorialist | 116 | 15.45 | 0 |
| GreedyOptimizer | 111 | 14.78 | 5 |
| DynamicProgrammer | 99 | 13.18 | 1 |
| SystemsArchitect | 122 | 16.25 | 2 |
| TopologySpecialist | 114 | 15.18 | 4 |
| StringAnalyst | 115 | 15.31 | 5 |
| ComputationalGeometer | 112 | 14.91 | 3 |
| ComplexityAnalyst | 110 | 14.65 | 1 |
| NumberTheorist | 111 | 14.78 | 2 |
| PermutationMathematician | 105 | 13.98 | 3 |
| BacktrackingSpecialist | 111 | 14.78 | 2 |
| Oracle union | 189 | 25.17 | — |

SystemsArchitect was the best single trained expert at 16.25%. The Top-1 router obtained 14.65%, 1.60 percentage points below that best fixed expert and 10.52 points below the actual 12-expert union.

### Accuracy on original roster expert slices

Each column contains problems the named original roster expert solved; columns overlap.

| Method | LUCA [549] | Combinatorialist [546] | GreedyOptimizer [545] | DynamicProgrammer [547] | SystemsArchitect [546] | TopologySpecialist [553] | StringAnalyst [545] | ComputationalGeometer [552] | ComplexityAnalyst [554] | NumberTheorist [533] | PermutationMathematician [548] | BacktrackingSpecialist [552] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original roster Top-1 | 93.26 | 94.51 | 94.13 | 94.88 | 93.22 | 93.49 | 94.13 | 93.66 | 93.32 | 95.87 | 93.61 | 94.75 |
| Soft parameter merge | 19.31 | 19.41 | 19.45 | 19.56 | 19.41 | 19.89 | 19.63 | 19.20 | 19.68 | 19.89 | 19.53 | 19.20 |
| SFT Top-1 | 18.40 | 18.68 | 18.53 | 18.46 | 18.68 | 18.99 | 18.90 | 18.48 | 18.59 | 18.95 | 18.61 | 18.30 |
| SFT oracle | 30.24 | 31.14 | 30.83 | 30.71 | 30.40 | 31.28 | 30.64 | 30.43 | 30.69 | 30.58 | 30.47 | 30.43 |
| Vanilla Llama3-8B | 21.68 | 20.88 | 21.65 | 20.84 | 21.43 | 21.34 | 21.47 | 20.65 | 21.48 | 21.20 | 20.99 | 21.20 |
| Vanilla Gemma4-26B-A4B | 95.63 | 94.51 | 94.86 | 94.70 | 95.60 | 94.21 | 95.41 | 94.75 | 95.13 | 95.50 | 93.98 | 94.93 |
| Dense all-data LoRA | 18.03 | 18.86 | 18.90 | 18.65 | 19.05 | 18.81 | 18.90 | 18.48 | 18.59 | 18.95 | 18.61 | 18.84 |

## Debugging History

### Incorrect shared Hugging Face cache

The first SFT submission hard-coded another user's cache at `/data5/jaehoonjeong/.cache/huggingface`. Jobs 227203–227210 failed reading its gated-model token, and the remaining queued jobs were cancelled. The reusable training wrapper was changed to use the current user's `${HOME}/.cache/huggingface`; jobs 227215–227226 then completed.

### ACC predictions initially scored as zero

The ACC generation prompt intentionally requested fenced Python. All 751 Llama outputs and all 751 Gemma outputs contained Markdown fences, but the ACC execution interface expected raw Python. The original evaluation therefore reported 0/751 for both models. Existing predictions were joined to the source execution specs and rescored after stripping fences:

- Llama: 0 → 128/751 (17.04%)
- Gemma: 0 → 557/751 (74.17%)

The permanent fix and regression test are commit `7e218b7`. The one-time rescore programs were discarded after their method and results were recorded here.

### Parameter merge semantics

PEFT `linear` weighted adapter combination mixes A and B matrices separately and introduces cross-adapter products. The experiment switched to `cat`, which preserves the intended weighted sum of complete LoRA deltas. A one-example smoke test preceded the full run.

### Repetition guard

The first full `cat` merge inference stopped at 40 processed rows after detecting a degenerate repeated generation. At the user's direction, the safety guard was removed and inference resumed from existing output until all 751 rows were complete. This allowed measurement but retained the degenerate output as a failure rather than curing generation instability.

## Interpretation

1. **The original evolved roster remains strong because it uses Gemma4-26B-A4B.** Its learned Top-1 result was 72.04%, close to vanilla Gemma's 74.17%. The label-level router selected a known original solver for 72.04% of all rows and 54.02% of contested rows.
2. **The trained Llama LoRA experts did not preserve the original roster competence.** Individual SFT experts clustered between 13.18% and 16.25%, and even the 12-attempt union reached only 25.17%.
3. **Routing is a secondary bottleneck.** Top-1 routing loses 10.52 points relative to the trained-expert union, but fixing routing alone cannot bridge the much larger gap to vanilla Gemma.
4. **Soft parameter merging did not improve over the best baseline.** It slightly exceeded exclusive Top-1 (15.98% vs. 14.65%) but remained below vanilla Llama (17.04%). High label-level solver probability mass did not translate into executable accuracy after merging weak/adapted experts.
5. **Dense SFT traded hard-set gains for common/easy-set losses.** Relative to vanilla Llama it improved All Failed from 3 to 9 solved problems but dropped All Passed from 111 to 95. Problemwise, vanilla-only successes were 57 and dense-only successes were 46, a net loss of 11 problems.
6. **Dense generation stability worsened.** There were 24 dense outputs of at least 4,000 tokenizer tokens versus one for vanilla Llama. These runaway generations likely contributed to the regression.
7. **Token-level SFT loss is not execution pass@1.** The best eval-loss checkpoint need not be the best checkpoint for executable code correctness. Reference imitation can also narrow solution diversity and partially overwrite useful base-model behavior.

## Caveats

- The 751-example test difference between vanilla Llama and dense LoRA is only 11 problems (1.46 percentage points); a paired McNemar approximation gives `p ≈ 0.32`, so this run alone is not decisive evidence that dense adaptation is intrinsically worse.
- The SFT “oracle” uses up to 12 attempts per problem. It is an upper bound on this generated pool, not a fair single-attempt deployment method.
- Original roster labels were generated by Gemma4-26B-A4B personas, whereas the SFT experts used a Llama3.1-8B backbone. The backbone change confounds the effect of SFT and routing.
- Expert-slice columns are overlapping solvability slices, not mutually exclusive semantic task categories.
- The soft merge run deliberately proceeded without a repetition abort guard, so some outputs hit long/degenerate generation behavior.
- Sparse-upcycled MoE did not receive an ACC evaluation in this experiment.
- The reference corpus may contain heterogeneous solution styles and sequences truncated at 3,072 tokens; truncation/noise was not separately quantified.

## Next Steps

1. Evaluate checkpoints 500, 1,000, 1,500, 2,000, and 2,500 using actual pass@1 rather than selecting only by eval loss.
2. Try one epoch and lower LoRA learning rates (`5e-6` or `1e-5`) to reduce forgetting.
3. Add an explicit, symmetric generation stopping/repetition policy to every method and rerun vanilla/dense/expert comparisons under identical decoding.
4. Compare base Llama with the adapter disabled inside the exact same evaluator to exclude remaining inference-configuration differences.
5. Consider KL regularization or mixing general/base demonstrations with specialization data.
6. Train and evaluate experts without a backbone mismatch, or evaluate whether Gemma-compatible adaptation preserves roster competence.
7. Select router/checkpoint jointly against execution pass@1; label-level routing accuracy alone is insufficient.
8. Add the missing ACC sparse-upcycled MoE baseline before producing a final publication table.

## Validation Commands

```bash
pytest -q tests/test_scorer.py
bash -n scripts/sbatch/eval_sft_model.sh
sha256sum -c docs/journals/2026-08-21_acc-seed20211004-sft-routing/configs/source_sha256.txt
git apply --check docs/journals/2026-08-21_acc-seed20211004-sft-routing/patches/acc_sft_training.patch
pytest -q docs/journals/2026-08-21_acc-seed20211004-sft-routing/tests
for script in docs/journals/2026-08-21_acc-seed20211004-sft-routing/scripts/sbatch/*.sh; do bash -n "$script"; done
```

At journal creation, `tests/test_scorer.py` passed all six tests, the evaluation wrapper passed `bash -n`, and commit `7e218b7` contained exactly the scorer, wrapper, and regression-test changes. After the archival moves on 2026-08-25, all recorded SHA-256 checks passed, the patch passed `git apply --check` against the restored working tree, archived launchers passed `bash -n`, and the archived regression tests passed.
