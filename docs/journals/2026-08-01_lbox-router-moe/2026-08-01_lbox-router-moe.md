# 2026-08-01 LBox Router MoE Research Journal

## Provenance

- Repository: `/home/jongbinwon/data/evolving_MoE`
- Branch: `jb/lbox_MoE`
- Current commit: `0d3a8314860ef2bb665df22c5238bff4fc18347e`
- Current commit title: `feat(eval): reproduce LBox teacher-roster test coverage`
- Router workflow commit: `78a07fe696ee67f439baba08a3ebe2b83efff9bc`
- Router workflow title: `feat(router): add LBox pre-generation routing workflows`

Relevant source paths:

- `configs/lbox_router/lbox_router_banks.json`
- `configs/lbox_router/lbox_eval_a4b_test_binning.yaml`
- `scripts/lbox_router/README.md`
- `scripts/lbox_router/extract_router_encoder_embeddings.py`
- `scripts/lbox_router/run_lbox_top1_routed_inference.py`
- `scripts/lbox_router/summarize_lbox_router_baselines.py`
- `scripts/lbox_router/train_lbox_router_baseline.py`
- `scripts/lbox_router/train_lbox_task_prior_router.py`
- `scripts/lbox_router/build_roster_from_agent_mapping.py`
- `scripts/sbatch/lbox_router/*.sh`

## Objective

This session tested whether LBox expert routing can recover useful specialization from a Gemma teacher roster after SFT transfer into Llama-3.1-8B LoRA experts.

The main comparisons were:

- Vanilla `meta-llama/Llama-3.1-8B-Instruct`
- Dense LBox SFT model: `Jongbin-kr/llama3_lbox_baseline_eval500`
- Evolved roster MoE: 10 low5 specialists plus one high6 generalist, routed before generation
- Task-prior MoE: civil, criminal, and statute experts, routed before generation

## Artifacts

Copied artifacts are under this journal folder:

- `results/original/router/lbox_router_baseline_20260729_summary.md`
- `results/original/router/lbox_task_prior_router_20260730_metrics.json`
- `results/original/inference/lbox_low5_high6_routed_top1_summary.md`
- `results/original/inference/lbox_low5_high6_routed_top1_metrics.json`
- `results/original/inference/lbox_task_prior_routed_top1_summary.md`
- `results/original/inference/lbox_task_prior_routed_top1_metrics.json`
- `results/original/teacher/lbox_binning_seed20210311_train_summary.json`
- `results/original/teacher/lbox_binning_seed20210311_agent_mapping.csv`
- `results/original/cross_eval/lbox_low5_low7_cross_eval_summary.md`
- `logs/lbox_router/train_lbox_router.216198_[0-3].log`
- `logs/lbox_router/lbox_task_router.216509.log`
- `logs/lbox_router/lbox_low5h6_routed.216576.log`
- `logs/lbox_router/lbox_task_routed.216577.log`
- `logs/lbox_router/lbox_test_roster_binning.216594.log`

Large prediction JSONL files were not copied. They remain in the original `results/` directories and are referenced by the copied metrics.

## Method

Base model for Phase 2 was `meta-llama/Llama-3.1-8B-Instruct`.

Teacher roster annotations came from 10 evolved Gemma personas using `google/gemma-4-26B-A4B-it` with thinking off. For the low5/high6 experiment:

- A specialist LoRA was trained on a problem when `1 <= n_solved <= 5` and that teacher expert solved the problem.
- The generalist LoRA was trained on `n_solved >= 6`.
- `n_solved = 0` problems were excluded from SFT target construction.
- The same problem can appear in multiple specialist SFT sets when multiple teacher experts solved it.

Router feature extraction:

- Input prompt: the same baseline generation prompt used for LBox.
- Encoder: base Llama-3.1-8B-Instruct, adapters off.
- Feature: final-layer hidden states mean-pooled over the prompt tokens, referred to as `hs_mean`.
- Normalization: z-score using statistics from all 46,019 train inputs.

Router architecture and training:

- MLP: `Linear(d,512) -> ReLU -> Dropout(0.3) -> Linear(512,num_experts)`
- Optimizer: AdamW, `lr=1e-3`, `weight_decay=1e-2`
- Epochs: 120
- Batch size: 256
- Roster router loss: multi-label BCE for specialist targets plus the high6 generalist target.
- Task-prior router loss: cross entropy over civil, criminal, and statute task labels.
- E2E inference used router seed 42 and top-1 argmax routing.

## Debugging History

Several Slurm and runtime issues were fixed during the workflow:

- Router scripts were consolidated under `scripts/lbox_router/` and `scripts/sbatch/lbox_router/`.
- Convenience submit wrappers were not committed, because they were run-specific rather than reusable workflow code.
- Hardcoded dated router result directories were replaced with required `ROUTER_DIR` arguments.
- Routed inference was changed to initialize vLLM before CPU-side PyTorch router loading. This avoided the fork/OpenMP interaction that made earlier jobs stall.
- Low5/high6 routed inference succeeded as job `216576`, with final metrics in `results/lbox_routed_top1/20260730_125326`.
- Task-prior routed inference succeeded as job `216577`, with final metrics in `results/lbox_task_prior_routed_top1/20260730_125326`.
- Teacher test binning succeeded as job `216594`, reconstructing the test solve annotation from `agent_mapping.csv`.

Direct `sacct` accounting could not be re-queried in the current session because the escalation request was rejected by the workspace spend policy. Completion status here is inferred from final metrics files and logs ending with vLLM shutdown.

## Results

### Main E2E Test Results

Test split size: 8,203 examples.

| Method | Correct | Overall | Civil (1,766) | Criminal (4,024) | Statute (2,413) |
|---|---:|---:|---:|---:|---:|
| Vanilla Llama-3.1-8B-Instruct | 66 / 8,203 | 0.80% | 0.00% | 1.64% | 0.00% |
| Dense SFT | 6,534 / 8,203 | 79.65% | 66.19% | 87.38% | 76.63% |
| Roster routing: low5 + high6 | 4,482 / 8,203 | 54.64% | 35.33% | 67.72% | 46.95% |
| Task-prior routing: Top-1 | 6,610 / 8,203 | 80.58% | 66.82% | 89.59% | 75.63% |

The task-prior MoE slightly beat dense SFT overall by +0.93 percentage points. The evolved roster MoE underperformed dense SFT by -25.01 percentage points.

### Teacher Roster Coverage

| Split | Total | Gemma teacher roster solved by at least one expert | Never solved |
|---|---:|---:|---:|
| Train | 46,019 | 25,781 (56.02%) | 20,238 (43.98%) |
| Test | 8,203 | 4,559 (55.58%) | 3,644 (44.42%) |

This means the evolved Gemma teacher roster has a hard upper bound near 55.6% on the current test split before any Llama SFT transfer is considered.

### SFT Data Construction

| Group | Unique problems | SFT assignments | Notes |
|---|---:|---:|---|
| low5 specialists | 8,467 | 23,432 | Average 2.77 specialist assignments per selected problem |
| high6 generalist | 17,314 | 17,314 | One generalist assignment per high6 problem |
| Total SFT assignments | 25,781 unique | 40,746 | Duplicate specialist assignments included |

### Router Baseline Heldout Accuracy

These are target-hit rates against routing labels on heldout train data, not pass@1 generation accuracy.

| Bank | Feature | Best single | Top-1 target hit | Top-2 target hit | Oracle |
|---|---|---:|---:|---:|---:|
| low7 + high8 | hs_mean | 53.18% | 75.44% | 86.25% | 100.00% |
| low7 + high8 | encoder | 53.18% | 74.66% | 86.47% | 100.00% |
| low5 + high6 | hs_mean | 67.16% | 79.29% | 87.86% | 100.00% |
| low5 + high6 | encoder | 67.16% | 79.10% | 88.21% | 100.00% |

The low5/high6 router generalized reasonably on label prediction. Its heldout target-hit was about 79%, and earlier test target-hit was about 77.5%.

### Task-Prior Router

| Router | Train examples | Heldout examples | Heldout task accuracy | Test routed E2E |
|---|---:|---:|---:|---:|
| Task-prior hs_mean MLP | 36,816 | 9,203 | 99.88% | 80.58% |

The task-prior router is effectively solving task identification. The E2E gain comes from matching each task to the corresponding task-prior SFT expert.

### Post-SFT Expert Bank Replay

These replay numbers use saved full-test outputs from each Llama expert and simulate routing or oracle selection. They are not a new generation run.

| Split | Generalist only | Top-1 replay | Top-2 union | Top-3 union | Top-5 union | Oracle bank |
|---|---:|---:|---:|---:|---:|---:|
| Train | 49.36% | 25,703 / 46,019 (55.85%) | 60.13% | 61.86% | 63.68% | 30,173 / 46,019 (65.57%) |
| Test | 48.73% | 4,492 / 8,203 (54.76%) | 59.32% | 60.89% | 62.84% | 5,332 / 8,203 (65.00%) |

The actual low5/high6 E2E result was 4,482 / 8,203 (54.64%), close to the replay top-1 value of 4,492 / 8,203 (54.76%).

### Roster Router Selection Analysis on Test

| Expert | Top-1 selected | Top-1 correct | Routed accuracy | SFT train examples |
|---|---:|---:|---:|---:|
| Judicial Precedent Classifier | 282 (3.44%) | 101 | 35.82% | 3,225 |
| Legal Case Typology Architect | 733 (8.94%) | 322 | 43.93% | 3,535 |
| Statutory Element Matcher | 215 (2.62%) | 37 | 17.21% | 1,286 |
| Legal Nomenclature Purist | 320 (3.90%) | 122 | 38.13% | 3,299 |
| Legal Fact Synthesis Engine | 502 (6.12%) | 206 | 41.04% | 2,375 |
| Legal Provision Auditor | 293 (3.57%) | 132 | 45.05% | 1,476 |
| Civil Dispute Taxonomy Expert | 349 (4.25%) | 132 | 37.82% | 2,190 |
| Judicial Labeling Precisionist | 400 (4.88%) | 209 | 52.25% | 3,174 |
| Legal Recidivism Analyst | 105 (1.28%) | 34 | 32.38% | 1,042 |
| Criminal Charge Aggregator | 369 (4.50%) | 171 | 46.34% | 1,830 |
| High6 Generalist | 4,635 (56.50%) | 3,026 | 65.29% | 17,314 |
| Total | 8,203 (100.00%) | 4,492 | 54.76% | 40,746 assignments |

The router sends 56.5% of test examples to the high6 generalist. This is not necessarily a bug: the train label distribution also heavily favors high6 because high6 covers 17,314 of 25,781 teacher-solved unique train problems.

### Unique Solve Recovery

Unique solve means a problem that exactly one post-SFT Llama expert solved when all expert outputs are evaluated on the full test set.

| Expert | Unique solves | Top-1 recovered | Top-2 recovered |
|---|---:|---:|---:|
| Judicial Precedent Classifier | 29 | 3 | 7 |
| Legal Case Typology Architect | 26 | 7 | 12 |
| Statutory Element Matcher | 36 | 8 | 17 |
| Legal Nomenclature Purist | 16 | 2 | 4 |
| Legal Fact Synthesis Engine | 41 | 4 | 13 |
| Legal Provision Auditor | 60 | 14 | 31 |
| Civil Dispute Taxonomy Expert | 9 | 2 | 3 |
| Judicial Labeling Precisionist | 46 | 8 | 15 |
| Legal Recidivism Analyst | 23 | 5 | 6 |
| Criminal Charge Aggregator | 27 | 3 | 9 |
| High6 Generalist | 235 | 208 | 211 |
| Total | 548 | 264 | 328 |

Specialist-only unique solves were 313, but top-1 routing recovered only 56 of them. The generalist had 235 unique solves and top-1 routing recovered 208. This shows the current router objective favors the broad generalist and does not fully exploit specialist complementarity.

### Phase 1 and Phase 2 Routing Table

Some Phase 1 values are from a historical valid split of 7,651 examples, while Phase 2 values here are from the current test split of 8,203 examples. They should not be treated as same-split comparisons.

| Phase | Backbone | Method | Top-1 routing | Top-2 routing | Upper bound |
|---|---|---|---:|---:|---:|
| Phase 1 | Gemma-4-26B-A4B | Evolved Roster, 10 experts, valid split | 39.1% | 47.7% | 56.9% |
| Phase 1 | Gemma-4-26B-A4B | Evolved Roster teacher coverage, test split | Not rerun | Not rerun | 55.58% |
| Phase 1 | Gemma-4-26B-A4B | Task-prior Roster, 3 experts, test split | 43.95% | Not computed | Not computed |
| Phase 1 | Gemma-4-26B-A4B | Vanilla Gemma baseline, valid split | 38.5% | N/A | N/A |
| Phase 2 | Llama-3.1-8B | Evolved MoE, 10 specialists + 1 generalist, test split | 54.64% | 59.32% replay | 65.00% replay |
| Phase 2 | Llama-3.1-8B | Task-prior MoE, 3 experts, test split | 80.58% | 80.62% replay | 80.62% replay |
| Phase 2 | Llama-3.1-8B | Dense SFT, test split | 79.65% | N/A | N/A |
| Phase 2 | Llama-3.1-8B | Vanilla Llama baseline, test split | 0.80% | N/A | N/A |

## Interpretation

The low5/high6 router is not the main bottleneck if measured only as target-label prediction. It reaches roughly 77 to 79% target hit. The larger issue is that the teacher roster coverage itself is only about 55.6% on test, and after Llama SFT the expert bank oracle reaches only 65.0%.

The evolved roster MoE therefore cannot currently compete with dense SFT. The best evidence is:

- Dense SFT: 79.65%
- low5/high6 routed E2E: 54.64%
- low5/high6 post-SFT oracle replay: 65.00%

Even a perfect oracle over the current Llama expert bank would remain far below dense SFT.

The task-prior MoE is much stronger because the task boundary is clean, the router is nearly perfect, and each task-prior expert maps to a real dataset mode. It slightly improves over dense SFT overall, mostly from criminal task gains, while statute drops by about 1 point.

## Caveats

- Phase 1 historical valid numbers and Phase 2 test numbers use different splits. Only same-split values should be used for strict comparison.
- Replay top-k and oracle values are computed from saved expert outputs. They are useful diagnostics, but they are not deployed single-answer generation unless a top-k selection strategy is implemented.
- The low5/high6 SFT assignment count of 40,746 includes duplicate specialist assignments. The unique teacher-solved train problem count is 25,781.
- `n_solved = 0` examples are excluded from roster SFT targets, but they still appear at E2E evaluation time.
- The task-prior phase1 result retains only selected output, so its top-2 and oracle values were not computed here.

## Next Steps

1. Train a router against post-SFT expert correctness, not only teacher roster solve labels. This directly targets the bank that is used at inference time.
2. Add a specialist-aware objective or sampling scheme so unique specialist solves are not hidden by the high6 generalist prior.
3. Evaluate top-2 routed generation or reranking, because replay suggests a test gain from 54.76% to about 59.32%.
4. Keep task-prior MoE as the main positive baseline. It is currently the only MoE setup that beats dense SFT.
5. For evolved roster MoE, report teacher coverage, post-SFT oracle, and unique-solve recovery before presenting E2E accuracy. Those diagnostics explain why the current result is low.
