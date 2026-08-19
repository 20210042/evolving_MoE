# 2026-07-29 LBox Specialist SFT Research Journal

## Provenance

- Repository: `/home/jongbinwon/data/evolving_MoE`
- Branch: `jb/lbox_MoE`
- Commit: `92b19c61a8b448a1808174f1707f9730159409ab` (`fix(sft): push best model to hub after training`)
- Base model: `meta-llama/Llama-3.1-8B-Instruct`
- Training implementation: `src/train_sft.py`
- Training launcher: `scripts/sbatch/train_sft_qasc_lbox_luca.sh`
- Hub verification utility: `scripts/sync_lbox_hub_best_adapters.py`

The low7/high8 and low5/high6 evaluation manifests and one-off evaluation scripts were intentionally removed from the working tree after the experiment. Their copied result artifacts are retained below; original prediction JSONL files remain under `results/`.

## Objective

Compare two roster-derived SFT allocations for ten LBox specialist adapters and one shared/generalist adapter:

- Previous allocation: specialists use solved problems with `n_solved <= 7`; shared model uses `n_solved >= 8`.
- More exclusive allocation: specialists use solved problems with `n_solved <= 5`; generalist uses `n_solved >= 6`.

The key question was whether the more exclusive low5 allocation improves specialist behavior enough to justify its smaller per-expert datasets.

## Artifacts

Copied compact artifacts:

- `results/low7_high8_train/{summary.md,model_metrics.csv}`
- `results/low7_high8_test/{summary.md,model_metrics.csv}`
- `results/low5_high6_train/{summary.md,model_metrics.csv}`
- `results/low5_high6_test/{summary.md,model_metrics.csv}`
- `results/cross_eval/{summary.md,cross_metrics.csv}`

Original result roots:

- `results/lbox_eval_14/train_32k_20260727_172515/`
- `results/lbox_eval_14/20260727_155355/test/`
- `results/lbox_low5_high6_eval/20260728_114325/{train,test}/`
- `results/lbox_low5_low7_cross_eval/20260729_cross/`

## Method

All models used LLaMA 3.1 8B Instruct with LoRA rank 16, alpha 32, dropout 0.05, six epochs, and two PRO6000 GPUs per SFT run. Evaluation and saving ran every 250 steps. Best checkpoint selection used validation `eval_loss` with `load_best_model_at_end=true`.

Full-split evaluations used `prompt_system=baseline` and `temperature=0.0`:

- Train: 46,019 examples (civil 9,377; criminal 21,117; statute 15,525)
- Test: 8,203 examples (civil 1,766; criminal 4,024; statute 2,413)

Low5/high6 SFT jobs were `215214` through `215224`. Full test and train evaluation arrays were `215528` and `215530`; dependent analysis jobs were `215529` and `215531`. The no-inference low5/low7 cross-score CPU job was `216092`.

The cross-score analysis re-used existing full-train prediction JSONL files. For each expert it computed both low5 and low7 adapter accuracy on the identical low5 and low7 roster-derived SFT problem sets.

## Results

### Full-split macro accuracy (%)

| Allocation | Split | Specialist macro | Civil | Criminal | Statute | Shared/generalist |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| low7 / high8 | train | 44.84 | 29.31 | 58.79 | 35.25 | 46.83 |
| low7 / high8 | test | 45.62 | 29.59 | 57.56 | 37.45 | 46.41 |
| low5 / high6 | train | 36.89 | 24.71 | 46.80 | 30.75 | 49.36 |
| low5 / high6 | test | 38.04 | 25.17 | 46.07 | 34.07 | 48.73 |

Relative to low7, low5 specialist macro accuracy changed by -7.95 points on train and -7.58 points on test. The expanded high6 generalist improved by +2.53 points on train and +2.32 points on test relative to high8.

### Specialist cross-score on identical SFT sets

| Evaluation set | low7 model macro | low5 model macro | low5 - low7 |
| --- | ---: | ---: | ---: |
| low7 SFT sets | 81.76 | 71.67 | -10.09 |
| low5 SFT sets | 78.03 | 75.28 | -2.74 |

Low5 was better on its own low5 set for four specialists: Legal Provision Auditor (+4.95), Judicial Labeling Precisionist (+4.69), Legal Recidivism Analyst (+3.36), and Legal Fact Synthesis Engine (+1.77). It was lower for the other six; Civil Dispute Taxonomy Expert had the largest own-low5 drop (-19.00).

### Best checkpoint and Hub verification

Trainer state records best checkpoints by validation loss. Examples: `sft_lbox_roster_c_29934_low5` selected `checkpoint-500` rather than its final step 1212; `sft_lbox_generalist_high6` selected `checkpoint-750` rather than its final step 6498.

The Hub API was queried for the `adapter_model.safetensors` LFS SHA-256 of all 22 roster adapters (11 low7/high8 and 11 low5/high6). Every remote adapter hash matched the corresponding local output-root adapter used for evaluation. No repair upload was required.

## Debugging History

- Test analysis job `215529` initially failed because the CSV writer did not include new manifest metadata fields (`roster_expert_id`, consensus cutoffs). The analyzer was fixed and raw predictions were re-aggregated; no model inference was lost.
- The initial analyzer invocation omitted the train prediction root for own-SFT scoring. Re-running analysis against existing full-train predictions populated the own-SFT metrics without new inference.
- Existing `hub_strategy=every_save` behavior was confirmed from training logs. The committed training change makes this explicit and performs a final blocking Hub push after the best model has been restored.

## Interpretation

The low5 cutoff reduces overlap and is structurally more exclusive, but it reduces specialist SFT set sizes from 2,616-6,836 to 1,042-3,535 examples. With the current six-epoch recipe, that data reduction dominates: low7 is better both on full train/test and on the identical low5 SFT subsets.

The high6 generalist benefits from receiving the additional high-consensus examples. For the next MoE baseline, retain low7 specialists and the stronger high6 generalist only if routing experiments are designed to compare that mixed allocation explicitly.

## Caveats

- Full-split accuracy is based on the baseline prompt, deterministic decoding, and `pass_score >= 100`; it is not a learned-router MoE result.
- Best model selection used validation loss, not LBox train/test pass accuracy.
- The low5/low7 cross-score measures independently trained adapters, not a controlled dataset-size-matched ablation.
- Roster expert names describe evolved roles but do not guarantee task-pure behavior; learned routing and oracle routing must be evaluated separately.

## Next Steps

1. Use low7 specialist adapters as the initial MoE expert bank and compare router oracle and learned-routing accuracy against the high6 generalist.
2. Test owner assignment for `n_solved` 3-5 problems to reduce overlap without shrinking each specialist's training set as aggressively as low5.
3. Add a dataset-size-matched low5 ablation before attributing the low5 degradation solely to consensus cutoff.
4. Keep `hub_strategy=every_save` plus the final best-model Hub push for all subsequent SFT runs.
