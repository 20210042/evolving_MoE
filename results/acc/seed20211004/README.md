# ACC roster + binning assets for collaborator use

This branch contains the delivery bundle for the final ACC evolution run at seed20211004.

## Contents

- `roster_final.json`: final expert roster after evolution
- `binning_train_full.binned.jsonl`: per-problem train labels, one row per problem
- `binning_train_full.binned.summary.json`: aggregate coverage and union upper bound
- `binning_train_full.binned.agent_solves.json`: per-agent solve index
- `binning_test_full.binned.jsonl`: per-problem test labels
- `binning_test_full.binned.summary.json`: aggregate coverage and union upper bound
- `binning_test_full.binned.agent_solves.json`: per-agent solve index

## What this is for

These assets are intended for downstream LoRA training and label-based routing experiments.

- `roster_final.json` provides the final roster of experts
- `*.binned.jsonl` provides binary solved-by labels for each problem
- `*.binned.summary.json` gives quick coverage statistics
- `*.binned.agent_solves.json` gives the inverse mapping from expert to solved problems

## Summary statistics (train)

- total problems: 11097
- union upper bound: 9297 / 11097 = 83.78%
- all-fail count: 1800 (16.22%)
- all-pass count: 6458 (58.20%)

## Notes

This branch is intentionally limited to the collaborator-facing artifacts and excludes experimental logs, scratch analysis, and in-progress work.
