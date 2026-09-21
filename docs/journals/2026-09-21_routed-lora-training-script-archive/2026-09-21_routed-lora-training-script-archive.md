# 2026-09-21 Routed-LoRA Training Script Archive Research Journal

## Provenance

- Repository: `/data6/jongbinwon/evolving_MoE`
- Branch: `jb/sparse-upcycling-MoE_260908`
- HEAD: `b39932f85a64ca7d7d2d361ac9043e18a3628d68`
- HEAD title: `Archive LBox category SFT job scripts`
- Archive created: 2026-09-21 (Asia/Seoul)

## Objective

Archive the one-off SNI/LBox routed-LoRA training entrypoints while preserving an exact local code snapshot for audit and future migration to one official recipe-driven training entrypoint.

## What `routed_lora_training.py` is intended to contain

The proposed module is the shared training engine, not another experiment definition. It should own the code duplicated across the legacy scripts:

- completion-only route collator;
- supervised router, load-balancing, and router-z losses;
- the routed-LoRA `Trainer` subclass and optimizer parameter groups;
- checkpoint save/load and gradient verification helpers.

Dataset-specific row construction, adapter naming, expert roster, top-k choice, and fallback-specific loss behavior should remain recipe/config concerns.

## Archived scripts

The following uncommitted scripts were copied byte-for-byte to `legacy-scripts/`:

| Original path | Archived path | SHA-256 |
|---|---|---|
| `scripts/train_sni_17x2_lora_moe.py` | `legacy-scripts/train_sni_17x2_lora_moe.py` | `c17552dad46c0d8ec39f0f3b1547330f0c5da902f983e09d3994fc0d7cc5fc01` |
| `scripts/train_lbox_11x2_lora_moe.py` | `legacy-scripts/train_lbox_11x2_lora_moe.py` | `9b60f6d9501fafca678e6ddf7915f96f04048fe6706559c7154e9a2d1ddc817a` |
| `scripts/train_lbox_11x2_fallback_lora_moe.py` | `legacy-scripts/train_lbox_11x2_fallback_lora_moe.py` | `c6a0546547356dc4e168d091104a7992c12c94424abaa21a8343166c5429a02e` |
| `scripts/train_lbox_category_prior_8x1_lora_moe.py` | `legacy-scripts/train_lbox_category_prior_8x1_lora_moe.py` | `115c26a290db8a1f3b19d0170d9fb6e82f24b2f479cb4f0babdde71f2fad1906` |
| `scripts/train_lbox_task_prior_3x1_lora_moe.py` | `legacy-scripts/train_lbox_task_prior_3x1_lora_moe.py` | `c4e8efd6f9fa8d3890ffec1ffb3cd59e66527216c33424a69743c6fdbc63de85` |

The original files remain in place temporarily because the current LBox scripts and `tests/test_routed_lora_ffn.py` import shared symbols from `train_sni_17x2_lora_moe.py`. Removing them before extracting the shared module would break imports.

## Additional experiment scripts moved into this journal

The following experiment-only files were copied byte-for-byte and then removed from the active `scripts/` tree:

- Evaluation entrypoints: `scripts/eval_lbox_routed_lora_moe.py`, `scripts/eval_sni_17x2_lora_moe.py`.
- Slurm launchers under `scripts/sbatch/` (27 files), including the SNI/LBox training, evaluation, smoke, export, analysis, and retry launchers.
- Post-hoc analysis scripts: `scripts/analyze_4x1_router_clustering.py`, `scripts/audit_4x1_router.py`, `scripts/export_sni_4x1_router_groups.py`, `scripts/extract_sni_test_hidden_states.py`, and `scripts/plot_4x1_expert_embedding_tsne.py`.

Their snapshots are under `legacy-scripts/evaluation/`, `legacy-scripts/sbatch/`, and `legacy-scripts/analysis/`. Each snapshot was checked with `cmp` against its source before removal.

The five training entrypoints listed above were intentionally left in `scripts/` for now because they still provide shared imports. They will be removed after the common training module and official recipe-driven entrypoint are implemented.

## Current validation

- The archived copies have matching SHA-256 hashes with their originals.
- The additional evaluation, Slurm, and analysis snapshots match their source files byte-for-byte.
- The current focused routed-LoRA/data/scorer test set passed: 14 tests.
- Python and Slurm syntax checks passed for the SNI/LBox workflow files.

## Next steps

1. Extract the shared training engine into `src/moe/routed_lora_training.py`.
2. Update LBox scripts and tests to import the new module.
3. Add recipe-driven official train/eval entrypoints.
4. Re-run focused tests and then remove the legacy copies from active `scripts/` only after the new entrypoints are verified.
