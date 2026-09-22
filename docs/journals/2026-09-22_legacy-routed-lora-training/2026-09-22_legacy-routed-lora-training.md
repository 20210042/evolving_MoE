# 2026-09-22 Legacy Routed-LoRA Training Archive Research Journal

## Provenance

- Repository: `/data6/jongbinwon/evolving_MoE`
- Branch: `jb/sparse-upcycling-MoE_260908`
- Source commit before archiving: `8001bc9a6b9c07eaa136b80ac142081b55731513` (`Extract routed LoRA training engine`)
- Archive date: 2026-09-22 (Asia/Seoul)
- Existing archive: `docs/journals/2026-09-21_routed-lora-training-script-archive/` is preserved unchanged; this journal contains the later common-engine-import versions.

## Objective

Archive the one-off SNI and LBox routed-LoRA training entrypoints after the shared training engine was extracted into `src/moe/routed_lora_training.py`. These scripts are retained for audit and reproducibility, but are no longer active entrypoints under `scripts/`.

## Archived artifacts

All snapshots are under `legacy-scripts/` and were moved byte-for-byte from the active `scripts/` paths.

| Original path | Archived path | Bytes | SHA-256 |
|---|---|---:|---|
| `scripts/train_sni_17x2_lora_moe.py` | `legacy-scripts/train_sni_17x2_lora_moe.py` | 11,206 | `84b321bb3ef2e51320a95a2444ccae330be2221db94f313a0cc4dafc432ac759` |
| `scripts/train_lbox_11x2_fallback_lora_moe.py` | `legacy-scripts/train_lbox_11x2_fallback_lora_moe.py` | 14,323 | `afc67187dcaa30423f6744117f585eb404d784a7e5a2744060a77c30f71a5a95` |
| `scripts/train_lbox_11x2_lora_moe.py` | `legacy-scripts/train_lbox_11x2_lora_moe.py` | 8,147 | `405967bafb3a155ad37c5588a8ba12f1145e95f68757b92b5cdbcbc4af672b7e` |
| `scripts/train_lbox_category_prior_8x1_lora_moe.py` | `legacy-scripts/train_lbox_category_prior_8x1_lora_moe.py` | 7,008 | `8cd73deb6250735c02dec04e0b6f88f38efc6cee13f6a647990a32fa6e8fe127` |
| `scripts/train_lbox_task_prior_3x1_lora_moe.py` | `legacy-scripts/train_lbox_task_prior_3x1_lora_moe.py` | 7,227 | `cbb1ecfd2aa4d02c6509ee931b3a778fee5724f39a83fc761058464ba12f4925` |

## Method and dependency change

- The shared completion-only collator, routed-LoRA Trainer, checkpoint handling, gradient verification, and router objectives now live in `src/moe/routed_lora_training.py`.
- The archived snapshots were the working-tree versions that imported the shared engine directly.
- The LBox fallback entrypoint keeps its fallback-specific route and slot objectives in the archived script.
- No model weights, checkpoints, datasets, Slurm logs, or generated exports were copied into this archive.

## Validation

- The five snapshots retain the recorded SHA-256 hashes after the move.
- The common-engine/import validation completed before archiving:
  - focused routed-LoRA/SNI/evaluation tests: 25 passed;
  - all five training modules imported successfully;
  - Python compilation and `git diff --check` passed.
- The archived scripts are historical entrypoints, not the supported production training interface.

## Caveats

- The scripts still contain experiment-specific local checkpoint layouts and optional Hub/W&B publishing flags.
- Their dataset, model revision, and adapter provenance must be checked before any rerun.
- The prior 2026-09-21 archive remains the byte-for-byte record of the pre-common-engine versions.

## Next steps

- Keep `training_histories.md` as the accumulating root-level learning log.
- Review registry tooling and repository configuration separately.
- Add a recipe-driven official training entrypoint if future runs need a maintained CLI.
