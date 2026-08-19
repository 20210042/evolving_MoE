# 2026-07-24 LBox Consensus SFT Research Journal

## Provenance

- Repository: `/home/jongbinwon/data/evolving_MoE`
- Training branch: `jb/lbox_MoE`
- Training commit: `32adf19242d49e8a6dd1770b9c32c200d258a6d6` (`refactor: clarify consensus-based expert SFT modes`)
- Relevant code: `src/train_sft.py`, `scripts/sbatch/train_sft_qasc_lbox_luca.sh`
- Label package: `results/lbox_binning_seed20210311/`
- Base model: `meta-llama/Llama-3.1-8B-Instruct`

The current worktree has one post-run, uncommitted change in `src/train_sft.py`: a help metadata key was changed from `help` to `ㅂhelp`. It was not part of the recorded training commit.

## Objective

Train LBox LoRA experts from evolved-roster solve labels while separating specialist examples from high-consensus examples:

- Ten roster specialists use only examples they solved with `n_solved <= 7`.
- One shared consensus expert uses all examples with `n_solved >= 8`.

The resulting eleven LoRAs are intended for later MoE-LoRA composition and router supervision.

## Artifacts

- `logs/`: copied Slurm stdout/stderr logs for all eleven completed jobs.
- `artifacts/trainer_state/`: copied trainer states, including best validation loss and checkpoint selection.
- `artifacts/adapter_config/`: copied LoRA adapter configurations.
- `artifacts/model_cards/`: copied generated model cards with Hub model IDs and W&B links.
- `config/summary.json`, `config/agent_mapping.csv`: copied roster-label package metadata.

The 11 adapter weight files remain in their original `checkpoints/sft_lbox_*` directories. Each `adapter_model.safetensors` is 167,832,240 bytes; weights were not duplicated into this journal.

## Method

### Data selection

| Role | Data mode | Selection rule | Examples |
|---|---|---|---:|
| `c_29934` | `roster_expert_with_low_consensus_solved` | solved by expert and `n_solved <= 7` | 6,333 |
| `c_28126` | same | same | 6,836 |
| `c_63621` | same | same | 3,368 |
| `c_24222` | same | same | 6,439 |
| `c_47388` | same | same | 4,815 |
| `c_27344` | same | same | 2,616 |
| `c_16504` | same | same | 4,886 |
| `c_31181` | same | same | 5,586 |
| `c_4799` | same | same | 2,993 |
| `c_31573` | same | same | 3,216 |
| `shared_consensus` | `shared_with_high_consensus_solved` | `n_solved >= 8` | 13,712 |

### Training configuration

- 6 epochs; bf16; SDPA attention.
- LoRA: rank 16, alpha 32, dropout 0.05, target `all-linear`.
- 2 PRO6000 GPUs/job; train/eval batch size 2 per device; gradient accumulation 4.
- Learning rate `2e-5`, cosine scheduler.
- Evaluation and checkpoint save every 250 steps; `load_best_model_at_end=true`, selected by `eval_loss`.
- Full LBox validation split was used for evaluation.
- Hub push enabled with `hub_strategy=every_save`.

## Debugging History

The first submission requested 8 CPU cores/job and hit `QOSMaxCpuPerUserLimit`. Jobs `210954` through `210964` were cancelled and replaced with 4 CPU cores/job. The final job set below completed cleanly; no `Traceback`, OOM, timeout, or non-zero exit code was found in the final logs.

## Results

All final jobs completed with Slurm `COMPLETED` and exit code `0:0`.

| Job | Expert | Elapsed | Global step | Best eval loss | Best checkpoint |
|---:|---|---:|---:|---:|---|
| 210965 | `c_29934` | 2:12:46 | 2,376 | 0.5893 | checkpoint-750 |
| 210966 | `c_28126` | 2:23:29 | 2,568 | 0.5831 | checkpoint-500 |
| 210967 | `c_63621` | 1:14:14 | 1,266 | 0.5802 | checkpoint-250 |
| 210968 | `c_24222` | 2:13:18 | 2,418 | 0.5708 | checkpoint-750 |
| 210969 | `c_47388` | 1:44:01 | 1,806 | 0.5720 | checkpoint-250 |
| 210970 | `c_27344` | 0:53:49 | 984 | 0.6080 | checkpoint-500 |
| 210971 | `c_16504` | 1:43:51 | 1,836 | 0.6566 | checkpoint-750 |
| 210972 | `c_31181` | 1:59:58 | 2,100 | 0.7190 | checkpoint-250 |
| 210973 | `c_4799` | 1:01:55 | 1,128 | 0.6595 | checkpoint-250 |
| 210974 | `c_31573` | 1:07:46 | 1,206 | 0.6763 | checkpoint-750 |
| 210975 | `shared_consensus` | 4:13:25 | 5,142 | 0.8275 | checkpoint-750 |

Local model cards and logs identify the intended Hub repositories as `Jongbin-kr/llama3_lbox_roster_<expert_id>` and `Jongbin-kr/llama3_lbox_shared_consensus`. The local run logs confirm push configuration; remote repository contents were not independently queried for this journal.

## Interpretation

- The specialist datasets were successfully restricted to low-consensus examples, and the shared expert received the 13,712 high-consensus examples.
- All runs reached the requested six epochs, but the selected best checkpoints generally occurred early. This is especially pronounced for the shared expert: its best `eval_loss` was 0.8275 at checkpoint 750, while the final logged validation loss was 1.0640 at epoch 6.
- Because `load_best_model_at_end=true`, the root adapter outputs should correspond to the best selected checkpoint rather than the final-step model. This should be verified during downstream composition if checkpoint lineage matters.

## Caveats

- `eval_loss` is language-model loss on the full LBox validation split, not LBox exact-match task accuracy. It cannot establish specialist quality or MoE routing quality on its own.
- The full validation distribution includes examples outside an individual specialist's low-consensus training subset; cross-run loss comparison is therefore only a coarse training-health signal.
- The early best checkpoints indicate that six epochs may be more than necessary for several experts. Subsequent ablations should test fewer epochs or early-stop based on a task metric.
- The journal records the exact training commit, but the working tree has a later uncommitted metadata change as noted above.

## Next Steps

1. Run LBox exact-match evaluation for each local adapter and validate the corresponding Hub checkpoints.
2. Compose the ten specialists plus shared LoRA into the MoE-LoRA checkpoint.
3. Add router supervision: high-consensus examples target shared; low-consensus examples use the roster experts that solved them as multi-positive targets.
4. Compare router-free mixture, learned router, and a simpler task-based three-expert baseline.
