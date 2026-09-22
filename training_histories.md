# Training histories

## Current SFT reference settings

| 항목 | 설정 | 출처 |
|---|---|---|
| 방법/로스 | assistant-only SFT, completion-only labels; FFN LoRA (`gate_proj`, `up_proj`, `down_proj`) + full router | historical runtime: `experiments/switch_top1_4x1_router_ffn_lora_20260914/runtime/upcycled_llama/training.py` (removed; exact snapshot unavailable); current equivalent: [`src/moe/routed_lora_training.py`](src/moe/routed_lora_training.py) + [`src/train_sft.py`](src/train_sft.py) |
| 최적화 | AdamW fused, LR `2e-5`, linear scheduler, warmup `3%`, weight decay `0.1` | same launcher |
| 배치/길이 | per-device `1`, grad accumulation `16`, world size `2`, effective global batch `32`, max length `2048` | same launcher |
| 평가/저장 | every `200` steps, `eval_loss` 최소값, `load_best_model_at_end=true`, save limit `2` | same launcher |
| 자원 | `2x PRO6000`, 2 CPU, 40GB RAM, 48h | Slurm launchers |

> Provenance note: The original `runtime/upcycled_llama/training.py` was an uncommitted launch-time file and is not retained byte-for-byte. Historical launcher snapshots are archived in [`docs/journals/2026-09-22_legacy-routed-lora-training/`](docs/journals/2026-09-22_legacy-routed-lora-training/), while run metadata and the recorded `b39932f`/dirty-tree state are preserved in [`docs/journals/2026-09-22-experiment-artifact-provenance/`](docs/journals/2026-09-22-experiment-artifact-provenance/).

## Current GRPO reference settings

| 항목 | 설정 | 출처 |
|---|---|---|
| 상태 | 해당 없음 | 현재 기록된 논리적 run에 GRPO 없음 |

### sni_4x1_switch_top1_router_ffn_lora_20260914

| 항목 | 값 |
|---|---|
| 상태/목적 | COMPLETED; SNI 4x1 Switch-style top-1 router + FFN-LoRA SFT |
| 모델/데이터 | `Jongbin-kr/llama-3.1-8b-instruct-4x1-moe-switch-top1` revision `a52ca51456c7557b0196e0f6ae9f6c1e374200c3`; local SNI snapshot (train 69,542 / validation 8,692) |
| 학습 계약 | 5 epochs / ~10,870 steps; assistant completion-only loss; prompt `definition + 2 positives + instruction -> assistant target` |
| 평가/best | validation `eval_loss`; best checkpoint-4200, `0.3524`; router max load `0.2668` at final eval |
| Slurm/W&B | job `250586` (training/eval reached 100%; wrapper failed only at initial Hub metadata validation); W&B [run](https://wandb.ai/jongbin-kr-skiml_moe/llama-sparse-upcycling-sni/runs/sni4x1-switch-top1-20260914) |
| Hub/artifact | [Hub model](https://huggingface.co/Jongbin-kr/llama-3.1-8b-instruct-4x1-moe-sni-switch-top1-router-ffn-lora-sft-5ep), revision `13df4ac78188059500510cd76598bb0b2f5d900c`; repaired README metadata and verified adapter output; local output deleted after verification |
| Provenance/다음 조치 | branch `jb/sparse-upcycling-MoE_260908`, HEAD `b39932f85a64ca7d7d2d361ac9043e18a3628d68`, dirty snapshot; no retraining required |

### lbox_4x1_switch_top1_router_ffn_lora_20260914

| 항목 | 값 |
|---|---|
| 상태/목적 | SUBMITTED/PENDING; LBox Switch-style top-1 router + FFN-LoRA SFT 재개 (job `253229`) |
| 모델/데이터 | same Switch base revision; LBox local snapshot (train 46,006 / validation 7,648) |
| 학습 계약 | 5 epochs / ~7,190 steps; same completion-only loss and optimizer settings as SNI |
| 재개/원인 | parent `250587`, retry `251364`, `251804`가 CUDA/CUBLAS timeout; latest intact `checkpoint-4200`에서 재개하며 optimizer/scheduler/RNG state 유지 |
| Slurm/W&B | job `253229`, currently `PENDING (Priority)`; log `/data6/jongbinwon/evolving_MoE/logs/lbox_4x1_switch_r3.253229.log`; no node exclusion; same W&B [run](https://wandb.ai/jongbin-kr-skiml_moe/llama-sparse-upcycling-lbox/runs/lbox4x1-switch-top1-20260914) |
| Hub/local | [Hub target](https://huggingface.co/Jongbin-kr/llama-3.1-8b-instruct-4x1-moe-lbox-switch-top1-router-ffn-lora-sft-5ep); output `/data6/jongbinwon/evolving_MoE/checkpoints/lbox_4x1_switch_top1_router_ffn_lora_5ep_retry_20260917`; resume source `/data6/jongbinwon/evolving_MoE/checkpoints/lbox_4x1_switch_top1_router_ffn_lora_5ep_retry_20260916/checkpoint-4200` |
| 코드/다음 조치 | launcher `experiments/switch_top1_4x1_router_ffn_lora_20260914/scripts/train_lbox_switch_top1_4x1_retry_20260917.sh`; `CUDA_LAUNCH_BLOCKING=0`, final README metadata repair included; monitor and verify best Hub push |
