# 2026-07-06 MoE-on-LoRA Research Journal

## Provenance

| Field | Value |
| --- | --- |
| Repository | `/home/jongbinwon/data/evolving_MoE` |
| Branch | `jb/MoE` |
| HEAD commit | `2320e6b3f154215127cffcb2fb0b5e372b4240fb` |
| Commit title | `update .gitignore to include journals directory  (종빈 연구 기록용)` |
| Session outputs | `environment-mergoo.yml`, `requirements-mergoo-overlay.txt`, `docs/mergoo_environment.md`, `docs/mergoo_pip_freeze.txt` |
| Journal path | `journals/2026-07-06_moe-on-lora-research/` |

This journal folder is intended to collect the broader MoE-on-LoRA research
thread: environment setup, LoRA expert composition, router training, evaluation,
and follow-up debugging. The first entry records the mergoo environment setup.
The mergoo environment files were uncommitted at the time of this entry and
were copied into this journal under `artifacts/` for auditability.

## Objective

Start the MoE-on-LoRA research track using `Leeroo-AI/mergoo`. The immediate
goal for this entry was to prepare a working mergoo environment while preserving
the existing `MoE` environment for current LoRA SFT, evaluation, and vLLM
workflows.

The immediate question was whether the installed `mergoo` package in the
existing `MoE` Conda environment was usable. After diagnosing dependency
conflicts, the objective shifted to creating a separate `MoE_mergoo` environment
with versions compatible with `mergoo==0.0.10`.

## Artifacts

| Artifact | Description |
| --- | --- |
| `artifacts/environment-mergoo.yml` | From-scratch Conda environment record for mergoo experiments. |
| `artifacts/requirements-mergoo-overlay.txt` | Preferred cluster overlay requirements used after cloning `MoE`. |
| `artifacts/mergoo_environment.md` | Operational notes, setup commands, smoke test, and caveats. |
| `artifacts/mergoo_pip_freeze.txt` | Full observed `pip freeze` snapshot from `MoE_mergoo`. |

The original source copies remain in the repository root and `docs/`.

## Entry 1: Mergoo Environment Setup

### Method

Initial inspection showed that the repository was on `main`, with a local
`jb/MoE` branch available. Switching branches required elevated filesystem
permissions because the Git index lock lives under the real `.git` path.

After switching to `jb/MoE`, the existing SFT flow was inspected:

| Path | Role |
| --- | --- |
| `src/train_sft.py` | Existing PEFT LoRA SFT script. |
| `scripts/sbatch/train_sft_by_category.sh` | Category-specific Numina CoT LoRA training. |
| `scripts/sbatch/launch_sft_by_categories.sh` | Multi-category Slurm launcher. |
| `src/data/loader.py` | Dataset/category filtering used by SFT. |

The intended research path is to treat category-specific LoRAs as experts,
compose them with mergoo as a MoE-on-LoRA model, then train the router/gate on
mixed or held-out data.

### Debugging History

The existing `MoE` environment had `mergoo==0.0.10` installed, but the stack was
internally inconsistent.

Observed `MoE` versions:

```text
mergoo 0.0.10
transformers 5.9.0
accelerate 0.27.2
peft 0.19.1
trl 0.29.1
torch 2.11.0
datasets 4.4.1
numpy 2.2.6
```

Observed failures:

```text
ImportError: cannot import name 'clear_device_cache' from accelerate.utils.memory
```

This occurred while importing `peft`. The installed `peft==0.19.1` expected a
newer `accelerate`, while mergoo's declared dependency range had pulled
`accelerate` down to `0.27.2`.

```text
ImportError: cannot import name 'shard_checkpoint' from transformers.modeling_utils
```

This occurred while importing `mergoo.compose_experts`. The installed
`transformers==5.9.0` no longer exposes the older internal API used by
`mergoo==0.0.10`.

Two environment creation strategies were tried:

| Attempt | Outcome |
| --- | --- |
| `conda env create -f environment-mergoo.yml` | Too slow during pip dependency installation because it tried to build/download the full stack, including PyTorch. Interrupted. |
| `conda create -n MoE_mergoo --clone MoE` plus overlay install | Succeeded. Reused the known PyTorch/CUDA stack and downgraded only the Hugging Face packages needed by mergoo. |

The interrupted partial `MoE_mergoo` environment was removed before cloning
again.

### Results

Final verified `MoE_mergoo` versions:

```text
mergoo 0.0.10
transformers 4.38.2
accelerate 0.27.2
peft 0.9.0
trl 0.7.11
torch 2.11.0
datasets 2.18.0
numpy 1.26.4
```

Final smoke test:

```text
OK mergoo.compose_experts
OK mergoo.models.modeling_llama
OK mergoo.models.modeling_mistral
smoke ok
```

The repository was also installed into `MoE_mergoo` with:

```bash
conda run -n MoE_mergoo python -m pip install -e . --no-deps
```

`--no-deps` was important because the repository's normal dependency set could
otherwise cause pip to revisit the incompatible vLLM/Transformers stack.

### Interpretation

The working split is:

| Environment | Intended use |
| --- | --- |
| `MoE` | Existing LoRA SFT, evaluation, and vLLM inference. |
| `MoE_mergoo` | mergoo compose, MoE-on-LoRA checkpoint loading, and router/gate training. |

`MoE_mergoo` should not be used for vLLM inference. Because it was cloned from
`MoE`, `vllm` remains installed, but the overlay pins `transformers==4.43.4` and
`tokenizers==0.19.1`, which are older than current vLLM requirements.

The successful mergoo imports suggest the next blocker is no longer installation
but actual model composition with real category LoRA checkpoints.

### Caveats

- `MoE` is still in a conflicted state from the earlier mergoo install:
  `peft==0.19.1` with `accelerate==0.27.2` is not a valid combination.
- `MoE_mergoo` intentionally contains leftover cloned packages such as `vllm`;
  those packages may report dependency conflicts and should be ignored unless
  needed for mergoo.
- Only import-level smoke tests were run. No full `ComposeExperts.compose()`
  run has been attempted yet.
- `environment-mergoo.yml` records a from-scratch path, but the verified path on
  this cluster is clone plus overlay via `requirements-mergoo-overlay.txt`.

## Next Steps

1. Add a router-only training script, likely `src/train_mergoo_router.py`, by
   reusing the existing dataset and prompt construction from `src/train_sft.py`.
2. Decide whether to patch mergoo's Llama 3.1 RoPE implementation properly
   before real router training, or use the temporary linear RoPE compatibility
   mode only for smoke tests.
3. Restore or repair the original `MoE` environment for existing SFT/eval if
   needed, especially `accelerate>=1.4.0` for `trl==0.29.1` and `peft==0.19.1`.

## Entry 2: Two-Expert Compose Smoke Setup

### Objective

Create the smallest real MoE-on-LoRA composition path using two local Llama 3.1
category LoRA experts:

| Expert | Adapter path |
| --- | --- |
| Algebra | `checkpoints/sft_llama3_numina_cot_algebra` |
| Geometry | `checkpoints/sft_llama3_numina_cot_geometry` |

Both adapters target `meta-llama/Llama-3.1-8B-Instruct` with rank `16`,
`lora_alpha=32`, and the same target modules:

```text
down_proj, gate_proj, k_proj, o_proj, q_proj, up_proj, v_proj
```

### Method

Added:

| Path | Purpose |
| --- | --- |
| `scripts/compose_lora_moe.py` | Builds mergoo compose configs, sanitizes newer PEFT adapter configs for `peft==0.9.0`, and optionally runs `ComposeExperts`. |
| `scripts/sbatch/compose_lora_moe_smoke.sh` | Slurm wrapper for a 1-GPU PRO6000 2-expert compose smoke job. |

The compose script creates sanitized adapter copies under
`<output_dir>/_sanitized_adapters/` and leaves the original LoRA checkpoints
untouched. This is needed because the original adapters were saved by
`peft==0.19.1`, whose `adapter_config.json` contains keys that `peft==0.9.0`
cannot parse.

Dry-run command:

```bash
conda run -n MoE_mergoo python scripts/compose_lora_moe.py \
    --dry_run \
    --output_dir /tmp/mergoo_lora_moe_dryrun
```

Dry-run result:

```text
Preparing adapters:
  - algebra: r=16, alpha=32
  - geometry: r=16, alpha=32
Wrote compose config: /tmp/mergoo_lora_moe_dryrun/mergoo_compose_config.json
Dry run complete; base model was not loaded.
```

Unsupported PEFT 0.19 config keys were dropped only in sanitized copies, not in
the original checkpoints.

### Slurm Submission

First submission:

| Job | State | Reason |
| --- | --- | --- |
| `203838` | `PENDING` | `QOSMaxMemoryPerUser` |

The initial script requested `--mem=96G`, which exceeded the active QOS memory
limit. The job was cancelled and the script was adjusted to `--mem=64G`.

Second submission:

| Job | State at last check | Reason |
| --- | --- | --- |
| `203839` | `FAILED` | Llama 3.1 `rope_scaling` unsupported by `transformers==4.38.2` |

The second job passed resource validation but failed after 15 seconds on node
`n04`. The failure happened before model weights loaded:

```text
ValueError: `rope_scaling` must be a dictionary with with two fields, `type` and `factor`,
got {'factor': 8.0, 'low_freq_factor': 1.0, 'high_freq_factor': 4.0,
'original_max_position_embeddings': 8192, 'rope_type': 'llama3'}
```

`MoE_mergoo` was updated from `transformers==4.38.2` to
`transformers==4.43.4`. This version successfully reads the Llama 3.1
`rope_scaling` config while still exposing the deprecated
`transformers.modeling_utils.shard_checkpoint` API required by mergoo 0.0.10.

Third submission:

| Job | Final state | Runtime | Node |
| --- | --- | --- | --- |
| `203847` | `COMPLETED`, exit code `0:0` | `00:05:50` | `n04` |

Output checkpoint:

```text
checkpoints/mergoo_lora_moe_algebra_geometry_top1
```

Key generated files:

| File | Size |
| --- | ---: |
| `model-00001-of-00002.safetensors` | `8,997,159,384` bytes |
| `model-00002-of-00002.safetensors` | `7,231,282,144` bytes |
| `model.safetensors.index.json` | `107,158` bytes |
| `config.json` | `3,977` bytes |
| `tokenizer.json` | `9,085,657` bytes |

Checkpoint directory size: `16G`.

Generated mergoo config summary:

```text
model_type llama
num_experts 2
num_experts_per_tok 1
router_layers ['gate_proj', 'down_proj', 'v_proj', 'q_proj', 'o_proj', 'k_proj', 'up_proj']
router_layers_index_len 32
adapter_configs 2
```

Log summary:

```text
count_averaged_layers : 67
count_router_layers : 1120
count_total_router_layers : 1120
checkpoint saved at checkpoints/mergoo_lora_moe_algebra_geometry_top1
Saved composed checkpoint: checkpoints/mergoo_lora_moe_algebra_geometry_top1
```

## Entry 3: Router-Only Trainability Inspection

### Objective

Verify that the composed MoE-on-LoRA checkpoint can be loaded and that only the
router/gate parameters are marked trainable.

### Method

Added:

| Path | Purpose |
| --- | --- |
| `scripts/inspect_mergoo_trainable_params.py` | Loads a mergoo checkpoint, freezes all parameters, unfreezes `*.gate.weight`, and writes a JSON summary. |
| `scripts/sbatch/inspect_mergoo_trainable_params.sh` | Slurm wrapper for the trainability inspection. |

The mergoo Llama implementation does not directly support the Llama 3.1
`rope_scaling` schema:

```text
{'factor': 8.0, 'low_freq_factor': 1.0, 'high_freq_factor': 4.0,
 'original_max_position_embeddings': 8192, 'rope_type': 'llama3'}
```

For this inspection only, the script used `--rope_compat linear`, which maps
the config to:

```text
{'type': 'linear', 'factor': 8.0}
```

This is enough to instantiate and inspect parameter names, but should not be
treated as a validated Llama 3.1 runtime for final router training.

### Slurm Result

| Job | Final state | Runtime | Node |
| --- | --- | --- | --- |
| `203858` | `COMPLETED`, exit code `0:0` | `00:00:24` | `n04` |

Output summary:

```text
results/mergoo_trainable_params/algebra_geometry_top1.json
```

Parameter summary:

| Parameter group | Count |
| --- | ---: |
| Total parameters | `8,115,982,336` |
| Trainable parameters | `1,835,008` |
| Trainable fraction | `0.000226098` |
| Trainable group | `router_gate` only |

Breakdown:

```text
embedding_or_head 1,050,673,152
lora_expert          83,886,080
base_layer        6,979,321,856
router_gate           1,835,008
other                   266,240
```

Example trainable parameters:

```text
model.layers.0.self_attn.q_proj.gate.weight
model.layers.0.self_attn.k_proj.gate.weight
model.layers.0.self_attn.v_proj.gate.weight
model.layers.0.self_attn.o_proj.gate.weight
model.layers.0.mlp.gate_proj.gate.weight
model.layers.0.mlp.up_proj.gate.weight
model.layers.0.mlp.down_proj.gate.weight
```

The load log reported that all `*.gate.weight` parameters were newly
initialized rather than loaded from checkpoint. This is expected for router
training: the composed checkpoint carries the base and expert LoRA weights, and
the router gates start from random initialization.

## Entry 4: Router Training Smoke on A6000

### Objective

Verify that the two-expert mergoo MoE-on-LoRA checkpoint can run actual
router-only training on GPU, not just instantiate on CPU. The key success
criterion was nonzero router gradients and a saved router-only state dict.

### Environment Update

The original `MoE_mergoo` environment imports mergoo after the Hugging Face
overlay, but it inherited `torch==2.11.0+cu130`. The local cluster driver stack
is CUDA 12.4, so this environment cannot train on GPU.

A new environment was created by cloning the working A6000 environment and
overlaying the mergoo-compatible stack:

```text
Conda env: MoE_mergoo_a6000
Base env: MoE_a6000
torch: 2.6.0+cu124
transformers: 4.43.4
accelerate: 0.33.0
peft: 0.9.0
trl: 0.7.11
mergoo: 0.0.10
datasets: 4.4.1
CUDA available: True
```

Environment artifacts:

| Artifact | Description |
| --- | --- |
| `artifacts/docs/mergoo_a6000_environment.md` | Verified A6000 mergoo runtime notes. |
| `artifacts/docs/mergoo_a6000_pip_freeze.txt` | Full `pip freeze` from `MoE_mergoo_a6000`. |

### Method

Added and updated:

| Path | Purpose |
| --- | --- |
| `src/mergoo_compat.py` | Runtime patch for mergoo's vendored Llama attention to use Transformers' Llama 3.1 RoPE implementation. |
| `src/train_mergoo_router.py` | Router-only training script using existing Numina CoT data/prompt utilities and `Trainer`. |
| `scripts/sbatch/train_mergoo_router_smoke.sh` | A6000 smoke job for 16 Algebra/Geometry examples. |

Smoke parameters:

```text
model_name_or_path: checkpoints/mergoo_lora_moe_algebra_geometry_top1
categories: Algebra Geometry
max_train_samples: 16
max_eval_samples: 8
max_seq_length: 1024
batch size: 1
learning rate: 1e-3
dtype: bfloat16
report_to: none
save_full_model: false
router_num_experts_per_tok_for_training: 2
```

### Debugging History

| Job | Final state | Diagnosis | Fix |
| --- | --- | --- | --- |
| `204357` | `FAILED` | `Trainer` import failed because `accelerate==0.27.2` lacked `is_mlu_available`. | Upgraded mergoo overlay to `accelerate==0.33.0`. |
| `204359` | `FAILED` | Dataset/cache compatibility issue after earlier package mix. | Restored `datasets==4.4.1` in the mergoo runtime. |
| `204368` | `FAILED` | CUDA OOM: mergoo allocated a full Llama 3.1 131072 x 131072 causal mask. | Reduced runtime `max_position_embeddings` to `max_seq_length` before model construction. |
| `204371` | `FAILED` | `mlp.down_proj` router gate had shape `4096 x 2`, but the layer input is `14336`. | Repaired LoRAMoe gates after load when `gate.in_features != module.in_features`. |
| `204372` | `COMPLETED` | Top-1 router smoke ran, but `grad_norm` was `0.0`. | Identified top-1 hard routing as nondifferentiable in mergoo's implementation. |
| `204373` | `COMPLETED` | Top-2 router smoke ran with nonzero gradients. | Use `--router_num_experts_per_tok_for_training 2` for differentiable 2-expert training. |

The top-1 issue is important. In mergoo's current LoRA MoE forward pass, routing
uses `topk` followed by `softmax` over the selected experts. With
`num_experts_per_tok=1`, the softmax is over a single value and is always `1`,
while the discrete selected expert index is not differentiable. As a result,
router gate gradients are zero even though training appears to run.

### Results

Successful top-2 smoke:

| Job | State | Runtime | Node |
| --- | --- | --- | --- |
| `204373` | `COMPLETED`, exit code `0:0` | `00:00:38` | `n02` |

Training metrics:

```text
train_loss: 0.44189630076289177
train_runtime: 10.7832 seconds
train_samples_per_second: 1.484
train_steps_per_second: 1.484
```

Router parameter summary after dimension repair:

| Parameter group | Count |
| --- | ---: |
| Total parameters | `8,116,637,696` |
| Trainable parameters | `2,490,368` |
| Trainable fraction | `0.000306823` |
| Trainable group | `router_gate` only |

Observed nonzero gradient norms in the successful top-2 run:

```text
0.09375
0.12353515625
0.057373046875
0.052734375
...
```

Saved output:

```text
checkpoints/router_smoke_algebra_geometry_top1/router_model.safetensors
checkpoints/router_smoke_algebra_geometry_top1/router_trainable_summary.json
checkpoints/router_smoke_algebra_geometry_top1/train_results.json
```

Copied journal artifacts:

| Artifact | Description |
| --- | --- |
| `artifacts/logs/train_mergoo_router_smoke.204368.log` | causal-mask OOM failure log. |
| `artifacts/logs/train_mergoo_router_smoke.204371.log` | down-proj gate dimension failure log. |
| `artifacts/logs/train_mergoo_router_smoke.204372.log` | successful top-1 run with zero gradients. |
| `artifacts/logs/train_mergoo_router_smoke.204373.log` | successful top-2 run with nonzero gradients. |
| `artifacts/results/router_smoke_algebra_geometry_top1/` | final router state, train metrics, trainer state, and trainability summary. |
| `artifacts/scripts/train_mergoo_router.py` | copied script snapshot for this smoke run. |
| `artifacts/scripts/train_mergoo_router_smoke.sh` | copied Slurm script snapshot for this smoke run. |
| `artifacts/scripts/mergoo_compat.py` | copied Llama 3.1 RoPE compatibility patch. |

### Interpretation

The system now has a verified end-to-end path for:

1. composing two category LoRA experts into a mergoo MoE-on-LoRA checkpoint;
2. loading it under Llama 3.1-compatible runtime patches;
3. freezing all base and LoRA expert weights;
4. training only router gates on A6000;
5. saving a router-only safetensors checkpoint.

The biggest practical lesson is that top-1 router training is silently
ineffective with the current mergoo LoRA MoE implementation. For router
training, use top-2 or implement a separate differentiable/supervised router
loss. For inference, a later run can still evaluate top-1 routing after the
router has been trained under a differentiable regime.

### Next Steps

1. Promote the smoke into a real router training job with more Numina CoT data,
   still using `MoE_mergoo_a6000`.
2. Decide whether final inference should use top-1 after training or keep top-2
   for all comparisons.
3. Compose the full category expert set once the two-expert training path is
   stable.
4. Add an evaluation script that loads the composed checkpoint plus
   `router_model.safetensors` and compares expert routing behavior by category.

## Entry 5: Full Checkpoint Apply and 5-Expert Job Chain

### Objective

Move from router-only smoke artifacts toward the requested final deliverable:
a complete mergoo MoE-on-LoRA checkpoint directory with trained router weights,
then a five-category expert version ready for Hugging Face Hub upload.

### Added Tooling

| Path | Purpose |
| --- | --- |
| `scripts/apply_mergoo_router.py` | Loads a composed mergoo checkpoint, repairs LoRAMoe gate dimensions, applies `router_model.safetensors`, restores the original Llama 3.1 context length in config, and saves a full checkpoint. |
| `scripts/sbatch/apply_mergoo_router_smoke.sh` | Applies the successful top-2 smoke router to the two-expert composed checkpoint. |
| `scripts/sbatch/train_mergoo_router_algebra_geometry.sh` | Medium Algebra/Geometry router training job using 2048 samples and gradient accumulation. |
| `scripts/sbatch/compose_lora_moe_5cat.sh` | Composes Algebra, Calculus, Combinatorics, Geometry, and Number Theory LoRA experts into a five-expert mergoo checkpoint. |
| `scripts/sbatch/train_mergoo_router_5cat.sh` | Five-category router training job using 5000 Numina CoT samples. |
| `scripts/sbatch/apply_mergoo_router_5cat.sh` | Applies the trained five-category router to produce the full final local checkpoint. |
| `scripts/verify_mergoo_moe_checkpoint.py` | Loads a completed checkpoint, repairs runtime mergoo gate metadata if needed, and runs a short forward pass to verify finite logits. |
| `scripts/sbatch/verify_mergoo_moe_5cat.sh` | Verification job for the final five-category checkpoint after router application. |
| `scripts/upload_mergoo_moe_to_hf.py` | Upload helper for the completed checkpoint directory. Requires `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` and a target `--repo_id`. |
| `scripts/sbatch/upload_mergoo_moe_5cat_to_hf.sh` | Upload wrapper requiring `HF_REPO_ID`; intended to run only after final verification succeeds. |

All new Python scripts passed `py_compile`, and all new Slurm wrappers passed
`bash -n`.

### Submitted Slurm Jobs

| Job | Name | Purpose | State at submission check |
| --- | --- | --- | --- |
| `204378` | `apply_mergoo_router_smoke` | Produce a two-expert full checkpoint with the successful smoke router applied. | `PENDING (Priority)` |
| `204379` | `train_mergoo_router_ag` | Train a stronger two-expert Algebra/Geometry router on 2048 examples. | `PENDING (Priority)` |
| `204380` | `compose_lora_moe_5cat` | Compose the five base category LoRA experts into one mergoo MoE checkpoint. | `PENDING (Priority)` |
| `204383` | `train_mergoo_router_5cat` | Train the five-category router after `204380` succeeds. | `PENDING (Dependency)` |
| `204384` | `apply_mergoo_router_5cat` | Apply the five-category trained router after `204383` succeeds. | `PENDING (Dependency)` |
| `204388` | `verify_mergoo_moe_5cat` | Verify the final full checkpoint after `204384` succeeds. | `PENDING (Dependency)` |

Dependency chain:

```text
204380 compose_lora_moe_5cat
  -> 204383 train_mergoo_router_5cat
      -> 204384 apply_mergoo_router_5cat
          -> 204388 verify_mergoo_moe_5cat
```

### Expected Outputs

| Output | Meaning |
| --- | --- |
| `checkpoints/mergoo_lora_moe_algebra_geometry_top2_router_smoke/` | Full two-expert checkpoint with smoke-trained router applied. |
| `checkpoints/router_algebra_geometry_top2_numina_2k/` | Medium two-expert router-only training output. |
| `checkpoints/mergoo_lora_moe_5cat_top2/` | Full composed five-expert MoE-on-LoRA checkpoint before router training. |
| `checkpoints/router_5cat_top2_numina_5k/` | Five-category router-only training output. |
| `checkpoints/mergoo_lora_moe_5cat_top2_router_trained/` | Final local full five-expert MoE checkpoint with trained router weights. |

### Caveats

- The jobs were still pending at the time of this entry due to scheduler
  priority. No result logs existed yet for `204378` through `204384`.
- Hugging Face upload is not yet complete. The upload script is ready, but the
  final checkpoint must exist first, and the target Hub repo id plus an
  authenticated token are required. A local Hugging Face token file exists at
  the time of this entry, but no `HF_REPO_ID` has been selected yet.
- The current training strategy uses top-2 routing so router gradients are
  nonzero under mergoo's current implementation.

## Entry 6: Final Checkpoint Reload Compatibility Fix

### Objective

Prevent a subtle final-checkpoint failure before the pending Slurm jobs start.
The router training smoke repaired `mlp.down_proj` gates after loading, but a
fully saved checkpoint with corrected `down_proj` gate weights must also be
loadable later for verification and Hugging Face use.

### Issue

mergoo 0.0.10 constructs every `LoRAMoeLayer.gate` with
`config.hidden_size` input features. For Llama 3.1 8B this is `4096`. That is
wrong for `mlp.down_proj`, whose router input is the MLP intermediate size
`14336`.

Earlier training survived by repairing gates after `from_pretrained`. However,
once a final checkpoint is saved with corrected `14336 x num_experts`
`down_proj` gates, a future `from_pretrained` call would instantiate `4096 x
num_experts` gates and fail during state-dict loading before post-load repair
could run.

### Fix

Added `patch_mergoo_lora_moe_gate_dimensions()` and
`patch_mergoo_for_llama31_lora_moe()` to `src/mergoo_compat.py`.

The new patch wraps `mergoo.compose_layers.LoRAMoeLayer.__init__` so the gate is
created with the projection's real `in_features` before checkpoint weights are
loaded. This makes final checkpoints with repaired router gate shapes
loadable.

Updated:

| Path | Change |
| --- | --- |
| `src/mergoo_compat.py` | Adds LoRAMoe gate constructor patch and an all-in-one mergoo patch function. |
| `src/train_mergoo_router.py` | Applies the all-in-one patch before model load and uses `ignore_mismatched_sizes=True` for old composed checkpoints. |
| `scripts/apply_mergoo_router.py` | Applies the same constructor patch before model load and allows old composed router mismatch before trained router state is applied. |
| `scripts/verify_mergoo_moe_checkpoint.py` | Applies the constructor patch before verifying final checkpoint reload and forward pass. |

### Validation

Local syntax checks:

```text
python -m py_compile src/mergoo_compat.py src/train_mergoo_router.py \
  scripts/apply_mergoo_router.py scripts/verify_mergoo_moe_checkpoint.py
```

Runtime import check in `MoE_mergoo_a6000`:

```text
compat patch import ok True
```

The pending Slurm jobs had not started at the time of this entry, so they will
pick up this compatibility fix when they eventually run.

## Entry 7: Router Training Chain Status

### Objective

Move from smoke validation to the first useful router-trained MoE-on-LoRA
checkpoints:

1. Apply the smoke router into a full two-expert Algebra/Geometry checkpoint.
2. Train a medium two-expert Algebra/Geometry router on 2048 Numina CoT samples.
3. Compose the five category LoRA experts into one mergoo MoE checkpoint.
4. Start five-category top-2 router training on 5000 Numina CoT samples.

### Completed Results

| Job | Name | Result |
| --- | --- | --- |
| `204378` | `apply_mergoo_router_smoke` | Completed successfully. Saved a full two-expert checkpoint at `checkpoints/mergoo_lora_moe_algebra_geometry_top2_router_smoke/`. |
| `204379` | `train_mergoo_router_ag` | Completed successfully. Saved router weights at `checkpoints/router_algebra_geometry_top2_numina_2k/router_model.safetensors`. |
| `204380` | `compose_lora_moe_5cat` | Completed successfully in 6 minutes 13 seconds. Saved the five-expert composed checkpoint at `checkpoints/mergoo_lora_moe_5cat_top2/`. |

Two-expert medium router training metrics from job `204379`:

```text
train_loss = 0.43062272295355797
train_runtime = 1367.332 seconds
train_samples_per_second = 1.498
train_steps_per_second = 0.187
```

Nonzero gradient norms were observed throughout the two-expert top-2 run, for
example `0.01239013671875`, `0.0184326171875`, and `0.031982421875`. This
continues to support the earlier conclusion that mergoo router training needs
top-2 routing for meaningful gradient flow in this setup.

Five-expert composition output from job `204380`:

```text
count_averaged_layers : 67
count_router_layers : 2464
count_total_router_layers : 2464
checkpoint saved at checkpoints/mergoo_lora_moe_5cat_top2
```

The five-expert checkpoint was split into two safetensors shards:

```text
model-00001-of-00002.safetensors 8898011632 bytes
model-00002-of-00002.safetensors 7582253888 bytes
model.safetensors.index.json 228274 bytes
```

### Current Running State

At the last Slurm check on 2026-07-07, job `204383`
`train_mergoo_router_5cat` was running on node `n02`.

Five-category router training setup:

```text
categories = Algebra, Calculus, Combinatorics, Geometry, Number Theory
max_train_samples = 5000
max_eval_samples = 256
max_seq_length = 1024
router top-k = 2
optimizer steps = 625
```

The training script confirmed that only router gates are trainable:

```text
total_params = 8246202368
trainable_params = 6225920
trainable_fraction = 0.0007550045126421036
trainable_params_by_kind = {"router_gate": 6225920}
```

The 5-category job reached the training loop and was observed at `32/625`
optimizer steps, around `8.5` seconds per step. This implies roughly 1.5 hours
of router training after warmup/loading, well within the requested 8-hour time
limit.

### Pending Follow-up Jobs

| Job | Name | Dependency / Status |
| --- | --- | --- |
| `204384` | `apply_mergoo_router_5cat` | Pending on successful completion of `204383`. |
| `204388` | `verify_mergoo_moe_5cat` | Pending on successful completion of `204384`. |
| `204397` | `apply_mergoo_router_ag` | Pending with `QOSMaxCpuPerUserLimit`; intended to apply the completed 2k Algebra/Geometry router. |
| `204398` | `verify_mergoo_moe_ag` | Pending on successful completion of `204397`. |

### Notes

- The five-category composed checkpoint currently loads with fresh router gates
  during router training. This is expected because the compatibility patch
  constructs corrected gate dimensions and old composed router weights may not
  match those corrected dimensions.
- Hugging Face upload is still intentionally blocked until the final
  five-category checkpoint is applied and verified. A target `HF_REPO_ID` is
  still needed.
