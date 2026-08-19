# Mergoo Environment Notes

Use a separate Conda environment for mergoo experiments. The regular `MoE`
environment is pinned for the current SFT/evaluation stack, while mergoo 0.0.10
depends on older Hugging Face APIs.

## Why a Separate Environment

The existing `MoE` environment currently has conflicting packages after mergoo
was installed:

| Package | Observed version |
| --- | --- |
| `mergoo` | `0.0.10` |
| `transformers` | `5.9.0` |
| `peft` | `0.19.1` |
| `trl` | `0.29.1` |
| `accelerate` | `0.27.2` |

Observed failures:

```text
ImportError: cannot import name 'clear_device_cache' from accelerate.utils.memory
```

This happens because `peft==0.19.1` expects a newer `accelerate`, while mergoo
pulled the environment back to `accelerate~=0.27.2`.

```text
ImportError: cannot import name 'shard_checkpoint' from transformers.modeling_utils
```

This happens because `mergoo.compose_experts` imports
`transformers.modeling_utils.shard_checkpoint`, which is not available in the
installed `transformers==5.9.0`.

## Create the Mergoo Environment

Preferred cluster setup: clone the existing `MoE` environment so the working
PyTorch/CUDA stack is reused, then overlay the older Hugging Face stack that
mergoo needs.

```bash
conda create -n MoE_mergoo --clone MoE
conda run -n MoE_mergoo python -m pip install -r requirements-mergoo-overlay.txt
conda activate MoE_mergoo
pip install -e ".[dev]"
export PYTHONPATH="$PWD/src"
```

From-scratch setup is recorded in `environment-mergoo.yml`, but it may be much
slower because it has to download and install PyTorch:

```bash
conda env create -f environment-mergoo.yml
conda activate MoE_mergoo
pip install -e ".[dev]"
export PYTHONPATH="$PWD/src"
```

If an old attempt exists, remove it first:

```bash
conda env remove -n MoE_mergoo
```

## Smoke Test

Run this before launching any Slurm job:

```bash
python - <<'PY'
import importlib
from importlib.metadata import version

for pkg in ["mergoo", "transformers", "accelerate", "peft", "trl", "torch"]:
    print(pkg, version(pkg))

for mod in [
    "mergoo.compose_experts",
    "mergoo.models.modeling_llama",
    "mergoo.models.modeling_mistral",
]:
    importlib.import_module(mod)
    print("OK", mod)
PY
```

Expected key versions:

```text
mergoo 0.0.10
transformers 4.38.2
accelerate 0.27.2
peft 0.9.0
trl 0.7.11
```

## Recommended Workflow

Use `MoE` for the existing LoRA SFT and evaluation scripts. Use `MoE_mergoo` for:

1. composing LoRA experts with `mergoo.compose_experts.ComposeExperts`;
2. loading the composed MoE-on-LoRA checkpoint;
3. training only the router/gate parameters.

Do not use `MoE_mergoo` for vLLM inference. Because this environment is cloned
from `MoE`, packages such as `vllm` remain installed, but the overlay pins
`transformers==4.43.4` and `tokenizers==0.19.1`, which are intentionally older
than the versions required by the current vLLM stack.

## Verified State

Created and checked on 2026-07-06:

```text
mergoo 0.0.10
transformers 4.43.4
accelerate 0.27.2
peft 0.9.0
trl 0.7.11
torch 2.11.0
datasets 2.18.0
numpy 1.26.4
OK mergoo.compose_experts
OK mergoo.models.modeling_llama
OK mergoo.models.modeling_mistral
```

The full observed pip package snapshot is recorded in
`docs/mergoo_pip_freeze.txt`.
