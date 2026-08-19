# MoE Mergoo A6000 Environment

Date: 2026-07-07

This environment is the verified GPU-capable runtime for mergoo MoE-on-LoRA
router training on this lab cluster.

## Environment

| Field | Value |
| --- | --- |
| Conda env | `MoE_mergoo_a6000` |
| Base env | cloned from `MoE_a6000` |
| GPU target | A6000 / CUDA 12.4 driver stack |
| Repository install | `pip install -e . --no-deps` |
| Freeze | `docs/mergoo_a6000_pip_freeze.txt` |

Key verified packages:

```text
mergoo==0.0.10
torch==2.6.0+cu124
transformers==4.43.4
accelerate==0.33.0
peft==0.9.0
trl==0.7.11
tokenizers==0.19.1
datasets==4.4.1
numpy==1.26.4
```

CUDA smoke:

```text
torch 2.6.0+cu124
torch.version.cuda 12.4
torch.cuda.is_available() True
mergoo imports ok
```

## Why This Env Exists

The earlier `MoE_mergoo` environment can import mergoo after the Hugging Face
overlay, but it inherited `torch==2.11.0+cu130`. The local driver exposes CUDA
12.4, so PyTorch reports CUDA unavailable there. `MoE_mergoo_a6000` keeps the
known-good `torch==2.6.0+cu124` stack from `MoE_a6000` and overlays only the
mergoo-compatible Hugging Face packages.

## Router Training Compatibility Notes

Two runtime compatibility patches are used by `src/train_mergoo_router.py`:

- Llama 3.1 RoPE: `src/mergoo_compat.py` patches mergoo's vendored Llama
  attention to use the Transformers Llama rotary embedding implementation.
- Causal mask size: the training script reduces runtime
  `max_position_embeddings` to the requested `max_seq_length` to avoid mergoo
  allocating a full 131072 x 131072 causal mask for Llama 3.1.
- LoRA MoE gate dimensions: mergoo initializes every LoRA router gate with
  `config.hidden_size`; `mlp.down_proj` actually receives
  `config.intermediate_size`, so the training script repairs those gates after
  model load.

For differentiable router training with the current mergoo implementation,
top-1 routing is not enough: top-1 softmax over one selected expert gives a
constant weight of 1 and produces zero router gradients. The smoke script uses
`--router_num_experts_per_tok_for_training 2` for the 2-expert model so gate
gradients are nonzero.
