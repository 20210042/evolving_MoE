#!/usr/bin/env python
"""Inspect trainable parameters for a mergoo MoE-on-LoRA checkpoint."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoConfig


def dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return mapping[name.lower()]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(f"unknown dtype: {name}") from exc


def apply_rope_compat(config, mode: str) -> None:
    rope_scaling = getattr(config, "rope_scaling", None)
    if not isinstance(rope_scaling, dict):
        return
    if "type" in rope_scaling:
        return
    if rope_scaling.get("rope_type") != "llama3":
        return

    if mode == "error":
        raise ValueError(
            "Checkpoint uses Llama 3.1 rope_scaling, but mergoo 0.0.10 expects "
            "the older {'type', 'factor'} schema. Use --rope_compat linear "
            "for parameter inspection only."
        )
    if mode == "linear":
        factor = float(rope_scaling.get("factor", 1.0))
        config.rope_scaling = {"type": "linear", "factor": factor}
        return
    if mode == "none":
        config.rope_scaling = None
        return
    raise ValueError(f"unknown rope compat mode: {mode}")


def parameter_kind(name: str) -> str:
    if ".gate.weight" in name:
        return "router_gate"
    if ".lora_A." in name or ".lora_B." in name:
        return "lora_expert"
    if ".base_layer." in name:
        return "base_layer"
    if "embed_tokens" in name or "lm_head" in name:
        return "embedding_or_head"
    return "other"


def set_router_only_trainable(model: torch.nn.Module) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = parameter_kind(name) == "router_gate"


def summarize(model: torch.nn.Module, max_names: int) -> dict:
    total_params = 0
    trainable_params = 0
    counts: Counter[str] = Counter()
    trainable_counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = {}

    for name, param in model.named_parameters():
        kind = parameter_kind(name)
        n = param.numel()
        total_params += n
        counts[kind] += n
        if param.requires_grad:
            trainable_params += n
            trainable_counts[kind] += n
            examples.setdefault(kind, [])
            if len(examples[kind]) < max_names:
                examples[kind].append(name)

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_fraction": trainable_params / total_params if total_params else 0.0,
        "params_by_kind": dict(counts),
        "trainable_params_by_kind": dict(trainable_counts),
        "trainable_examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load a mergoo checkpoint and report router-only trainability."
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/mergoo_lora_moe_algebra_geometry_top1",
        help="Path to mergoo MoE-on-LoRA checkpoint.",
    )
    parser.add_argument(
        "--dtype",
        type=dtype_from_name,
        default=torch.bfloat16,
        help="Load dtype: bfloat16, float16, or float32.",
    )
    parser.add_argument(
        "--device_map",
        default="cpu",
        help="device_map for from_pretrained. Use cpu for inspection.",
    )
    parser.add_argument(
        "--rope_compat",
        choices=["linear", "none", "error"],
        default="linear",
        help=(
            "Compatibility mode for Llama 3.1 rope_scaling in mergoo 0.0.10. "
            "'linear' is for parameter inspection only."
        ),
    )
    parser.add_argument(
        "--max_names",
        type=int,
        default=12,
        help="Maximum trainable parameter examples to print per kind.",
    )
    parser.add_argument(
        "--write_json",
        default=None,
        help="Optional output JSON path for the summary.",
    )
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
    original_rope = getattr(config, "rope_scaling", None)
    apply_rope_compat(config, args.rope_compat)

    from mergoo.models.modeling_llama import LlamaForCausalLM

    model = LlamaForCausalLM.from_pretrained(
        checkpoint,
        config=config,
        torch_dtype=args.dtype,
        device_map=args.device_map,
        trust_remote_code=True,
    )
    set_router_only_trainable(model)
    summary = summarize(model, args.max_names)
    summary["checkpoint"] = str(checkpoint)
    summary["dtype"] = str(args.dtype)
    summary["device_map"] = args.device_map
    summary["rope_compat"] = args.rope_compat
    summary["original_rope_scaling"] = original_rope
    summary["effective_rope_scaling"] = getattr(config, "rope_scaling", None)

    print(json.dumps(summary, indent=2))
    if args.write_json:
        output_path = Path(args.write_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")
        print(f"Wrote summary: {output_path}")


if __name__ == "__main__":
    main()
