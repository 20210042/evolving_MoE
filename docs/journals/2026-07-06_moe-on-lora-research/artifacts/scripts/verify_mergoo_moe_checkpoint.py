#!/usr/bin/env python
"""Verify that a mergoo MoE-on-LoRA checkpoint can load and run a short forward pass."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mergoo_compat import patch_mergoo_for_llama31_lora_moe
from train_mergoo_router import repair_lora_moe_gate_dimensions, torch_dtype


def gate_summary(model: torch.nn.Module) -> dict:
    gates = []
    for name, module in model.named_modules():
        gate = getattr(module, "gate", None)
        if isinstance(gate, torch.nn.Linear):
            gates.append(
                {
                    "name": name,
                    "in_features": gate.in_features,
                    "out_features": gate.out_features,
                    "weight_shape": list(gate.weight.shape),
                }
            )
    return {
        "gate_count": len(gates),
        "gate_examples": gates[:20],
        "unique_shapes": sorted({tuple(item["weight_shape"]) for item in gates}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint_dir",
        default="checkpoints/mergoo_lora_moe_5cat_top2_router_trained",
        help="Full mergoo MoE checkpoint directory to verify.",
    )
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--max_seq_length_for_load",
        type=int,
        default=1024,
        help="Temporary load-time max_position_embeddings to avoid mergoo causal-mask OOM.",
    )
    parser.add_argument(
        "--prompt",
        default="Solve: If x + 3 = 7, what is x?",
        help="Short prompt used for a one-step forward smoke test.",
    )
    parser.add_argument("--write_json", default=None)
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {checkpoint_dir}")

    patch_mergoo_for_llama31_lora_moe()

    from mergoo.models.modeling_llama import LlamaForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(checkpoint_dir, trust_remote_code=args.trust_remote_code)
    original_max_position_embeddings = getattr(config, "max_position_embeddings", None)
    if (
        original_max_position_embeddings is not None
        and args.max_seq_length_for_load < original_max_position_embeddings
    ):
        config.max_position_embeddings = args.max_seq_length_for_load
    config.use_cache = False

    model = LlamaForCausalLM.from_pretrained(
        checkpoint_dir,
        config=config,
        torch_dtype=torch_dtype(args.dtype),
        trust_remote_code=args.trust_remote_code,
    )
    model.config.use_cache = False
    repair_summary = repair_lora_moe_gate_dimensions(model)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    messages = [{"role": "user", "content": args.prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_seq_length_for_load)
    encoded = {key: value.to(model.device) for key, value in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)

    logits = outputs.logits
    if not torch.isfinite(logits[:, -1, :]).all():
        raise RuntimeError("forward pass produced non-finite final-token logits")

    summary = {
        "checkpoint_dir": str(checkpoint_dir),
        "device": str(model.device),
        "dtype": str(next(model.parameters()).dtype),
        "original_max_position_embeddings": original_max_position_embeddings,
        "runtime_max_position_embeddings": getattr(model.config, "max_position_embeddings", None),
        "num_experts": getattr(model.config, "num_experts", None),
        "num_experts_per_tok": getattr(model.config, "num_experts_per_tok", None),
        "repair_summary": repair_summary,
        "gate_summary": gate_summary(model),
        "input_tokens": int(encoded["input_ids"].shape[-1]),
        "logits_shape": list(logits.shape),
        "final_token_logits_finite": True,
    }

    if args.write_json:
        path = Path(args.write_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
