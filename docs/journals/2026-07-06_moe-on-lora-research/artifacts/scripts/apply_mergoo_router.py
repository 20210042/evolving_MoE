#!/usr/bin/env python
"""Apply a trained mergoo router state to a composed MoE-on-LoRA checkpoint."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from train_mergoo_router import (
    override_router_topk_for_training,
    repair_lora_moe_gate_dimensions,
    torch_dtype,
)
from mergoo_compat import patch_mergoo_for_llama31_lora_moe


def copy_auxiliary_files(input_dir: Path, output_dir: Path) -> None:
    for filename in ("mergoo_compose_config.json",):
        src = input_dir / filename
        if src.exists():
            shutil.copy2(src, output_dir / filename)


def load_router_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"router state not found: {path}")
    return load_file(str(path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_name_or_path",
        default="checkpoints/mergoo_lora_moe_algebra_geometry_top1",
        help="Composed mergoo MoE checkpoint with base and LoRA experts.",
    )
    parser.add_argument(
        "--router_state",
        default="checkpoints/router_smoke_algebra_geometry_top1/router_model.safetensors",
        help="Router-only safetensors file produced by src/train_mergoo_router.py.",
    )
    parser.add_argument(
        "--output_dir",
        default="checkpoints/mergoo_lora_moe_algebra_geometry_top2_router_smoke",
        help="Output directory for the full MoE checkpoint with trained router weights.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        help="Model load dtype: bfloat16, float16, or float32.",
    )
    parser.add_argument(
        "--max_seq_length_for_load",
        type=int,
        default=1024,
        help="Temporary load-time max_position_embeddings to avoid mergoo causal-mask OOM.",
    )
    parser.add_argument(
        "--num_experts_per_tok",
        type=int,
        default=None,
        help="Optional final top-k to store in the model config and layers.",
    )
    parser.add_argument(
        "--max_shard_size",
        default="9GB",
        help="Maximum shard size for save_pretrained.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--patch_llama3_rope",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    model_dir = Path(args.model_name_or_path)
    router_state_path = Path(args.router_state)
    output_dir = Path(args.output_dir)

    if args.patch_llama3_rope:
        patch_mergoo_for_llama31_lora_moe()

    from mergoo.models.modeling_llama import LlamaForCausalLM

    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=args.trust_remote_code)
    original_max_position_embeddings = getattr(config, "max_position_embeddings", None)
    config.use_cache = False
    if (
        original_max_position_embeddings is not None
        and args.max_seq_length_for_load < original_max_position_embeddings
    ):
        config.max_position_embeddings = args.max_seq_length_for_load

    model = LlamaForCausalLM.from_pretrained(
        model_dir,
        config=config,
        torch_dtype=torch_dtype(args.dtype),
        trust_remote_code=args.trust_remote_code,
        ignore_mismatched_sizes=True,
    )
    model.config.use_cache = False

    gate_repair_summary = repair_lora_moe_gate_dimensions(model)
    topk_summary = override_router_topk_for_training(model, args.num_experts_per_tok)

    router_state = load_router_state(router_state_path)
    load_result = model.load_state_dict(router_state, strict=False)
    missing_router_keys = [
        name
        for name, _ in model.named_parameters()
        if ".gate.weight" in name and name not in router_state
    ]
    unexpected_keys = list(load_result.unexpected_keys)
    if unexpected_keys:
        raise RuntimeError(f"unexpected router keys: {unexpected_keys[:20]}")
    if missing_router_keys:
        raise RuntimeError(f"missing router keys: {missing_router_keys[:20]}")

    if original_max_position_embeddings is not None:
        model.config.max_position_embeddings = original_max_position_embeddings
    if args.num_experts_per_tok is not None and hasattr(model.config, "num_experts_per_tok"):
        model.config.num_experts_per_tok = args.num_experts_per_tok

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, max_shard_size=args.max_shard_size)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=args.trust_remote_code)
    tokenizer.save_pretrained(output_dir)
    copy_auxiliary_files(model_dir, output_dir)

    summary = {
        "source_model": str(model_dir),
        "router_state": str(router_state_path),
        "output_dir": str(output_dir),
        "router_key_count": len(router_state),
        "gate_repair": gate_repair_summary,
        "topk": topk_summary,
        "original_max_position_embeddings": original_max_position_embeddings,
        "saved_max_position_embeddings": getattr(model.config, "max_position_embeddings", None),
        "missing_non_router_key_count": len(load_result.missing_keys),
    }
    with (output_dir / "router_apply_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
