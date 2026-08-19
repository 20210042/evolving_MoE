#!/usr/bin/env python
"""Compose LoRA adapters into a mergoo MoE-on-LoRA checkpoint.

This script keeps the original LoRA checkpoints untouched. If an adapter config
was written by a newer PEFT version than the mergoo environment supports, it
creates a sanitized adapter copy under the output directory and points mergoo at
that copy.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import shutil
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftConfig


DEFAULT_EXPERTS = [
    "algebra=checkpoints/sft_llama3_numina_cot_algebra",
    "geometry=checkpoints/sft_llama3_numina_cot_geometry",
]


def parse_expert(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"expert must be NAME=PATH, got: {value!r}"
        )
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("expert name cannot be empty")
    return name, Path(path).expanduser()


def torch_dtype(name: str) -> torch.dtype:
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return dtype_map[name.lower()]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(f"unknown dtype: {name}") from exc


def validate_adapter_dir(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"adapter path does not exist: {path}")
    if not (path / "adapter_config.json").is_file():
        raise FileNotFoundError(f"missing adapter_config.json in {path}")
    if not (
        (path / "adapter_model.safetensors").is_file()
        or (path / "adapter_model.bin").is_file()
    ):
        raise FileNotFoundError(f"missing adapter_model.safetensors/bin in {path}")


def lora_config_allowed_keys() -> set[str]:
    signature = inspect.signature(LoraConfig.__init__)
    return {
        key
        for key, parameter in signature.parameters.items()
        if key != "self"
        and parameter.kind
        in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY)
    }


def normalize_jsonable(value: Any) -> Any:
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def sanitize_adapter_config(src: Path, dst: Path) -> list[str]:
    """Create a PEFT-compatible adapter copy and return dropped config keys."""
    validate_adapter_dir(src)
    dst.mkdir(parents=True, exist_ok=True)

    with (src / "adapter_config.json").open() as f:
        raw_config = json.load(f)

    allowed = lora_config_allowed_keys()
    sanitized = {
        key: normalize_jsonable(value)
        for key, value in raw_config.items()
        if key in allowed
    }
    dropped = sorted(set(raw_config) - set(sanitized))

    with (dst / "adapter_config.json").open("w") as f:
        json.dump(sanitized, f, indent=2, sort_keys=True)
        f.write("\n")

    for filename in ("adapter_model.safetensors", "adapter_model.bin"):
        src_file = src / filename
        if not src_file.exists():
            continue
        dst_file = dst / filename
        if dst_file.exists() or dst_file.is_symlink():
            dst_file.unlink()
        try:
            os.symlink(src_file.resolve(), dst_file)
        except OSError:
            shutil.copy2(src_file, dst_file)
        break

    return dropped


def build_config(args: argparse.Namespace, experts: list[tuple[str, Path]]) -> dict:
    config: dict[str, Any] = {
        "model_type": args.model_type,
        "num_experts_per_tok": args.num_experts_per_tok,
        "base_model": args.base_model,
        "experts": [
            {"expert_name": f"adapter_{name}", "model_id": str(path)}
            for name, path in experts
        ],
    }
    if args.router_layers_index:
        config["router_layers_index"] = args.router_layers_index
    return config


def write_config_snapshot(output_dir: Path, config: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "mergoo_compose_config.json").open("w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compose LoRA adapters into a mergoo MoE-on-LoRA checkpoint."
    )
    parser.add_argument(
        "--base_model",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Base model ID/path used by all LoRA experts.",
    )
    parser.add_argument(
        "--model_type",
        default="llama",
        choices=["llama", "mistral", "bert", "phi3"],
        help="mergoo model_type value.",
    )
    parser.add_argument(
        "--expert",
        action="append",
        type=parse_expert,
        default=None,
        help="Expert adapter as NAME=PATH. Can be repeated.",
    )
    parser.add_argument(
        "--output_dir",
        default="checkpoints/mergoo_lora_moe_algebra_geometry_top1",
        help="Directory for the composed mergoo checkpoint.",
    )
    parser.add_argument(
        "--num_experts_per_tok",
        type=int,
        default=1,
        help="Number of LoRA experts selected per token.",
    )
    parser.add_argument(
        "--router_layers_index",
        type=int,
        nargs="*",
        default=None,
        help="Optional decoder layer indexes that receive MoE routing.",
    )
    parser.add_argument(
        "--dtype",
        type=torch_dtype,
        default=torch.bfloat16,
        help="Model load dtype: bfloat16, float16, or float32.",
    )
    parser.add_argument(
        "--device_map",
        default="auto",
        help="device_map passed to mergoo ComposeExperts.",
    )
    parser.add_argument(
        "--max_shard_size",
        default="9GB",
        help="Maximum checkpoint shard size.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate and write mergoo_compose_config.json without loading the base model.",
    )
    args = parser.parse_args()

    requested_experts = args.expert or [parse_expert(x) for x in DEFAULT_EXPERTS]
    if len(requested_experts) < 2:
        raise ValueError("at least two experts are required for MoE composition")
    if args.num_experts_per_tok < 1 or args.num_experts_per_tok > len(requested_experts):
        raise ValueError("--num_experts_per_tok must be between 1 and number of experts")

    output_dir = Path(args.output_dir)
    sanitized_root = output_dir / "_sanitized_adapters"

    sanitized_experts: list[tuple[str, Path]] = []
    print("Preparing adapters:")
    for name, src in requested_experts:
        src = src.resolve()
        dst = sanitized_root / name
        dropped = sanitize_adapter_config(src, dst)
        config = PeftConfig.from_pretrained(dst)
        print(
            f"  - {name}: {src} -> {dst} "
            f"(r={getattr(config, 'r', '?')}, alpha={getattr(config, 'lora_alpha', '?')}, "
            f"targets={sorted(config.target_modules)})"
        )
        if dropped:
            print(f"    dropped unsupported PEFT config keys: {', '.join(dropped)}")
        sanitized_experts.append((name, dst))

    config = build_config(args, sanitized_experts)
    write_config_snapshot(output_dir, config)
    print(f"Wrote compose config: {output_dir / 'mergoo_compose_config.json'}")

    if args.dry_run:
        print("Dry run complete; base model was not loaded.")
        return

    from mergoo.compose_experts import ComposeExperts

    print("Composing mergoo MoE checkpoint...")
    composer = ComposeExperts(
        config,
        torch_dtype=args.dtype,
        device_map=args.device_map,
        max_shard_size=args.max_shard_size,
    )
    composer.compose()
    composer.save_checkpoint(str(output_dir))
    print(f"Saved composed checkpoint: {output_dir}")


if __name__ == "__main__":
    main()
